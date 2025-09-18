import json
import boto3
import pymongo
import os

#utility 
newline, bold, unbold = '\n', '\033[1m', '\033[0m'

# to read from the .env file
from dotenv import load_dotenv
load_dotenv()

#MongoDB Credentials

mongo_uri= os.getenv("ATLAS_URI")
print(mongo_uri)

mongo_db= os.getenv("MONGO_DB")
mongo_collection = os.getenv("MONGO_COLLECTION")

#MongoDB Vector Parameters
index_name = os.getenv("MONGO_INDEX")
field_name_to_be_vectorized=os.getenv("FIELD_NAME_TO_BE_VECTORIZED")
vectorized_field_name = os.getenv("VECTORIZED_FIELD_NAME") 

# What you want to search (Semantic Search) in the MongoDB Atlas Collection
search_variable = os.getenv("SEARCH_VARIABLE")


# Model used for Embedding
embedding_endpoint_name=os.getenv("EMBEDDING_ENDPOINT_NAME")


# Connect to the MongoDB database
client = pymongo.MongoClient(mongo_uri)
db = client[mongo_db]
collection = db[mongo_collection]

print("Collection:"+ str(collection))
      
# Get all documents from the collection and start the embedding
documents = collection.find()
print("started processing...")

# Declare the endpoint
newline, bold, unbold = '\n', '\033[1m', '\033[0m'

# Function to call model endpoint with json payload
def query_endpoint_with_json_payload(encoded_json):
    client = boto3.client('runtime.sagemaker')
    # Invoke endpoint with json payload 
    response = client.invoke_endpoint(EndpointName=embedding_endpoint_name, ContentType='application/json', Body=encoded_json)
    return response

# Parse and return model response
def parse_response_multiple_texts(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    return model_predictions

i = 0
# Loop over all documents
for document in documents:

    i += 1
    query = {'_id': document['_id']} 

 ##########################################################################
 # This code to be used if the schema is flat    
    if field_name_to_be_vectorized in document:
        payload = {"inputs": [document[field_name_to_be_vectorized]]}
        query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))
        embeddings = parse_response_multiple_texts(query_response)
        # print("embeddings: " + str(embeddings[0]))

        # Flatten the embedding to ensure it's a simple array
        vector = embeddings[0]
        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
            # If nested, flatten to get the actual vector
            vector = vector[0]
        
        # update the document
        update = {'$set': {vectorized_field_name : vector}}
        collection.update_one(query, update)

    if i % 100 == 0:
        print("processed: " + str(i) + " records")
    
##########################################################################

print("finished processing: " + str(i) + " records")

print(newline + bold+ "Vector index to be created manually. Please ensure vector search index ~ " + index_name + "  ~ is created in MongoDB Atlas "+ unbold + newline)

#Query based on the vector
payload = {"inputs": [search_variable]}
query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))
embeddings = parse_response_multiple_texts(query_response)

# Debug the embeddings structure
print("Raw embeddings:", embeddings)
print("Embeddings type:", type(embeddings))
if isinstance(embeddings, list):
    print("Embeddings length:", len(embeddings))
    if len(embeddings) > 0:
        print("First element type:", type(embeddings[0]))
        print("First element:", embeddings[0])
        if isinstance(embeddings[0], list) and len(embeddings[0]) > 0:
            print("First vector element type:", type(embeddings[0][0]))
            print("First few vector values:", embeddings[0][:5])

# Ensure we have a proper numeric vector
if isinstance(embeddings, list) and len(embeddings) > 0:
    vector_for_search = embeddings[0]
    if isinstance(vector_for_search, list):
        # Convert all elements to float and filter out any non-numeric values
        try:
            vector_for_search = [float(x) for x in vector_for_search if x is not None]
            print("Converted vector length:", len(vector_for_search))
            print("First few converted values:", vector_for_search[:5])
        except (ValueError, TypeError) as e:
            print("Error converting vector elements:", e)
            print("Problematic elements:", [x for x in vector_for_search if not isinstance(x, (int, float))][:5])
            exit(1)
    else:
        print("Vector is not a list, got:", type(vector_for_search))
        exit(1)
else:
    print("Embeddings structure is unexpected")
    exit(1)

# Test the vector search
response = collection.aggregate([
    {
        '$search': {
            'index': index_name, 
            'knnBeta': {
                'vector': vector_for_search, 
                'path': vectorized_field_name, 
                'k': 3,
            }
        }
    }, {
        '$project': {
            'score': {'$meta': 'searchScore'}, 
            field_name_to_be_vectorized : 1
        }
    }
])

print("\n=== Test Search Results ===")
for result in response:
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Score: {result.get('score', 'N/A')}")
    print(f"Plot: {result.get(field_name_to_be_vectorized, 'N/A')[:100]}...")
    print("---")
print("Test completed.")
