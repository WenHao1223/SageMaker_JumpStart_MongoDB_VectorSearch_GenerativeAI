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
    embeddings = model_predictions['embedding']
    return embeddings

i = 0
# Loop over all documents
for document in documents:

    i += 1
    query = {'_id': document['_id']} 

 ##########################################################################
 # This code to be used if the schema is flat    
    if field_name_to_be_vectorized in document and vectorized_field_name not in document:
        payload = {"text_inputs": [document[field_name_to_be_vectorized]]}
        query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))
        embeddings = parse_response_multiple_texts(query_response)
        # print("embeddings: " + str(embeddings[0]))

        # update the document
        update = {'$set': {vectorized_field_name :  embeddings[0]}}
        collection.update_one(query, update)

    if i % 100 == 0:
        print("processed: " + str(i) + " records")
    
##########################################################################

print("finished processing: " + str(i) + " records")

print(newline + bold+ "Vector index to be created manually. Please ensure vector search index ~ " + index_name + "  ~ is created in MongoDB Atlas "+ unbold + newline)

#Query based on the vector

payload = {"text_inputs": [search_variable]}
query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))
embeddings = parse_response_multiple_texts(query_response)

#print(newline + "embeddings:" + str(embeddings[0]) + newline)

    
# get the vector search results based on the filter conditions.

response = collection.aggregate([
        {
            '$search': {
                'index': index_name, 
                'knnBeta': {
                    'vector': embeddings[0], 
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



# Result is a list of docs with the array fields
docs = list(response)

print(newline + "Vector Search Criteria:  "+ str(search_variable)+ newline)

#print(newline + "Vector Search Result: "+ str(docs)+ newline)

# Extract an array field from the docs
array_field = [doc[field_name_to_be_vectorized] for doc in docs]

# Join array elements into a string  
llm_input_text = '\n \n'.join(str(elem) for elem in array_field)

print(newline + bold + 'Given Input : ' +  unbold + newline + llm_input_text + newline )