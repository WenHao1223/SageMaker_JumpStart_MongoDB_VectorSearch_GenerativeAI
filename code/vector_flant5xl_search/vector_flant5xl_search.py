import json
import boto3
import pymongo
import os

#utility 
newline, bold, unbold = '\n', '\033[1m', '\033[0m'

# to read from the .env file
from dotenv import load_dotenv
#load_dotenv()
load_dotenv('./vector_flant5xl_search.env')

##define the parameters

#MongoDB Credentials
mongo_uri= os.getenv("MONGO_URI")
print(mongo_uri)

mongo_db= os.getenv("MONGO_DB")
mongo_collection = os.getenv("MONGO_COLLECTION")

#MongoDB Vector Parameters
index_name = os.getenv("INDEX_NAME")
field_name_to_be_vectorized=os.getenv("FIELD_NAME_TO_BE_VECTORIZED")
vector_field_name = os.getenv("VECTOR_FIELD_NAME") 

# What you want to search (Semantic Search) in the MongoDB Atlas Collection
search_variable = os.getenv("SEARCH_VARIABLE")

# What you want to ask the model - Chatbot, Translation to MultiLanguage , Text Summary
prompt_question_to_chatbot = os.getenv("PROMPT_QUESTION_TO_CHATBOT")

# Model used for Embedding
embedding_endpoint_name=os.getenv("EMBEDDING_ENDPOINT_NAME")
chatbot_endpoint_name =os.getenv("CHATBOT_ENDPOINT_NAME")

#Parameters for the LLM MOdel
max_length = os.getenv("MAX_LENGTH")
num_return_sequences=os.getenv("NUM_RETURN_SEQUENCES")
top_k= os.getenv("TOP_K")
top_p = os.getenv("TOP_P")
do_sample=os.getenv("DO_SAMPLE")
temperature = os.getenv("TEMPERATURE")

#How you want to filter the data 
add_filter =os.getenv("ADD_FILTER")
filter_clause1 =os.getenv("FILTER_CLAUSE1")
filter_clause2 =os.getenv("FILTER_CLAUSE2")
filter_clause3 =os.getenv("FILTER_CLAUSE3")



# Connect to the MongoDB database
client = pymongo.MongoClient(mongo_uri)
db = client[mongo_db]
collection = db[mongo_collection]


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
#    # This code is to be used if the schema is nested.
#     # Loop over all documents
#     if "claims" in document :
#             #print("claims is there in the document")
           
#             # Assuming "claims" is an array of objects
#             for claim in document["claims"]:
#                 if "claimant_summary" in claim and vector_field_name not in document:
#                     #print("claimant_summary is there in the document")

#                     payload = {"text_inputs": [claim["claimant_summary"]]}
#                     query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))
#                     embeddings = parse_response_multiple_texts(query_response)

    
#             if i % 1 == 0:
#                 print("processed: " + str(i) + " records")
    
    
##########################################################################    
 # This code to be used if the schema is flat    
    if field_name_to_be_vectorized in document and vector_field_name not in document:
        payload = {"text_inputs": [document[field_name_to_be_vectorized]]}
        query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))
        embeddings = parse_response_multiple_texts(query_response)
        # print("embeddings: " + str(embeddings[0]))




        # update the document
        update = {'$set': {vector_field_name :  embeddings[0]}}
        collection.update_one(query, update)

    if i % 1000 == 0:
        print("processed: " + str(i) + " records")
    
    # if i > 200:
    #     break
##########################################################################

print("finished processing: " + str(i) + " records")

##########################################################################
#Check if index already exists, if not create the vector index

# index_info = collection.list_indexes()
# for index in index_info:
#   if 'spec' in index and 'knnVector' in index['spec']:
#     print(index['name'])

# if index_name in index_info:
#     print("Index already exists!")
# else:
#     index_def = { "mappings": { "dynamic": true, "fields": { vector_field_name : { "dimensions": 384, "similarity": "euclidean", "type": "knnVector"} } } }
#     # Create vector index
#     #collection.create_search_index(index_def, name=index_name)
##########################################################################


print(newline + bold+ "Vector index to be created manually. Please ensure vector search index ~ " + index_name + "  ~ is created in MongoDB Atlas "+ unbold + newline)

#Query based on the vector

payload = {"text_inputs": [search_variable]}
query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))
embeddings = parse_response_multiple_texts(query_response)

#print(newline + "embeddings:" + str(embeddings[0]) + newline)

    
# get the vector search results based on the filter conditions.
if add_filter == "Y": 
    response = collection.aggregate([
        {
            '$search': {
                'index': index_name, 
                'knnBeta': {
                    'vector': embeddings[0], 
                    'path': vector_field_name,
                    'filter': {filter_clause1: { 'path': filter_clause2,'query': filter_clause3}}, 
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
else:
    response = collection.aggregate([
        {
            '$search': {
                'index': index_name, 
                'knnBeta': {
                    'vector': embeddings[0], 
                    'path': vector_field_name, 
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


#function to convert to boolean
def str_to_bool(input_str):
    if input_str.lower() in ["true", "yes", "1", "on"]:
        return True
    elif input_str.lower() in ["false", "no", "0", "off"]:
        return False
    else:
        raise ValueError("Invalid boolean string")
    
# Parameters for the LLM model
parameters = {
    "max_length" : int(max_length),
    "num_return_sequences": int(num_return_sequences),
    "top_k": int(top_k),
    "top_p": float(top_p),
    "do_sample": str_to_bool(do_sample),
    "temperature": float(temperature),
}

# Create the payload for the model
payload = {"text_inputs": prompt_question_to_chatbot + ":" + llm_input_text, **parameters}

#function to build the payload
def query_endpoint_with_json_payload(encoded_json):
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=chatbot_endpoint_name, ContentType='application/json', Body=encoded_json)
    return response

# build the query response
query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))

#parse the query response to the LLM model
def parse_response_multiple_texts(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    generated_text = model_predictions['generated_texts']
    return generated_text

#output
generated_texts = parse_response_multiple_texts(query_response)
print(bold +'Query : '+ unbold + prompt_question_to_chatbot + newline )
print(bold + 'LLM generated Output : '+ unbold + str(generated_texts) + newline)

