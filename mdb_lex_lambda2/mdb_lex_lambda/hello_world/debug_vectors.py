import os
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

mongo_uri = os.getenv("ATLAS_URI")
mongo_db = os.getenv("MONGO_DB")
mongo_collection = os.getenv("MONGO_COLLECTION")

# Connect to MongoDB
client = pymongo.MongoClient(mongo_uri)
db = client[mongo_db]
collection = db[mongo_collection]

# Check a few documents to see the vector format
print("Checking vector format in database...")
docs = list(collection.find({"egVector": {"$exists": True}}).limit(3))

for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}:")
    print(f"Title: {doc.get('title', 'N/A')}")
    
    eg_vector = doc.get('egVector')
    if eg_vector:
        print(f"Vector type: {type(eg_vector)}")
        print(f"Vector length: {len(eg_vector) if hasattr(eg_vector, '__len__') else 'N/A'}")
        
        if isinstance(eg_vector, list):
            if len(eg_vector) > 0:
                print(f"First element type: {type(eg_vector[0])}")
                print(f"First few values: {eg_vector[:5]}")
                
                # Check if it's a nested array
                if isinstance(eg_vector[0], list):
                    print("WARNING: Vector appears to be nested array!")
                    print(f"Inner array length: {len(eg_vector[0])}")
                    print(f"Inner array first few values: {eg_vector[0][:5]}")
        else:
            print(f"Vector value: {eg_vector}")

print(f"\nTotal documents with egVector: {collection.count_documents({'egVector': {'$exists': True}})}")