import pymongo
import os
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("ATLAS_URI")
mongo_db = os.getenv("MONGO_DB")
mongo_collection = os.getenv("MONGO_COLLECTION")
vectorized_field_name = os.getenv("VECTORIZED_FIELD_NAME")

client = pymongo.MongoClient(mongo_uri)
collection = client[mongo_db][mongo_collection]

# Remove all existing egVector fields
result = collection.update_many({}, {"$unset": {vectorized_field_name: ""}})
print(f"Removed {vectorized_field_name} from {result.modified_count} documents")

# Verify removal
count = collection.count_documents({vectorized_field_name: {"$exists": True}})
print(f"Documents with {vectorized_field_name}: {count}")
print("Ready for re-vectorization")