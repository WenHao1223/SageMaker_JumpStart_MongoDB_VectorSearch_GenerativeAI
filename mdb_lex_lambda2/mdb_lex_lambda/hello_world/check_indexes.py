import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def check_indexes():
    atlas_uri = os.getenv("ATLAS_URI")
    client = MongoClient(atlas_uri)
    db = client[os.getenv("MONGO_DB")]
    collection = db[os.getenv("MONGO_COLLECTION")]
    
    print("=== Collection Indexes ===")
    indexes = collection.list_indexes()
    for idx in indexes:
        print(f"Index: {idx}")
    
    print(f"\n=== Environment Variables ===")
    print(f"MONGO_INDEX: {os.getenv('MONGO_INDEX')}")
    print(f"VECTORIZED_FIELD_NAME: {os.getenv('VECTORIZED_FIELD_NAME')}")
    
    # Test if Atlas Search Tester might be using text search
    print(f"\n=== Testing Text Search ===")
    try:
        text_pipeline = [
            {
                "$search": {
                    "index": "default",  # Common text search index name
                    "text": {
                        "query": "action adventure",
                        "path": ["fullplot", "genres", "title"]
                    }
                }
            },
            {
                "$project": {
                    "title": 1,
                    "year": 1,
                    "genres": 1,
                    "score": {"$meta": "searchScore"}
                }
            },
            {"$limit": 3}
        ]
        
        results = list(collection.aggregate(text_pipeline))
        print("Text search results:")
        for r in results:
            print(f"- {r.get('title')} ({r.get('year')}) - Score: {r.get('score', 0):.4f}")
    except Exception as e:
        print(f"Text search failed: {e}")
    
    client.close()

if __name__ == "__main__":
    check_indexes()