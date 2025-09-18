import os
import sys
import pymongo
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv('../.env')

def test_vectorization_status():
    mongo_uri = os.getenv("ATLAS_URI")
    client = pymongo.MongoClient(mongo_uri)
    collection = client["my_mflix"]["movies"]
    
    total_docs = collection.count_documents({})
    vectorized_docs = collection.count_documents({"egVector": {"$exists": True}})
    
    print(f"Total documents: {total_docs}")
    print(f"Vectorized documents: {vectorized_docs}")
    print(f"Progress: {vectorized_docs/total_docs*100:.1f}%")
    
    if vectorized_docs > 0:
        sample = collection.find_one({"egVector": {"$exists": True}})
        vector = sample["egVector"]
        print(f"Vector dimensions: {len(vector)}")
        print(f"Sample movie: {sample.get('title', 'Unknown')}")
        return True
    return False

def test_search():
    from mongodb_retriever import MDBContextRetriever
    
    atlas_uri = os.getenv("ATLAS_URI")
    retriever = MDBContextRetriever(atlas_uri, k=3)
    
    test_queries = ["greedy tycoon", "love story", "action adventure"]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        docs = retriever._get_relevant_documents(query)
        
        if docs:
            for i, doc in enumerate(docs, 1):
                print(f"{i}. {doc.metadata.get('title', 'No title')}")
                print(f"   Score: {doc.metadata.get('score', 'N/A')}")
                print(f"   Plot: {doc.page_content[:100]}...")
        else:
            print("No results found")

if __name__ == "__main__":
    print("=== System Status Check ===")
    if test_vectorization_status():
        print("\n=== Testing Search ===")
        test_search()
    else:
        print("❌ No vectorized documents found.")