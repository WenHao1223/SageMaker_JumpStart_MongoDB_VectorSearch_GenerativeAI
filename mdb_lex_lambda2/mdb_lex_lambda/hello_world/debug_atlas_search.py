import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
import json

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: dict) -> bytes:
        return json.dumps({"inputs": inputs}).encode("utf-8")

    def transform_output(self, output: bytes) -> list[list[float]]:
        return json.loads(output.read().decode("utf-8"))

def debug_search():
    atlas_uri = os.getenv("ATLAS_URI")
    client = MongoClient(atlas_uri)
    collection = client[os.getenv("MONGO_DB")][os.getenv("MONGO_COLLECTION")]
    
    embeddings = SagemakerEndpointEmbeddings(
        endpoint_name=os.getenv("EMBEDDING_ENDPOINT_NAME"),
        region_name=os.getenv("AWS_REGION1"),
        content_handler=ContentHandler(),
    )
    
    query = "action adventure"
    query_embedding = embeddings.embed_query(query)
    
    # Flatten embedding
    while isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]
    query_embedding = [float(x) for x in query_embedding]
    
    print(f"Debugging search for: {query}")
    
    # Check if the expected movies exist in collection
    expected_titles = ["State Fair", "The Black Pirate", "Chapayev"]
    print(f"\n=== Checking if expected movies exist ===")
    for title in expected_titles:
        doc = collection.find_one({"title": title})
        if doc:
            print(f"✓ Found: {title} ({doc.get('year')})")
            print(f"  Genres: {doc.get('genres', [])}")
            print(f"  Has egVector: {'egVector' in doc}")
            if 'egVector' in doc:
                vector = doc['egVector']
                if isinstance(vector, list):
                    print(f"  Vector length: {len(vector)}")
        else:
            print(f"✗ Not found: {title}")
    
    # Try different index names
    index_names = ["vector-index", "default", "movies-vector-index", "search-index"]
    
    for index_name in index_names:
        print(f"\n=== Testing index: {index_name} ===")
        pipeline = [{
            "$vectorSearch": {
                "index": index_name,
                "path": "egVector",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": 3
            }
        }, {
            "$project": {
                "title": 1,
                "year": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }]
        
        try:
            results = list(collection.aggregate(pipeline))
            if results:
                print(f"SUCCESS with {index_name}:")
                for r in results:
                    print(f"  {r.get('title')} ({r.get('year')}) - {r.get('score'):.4f}")
            else:
                print(f"No results with {index_name}")
        except Exception as e:
            print(f"Failed with {index_name}: {e}")
    
    client.close()

if __name__ == "__main__":
    debug_search()