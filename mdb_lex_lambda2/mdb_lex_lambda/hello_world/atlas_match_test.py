import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
import json

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: dict) -> bytes:
        payload = {"inputs": inputs}
        return json.dumps(payload).encode("utf-8")

    def transform_output(self, output: bytes) -> list[list[float]]:
        return json.loads(output.read().decode("utf-8"))

def test_atlas_match():
    # Setup
    atlas_uri = os.getenv("ATLAS_URI")
    client = MongoClient(atlas_uri)
    db = client[os.getenv("MONGO_DB")]
    collection = db[os.getenv("MONGO_COLLECTION")]
    
    # Setup embeddings
    embeddings = SagemakerEndpointEmbeddings(
        endpoint_name=os.getenv("EMBEDDING_ENDPOINT_NAME"),
        region_name=os.getenv("AWS_REGION1"),
        content_handler=ContentHandler(),
    )
    
    # Test query
    query = "action adventure"
    query_embedding = embeddings.embed_query(query)
    
    # Debug and flatten embedding completely
    print(f"Raw embedding type: {type(query_embedding)}")
    print(f"Raw embedding structure: {query_embedding[:2] if len(query_embedding) > 0 else query_embedding}")
    
    # Flatten completely nested arrays
    def flatten_completely(embedding):
        while isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = embedding[0]
        return embedding
    
    query_embedding = flatten_completely(query_embedding)
    
    # Ensure it's a flat list of numbers
    if isinstance(query_embedding, list) and all(isinstance(x, (int, float)) for x in query_embedding):
        query_embedding = [float(x) for x in query_embedding]
    else:
        print(f"Error: embedding is not a flat list of numbers: {type(query_embedding[0]) if query_embedding else 'empty'}")
        return
    
    print(f"Query: {query}")
    print(f"Final embedding dimensions: {len(query_embedding)}")
    print(f"First few values: {query_embedding[:5]}")
    
    # Exact Atlas Search Tester pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": os.getenv("MONGO_INDEX"),
                "path": os.getenv("VECTORIZED_FIELD_NAME"),
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": 10
            }
        },
        {
            "$project": {
                "_id": 1,
                "title": 1,
                "fullplot": 1,
                "genres": 1,
                "year": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('title', 'No title')} ({result.get('year', 'N/A')})")
            print(f"   Score: {result.get('score', 0):.4f}")
            print(f"   Genres: {result.get('genres', [])}")
            print()
    except Exception as e:
        print(f"Error: {e}")
    
    client.close()

if __name__ == "__main__":
    test_atlas_match()