import os
import numpy as np
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

def normalize_vector(vector):
    """Normalize vector to unit length"""
    norm = np.linalg.norm(vector)
    return (vector / norm).tolist() if norm > 0 else vector

def test_normalization():
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
    
    print(f"Testing normalization for: {query}")
    print(f"Original embedding norm: {np.linalg.norm(query_embedding):.4f}")
    
    # Test with and without normalization
    test_cases = [
        ("Original", query_embedding),
        ("Normalized", normalize_vector(query_embedding))
    ]
    
    for name, embedding in test_cases:
        print(f"\n=== {name} embedding ===")
        if name == "Normalized":
            print(f"Normalized embedding norm: {np.linalg.norm(embedding):.4f}")
        
        pipeline = [{
            "$vectorSearch": {
                "index": os.getenv("MONGO_INDEX"),
                "path": os.getenv("VECTORIZED_FIELD_NAME"),
                "queryVector": embedding,
                "numCandidates": 150,
                "limit": 5
            }
        }, {
            "$project": {
                "title": 1,
                "year": 1,
                "genres": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }]
        
        try:
            results = list(collection.aggregate(pipeline))
            for i, r in enumerate(results, 1):
                print(f"{i}. {r.get('title')} ({r.get('year')}) - {r.get('score'):.4f}")
        except Exception as e:
            print(f"Error: {e}")
    
    client.close()

if __name__ == "__main__":
    test_normalization()