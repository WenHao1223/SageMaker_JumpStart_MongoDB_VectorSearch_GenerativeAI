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

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compare_embeddings():
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
    
    # Flatten query embedding
    while isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]
    query_embedding = [float(x) for x in query_embedding]
    
    print(f"Comparing embeddings for query: {query}")
    
    # Get the expected movies and their stored vectors
    expected_movies = ["State Fair", "The Black Pirate", "Chapayev"]
    
    print(f"\n=== Manual Similarity Calculation ===")
    similarities = []
    
    for title in expected_movies:
        doc = collection.find_one({"title": title})
        if doc and 'egVector' in doc:
            stored_vector = doc['egVector']
            
            # Calculate cosine similarity manually
            similarity = cosine_similarity(query_embedding, stored_vector)
            similarities.append((title, similarity, doc.get('year')))
            
            print(f"{title} ({doc.get('year')}): {similarity:.4f}")
            
            # Also check what your current search would return for this specific movie
            # by generating embedding for its fullplot
            fullplot = doc.get('fullplot', '')
            if fullplot:
                movie_embedding = embeddings.embed_query(fullplot[:100])  # First 100 chars
                while isinstance(movie_embedding, list) and len(movie_embedding) > 0 and isinstance(movie_embedding[0], list):
                    movie_embedding = movie_embedding[0]
                movie_embedding = [float(x) for x in movie_embedding]
                
                plot_similarity = cosine_similarity(query_embedding, movie_embedding)
                print(f"  Query vs Movie Plot Embedding: {plot_similarity:.4f}")
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"\n=== Ranked by Manual Similarity ===")
    for title, sim, year in similarities:
        print(f"{title} ({year}): {sim:.4f}")
    
    # Compare with what MongoDB returns
    print(f"\n=== What MongoDB Vector Search Returns ===")
    pipeline = [{
        "$vectorSearch": {
            "index": "vector-index",
            "path": "egVector",
            "queryVector": query_embedding,
            "numCandidates": 150,
            "limit": 10
        }
    }, {
        "$project": {
            "title": 1,
            "year": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }]
    
    results = list(collection.aggregate(pipeline))
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.get('title')} ({r.get('year')}) - MongoDB Score: {r.get('score'):.4f}")
    
    client.close()

if __name__ == "__main__":
    compare_embeddings()