from langchain.schema import BaseRetriever, Document
from pymongo import MongoClient
from pymongo.collection import Collection
from typing import Dict, List, Optional
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# retrieve the variables from template.yaml
mongo_db = os.environ["MONGO_DB"]
aws_region = os.environ["AWS_REGION1"]
embedding_endpoint_name = os.environ["EMBEDDING_ENDPOINT_NAME"]
mongo_collection = os.environ["MONGO_COLLECTION"]
mongo_index = os.environ["MONGO_INDEX"]

print("MongoDB : " + str(mongo_db))

class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        payload = {"inputs": inputs}
        input_str = json.dumps(payload)
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json


content_handler = ContentHandler()


embeddings = SagemakerEndpointEmbeddings(
    endpoint_name=embedding_endpoint_name,
    region_name=aws_region,
    content_handler=content_handler,
)

class MDBContextRetriever(BaseRetriever):
    """Retriever to retrieve documents from MongoDB using Vector index."""

    k: int = 2
    return_source_documents: bool = False
    client: Optional[MongoClient] = None
    collection: Optional[Collection] = None
    embeddings: Optional[SagemakerEndpointEmbeddings] = None

    def __init__(self, mongodb_uri, k=2, return_source_documents=False):
        super().__init__()
        self.k = k
        self.return_source_documents = return_source_documents
        self.client = MongoClient(mongodb_uri)
        self.collection = self.client[mongo_db][mongo_collection]
        self.embeddings = embeddings

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Run search in MongoDB Atlas and get top k documents"""
        # Check if collection has documents
        doc_count = self.collection.count_documents({})
        print(f"Collection has {doc_count} documents")
        
        if doc_count == 0:
            print("No documents in collection")
            return []
        
        # Generate embedding for the query
        query_embedding = self.embeddings.embed_query(query)
        
        # Debug the embedding structure
        # print(f"Raw embedding type: {type(query_embedding)}")
        # print(f"Raw embedding: {query_embedding[:2] if hasattr(query_embedding, '__len__') else query_embedding}")
        
        # Handle nested array structures - flatten completely
        def flatten_embedding(embedding):
            if isinstance(embedding, list):
                if len(embedding) == 1 and isinstance(embedding[0], list):
                    return flatten_embedding(embedding[0])
                elif all(isinstance(x, (int, float)) for x in embedding):
                    return embedding
                elif len(embedding) > 0 and isinstance(embedding[0], list):
                    return flatten_embedding(embedding[0])
            return embedding
        
        query_embedding = flatten_embedding(query_embedding)
        
        # Convert all elements to float
        try:
            if not isinstance(query_embedding, list):
                return self._simple_search(query)
            query_embedding = [float(x) for x in query_embedding]
            print(f"Query embedding length: {len(query_embedding)}")
            
            if len(query_embedding) == 0:
                return self._simple_search(query)
                
        except (ValueError, TypeError) as e:
            print(f"Error converting embedding to float: {e}")
            return self._simple_search(query)
        
        # Use exact $vectorSearch syntax matching MongoDB Atlas Search Tester
        pipeline = [
            {
                "$vectorSearch": {
                    "index": mongo_index,
                    "path": os.getenv("VECTORIZED_FIELD_NAME"),
                    "queryVector": query_embedding,
                    "numCandidates": 150,  # Match Atlas Search Tester default
                    "limit": self.k,
                    "filter": {}  # Empty filter like Atlas Search Tester
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "fullplot": 1,
                    "title": 1,
                    "genres": 1,
                    "cast": 1,
                    "year": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            print(f"Using index: {mongo_index}")
            # print(f"Pipeline: {pipeline}")
            results = list(self.collection.aggregate(pipeline))
            docs = []
            for result in results:
                raw_score = result.get("score", 0)
                print(f"Raw result: {result.get('title')} - Score: {raw_score}")
                doc = Document(
                    page_content=result.get("fullplot", ""),
                    metadata={
                        "title": result.get("title", ""),
                        "score": raw_score,
                        "_id": str(result.get("_id", ""))
                    }
                )
                docs.append(doc)
            print(f"Vector search found {len(docs)} documents")
            if len(docs) == 0:
                print("Vector search returned 0 results")
                return self._simple_search(query)
            return docs
        except Exception as e:
            print(f"Vector search failed: {e}")
            print("Trying fallback knnBeta syntax...")
            # Fallback to knnBeta
            fallback_pipeline = [
                {
                    "$search": {
                        "index": mongo_index,
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": os.getenv("VECTORIZED_FIELD_NAME"),
                            "k": self.k
                        }
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "fullplot": 1,
                        "title": 1,
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]
            try:
                results = list(self.collection.aggregate(fallback_pipeline))
                docs = []
                for result in results:
                    raw_score = result.get("score", 0)
                    print(f"Fallback result: {result.get('title')} - Score: {raw_score}")
                    doc = Document(
                        page_content=result.get("fullplot", ""),
                        metadata={
                            "title": result.get("title", ""),
                            "score": raw_score,
                            "_id": str(result.get("_id", ""))
                        }
                    )
                    docs.append(doc)
                return docs
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return self._simple_search(query)

    def _simple_search(self, query: str) -> List[Document]:
        """Simple search fallback"""
        try:
            # Try different search strategies
            search_terms = query.lower().split()
            
            # First try: search for any of the terms in fullplot
            or_conditions = [{"fullplot": {"$regex": term, "$options": "i"}} for term in search_terms]
            results = list(self.collection.find(
                {"$or": or_conditions}
            ).limit(self.k))
            
            # If no results, try searching in other fields
            if not results:
                or_conditions = [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"genres": {"$regex": query, "$options": "i"}},
                    {"plot": {"$regex": query, "$options": "i"}}
                ]
                results = list(self.collection.find(
                    {"$or": or_conditions}
                ).limit(self.k))
            
            # If still no results, just get any documents
            if not results:
                results = list(self.collection.find().limit(self.k))
            
            docs = []
            for result in results:
                doc = Document(
                    page_content=result.get("fullplot", result.get("plot", "")),
                    metadata={
                        "title": result.get("title", ""),
                        "score": 1.0,
                        "_id": str(result.get("_id", ""))
                    }
                )
                docs.append(doc)
                print(f"Found: {result.get('title', 'No title')} - {result.get('fullplot', '')[:100]}...")
            print(f"Fallback search found {len(docs)} documents")
            return docs
        except Exception as e:
            print(f"Fallback search failed: {e}")
            return []

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


if __name__ == "__main__":
    # Test the retriever
    atlas_uri = os.environ.get("ATLAS_URI")
    if atlas_uri:
        retriever = MDBContextRetriever(atlas_uri)
        docs = retriever._get_relevant_documents("action movie")
        print(f"Found {len(docs)} documents")
    else:
        print("ATLAS_URI not found in environment variables")

