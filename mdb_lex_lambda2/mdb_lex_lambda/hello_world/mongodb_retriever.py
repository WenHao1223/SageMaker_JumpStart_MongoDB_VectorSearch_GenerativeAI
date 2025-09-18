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
        """Hybrid search: keyword search first, then semantic search"""
        doc_count = self.collection.count_documents({})
        print(f"\n{'='*60}")
        print(f"🔍 HYBRID SEARCH STARTED")
        print(f"Query: '{query}'")
        print(f"Collection has {doc_count} documents")
        print(f"{'='*60}")
        
        if doc_count == 0:
            return []
        
        # Step 1: Try keyword search first
        print(f"\n📝 STEP 1: KEYWORD SEARCH")
        print(f"{'-'*40}")
        keyword_docs = self._keyword_search(query)
        if keyword_docs:
            print(f"✅ Keyword search SUCCESS: {len(keyword_docs)} documents found")
            return keyword_docs
        
        # Step 2: Fall back to semantic search
        print(f"❌ Keyword search returned 0 results")
        print(f"\n🧠 STEP 2: SEMANTIC SEARCH")
        print(f"{'-'*40}")
        return self._semantic_search(query)
    
    def _keyword_search(self, query: str) -> List[Document]:
        """MongoDB Atlas text search"""
        try:
            pipeline = [{
                "$search": {
                    "index": mongo_index,
                    "text": {
                        "query": query,
                        "path": {"wildcard": "*"}
                    }
                }
            }, {
                "$project": {
                    "_id": 1,
                    "fullplot": 1,
                    "title": 1,
                    "genres": 1,
                    "cast": 1,
                    "year": 1,
                    "score": {"$meta": "searchScore"}
                }
            }, {
                "$limit": self.k
            }]
            
            print(f"Using MongoDB Atlas Text Search with index: {mongo_index}")
            results = list(self.collection.aggregate(pipeline))
            docs = []
            for i, result in enumerate(results, 1):
                score = result.get("score", 0)
                print(f"  📄 #{i} [{score:.4f}] {result.get('title')} ({result.get('year', 'N/A')})")
                doc = Document(
                    page_content=result.get("fullplot", ""),
                    metadata={
                        "title": result.get("title", ""),
                        "score": score,
                        "search_type": "KEYWORD",
                        "_id": str(result.get("_id", ""))
                    }
                )
                docs.append(doc)
            return docs
        except Exception as e:
            print(f"❌ Keyword search failed: {e}")
            return []
    
    def _semantic_search(self, query: str) -> List[Document]:
        """Vector/semantic search"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            # Flatten embedding
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
            query_embedding = [float(x) for x in query_embedding]
            print(f"Generated embedding vector: {len(query_embedding)} dimensions")
            
            pipeline = [{
                "$vectorSearch": {
                    "index": mongo_index,
                    "path": os.getenv("VECTORIZED_FIELD_NAME"),
                    "queryVector": query_embedding,
                    "numCandidates": 150,
                    "limit": self.k,
                    "filter": {}
                }
            }, {
                "$project": {
                    "_id": 1,
                    "fullplot": 1,
                    "title": 1,
                    "genres": 1,
                    "cast": 1,
                    "year": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }]
            
            print(f"Using MongoDB Vector Search with index: {mongo_index}")
            results = list(self.collection.aggregate(pipeline))
            docs = []
            for i, result in enumerate(results, 1):
                score = result.get("score", 0)
                print(f"  🎯 #{i} [{score:.4f}] {result.get('title')} ({result.get('year', 'N/A')})")
                doc = Document(
                    page_content=result.get("fullplot", ""),
                    metadata={
                        "title": result.get("title", ""),
                        "score": score,
                        "search_type": "SEMANTIC",
                        "_id": str(result.get("_id", ""))
                    }
                )
                docs.append(doc)
            
            if docs:
                print(f"✅ Semantic search SUCCESS: {len(docs)} documents found")
                return docs
            else:
                print(f"❌ Semantic search returned 0 results")
                print(f"\n🔧 STEP 3: SIMPLE FALLBACK SEARCH")
                print(f"{'-'*40}")
                return self._simple_search(query)
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            print(f"\n🔧 STEP 3: SIMPLE FALLBACK SEARCH")
            print(f"{'-'*40}")
            return self._simple_search(query)

    def _simple_search(self, query: str) -> List[Document]:
        """Simple regex search fallback"""
        try:
            search_terms = query.lower().split()
            print(f"Using MongoDB regex search for terms: {search_terms}")
            
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
            for i, result in enumerate(results, 1):
                print(f"  🔍 #{i} [1.0000] {result.get('title', 'No title')} ({result.get('year', 'N/A')})")
                doc = Document(
                    page_content=result.get("fullplot", result.get("plot", "")),
                    metadata={
                        "title": result.get("title", ""),
                        "score": 1.0,
                        "search_type": "SIMPLE",
                        "_id": str(result.get("_id", ""))
                    }
                )
                docs.append(doc)
            
            if docs:
                print(f"✅ Simple search SUCCESS: {len(docs)} documents found")
            else:
                print(f"❌ All search methods failed")
            return docs
        except Exception as e:
            print(f"❌ Simple search failed: {e}")
            return []

    def invoke(self, query: str) -> List[Document]:
        """Invoke the retriever with a query string"""
        return self._get_relevant_documents(query)
    
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

