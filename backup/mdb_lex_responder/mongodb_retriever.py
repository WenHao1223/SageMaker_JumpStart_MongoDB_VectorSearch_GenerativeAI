from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from typing import Dict, List
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
import json


class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"text_inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["embedding"]


content_handler = ContentHandler()


embeddings = SagemakerEndpointEmbeddings(
    endpoint_name="jumpstart-dft-hf-textembedding-all-minilm-l6-v2",
    region_name="us-east-1",
    content_handler=content_handler,
)

class MDBContextRetriever(BaseRetriever):
    """Retriever to retrieve documents from MongoDB using Vector index.
    Example:
        .. code-block:: python
            MDBContextRetriever = MDBContextRetriever()
    """

    k: int = 2
    """Number of documents to query for."""
    return_source_documents: bool = False
    """Whether source documents to be returned """
    client: MongoClient = None
    mdb_vectorestore: MongoDBAtlasVectorSearch = None

    def __init__(self, mongodb_uri, k=2, return_source_documents=False):
        super().__init__()
        self.k = k
        self.return_source_documents = return_source_documents
        self.client = MongoClient(mongodb_uri)

        db_name = "sample_mflix"
        collection_name = "movies"
        collection = self.client[db_name][collection_name]
        index_name = "vector-index"
        self.mdb_vectorestore = MongoDBAtlasVectorSearch(
            collection,
            embedding=embeddings,
            index_name=index_name,
            embedding_key="egVector",
            text_key="fullplot"
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Run search in MongoDB Atlas and get top k documents
        docs = get_relevant_documents('This is my query')
        """
        docs = self.mdb_vectorestore.similarity_search(query)
        print("context: "+str(docs))
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return await super().aget_relevant_documents(query)

