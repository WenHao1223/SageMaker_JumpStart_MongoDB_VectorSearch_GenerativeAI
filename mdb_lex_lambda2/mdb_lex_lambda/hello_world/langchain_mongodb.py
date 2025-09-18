import warnings
warnings.filterwarnings("ignore")

from langchain.chains import RetrievalQA
from mongodb_retriever import MDBContextRetriever
from langchain.prompts import PromptTemplate
try:
    from langchain_aws.llms import SagemakerEndpoint
    from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler
except ImportError:
    from langchain_community.llms import SagemakerEndpoint
    from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
import json
import os

class FallbackLLM:
    """Simple fallback LLM that summarizes documents without SageMaker"""
    
    def invoke(self, inputs):
        context = inputs.get('context', '')
        question = inputs.get('question', '')
        
        # Simple summarization based on retrieved documents
        if context and question:
            return f"Based on the retrieved documents about '{question}': {context[:500]}..."
        elif context:
            return f"Summary of retrieved documents: {context[:500]}..."
        else:
            return "No relevant information found in the documents."

def build_chain():
    mongodb_uri = os.environ["ATLAS_URI"]
    endpoint_name = os.environ.get("LLM_ENDPOINT", "")
    aws_region = os.environ["AWS_REGION1"]

    print("AWS Region: " + str(aws_region))

    # Try to use SageMaker endpoint, fallback to simple LLM if it fails
    try:
        class ContentHandler(LLMContentHandler):
            content_type = "application/json"
            accepts = "application/json"

            def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
                input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
                return input_str.encode('utf-8')

            def transform_output(self, output: bytes) -> str:
                response_json = json.loads(output.read().decode("utf-8"))
                return response_json["generated_texts"][0]

        content_handler = ContentHandler()
        llm = SagemakerEndpoint(
                endpoint_name=endpoint_name,
                region_name=aws_region,
                model_kwargs={"temperature":1e-10, "max_length": 500},
                content_handler=content_handler
            )
    except Exception as e:
        print(f"SageMaker endpoint failed, using fallback LLM: {e}")
        llm = FallbackLLM()

    retriever = MDBContextRetriever(mongodb_uri= mongodb_uri, k=3,
                                    return_source_documents=False
                                    )

    prompt_template = """
    The following is a friendly conversation between a human and an AI.
    The AI is talkative and provides lots of specific details from its context.
    If the AI does not know the answer to a question, it truthfully says it
    does not know.
    {context}
    Instruction: Based on the above documents, summarize the following statement, {question} Answer "don't know" if not present in the document. Solution:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    try:
        qa = RetrievalQA.from_chain_type(
            llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        return qa
    except Exception as e:
        print(f"Failed to create RetrievalQA chain: {e}")
        # Return a simple object that has the retriever
        class SimpleChain:
            def __init__(self, retriever):
                self.retriever = retriever
            
            def invoke(self, inputs):
                query = inputs.get('query', '')
                docs = self.retriever.invoke(query)
                context = "\n".join([doc.page_content for doc in docs])
                return {
                    'result': f"Found {len(docs)} documents about '{query}': {context[:500]}...",
                    'source_documents': docs
                }
        
        return SimpleChain(retriever)

def run_chain(chain, prompt: str, history=[]):
    try:
        # Try the normal chain first
        result = chain.invoke({"query": prompt})
        return {
            "answer": result['result'],
            "source_documents": result['source_documents']
        }
    except Exception as e:
        print(f"Chain failed: {e}")
        # Fallback: get documents directly and create simple response
        try:
            retriever = chain.retriever
            docs = retriever.invoke(prompt)
            
            if docs:
                # Create a simple summary from the documents
                context = "\n".join([f"Title: {doc.metadata.get('title', 'Unknown')}\nContent: {doc.page_content[:200]}..." for doc in docs[:3]])
                answer = f"Based on your query '{prompt}', I found the following relevant information:\n\n{context}"
            else:
                answer = f"No relevant documents found for '{prompt}'."
                
            return {
                "answer": answer,
                "source_documents": docs
            }
        except Exception as e2:
            return {
                "answer": f"Unable to process query '{prompt}'. Error: {str(e2)}",
                "source_documents": []
            }

if __name__ == "__main__":

    input_text = "Remember the movie where the kid fight with other and they won?"
    # input_text = "describe fish that lives in the ocean"
    chain = build_chain()
    result = run_chain(chain, input_text)

    print("Input text is:",input_text)
    print("LLM generated text is:",result['answer'])

