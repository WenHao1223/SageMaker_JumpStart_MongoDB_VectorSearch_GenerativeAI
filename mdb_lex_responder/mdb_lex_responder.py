import json
from mdb_context_retriever import MDBContextRetriever
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json
import os
from llm_with_mdb_semantic_context import build_chain, run_chain

# import requests

def lex_response(res):
    
    response = {
    'sessionState': {
        'dialogAction': {
            'type': 'Close'
        },
        'intent':{
            'name':'FallbackIntent',
            'state': 'Fulfilled'
        }
    },
    'messages': [
        {
            'contentType': 'PlainText',
            'content': res
        }
        ]
    }    
    
    return response

def lambda_handler(event, context):
    
    input_text = event['inputTranscript']

    chain = build_chain()
    result = run_chain(chain, input_text)

    print("Input text is:",input_text)
    print("LLM generated text is:",result['answer'])

    response = lex_response(result['answer'])
    
    return response
