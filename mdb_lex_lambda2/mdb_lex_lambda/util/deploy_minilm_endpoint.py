import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import os
import boto3
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import Session
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# role = "arn:aws:iam::<your-account-id>:role/<your-sagemaker-execution-role>"  # Replace with your SageMaker role ARN
role = os.getenv("SAGEMAKER_ROLE")

# Create boto3 session with explicit region
boto_session = boto3.Session(region_name=os.getenv("AWS_REGION1"))
sagemaker_session = Session(boto_session=boto_session)

# Specify the Hugging Face model and task
hub = {
    'HF_MODEL_ID':'sentence-transformers/all-MiniLM-L6-v2',
    'HF_TASK':'feature-extraction'
}

# Create the Hugging Face Model
huggingface_model = HuggingFaceModel(
    transformers_version='4.26.0',
    pytorch_version='1.13.1',
    py_version='py39',
    env=hub,
    role=role
)

# Deploy the model
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=os.getenv("EMBEDDING_ENDPOINT_NAME")
)

print("\nMiniLM Endpoint deployed!")