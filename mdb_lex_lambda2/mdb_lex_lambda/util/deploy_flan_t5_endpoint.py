import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import os
from sagemaker.huggingface import HuggingFaceModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# role = "arn:aws:iam::<your-account-id>:role/<your-sagemaker-execution-role>"  # Replace with your SageMaker role ARN
role = os.getenv("SAGEMAKER_ROLE")

hub = {
    'HF_MODEL_ID': 'google/flan-t5-xl',
    'HF_TASK': 'text2text-generation'
}

huggingface_model = HuggingFaceModel(
    transformers_version='4.26.0',
    pytorch_version='1.13.1',
    py_version='py39',
    env=hub,
    role=role
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',  # You can use a larger instance if needed
    endpoint_name=os.getenv("LLM_ENDPOINT")
)

print("Flan-T5 XL endpoint deployed!")
