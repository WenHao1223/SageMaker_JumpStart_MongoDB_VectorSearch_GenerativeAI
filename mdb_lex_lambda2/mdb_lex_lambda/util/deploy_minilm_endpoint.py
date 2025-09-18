import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import os
from sagemaker.huggingface import HuggingFaceModel

# role = "arn:aws:iam::<your-account-id>:role/<your-sagemaker-execution-role>"  # Replace with your SageMaker role ARN
role = os.getenv("SAGEMAKER_ROLE")

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

print("Endpoint deployed!")