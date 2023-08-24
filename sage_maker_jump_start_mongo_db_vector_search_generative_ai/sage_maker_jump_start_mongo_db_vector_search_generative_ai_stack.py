import os
from aws_cdk import (
    # Duration,
    Stack,
    # aws_sqs as sqs,
)
from constructs import Construct
from aws_cdk import ( aws_lambda as _lambda,
                     aws_iam as _iam,
                     Duration 
                     )

class SageMakerJumpStartMongoDbVectorSearchGenerativeAiStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        lambda_layer_mdb_lex_responder =_lambda.LayerVersion(self,id='mdb_lex_responder_layer', 
                                                   code=_lambda.Code.from_asset('package/.'), 
                                                   compatible_runtimes=[_lambda.Runtime.PYTHON_3_8],
                                                   description='requests Library',
                                                   layer_version_name='v1')

        role_name = "lex_responder_lambda_role"
        lambda_role = _iam.Role(scope=self, id=role_name,
                            
                                assumed_by =_iam.ServicePrincipal('lambda.amazonaws.com'),
                                managed_policies=[_iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
                                                  _iam.ManagedPolicy.from_aws_managed_policy_name("AmazonKendraFullAccess"),
                                                  _iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSESFullAccess"),
                                                  _iam.ManagedPolicy.from_aws_managed_policy_name("AmazonLexFullAccess"),
                                                  _iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchFullAccess")],
                                role_name=role_name)

        #lambda function to answer queries from lex
        lambda_function = _lambda.Function(self, "mdb_lex_responder",
                                           runtime=_lambda.Runtime.PYTHON_3_8,
                                           environment={
                                                    "LLM_ENDPOINT": "<PLACEHOLDER REPLACE WITH YOUR OWN>",
                                                    "ATLAS_URI": "<PLACEHOLDER REPLACE WITH YOUR OWN>",
                                                    "LLM_EMBEDDING_ENDPOINT": "<PLACEHOLDER REPLACE WITH YOUR OWN>"
                                            },
                                           handler="mdb_lex_responder.lambda_handler",
                                           code=_lambda.Code.from_asset("mdb_lex_responder"),
                                           layers=[lambda_layer_mdb_lex_responder],
                                           role=lambda_role,
                                           timeout = Duration.seconds(300)
        )
                                           
