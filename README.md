# Generative AI Solution: Lambda function to vectorize data

## Introduction

This repository demonstrates how to vectorize the data using the Amazon SageMaker Jumpstart models.

## Prerequisite

  [sam cli](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
  [MongoDB Atlas Databse](https://www.mongodb.com/docs/atlas/getting-started/) with [Sample Data](https://www.mongodb.com/docs/atlas/sample-data/) 

## Steps

### Update the environment variable for ATLAS_URI
  update the ATLAS_URI value in template.yaml and .env files.

### Generate Vector Embeddings

Generate the vector embedding(egVector) for fullplot field in sample_mflix.movies collection

    cd mdb_lex_lambda2/mdb_lex_lambda/util
    python3 mongodb_vectorization_search.py

### Create Index

Create the [Vector Search Index](https://www.mongodb.com/docs/atlas/atlas-search/field-types/knn-vector/) for the egVector field created in the previous step.

      {
        "mappings": {
          "dynamic": true,
          "fields": {
            "egVector": {
              "dimensions": 384,
              "similarity": "euclidean",
              "type": "knnVector"
            }
          }
        }
      }


### Build and Deploy

    cd ..
    
    sam build

    sam package

    sam deploy

### Troubleshoot

Refer to the Cloudformation Event for any errors