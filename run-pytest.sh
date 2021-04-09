#!/bin/bash

# Exit if any command fails
set -e

# Load environment variables
source .env

pipenv install --sequential

#pipenv run python nrt_predict.py $AWS_S3_DATASET_URL

# Run pytests - note that these tests currently depend on minio being installed.
pipenv run pytest ./nrtpredict/tests

