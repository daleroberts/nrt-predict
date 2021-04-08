#!/bin/bash

# Exit if any command fails
set -e

# Load environment variables
source .env

# Confirm dataset URL is configured
if [[ ! "$AWS_S3_DATASET_URL" ]]; then
    echo "Environment variable AWS_S3_DATASET_URL not set!"
    exit
fi

pipenv install --sequential
pipenv run python nrt_predict.py $AWS_S3_DATASET_URL

