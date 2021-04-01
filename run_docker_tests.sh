#!/bin/bash

## Check for existence of important environment variables.
[ -z "$MINIO_ROOT_USER" ] && echo "Environment variable MINIO_ROOT_USER not set!" && exit
[ -z "$MINIO_ROOT_PASSWORD" ] && echo "Environment variable MINIO_ROOT_PASSWORD not set!" && exit
[ -z "$MINIO_HOSTNAME" ] && echo "Environment variable MINIO_HOSTNAME not set!" && exit
[ -z "$MINIO_PORT" ] && echo "Environment variable MINIO_PORT not set!" && exit
[ -z "$AWS_ACCESS_KEY_ID" ] && echo "Environment variable AWS_ACCESS_KEY_ID not set!" && exit
[ -z "$AWS_SECRET_ACCESS_KEY" ] && echo "Environment variable AWS_SECRET_ACCESS_KEY not set!" && exit
[ -z "$AWS_S3_HOSTNAME" ] && echo "Environment variable AWS_S3_HOSTNAME not set!" && exit
[ -z "$AWS_S3_PORT" ] && echo "Environment variable AWS_S3_PORT not set!" && exit

## Remove existing docker containers
docker compose down

## Build the docker compose bundle
docker compose build

## Run the docker containers
docker compose up -d

## Run the tests within the container
docker exec -it nft-predict pytest ./tests --full-trace

## Destroy the existing docker containers
docker compose down

