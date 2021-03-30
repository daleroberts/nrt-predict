#!/bin/bash

# Remove existing container
docker kill nft-predict
docker rm nft-predict

# Build/run new container
docker build . -t nft-predict:latest
docker run --name nft-predict -dit nft-predict:latest

