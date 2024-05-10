#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Give docker permission
sudo chmod 666 /var/run/docker.sock

# Create a Docker volume for data persistence
echo "Creating Docker volume: heart-disease-data"
docker volume create --name heart-disease-data

# Create a Docker network for container communication
echo "Creating Docker network: heart-disease-network"
docker network create heart-disease-network

# Build PostgreSQL Docker image
echo "Building PostgreSQL Docker image from dockerfile-postgresql"
docker build -f dockerfiles/dockerfile-postgresql -t postgresql-heart-image .

# Build Jupyter Docker image
echo "Building Jupyter Docker image from dockerfile-jupyter"
docker build -f dockerfiles/dockerfile-jupyter -t jupyter-heart-image .

# Run PostgreSQL container with volume and network setup
echo "Starting PostgreSQL container"
docker run -d --network heart-disease-network \
           --name postgres-heart-container \
           -v heart-disease-data:/var/lib/postgresql/data \
           -p 5432:5432 \
           postgresql-heart-image

# Run Jupyter container with volume and network setup
echo "Starting Jupyter container"
docker run -it --network heart-disease-network \
           --name jupyter-heart-container \
           -v ~/DE300/homework2:/home/jovyan \
           -p 8889:8888 \
           jupyter-heart-image
