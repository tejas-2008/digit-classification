#!/bin/bash

# Build the Docker image
sudo docker build -t digits:v1 -f docker/dockerfile .

# Run the Docker container with the volume mounted
sudo docker run -v ./models:/digits/models -it digits:v1