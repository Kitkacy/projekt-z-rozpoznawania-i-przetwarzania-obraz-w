#!/bin/bash

# Allow X server connections from localhost
xhost +local:docker

# Build and run the Docker container using docker-compose
docker-compose up --build

# Revoke X server access when done
xhost -local:docker
