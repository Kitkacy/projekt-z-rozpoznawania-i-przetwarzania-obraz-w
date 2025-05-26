#!/bin/bash

# Script to run the face recognition app using Docker without docker-compose
# Used for environments where docker-compose is not available

# Check if the script is run as root or with sudo
if [ "$EUID" -ne 0 ]; then
  echo "This script requires Docker permissions."
  echo "You can either:"
  echo "1. Run with sudo: sudo $0"
  echo "2. Add your user to the docker group: sudo usermod -aG docker $USER (then log out and back in)"
  echo ""
  echo "Trying to continue anyway..."
fi

# Get the current user's DISPLAY variable
# This is important when running with sudo to maintain the correct display
CURRENT_DISPLAY=$DISPLAY
CURRENT_USER=$USER
CURRENT_HOME=$HOME

# Allow X server connections from localhost
xhost +local:docker

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Build the Docker image
echo "Building Docker image..."
docker build -t facex-zoo-gui:latest "$SCRIPT_DIR" || { echo "Docker build failed. Do you have permission to use Docker?"; exit 1; }

# Run the container
echo "Starting container..."
docker run --rm -it \
  --name facex-zoo-gui \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$SCRIPT_DIR/Twarze":/app/Twarze \
  -v "$SCRIPT_DIR/Logs":/app/Logs \
  -v "$SCRIPT_DIR/config":/app/config:ro \
  -e DISPLAY=$CURRENT_DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=$CURRENT_HOME/.Xauthority \
  --device /dev/video0:/dev/video0 \
  --network host \
  facex-zoo-gui:latest

# Revoke X server access when done
xhost -local:docker
