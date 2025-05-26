#!/bin/bash

# Script to set up Docker permissions and diagnose issues

echo "============================================"
echo "Docker and X11 Setup and Diagnostics Script"
echo "============================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Ubuntu/Debian: sudo apt install docker.io"
    echo "   Fedora: sudo dnf install docker"
    exit 1
else
    echo "✅ Docker is installed."
    docker_version=$(docker --version)
    echo "   $docker_version"
fi

# Check Docker service status
echo ""
echo "Checking Docker service status..."
if systemctl is-active --quiet docker; then
    echo "✅ Docker service is running."
else
    echo "❌ Docker service is not running."
    echo "   Starting Docker service..."
    sudo systemctl start docker
    if systemctl is-active --quiet docker; then
        echo "✅ Docker service started successfully."
    else
        echo "❌ Failed to start Docker service. Please check with: sudo systemctl status docker"
        exit 1
    fi
fi

# Set up Docker permissions
echo ""
echo "Setting up Docker permissions for user $USER..."
if groups $USER | grep -q docker; then
    echo "✅ User $USER is already in the docker group."
else
    echo "❌ User $USER is not in the docker group."
    echo "   Adding user to docker group..."
    sudo usermod -aG docker $USER
    echo "✅ User added to docker group."
    echo "   ⚠️ You need to log out and log back in for this to take effect."
    echo "   Alternatively, run: newgrp docker"
fi

# Check X11 configuration
echo ""
echo "Checking X11 configuration..."
if xhost &> /dev/null; then
    echo "✅ X11 server is accessible."
else
    echo "⚠️ Could not access X11 server. Make sure X11 is running."
fi

# Check webcam
echo ""
echo "Checking webcam devices..."
if [ -e /dev/video0 ]; then
    echo "✅ Webcam device found at /dev/video0."
    # Check permissions
    permissions=$(ls -l /dev/video0)
    echo "   $permissions"
    if [[ "$permissions" == *"rw"* ]]; then
        echo "   ✅ Webcam has read/write permissions."
    else
        echo "   ⚠️ Webcam might not have proper permissions."
        echo "      Consider running: sudo chmod 666 /dev/video0"
    fi
else
    echo "❌ No webcam device found at /dev/video0."
    echo "   Check if your webcam is connected."
    echo "   If your webcam is at a different location, edit the Docker scripts accordingly."
fi

# Final instructions
echo ""
echo "============================================"
echo "Setup completed. Please follow these steps:"
echo ""
echo "1. If your user was just added to the docker group:"
echo "   - Log out and log back in, OR"
echo "   - Run: newgrp docker"
echo ""
echo "2. Try running the application with:"
echo "   ./run_docker_no_compose.sh"
echo ""
echo "3. If you still have issues, you can run with sudo:"
echo "   sudo ./run_docker_no_compose.sh"
echo "============================================"
