#!/bin/bash

# Script to set up the face recognition app locally without Docker
# Useful if Docker setup is problematic

echo "============================================"
echo "Local Setup for Face Recognition Application"
echo "============================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Detected Python: $python_version"

# Check if virtual environment module is available
if ! python3 -m venv --help &> /dev/null; then
    echo "Installing Python virtual environment support..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

# Create a virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install opencv-python numpy pillow pyyaml matplotlib scikit-image torch torchvision torchaudio

# Install system dependencies for GUI and webcam
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    python3-tk \
    v4l-utils

# Download Arial font for Polish characters if not present
if [ ! -f "arial.ttf" ]; then
    echo "Downloading Arial font..."
    wget -q -O arial.ttf https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf
fi

# Create directories if they don't exist
mkdir -p Twarze Logs

echo "============================================"
echo "Setup completed. You can now run the application with:"
echo ""
echo "source venv/bin/activate  # Activate virtual environment"
echo "python main.py            # Run the application"
echo "============================================"
