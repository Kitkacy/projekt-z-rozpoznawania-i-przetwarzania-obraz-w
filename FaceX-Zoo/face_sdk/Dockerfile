FROM --platform=$BUILDPLATFORM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install dependencies for OpenCV, Tkinter, and X11
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libfontconfig1 \
    libtk8.6 \
    fonts-liberation \
    python3-tk \
    python3-pil.imagetk \
    v4l-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy face_sdk directory
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python \
    numpy \
    pillow \
    pyyaml \
    matplotlib \
    scikit-image \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Create the directories for face reference images and logs
RUN mkdir -p Twarze Logs config

# Download TrueType font for Polish characters
RUN wget -q -O /app/arial.ttf https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf

# Set the display environment variable
ENV DISPLAY=:0

# Command to run when the container starts
CMD ["python", "main.py"]
