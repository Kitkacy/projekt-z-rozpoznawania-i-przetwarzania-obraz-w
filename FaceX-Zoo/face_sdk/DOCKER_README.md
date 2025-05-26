# Face Recognition App - Docker Setup

This directory contains Docker configuration files to deploy the face recognition application in a container.

## Prerequisites

- Docker
- Docker Compose
- X11 server running on the host machine
- Webcam connected to the host machine

## Running the Application

### Option 1: Using the run script

1. Make sure your X11 server allows connections from localhost:
   ```bash
   xhost +local:docker
   ```

2. Run the provided script:
   ```bash
   ./run_docker.sh
   ```

### Option 2: Manual execution

1. Allow X11 connections:
   ```bash
   xhost +local:docker
   ```

2. Build and run the container:
   ```bash
   docker-compose up --build
   ```

3. When finished, revoke X11 access:
   ```bash
   xhost -local:docker
   ```

## Configuration

- Face images are stored in the `Twarze` directory
- Logs are stored in the `Logs` directory
- Both directories are shared between the container and the host

## Troubleshooting

If the GUI doesn't appear, try these steps:

1. Verify that X11 forwarding is properly set up:
   ```bash
   echo $DISPLAY  # Should output something like :0 or :0.0
   ```

2. Check if webcam is accessible:
   ```bash
   ls -l /dev/video*
   ```

3. Make sure the container has proper permissions to access the webcam:
   ```bash
   sudo chmod 666 /dev/video0
   ```

## Security Note

The current setup allows Docker containers to access your X11 server, which may pose security risks. This is necessary for the GUI to work but should be used cautiously. Always run `xhost -local:docker` when finished to revoke the permission.
