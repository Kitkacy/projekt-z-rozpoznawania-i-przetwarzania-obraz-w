# Desktop Shortcut Installation Instructions

This document provides instructions for installing and using the FaceX-Zoo Recognition desktop shortcuts.

## Available Shortcuts

1. **FaceRecognition.desktop** - Launches the application directly on your system
2. **FaceRecognition-Docker.desktop** - Launches the application using Docker

## Installation Methods

### Automated Installation

1. Open a terminal in the `face_sdk` directory
2. Run the installation script:
   ```bash
   ./install_shortcut.sh
   ```
3. Follow the prompts to complete the installation

### Manual Installation

To install the shortcuts manually:

1. Copy the desktop files to your applications directory:
   ```bash
   cp FaceRecognition.desktop ~/.local/share/applications/
   cp FaceRecognition-Docker.desktop ~/.local/share/applications/
   ```

2. Update the desktop database:
   ```bash
   update-desktop-database ~/.local/share/applications
   ```

3. Optionally, copy to your desktop:
   ```bash
   cp FaceRecognition.desktop ~/Desktop/
   # or
   cp FaceRecognition-Docker.desktop ~/Desktop/
   ```

## Troubleshooting

If the shortcuts don't work:

1. Make sure the scripts are executable:
   ```bash
   chmod +x main.py
   chmod +x run_docker_no_compose.sh
   ```

2. Check that the paths in the desktop files are correct:
   ```bash
   nano FaceRecognition.desktop
   nano FaceRecognition-Docker.desktop
   ```

3. Make sure the shortcut.png file exists in the face_sdk directory

## Using Different Icons

To use a different icon, simply replace the shortcut.png file or edit the desktop files to point to a different icon:

1. Open the desktop file for editing:
   ```bash
   nano FaceRecognition.desktop
   ```

2. Modify the Icon line to point to your preferred icon:
   ```
   Icon=/path/to/your/icon.png
   ```
