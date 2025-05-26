#!/bin/bash

# Script to install the FaceX-Zoo Recognition application shortcuts

echo "Installing FaceX-Zoo Recognition application shortcuts..."

# Make the main script and Docker script executable
echo "Making scripts executable..."
chmod +x "$(dirname "$0")/main.py"
chmod +x "$(dirname "$0")/run_docker_no_compose.sh"

# Copy the desktop files to the applications directory
echo "Copying shortcuts to applications directory..."
cp "$(dirname "$0")/FaceRecognition.desktop" ~/.local/share/applications/
cp "$(dirname "$0")/FaceRecognition-Docker.desktop" ~/.local/share/applications/

# Update desktop database
echo "Updating desktop database..."
update-desktop-database ~/.local/share/applications

# Copy to desktop if requested
echo "Which shortcut would you like to add to your desktop?"
echo "1. Standard version (local installation)"
echo "2. Docker version"
echo "3. Both versions"
echo "4. None"
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Adding standard shortcut to desktop..."
        cp "$(dirname "$0")/FaceRecognition.desktop" ~/Desktop/
        ;;
    2)
        echo "Adding Docker shortcut to desktop..."
        cp "$(dirname "$0")/FaceRecognition-Docker.desktop" ~/Desktop/
        ;;
    3)
        echo "Adding both shortcuts to desktop..."
        cp "$(dirname "$0")/FaceRecognition.desktop" ~/Desktop/
        cp "$(dirname "$0")/FaceRecognition-Docker.desktop" ~/Desktop/
        ;;
    4)
        echo "No shortcuts will be added to desktop."
        ;;
    *)
        echo "Invalid choice. No shortcuts will be added to desktop."
        ;;
esac

echo "Shortcut installation completed."
echo "You can now find 'FaceX-Zoo Recognition' in your applications menu."
