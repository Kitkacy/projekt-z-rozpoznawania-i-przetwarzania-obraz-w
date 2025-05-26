#!/bin/bash

# Script to install the FaceX-Zoo Recognition application with local dependencies

echo "============================================"
echo "Installing FaceX-Zoo Recognition Application"
echo "============================================"

# Set script directory
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR"

# Ask for username
read -p "Enter your username: " USERNAME
HOME_DIR="/home/$USERNAME"
echo "Using home directory: $HOME_DIR"

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

# Make the main script executable
echo "Making main script executable..."
chmod +x "$SCRIPT_DIR/main.py"

# Create activation script
cat > "$SCRIPT_DIR/run_app.sh" << EOL
#!/bin/bash
cd "$SCRIPT_DIR"
source venv/bin/activate
./main.py
EOL
chmod +x "$SCRIPT_DIR/run_app.sh"

# Create or update the desktop file with the correct paths
cat > "$SCRIPT_DIR/FaceRecognition.desktop" << EOL
[Desktop Entry]
Type=Application
Name=FaceX-Zoo Recognition
Comment=Face Recognition Application
Exec=$SCRIPT_DIR/run_app.sh
Icon=$SCRIPT_DIR/icon.png
Terminal=false
Categories=Graphics;Science;Education;
EOL

# Copy the desktop file to the applications directory
echo "Copying shortcut to applications directory..."
mkdir -p "$HOME_DIR/.local/share/applications"
cp "$SCRIPT_DIR/FaceRecognition.desktop" "$HOME_DIR/.local/share/applications/"

# Update desktop database
echo "Updating desktop database..."
update-desktop-database "$HOME_DIR/.local/share/applications"

# Copy to desktop if requested
read -p "Do you want to add a shortcut to your desktop? (y/n): " add_to_desktop
if [[ "$add_to_desktop" == "y" ]]; then
    echo "Adding shortcut to desktop..."
    cp "$SCRIPT_DIR/FaceRecognition.desktop" "$HOME_DIR/Desktop/"
fi

echo "Installation and shortcut setup completed."
echo "You can now find 'FaceX-Zoo Recognition' in your applications menu."
