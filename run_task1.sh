#!/bin/bash
# Setup and run Task 1 image filtering

cd "$(dirname "$0")"

echo "Setting up Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing packages..."
pip install opencv-python scikit-image numpy --quiet

# Run the script
echo ""
echo "Starting image filtering..."
echo ""
python task1_filter_images.py

echo ""
echo "Setup complete! Virtual environment is in ./venv"
