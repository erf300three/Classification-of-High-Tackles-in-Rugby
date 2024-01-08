#!/usr/bin/env bash

# Load the required python version
module load Python/3.11.0

# Create a virtual environment
if [[ $# == 1 ]]; then 
    # Remove the virtual environment if first argument is "clean"
    if [[ $1 == "clean" ]]; then
        echo "Removing virtual environment"
        rm -rf venv
        echo "Virtual environment removed"
    # Update the virtual environment if first argument is "update"
    elif [[ $1 == "update" ]]; then 
        # Activate the virtual environment
        source venv/bin/activate
        # Upgrade pip
        pip install --upgrade pip
        # Install the required packages
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install -r Classification-of-High-Tackles-in-Rugby/environment_creation/requirements.txt
        echo "Required packages installed"
        # Deactivate the virtual environment
        deactivate
    else
        echo "Invalid argument"
        echo "Usage: ./create.sh [clean]"
        exit 1
    fi
# If no arguments are provided, create the virtual environment if it does not exist
else 
    if ! [ -d "venv" ]; then
        python3 -m venv --system-site-packages venv
        echo "Virtual environment created"
        # Activate the virtual environment
        source venv/bin/activate
        echo "Virtual environment activated"

        # Display the python version
        python3 --version

        # Upgrade pip 
        pip install --upgrade pip 

        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        # Install the required packages
        pip install -r environment_creation/requirements.txt
        echo "Required packages installed"
        deactivate
    else
        echo "Virtual environment already exists"
    fi
fi





