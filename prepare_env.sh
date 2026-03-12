#!/bin/bash

# Prepare environment for HSIC project
# This script sets up the hsic_env conda environment with required packages

echo "Activating hsic_env environment..."
conda activate hsic_env

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate hsic_env. Please ensure the environment exists."
    echo "Create it with: conda create -n hsic_env python=3.8"
    exit 1
fi

echo "Installing required packages..."

# Core scientific computing
pip install numpy pandas

# Image processing
pip install tifffile

# Progress bars
pip install tqdm

# Machine learning libraries
pip install scikit-learn xgboost lightgbm

# Model serialization
pip install joblib

# Optional: PyTorch (if needed for some models)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Environment setup complete!"
echo "Current environment: $CONDA_DEFAULT_ENV"