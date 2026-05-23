#!/bin/bash
# Setup script for Autonomous AI Researcher

echo "=== Setting up Environment ==="

# Check Python version
python --version 2>&1 || { echo "Python is not installed. Exiting."; exit 1; }

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify GPU and imports
echo "Verifying GPU availability and library imports..."
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('GPU Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device Name:', torch.cuda.get_device_name(0))
"

python -c "
import transformers
import langgraph
import chromadb
import trafilatura
import tavily
print('Core packages successfully imported!')
"

echo "=== Setup Completed Successfully ==="
