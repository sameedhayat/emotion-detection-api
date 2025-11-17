#!/bin/bash

# RunPod Handler Script
# This script is used as an entry point for RunPod deployments

echo "Starting Emotion Detection API..."

# Check if running on GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi
else
    echo "Running on CPU"
fi

# Start the FastAPI application
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
