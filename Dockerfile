# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Pre-download and convert model to ONNX during build
RUN python -c "from transformers import AutoTokenizer; \
    from optimum.onnxruntime import ORTModelForSequenceClassification; \
    model_name = 'cardiffnlp/twitter-roberta-base-emotion'; \
    AutoTokenizer.from_pretrained(model_name); \
    ORTModelForSequenceClassification.from_pretrained(model_name, export=True)"

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
