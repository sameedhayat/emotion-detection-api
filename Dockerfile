# Stage 1: Build stage - Convert model to ONNX with tokenizer
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install conversion dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy conversion script
COPY convert_to_onnx.py .

# Convert model to ONNX with embedded tokenizer
RUN python convert_to_onnx.py

# Stage 2: Runtime stage - Minimal final image
FROM python:3.10-slim

WORKDIR /app

# Install only minimal system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy runtime requirements (no transformers!)
COPY requirements-runtime.txt .

# Install only runtime dependencies
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Copy application code
COPY main.py .

# Copy converted ONNX model and tokenizer from builder stage
COPY --from=builder /root/.cache/huggingface/emotion_model /root/.cache/huggingface/emotion_model

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
