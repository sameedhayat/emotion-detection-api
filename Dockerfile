FROM python:3.10 AS builder

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install conversion deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run ONNX conversion
COPY convert_to_onnx.py .
RUN python convert_to_onnx.py

# Stage 2: Runtime stage - Minimal final image
FROM python:3.10

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Copy app
COPY main.py .

# Copy the model from builder stage
COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]