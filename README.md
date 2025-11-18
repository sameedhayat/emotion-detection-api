# Emotion Detection API

A FastAPI-based emotion detection service using the Cardiff NLP Twitter RoBERTa emotion model with **ONNX Runtime** for fast inference and minimal container size. This API provides both single tweet and batch processing capabilities and is optimized for deployment on RunPod.

## Features

- **ONNX Optimized**: 2-5x faster inference with 60% smaller container size (~1.5-2 GB vs 4.5 GB)
- **Single Tweet Analysis**: Analyze individual tweets for emotion detection
- **Batch Processing**: Process multiple tweets in a single request (up to 100)
- **Emotion Categories**: Detects 4 emotions - anger, joy, optimism, sadness
- **CPU Optimized**: Efficient inference without requiring GPU
- **RunPod Ready**: Pre-configured for easy deployment on RunPod

## Model

This API uses the [cardiffnlp/twitter-roberta-base-emotion](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion) model, which is fine-tuned on ~58M tweets for emotion recognition.

## API Endpoints

### Root
- **GET** `/` - API information and available endpoints

### Health Check
- **GET** `/health` - Check if the service is running and model is loaded

### Single Tweet Prediction
- **POST** `/predict`
- **Request Body**:
  ```json
  {
    "text": "I am so happy today!"
  }
  ```
- **Response**:
  ```json
  {
    "text": "I am so happy today!",
    "emotion": "joy",
    "scores": {
      "anger": 0.001,
      "joy": 0.950,
      "optimism": 0.045,
      "sadness": 0.004
    }
  }
  ```

### Batch Tweet Prediction
- **POST** `/predict/batch`
- **Request Body**:
  ```json
  {
    "texts": [
      "I am so happy!",
      "This makes me angry.",
      "Feeling optimistic about the future!"
    ]
  }
  ```
- **Response**:
  ```json
  {
    "results": [
      {
        "text": "I am so happy!",
        "emotion": "joy",
        "scores": {...}
      },
      {
        "text": "This makes me angry.",
        "emotion": "anger",
        "scores": {...}
      },
      {
        "text": "Feeling optimistic about the future!",
        "emotion": "optimism",
        "scores": {...}
      }
    ]
  }
  ```

## Local Development

### Prerequisites
- Python 3.10+
- pip

### Installation

1. Clone the repository:
   ```bash
   cd /home/sameed/Documents/code/omniscent/container
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   python main.py
   ```
   Or with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Access the API:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Testing

Run the test script to verify all endpoints:
```bash
python test_api.py
```

## Docker Deployment

### Build the Docker image:
```bash
docker build -t emotion-detection-api .
```

### Run the container:
```bash
docker run -p 8000:8000 emotion-detection-api
```

### Run with GPU support:
```bash
docker run --gpus all -p 8000:8000 emotion-detection-api
```

## RunPod Deployment

### Method 1: Deploy from GitHub (Recommended)

This method automatically builds and deploys your Docker image whenever you push to GitHub.

1. **Set up GitHub repository**:
   ```bash
   cd /home/sameed/Documents/code/omniscent/container
   git init
   git add .
   git commit -m "Initial commit: Emotion detection API"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/emotion-detection-api.git
   git push -u origin main
   ```

2. **Configure GitHub Secrets**:
   - Go to your GitHub repository → Settings → Secrets and variables → Actions
   - Add the following secrets:
     - `DOCKERHUB_USERNAME`: Your Docker Hub username
     - `DOCKERHUB_TOKEN`: Your Docker Hub access token (create at https://hub.docker.com/settings/security)

3. **Automatic Build**:
   - GitHub Actions will automatically build and push the Docker image on every push to main
   - The image will be available at: `your-dockerhub-username/emotion-detection-api:latest`

4. **Deploy on RunPod**:
   - Go to [RunPod](https://www.runpod.io/)
   - Click "Deploy" → "Custom Container"
   - Enter your Docker image: `your-dockerhub-username/emotion-detection-api:latest`
   - Set container port: `8000`
   - Select GPU type (RTX 3090, A100, etc.) or CPU
   - Set environment variables if needed
   - Click "Deploy"

5. **Access your API**:
   - RunPod will provide a URL like: `https://your-pod-id-8000.proxy.runpod.net`
   - Test the health endpoint: `https://your-pod-id-8000.proxy.runpod.net/health`

### Method 2: Manual Docker Build

If you prefer to build locally without GitHub Actions:

1. **Build and push your Docker image**:
   ```bash
   # Build the image
   docker build -t your-dockerhub-username/emotion-detection-api:latest .
   
   # Push to Docker Hub
   docker push your-dockerhub-username/emotion-detection-api:latest
   ```

2. **Deploy on RunPod** (same as steps 4-5 above)

### Method 3: Using RunPod Template

1. Create a template in RunPod with:
   - **Docker Image**: `your-dockerhub-username/emotion-detection-api:latest`
   - **Container Disk**: 10 GB minimum
   - **Expose HTTP Ports**: `8000`
   - **Docker Command**: Leave empty (uses default CMD)

2. Deploy from template whenever needed

### Updating Your Deployment

When using GitHub Actions:
1. Make changes to your code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push
   ```
3. GitHub Actions automatically builds and pushes new image
4. In RunPod, restart your pod or create a new one to use the updated image

### Configuration for RunPod

The included `runpod.json` file contains recommended settings:
- Minimum 4GB VRAM for GPU instances
- Health check endpoint configured
- Port 8000 exposed

### Cost Optimization

- **CPU Instance**: ~$0.2-0.4/hour (suitable for low traffic)
- **GPU Instance** (RTX 3090): ~$0.4-0.6/hour (faster inference)
- **Spot Instances**: Save up to 70% with interruptible instances

## Performance

- **Container size**: ~1.5-2 GB (ONNX optimized vs 4.5 GB PyTorch)
- **Single prediction**: ~100-200ms on CPU (2-5x faster than PyTorch CPU)
- **Batch prediction**: Processes multiple tweets efficiently
- **Model size**: ~500MB (ONNX format)
- **Memory usage**: ~1-1.5GB RAM (60% less than PyTorch)

## Environment Variables

Optional environment variables:
- `PYTHONUNBUFFERED=1` - Prevents Python from buffering stdout/stderr
- `MAX_BATCH_SIZE=100` - Maximum number of texts in batch request

## Troubleshooting

### Model not loading
- Ensure you have enough disk space (~1GB for model)
- Check internet connectivity for first-time model download

### Out of memory errors
- Reduce batch size
- Use CPU instead of GPU
- Increase container memory allocation

### RunPod connection issues
- Verify the port (8000) is correctly exposed
- Check RunPod logs for startup errors
- Ensure health check endpoint returns 200

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

This project uses the Cardiff NLP model which is licensed under the model's original license. Please refer to the [model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion) for details.

## Citation

If you use this API or the underlying model, please cite:
```bibtex
@inproceedings{barbieri-etal-2020-tweeteval,
    title = "{T}weet{E}val: Unified Benchmark and Comparative Evaluation for Tweet Classification",
    author = "Barbieri, Francesco and Camacho-Collados, Jose and Espinosa Anke, Luis and Neves, Leonardo",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
```

## Support

For issues and questions:
- Check the logs: `docker logs <container-id>`
- Review RunPod documentation: https://docs.runpod.io/
- Model information: https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion
