from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import logging
import numpy as np
from pathlib import Path
from onnxruntime_extensions import get_library_path
import onnxruntime as ort

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Detection API (ONNX Optimized)",
    description="Fast emotion detection using ONNX Runtime with embedded tokenizer",
    version="2.0.0"
)

# Global variable for ONNX session
ort_session = None


class TweetRequest(BaseModel):
    text: str = Field(..., description="Tweet text to analyze", min_length=1)


class BatchTweetRequest(BaseModel):
    texts: List[str] = Field(..., description="List of tweet texts to analyze", min_length=1, max_length=100)


class EmotionResponse(BaseModel):
    text: str
    emotion: str
    scores: Dict[str, float]


class BatchEmotionResponse(BaseModel):
    results: List[EmotionResponse]


@app.on_event("startup")
async def load_model():
    """Load the complete ONNX model with embedded tokenizer."""
    global ort_session
    
    try:
        logger.info("Loading ONNX optimized emotion detection model with tokenizer...")
        
        # Find the complete ONNX model
        model_path = Path.home() / ".cache" / "huggingface" / "onnx_models" / "emotion_complete.onnx"
        
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}. Build container properly.")
        
        # Create session options with onnxruntime-extensions
        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        
        # Load ONNX model with tokenizer
        ort_session = ort.InferenceSession(str(model_path), so)
        
        logger.info("ONNX model with embedded tokenizer loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def predict_emotion(text: str) -> Dict:
    """
    Predict emotion for a single text using ONNX Runtime.
    The model includes the tokenizer, so we just pass raw text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing emotion label and scores
    """
    try:
        # Run inference with raw text (tokenizer is embedded in ONNX model)
        outputs = ort_session.run(None, {'text': [text]})
        probabilities = outputs[0][0]  # Get first (and only) batch item
        
        # Get emotion labels
        # Cardiff model labels: anger, joy, optimism, sadness
        labels = ["anger", "joy", "optimism", "sadness"]
        scores = {label: float(prob) for label, prob in zip(labels, probabilities)}
        
        # Get the emotion with highest score
        predicted_emotion = max(scores, key=scores.get)
        
        return {
            "emotion": predicted_emotion,
            "scores": scores
        }
    except Exception as e:
        logger.error(f"Error predicting emotion: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Emotion Detection API",
        "model": "cardiffnlp/twitter-roberta-base-emotion",
        "endpoints": {
            "health": "/health",
            "single": "/predict",
            "batch": "/predict/batch"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "runtime": "ONNX with embedded tokenizer"
    }


@app.post("/predict", response_model=EmotionResponse)
async def predict_single(request: TweetRequest):
    """
    Predict emotion for a single tweet.
    
    Args:
        request: TweetRequest containing the text to analyze
        
    Returns:
        EmotionResponse with predicted emotion and scores
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = predict_emotion(request.text)
        return EmotionResponse(
            text=request.text,
            emotion=result["emotion"],
            scores=result["scores"]
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchEmotionResponse)
async def predict_batch(request: BatchTweetRequest):
    """
    Predict emotions for multiple tweets.
    
    Args:
        request: BatchTweetRequest containing list of texts to analyze
        
    Returns:
        BatchEmotionResponse with results for all texts
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for text in request.texts:
            result = predict_emotion(text)
            results.append(EmotionResponse(
                text=text,
                emotion=result["emotion"],
                scores=result["scores"]
            ))
        
        return BatchEmotionResponse(results=results)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
