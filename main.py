from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from typing import List, Dict
import logging
from scipy.special import softmax

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Detection API (ONNX Optimized)",
    description="Fast emotion detection using ONNX Runtime for efficient inference",
    version="2.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None


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
    """Load the ONNX optimized emotion detection model on startup."""
    global model, tokenizer
    
    try:
        logger.info("Loading ONNX optimized emotion detection model...")
        model_name = "cardiffnlp/twitter-roberta-base-emotion"
        
        # Load tokenizer and ONNX model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True  # Auto-convert to ONNX if not already converted
        )
        
        logger.info("ONNX model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def predict_emotion(text: str) -> Dict:
    """
    Predict emotion for a single text using ONNX Runtime.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing emotion label and scores
    """
    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # Get predictions with ONNX
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()[0]
        
        # Apply softmax to get probabilities
        probabilities = softmax(logits)
        
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
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "runtime": "ONNX"
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
    if model is None or tokenizer is None:
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
    if model is None or tokenizer is None:
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
