from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import logging
import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Detection API (ONNX Optimized)",
    description="Fast emotion detection using ONNX Runtime with standalone tokenizer",
    version="2.0.0"
)

# Global variables
ort_model = None
tokenizer = None
emotion_labels = ["anger", "joy", "optimism", "sadness"]


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


def softmax(x):
    """Compute softmax values using numpy."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


@app.on_event("startup")
async def load_model():
    """Load the ONNX model and tokenizer using Optimum."""
    global ort_model, tokenizer
    
    try:
        logger.info("Loading ONNX model and tokenizer...")
        
        model_dir = "/app/model"
        
        # Load ONNX model using Optimum
        ort_model = ORTModelForSequenceClassification.from_pretrained(model_dir)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        logger.info("ONNX model and tokenizer loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def predict_emotion(text: str) -> Dict:
    """
    Predict emotion for a single text using Optimum ONNX Runtime.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing emotion label and scores
    """
    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # Run inference with Optimum ONNX model
        outputs = ort_model(**inputs)
        logits = outputs.logits.detach().numpy()[0]
        
        # Apply softmax to get probabilities
        probabilities = softmax(logits)
        
        # Create scores dictionary
        scores = {label: float(prob) for label, prob in zip(emotion_labels, probabilities)}
        
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
    if ort_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "runtime": "Optimum ONNX Runtime"
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
    if ort_model is None or tokenizer is None:
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
    if ort_model is None or tokenizer is None:
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
