"""
Convert Cardiff emotion model to ONNX and save tokenizer.
This script runs during Docker build.
"""
import os
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# Model configuration
model_name = "cardiffnlp/twitter-roberta-base-emotion"
output_dir = Path.home() / ".cache" / "huggingface" / "emotion_model"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Converting model: {model_name}")

# Load and save tokenizer
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(str(output_dir))
print(f"Tokenizer saved to: {output_dir}")

# Convert model to ONNX
print("Converting model to ONNX...")
model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    export=True
)
model.save_pretrained(str(output_dir))
print(f"ONNX model saved to: {output_dir}")

# Test the setup
print("\nTesting conversion...")
test_text = "I am so happy today!"
test_inputs = tokenizer(test_text, return_tensors="np", padding=True, truncation=True, max_length=512)
outputs = model(**test_inputs)
logits = outputs.logits[0]

from scipy.special import softmax
probs = softmax(logits)
labels = ["anger", "joy", "optimism", "sadness"]
predicted_idx = probs.argmax()

print(f"Test input: {test_text}")
print(f"Predicted emotion: {labels[predicted_idx]}")
print(f"Confidence: {probs[predicted_idx]:.4f}")
print("\n✅ Conversion complete!")
