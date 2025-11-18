"""
Convert Cardiff emotion model to ONNX and save tokenizer.
This script runs during Docker build.
"""
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Model configuration
model_name = "cardiffnlp/twitter-roberta-base-emotion"
output_dir = Path.home() / ".cache" / "huggingface" / "emotion_model"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Converting model: {model_name}")

# Load tokenizer and save
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(str(output_dir))
print(f"Tokenizer saved to: {output_dir}")

# Load PyTorch model
print("Loading PyTorch model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Prepare dummy input for ONNX export
dummy_text = "This is a sample text for ONNX export"
inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Export to ONNX
print("Exporting to ONNX...")
onnx_path = output_dir / "model.onnx"

torch.onnx.export(
    model,
    (inputs['input_ids'], inputs['attention_mask']),
    str(onnx_path),
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size'}
    },
    opset_version=14,
    do_constant_folding=True
)

print(f"ONNX model saved to: {onnx_path}")

# Test the export
print("\nTesting ONNX export...")
import onnxruntime as ort
import numpy as np
from scipy.special import softmax

session = ort.InferenceSession(str(onnx_path))
test_text = "I am so happy today!"
test_inputs = tokenizer(test_text, return_tensors="np", padding=True, truncation=True, max_length=512)

ort_inputs = {
    'input_ids': test_inputs['input_ids'].astype(np.int64),
    'attention_mask': test_inputs['attention_mask'].astype(np.int64)
}
outputs = session.run(None, ort_inputs)
logits = outputs[0][0]
probs = softmax(logits)

labels = ["anger", "joy", "optimism", "sadness"]
predicted_idx = probs.argmax()

print(f"Test input: {test_text}")
print(f"Predicted emotion: {labels[predicted_idx]}")
print(f"Confidence: {probs[predicted_idx]:.4f}")
print("\n✅ Conversion complete!")
