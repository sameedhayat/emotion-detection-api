"""
Convert Cardiff emotion model to ONNX with embedded tokenizer.
This script runs during Docker build to create a complete ONNX model.
"""
import onnx
import torch
from pathlib import Path
from onnxruntime_extensions import pnp, OrtPyFunction
from transformers import AutoTokenizer
from transformers.onnx import export, FeaturesManager

# Model configuration
model_name = "cardiffnlp/twitter-roberta-base-emotion"
output_dir = Path.home() / ".cache" / "huggingface" / "onnx_models"
output_dir.mkdir(parents=True, exist_ok=True)

base_model_path = output_dir / "emotion_base.onnx"
final_model_path = output_dir / "emotion_complete.onnx"

print(f"Loading model: {model_name}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Export base model to ONNX if not exists
if not base_model_path.exists():
    print("Converting base model to ONNX...")
    model = FeaturesManager.get_model_from_feature("sequence-classification", model_name)
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        model, feature="sequence-classification"
    )
    onnx_config = model_onnx_config(model.config)
    
    export(
        tokenizer,
        model=model,
        config=onnx_config,
        opset=14,
        output=base_model_path
    )
    print(f"Base model saved to: {base_model_path}")

# Post-processing: apply softmax to get probabilities
def post_processing(*pred):
    """Apply softmax to logits to get probabilities."""
    logits = pred[0]
    return torch.softmax(logits, dim=1)

# Mapping tokenizer output to model input
def mapping_token_output(input_ids, attention_mask, token_type_ids=None):
    """Map tokenizer outputs to model inputs."""
    # RoBERTa doesn't use token_type_ids, so we create a dummy one if needed
    batch_size = input_ids.shape[0]
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    return input_ids, attention_mask

# Create tokenizer for ONNX
print("Creating ONNX tokenizer...")
ort_tok = pnp.PreHuggingFaceTokenizer(tokenizer)

# Load the base ONNX model
onnx_model = onnx.load_model(str(base_model_path))

# Test sentence
test_sentence = ["I am so happy today!"]

# Create complete model with tokenizer and post-processing
print("Creating complete ONNX model with tokenizer...")
augmented_model = pnp.export(
    pnp.SequentialProcessingModule(
        ort_tok,
        mapping_token_output,
        onnx_model,
        post_processing
    ),
    test_sentence,
    opset_version=14,
    output_path=str(final_model_path)
)

print(f"Complete model saved to: {final_model_path}")

# Test the model
print("\nTesting the complete ONNX model...")
model_func = OrtPyFunction.from_model(str(final_model_path))
result = model_func(test_sentence)

print(f"Test input: {test_sentence[0]}")
print(f"Output shape: {result.shape}")
print(f"Probabilities: {result[0]}")

# Emotion labels
labels = ["anger", "joy", "optimism", "sadness"]
predicted_idx = result[0].argmax()
print(f"Predicted emotion: {labels[predicted_idx]}")
print(f"Confidence: {result[0][predicted_idx]:.4f}")

print("\n✅ Model conversion complete!")
print(f"Runtime will use: {final_model_path}")
