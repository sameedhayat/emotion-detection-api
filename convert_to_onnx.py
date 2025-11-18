"""
Convert Cardiff emotion model to ONNX and save tokenizer.
This script runs during Docker build using transformers.onnx package.
"""
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

# Load model
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Export to ONNX using transformers.onnx
print("Exporting to ONNX using transformers.onnx...")
from transformers.onnx import export, FeaturesManager

# Get the ONNX config for sequence classification
model_kind, onnx_config_constructor = FeaturesManager.check_supported_model_or_raise(
    model, feature="sequence-classification"
)
onnx_config = onnx_config_constructor(model.config)

# Export the model
onnx_path = output_dir / "model.onnx"
export(
    tokenizer,
    model,
    onnx_config,
    onnx_config.default_onnx_opset,
    onnx_path
)

print(f"ONNX model saved to: {onnx_path}")
print("\n✅ Conversion complete! Model and tokenizer ready for runtime.")
