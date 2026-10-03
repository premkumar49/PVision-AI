"""
PVision AI: Single-Image Inference Pipeline (EfficientNetV2-S)
Performs inference on a single PV fault image (GASF or I-V curve):
- Modality-aware preprocessing (direct resize for GASF, aspect-preserving pad for I-V)
- Prediction of fault category
- Calculation of confidence score
- Formatted output of all 7 class probabilities
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Path resolution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utilities.config import config
from src.utilities.logger import get_logger

logger = get_logger("CNN_Prediction")


def preprocess_single_image(
    image_path: Path,
    modality: str = "gasf",
    target_size: tuple = (224, 224)
) -> np.ndarray:
    """
    Applies the exact preprocessing required for the given modality.
    - GASF: Direct bilinear resize to 224x224
    - I-V: Aspect-ratio-preserving resize and padding to 224x224 (no stretching!)
    """
    img = Image.open(image_path).convert("RGB")
    target_w, target_h = target_size

    if "iv" in modality.lower():
        # Aspect-ratio-preserving resize and pad
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Pad with white (typical background for curve plots)
        padded_img = Image.new("RGB", target_size, (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded_img.paste(resized, (paste_x, paste_y))
        processed_img = padded_img
    else:
        # GASF 256x256 square direct resize
        processed_img = img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to NumPy array float32 in [0, 255]
    img_array = np.array(processed_img, dtype=np.float32)
    # Add batch dimension: (1, 224, 224, 3)
    batch_array = np.expand_dims(img_array, axis=0)
    return batch_array


def predict_fault(
    image_path: Path,
    modality: str = "gasf",
    model_path: Path = None,
    class_names_path: Path = None
):
    """
    Performs inference and prints formatted predictions.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image not found at: {image_path}")
        return None

    mod_key = "gasf" if "gasf" in modality.lower() else "iv"
    models_dir = config.PROJECT_ROOT / "models" / "cnn"

    if not model_path:
        model_path = models_dir / f"efficientnetv2s_{mod_key}_best.keras"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print("\n" + "=" * 60)
        print("                PVISION AI - PREDICTION NOTICE")
        print("=" * 60)
        print(f"Model checkpoint not found: {model_path}")
        print("\nBefore performing inference, the model must be trained in VS Code.")
        print(f"Execute training manually using:")
        print(f"  python src/cnn/train_cnn.py --modality {mod_key}")
        print("=" * 60 + "\n")
        return None

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow must be installed to load the model.")

    # Load Class Names
    if not class_names_path:
        class_names_path = models_dir / f"{mod_key}_class_names.json"
    if class_names_path and Path(class_names_path).exists():
        with open(class_names_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = config.PV_FAULT_CLASSES

    # Load Model
    model = tf.keras.models.load_model(str(model_path))

    # Preprocess Image
    input_tensor = preprocess_single_image(image_path, modality=mod_key)

    # Predict Probabilities
    probs = model.predict(input_tensor, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    predicted_class = class_names[top_idx]
    confidence_pct = float(probs[top_idx] * 100)

    # Format Output Exactly as Specified
    print("\n" + "-" * 40)
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence_pct:.1f}%")
    print("\nClass Probabilities:")
    for cls_name, prob in zip(class_names, probs):
        print(f"{cls_name}: {prob * 100:.1f}%")
    print("-" * 40 + "\n")

    return {
        "predicted_class": predicted_class,
        "confidence": confidence_pct,
        "probabilities": {cls_name: float(round(prob * 100, 2)) for cls_name, prob in zip(class_names, probs)}
    }


def parse_args():
    parser = argparse.ArgumentParser(description="PVision AI: Predict Photovoltaic Fault on a Single Image")
    parser.add_argument("--image-path", type=str, required=True,
                        help="Path to the query image (.png)")
    parser.add_argument("--modality", type=str, default="gasf", choices=["gasf", "iv"],
                        help="Image modality ('gasf' or 'iv')")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to custom .keras model file")
    parser.add_argument("--classes-path", type=str, default=None,
                        help="Path to class_names.json")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_fault(
        image_path=Path(args.image_path),
        modality=args.modality,
        model_path=Path(args.model_path) if args.model_path else None,
        class_names_path=Path(args.classes_path) if args.classes_path else None
    )
