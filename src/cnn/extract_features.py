"""
PVision AI: CNN Feature Extraction Pipeline (EfficientNetV2-S)
Extracts intermediate deep feature representations and fault classification predictions
from the trained EfficientNetV2-S model for both PV image modalities (GASF and I-V curves).

Architecture Flow:
    Input Image
        ↓
    Modality Preprocessing (GASF: Direct Resize | I-V: Aspect-Preserving Pad)
        ↓
    EfficientNetV2-S Backbone
        ↓
    Global Average Pooling Layer ('global_avg_pool') -> Feature Vector (1280-D)
        ↓
    Batch Normalization & Dropout Head
        ↓
    7-Class Softmax Predictions

Scientific Constraint:
    The PV image dataset and solar generation dataset originate from different sources.
    Artificial 1-to-1 sample mapping (e.g. image_001 -> generation_row_001) is strictly
    prohibited without a legitimate common identifier, timestamp, or experimental relationship.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image

# Path resolution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utilities.config import config
from src.utilities.logger import get_logger

logger = get_logger("CNN_FeatureExtraction")


def check_model_availability(
    model_path: Path,
    modality: str
) -> bool:
    """
    Verifies if the trained model checkpoint exists.
    If missing, displays a clear, informative message without raising an unhandled exception.
    """
    mod_key = "gasf" if "gasf" in modality.lower() else "iv"
    if not model_path.exists():
        print("\n" + "=" * 70)
        print("                  PVISION AI - MODEL STATUS NOTICE")
        print("=" * 70)
        if mod_key == "gasf":
            print("Trained GASF EfficientNetV2-S model not found. "
                  "Feature extraction will be performed after CNN training is completed.")
        else:
            print("Trained I-V EfficientNetV2-S model not found. "
                  "I-V feature extraction is pending CNN training.")
        print(f"\nExpected model location: {model_path}")
        print("\nPlease execute model training on Kaggle GPU using:")
        print("  notebooks/03_cnn_efficientnetv2s_kaggle.ipynb")
        print("or locally in VS Code using:")
        print(f"  python src/cnn/train_cnn.py --modality {mod_key}")
        print("=" * 70 + "\n")
        return False
    return True


def preprocess_image_array(
    image_path: Path,
    modality: str = "gasf",
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Applies the exact modality-specific spatial preprocessing used during model training:
    - GASF: Direct bilinear resize from 256x256 to 224x224.
    - I-V: Aspect-ratio-preserving resize and padding to 224x224 (prevents electrical curve distortion).
    Returns float32 NumPy array in range [0, 255].
    """
    img = Image.open(image_path).convert("RGB")
    target_w, target_h = target_size

    if "iv" in modality.lower():
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Pad with white (255, 255, 255) to match I-V curve plot background
        padded_img = Image.new("RGB", target_size, (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded_img.paste(resized, (paste_x, paste_y))
        processed_img = padded_img
    else:
        # GASF 256x256 square direct bilinear resize
        processed_img = img.resize(target_size, Image.Resampling.BILINEAR)

    img_array = np.array(processed_img, dtype=np.float32)
    return img_array


def build_feature_extractor_model(
    model,
    preferred_layer_name: Optional[str] = "global_avg_pool"
):
    """
    Constructs a dual-output Keras model that returns both:
    1. The final 7-class prediction probabilities (from the classification head)
    2. The deep feature vector from the intermediate pooling layer
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow must be installed to construct the feature extractor.")

    # Search for target intermediate feature layer
    target_layer = None
    if preferred_layer_name:
        try:
            target_layer = model.get_layer(preferred_layer_name)
        except ValueError:
            target_layer = None

    if target_layer is None:
        # Fallback search through candidate layers
        candidates = ["global_avg_pool", "head_bn", "head_dropout"]
        for cand in candidates:
            try:
                target_layer = model.get_layer(cand)
                break
            except ValueError:
                continue

    if target_layer is None:
        # Default to the layer immediately preceding the final Dense layer
        target_layer = model.layers[-2]

    logger.info(f"Identified feature extraction layer: '{target_layer.name}' "
                f"(output shape: {target_layer.output_shape})")

    # Create dual-output model (outputs: [predictions, feature_representation])
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.output, target_layer.output],
        name="PVision_FeatureExtractor"
    )
    return feature_extractor, target_layer.name, target_layer.output_shape[-1]


def discover_image_samples(
    data_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    modality: str = "gasf",
    split: str = "test"
) -> List[Dict[str, Any]]:
    """
    Discovers image file paths and true labels from a manifest file or directory scan.
    """
    samples = []
    mod_key = "gasf" if "gasf" in modality.lower() else "iv"

    # Priority 1: User-specified manifest CSV
    if manifest_path and Path(manifest_path).exists():
        logger.info(f"Loading image samples from manifest: {manifest_path}")
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            samples.append({
                "file_path": Path(row["file_path"]),
                "image_id": Path(row["file_path"]).name,
                "true_class": row.get("class_name", row.get("label", None))
            })
        return samples

    # Priority 2: Standard split manifest from data/processed/
    split_manifest = config.PROJECT_ROOT / "data" / "processed" / f"{mod_key}_images_{split}_manifest.csv"
    if split_manifest.exists():
        logger.info(f"Loading {split} split manifest: {split_manifest}")
        df = pd.read_csv(split_manifest)
        for _, row in df.iterrows():
            samples.append({
                "file_path": Path(row["file_path"]),
                "image_id": Path(row["file_path"]).name,
                "true_class": row.get("class_name", None)
            })
        return samples

    # Priority 3: Scan image directory
    scan_dir = data_dir if data_dir else (config.PROJECT_ROOT / "data" / "pv_fault" / f"{mod_key}_images")
    if scan_dir.exists():
        logger.info(f"Scanning image directory: {scan_dir}")
        for cls_name in config.PV_FAULT_CLASSES:
            cls_dir = scan_dir / cls_name
            if cls_dir.is_dir():
                for img_file in cls_dir.glob("*.png"):
                    samples.append({
                        "file_path": img_file,
                        "image_id": img_file.name,
                        "true_class": cls_name
                    })
        if not samples:
            # Check for direct images in scan_dir
            for img_file in scan_dir.glob("*.png"):
                samples.append({
                    "file_path": img_file,
                    "image_id": img_file.name,
                    "true_class": None
                })
        return samples

    logger.warning(f"No image samples could be located for modality '{modality}'.")
    return samples


def extract_features(
    modality: str = "gasf",
    model_path: Optional[Path] = None,
    classes_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    split: str = "test",
    output_dir: Optional[Path] = None,
    batch_size: int = 32,
    feature_layer_name: str = "global_avg_pool",
    include_feature_cols_in_csv: bool = False,
    visualize_pca: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Main execution pipeline for CNN feature extraction and prediction generation.
    """
    mod_key = "gasf" if "gasf" in modality.lower() else "iv"
    models_dir = config.PROJECT_ROOT / "models" / "cnn"

    if model_path is None:
        model_path = models_dir / f"efficientnetv2s_{mod_key}_best.keras"
    else:
        model_path = Path(model_path)

    # 1. Verify model availability
    if not check_model_availability(model_path, mod_key):
        return None

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow must be installed to run feature extraction.")

    # 2. Resolve output directories
    if output_dir is None:
        output_dir = config.PROJECT_ROOT / "results" / "cnn" / "features"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load class names
    if classes_path is None:
        classes_path = models_dir / f"{mod_key}_class_names.json"
    if classes_path and Path(classes_path).exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = config.PV_FAULT_CLASSES

    logger.info(f"Active classes ({len(class_names)}): {class_names}")

    # 4. Discover image samples
    samples = discover_image_samples(
        data_dir=data_dir,
        manifest_path=manifest_path,
        modality=mod_key,
        split=split
    )

    if not samples:
        logger.error(f"No images discovered for modality '{mod_key}'. Feature extraction aborted.")
        return None

    total_images = len(samples)
    logger.info(f"Loaded {total_images} image samples for {mod_key.upper()} feature extraction.")

    # 5. Load model & construct feature extractor
    logger.info(f"Loading trained CNN model: {model_path}")
    base_full_model = tf.keras.models.load_model(str(model_path))
    feature_extractor, active_layer_name, feature_dim = build_feature_extractor_model(
        base_full_model,
        preferred_layer_name=feature_layer_name
    )

    # 6. Batched Inference & Feature Extraction
    feature_vectors = []
    metadata_rows = []

    logger.info(f"Starting batched feature extraction (batch_size={batch_size})...")
    num_batches = int(np.ceil(total_images / batch_size))

    for b_idx in range(num_batches):
        batch_samples = samples[b_idx * batch_size : (b_idx + 1) * batch_size]
        batch_arrays = []

        for s in batch_samples:
            arr = preprocess_image_array(s["file_path"], modality=mod_key)
            batch_arrays.append(arr)

        batch_tensor = np.stack(batch_arrays, axis=0)

        # Forward pass: returns [probabilities, features]
        batch_probs, batch_feats = feature_extractor.predict(batch_tensor, verbose=0)

        for i, s in enumerate(batch_samples):
            feat = batch_feats[i].flatten()
            probs = batch_probs[i]
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            conf = float(probs[pred_idx])

            feat_idx = len(feature_vectors)
            feature_vectors.append(feat)

            row = {
                "image_id": s["image_id"],
                "true_class": s["true_class"] if s["true_class"] is not None else "unlabeled",
                "predicted_class": pred_class,
                "confidence": round(conf, 4),
                "feature_vector_reference": f"idx_{feat_idx:06d}"
            }

            # Add class probability columns
            for cls_idx, cls_name in enumerate(class_names):
                row[f"prob_{cls_name}"] = round(float(probs[cls_idx]), 4)

            metadata_rows.append(row)

        if (b_idx + 1) % max(1, (num_batches // 10)) == 0 or (b_idx + 1) == num_batches:
            logger.info(f"Processed batch {b_idx + 1}/{num_batches} ({min((b_idx + 1) * batch_size, total_images)}/{total_images} images)")

    # 7. Convert feature representations to NumPy matrix
    feature_matrix = np.array(feature_vectors, dtype=np.float32)
    logger.info(f"Extracted feature matrix shape: {feature_matrix.shape}")

    # 8. Save Features & Metadata
    npy_file = output_dir / f"{mod_key}_features.npy"
    csv_file = output_dir / f"{mod_key}_features.csv"
    meta_json_file = output_dir / f"{mod_key}_feature_metadata.json"

    # Save NumPy array
    np.save(npy_file, feature_matrix)
    logger.info(f"Saved feature matrix to: {npy_file}")

    # Build DataFrame
    df_meta = pd.DataFrame(metadata_rows)

    if include_feature_cols_in_csv:
        # Append feature columns feature_0001, feature_0002...
        feat_cols = [f"feature_{i+1:04d}" for i in range(feature_matrix.shape[1])]
        df_feats = pd.DataFrame(feature_matrix, columns=feat_cols)
        df_combined = pd.concat([df_meta, df_feats], axis=1)
        df_combined.to_csv(csv_file, index=False)
        logger.info(f"Saved metadata + {len(feat_cols)} feature columns to CSV: {csv_file}")
    else:
        df_meta.to_csv(csv_file, index=False)
        logger.info(f"Saved structured prediction metadata to CSV: {csv_file}")

    # Save extraction metadata JSON
    extraction_meta = {
        "model_name": "EfficientNetV2-S",
        "modality": mod_key,
        "input_image_size": [224, 224, 3],
        "feature_extraction_layer": active_layer_name,
        "feature_dimension": int(feature_matrix.shape[1]),
        "total_samples": int(feature_matrix.shape[0]),
        "class_names": class_names,
        "preprocessing_method": "direct_resize" if mod_key == "gasf" else "aspect_ratio_preserving_padding",
        "model_path": str(model_path),
        "extraction_datetime": datetime.now().isoformat(),
        "npy_features_path": str(npy_file),
        "metadata_csv_path": str(csv_file)
    }

    with open(meta_json_file, "w", encoding="utf-8") as f:
        json.dump(extraction_meta, f, indent=4)
    logger.info(f"Saved feature extraction metadata to: {meta_json_file}")

    # 9. Generate PCA Visualization (if requested and labels present)
    if visualize_pca and len(feature_matrix) >= 2:
        pca_plot_path = output_dir / f"{mod_key}_feature_pca.png"
        generate_feature_pca_visualization(
            feature_matrix=feature_matrix,
            labels=df_meta["true_class"].values,
            class_names=class_names,
            output_path=pca_plot_path,
            modality=mod_key
        )

    return extraction_meta


def generate_feature_pca_visualization(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_path: Path,
    modality: str
):
    """
    Computes 2D Principal Component Analysis (PCA) projection of extracted feature vectors
    and saves a high-resolution scatter plot.

    Note: PCA is employed solely as an unsupervised 2D projection tool for visual inspection.
    It does NOT serve as proof of classification quality or hybrid model performance.
    """
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("Scikit-learn or Matplotlib not available for PCA visualization.")
        return

    logger.info("Computing 2D PCA projection of extracted CNN features...")
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(feature_matrix)
    var_exp = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("tab10", n_colors=len(class_names))
    label_to_color = {cls: palette[i] for i, cls in enumerate(class_names)}

    for cls in class_names:
        mask = (labels == cls)
        if np.any(mask):
            plt.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                label=cls,
                alpha=0.6,
                s=25,
                color=label_to_color.get(cls, "#333333"),
                edgecolors="none"
            )

    # Plot unlabeled points if any
    unlabeled_mask = ~np.isin(labels, class_names)
    if np.any(unlabeled_mask):
        plt.scatter(
            coords_2d[unlabeled_mask, 0],
            coords_2d[unlabeled_mask, 1],
            label="unlabeled",
            alpha=0.3,
            s=15,
            color="#888888",
            edgecolors="none"
        )

    plt.title(
        f"EfficientNetV2-S Feature Representation — PCA ({modality.upper()})\n"
        f"Total Variance Explained: {(var_exp[0] + var_exp[1]) * 100:.1f}% "
        f"(PC1: {var_exp[0] * 100:.1f}%, PC2: {var_exp[1] * 100:.1f}%)",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    plt.xlabel(f"Principal Component 1 ({var_exp[0] * 100:.1f}% Variance)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_exp[1] * 100:.1f}% Variance)", fontsize=11)
    plt.legend(title="PV Fault Class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved PCA feature visualization to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="PVision AI: CNN Feature Extraction Pipeline (EfficientNetV2-S)"
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="gasf",
        choices=["gasf", "iv"],
        help="Image modality to extract features for ('gasf' or 'iv')"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained .keras model file (defaults to models/cnn/efficientnetv2s_{modality}_best.keras)"
    )
    parser.add_argument(
        "--classes-path",
        type=str,
        default=None,
        help="Path to class_names.json"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Path to manifest CSV"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train", "all"],
        help="Dataset split to extract features from if using standard manifests (default: test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for features and metadata (default: results/cnn/features)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction inference (default: 32)"
    )
    parser.add_argument(
        "--feature-layer",
        type=str,
        default="global_avg_pool",
        help="Intermediate layer name for feature extraction (default: global_avg_pool)"
    )
    parser.add_argument(
        "--include-feature-cols",
        action="store_true",
        help="Include full feature columns in the CSV output (may create large files)"
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Skip PCA 2D scatter plot visualization"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_features(
        modality=args.modality,
        model_path=Path(args.model_path) if args.model_path else None,
        classes_path=Path(args.classes_path) if args.classes_path else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        split=args.split,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        batch_size=args.batch_size,
        feature_layer_name=args.feature_layer,
        include_feature_cols_in_csv=args.include_feature_cols,
        visualize_pca=not args.no_pca
    )
