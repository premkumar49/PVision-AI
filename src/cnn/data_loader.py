"""
PVision AI: CNN Data Pipeline & Preprocessing Loader
Implements:
1. GASF Preprocessing: 256x256 RGB -> resize to 224x224 -> EfficientNetV2 preprocessing
2. I-V Preprocessing: 1343x808 RGB -> Aspect-Ratio-Preserving Resize & Padding to 224x224 -> EfficientNetV2 preprocessing
3. Data Augmentation: Applied strictly to training split (flips, slight rotations, zoom, brightness)
4. Split Manifest Loading: Reproducible 70% Train / 15% Val / 15% Test partitions
5. Class Weight Calculation: Inversely proportional to sample frequencies for handling I-V class imbalance
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np

# Path resolution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utilities.config import config
from src.utilities.logger import get_logger

logger = get_logger("CNN_DataLoader")


def compute_class_weights(labels: List[int], num_classes: int = 7) -> Dict[int, float]:
    """
    Computes balanced class weights:
    weight[c] = total_samples / (num_classes * count[c])
    """
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = {}

    for c in range(num_classes):
        if counts[c] > 0:
            class_weights[c] = float(total_samples / (num_classes * counts[c]))
        else:
            class_weights[c] = 1.0

    return class_weights


def load_manifests(
    manifest_dir: Path,
    modality: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, validation, and test manifest CSVs.
    Falls back to generating them if not present.
    """
    mod_key = "gasf_images" if "gasf" in modality.lower() else "iv_images"
    train_file = manifest_dir / f"{mod_key}_train_manifest.csv"
    val_file = manifest_dir / f"{mod_key}_val_manifest.csv"
    test_file = manifest_dir / f"{mod_key}_test_manifest.csv"

    if train_file.exists() and val_file.exists() and test_file.exists():
        logger.info(f"Loading pre-generated manifests for {mod_key} from {manifest_dir}")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
    else:
        logger.warning(f"Manifests missing in {manifest_dir}. Generating fresh stratified splits...")
        from src.preprocessing.clean_cnn_data import CNNDataCleaner
        cleaner = CNNDataCleaner()
        cleaner.clean_and_split()
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)

    return train_df, val_df, test_df


def build_tf_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    modality: str = "gasf",
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42
):
    """
    Builds optimized tf.data.Dataset pipelines for training, validation, and testing.
    Applies modality-specific aspect-ratio-preserving padding and training augmentations.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow must be installed to create tf.data pipelines.")

    is_iv = "iv" in modality.lower()
    target_h, target_w = image_size

    def preprocess_image(file_path, label, is_training: bool):
        # 1. Read and decode PNG
        img_bytes = tf.io.read_file(file_path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.cast(img, tf.float32)

        # 2. Modality-specific spatial transformation
        if is_iv:
            # Aspect-ratio-preserving resize and pad to 224x224 (No geometric stretching!)
            img = tf.image.resize_with_pad(
                img,
                target_height=target_h,
                target_width=target_w,
                method=tf.image.ResizeMethod.BILINEAR
            )
        else:
            # GASF images are native 256x256 squares -> Direct bilinear resize to 224x224
            img = tf.image.resize(
                img,
                [target_h, target_w],
                method=tf.image.ResizeMethod.BILINEAR
            )

        # 3. Data Augmentation (STRICTLY during training)
        if is_training:
            # Random horizontal flip
            img = tf.image.random_flip_left_right(img, seed=seed)
            # Random vertical flip (preserves symmetry in 2D fields/plots)
            img = tf.image.random_flip_up_down(img, seed=seed)
            # Subtle brightness adjustment (max delta = 8%)
            img = tf.image.random_brightness(img, max_delta=0.08, seed=seed)
            # Subtle contrast adjustment (range: 0.92 to 1.08)
            img = tf.image.random_contrast(img, lower=0.92, upper=1.08, seed=seed)

        # 4. EfficientNetV2 Preprocessing
        # EfficientNetV2 expects input pixel values in range [0, 255] or handles rescaling internally
        # We clip to valid range [0, 255]
        img = tf.clip_by_value(img, 0.0, 255.0)

        # One-hot encode label for 7 classes
        label_one_hot = tf.one_hot(label, depth=7)

        return img, label_one_hot

    # Build tf.data pipeline
    def create_dataset(df: pd.DataFrame, is_training: bool):
        file_paths = tf.constant(df["file_path"].values, dtype=tf.string)
        labels = tf.constant(df["label"].values, dtype=tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        if is_training:
            ds = ds.shuffle(buffer_size=min(len(df), 5000), seed=seed, reshuffle_each_iteration=True)

        ds = ds.map(
            lambda fp, lbl: preprocess_image(fp, lbl, is_training=is_training),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    train_ds = create_dataset(train_df, is_training=True)
    val_ds = create_dataset(val_df, is_training=False)
    test_ds = create_dataset(test_df, is_training=False)

    return train_ds, val_ds, test_ds
