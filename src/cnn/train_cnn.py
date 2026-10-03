"""
PVision AI: CNN Training Pipeline (EfficientNetV2-S)
Executes Two-Stage Transfer Learning for Photovoltaic Fault Classification:
- Stage 1: Feature Extraction (Backbone frozen, trains custom 7-class head)
- Stage 2: Fine-Tuning (Upper layers unfrozen, trained with reduced learning rate)

IMPORTANT:
This script is designed for manual execution in VS Code.
It does NOT execute automatically upon import.
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import numpy as np

# Path resolution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utilities.config import config
from src.utilities.logger import get_logger
from src.cnn.model_factory import build_cnn_model, configure_fine_tuning
from src.cnn.data_loader import load_manifests, build_tf_datasets, compute_class_weights

logger = get_logger("CNN_Training")


def detect_and_configure_hardware():
    """Detects available GPU devices and configures CPU fallback if unavailable."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"GPU Detected: {len(gpus)} device(s) found.")
            for gpu in gpus:
                logger.info(f" - {gpu.name}")
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    logger.warning(f"Could not enable memory growth: {e}")
            device_name = gpus[0].name
        else:
            logger.info("No GPU detected. Falling back to CPU for training.")
            device_name = "CPU"
        return device_name
    except ImportError:
        logger.warning("TensorFlow not installed. Hardware detection skipped.")
        return "Unknown"


def set_reproducible_seed(seed: int = 42):
    """Sets deterministic random seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def train_cnn(
    modality: str = "gasf",
    manifest_dir: Path = None,
    output_dir: Path = None,
    model_name: str = "efficientnetv2-s",
    image_size: int = 224,
    batch_size: int = 32,
    epochs_stage1: int = 10,
    epochs_stage2: int = 20,
    lr: float = 1e-3,
    fine_tune_lr: float = 1e-4,
    unfreeze_layers: int = 50,
    use_class_weights: bool = False,
    optimizer_name: str = "adam",
    seed: int = 42
):
    """
    Orchestrates Stage 1 and Stage 2 transfer learning for the specified image modality.
    """
    set_reproducible_seed(seed)
    hardware = detect_and_configure_hardware()

    try:
        import tensorflow as tf
        from tensorflow.keras import callbacks, optimizers, losses, metrics
    except ImportError:
        raise ImportError(
            "TensorFlow is required to train the model. "
            "Please install it in your VS Code environment: pip install tensorflow"
        )

    # Validate Modality
    mod_lower = modality.lower()
    if mod_lower not in ["gasf", "iv", "gasf_images", "iv_images"]:
        raise ValueError(f"Unknown modality: {modality}. Choose 'gasf' or 'iv'.")
    mod_key = "gasf" if "gasf" in mod_lower else "iv"

    # Directory Setup
    manifest_dir = Path(manifest_dir) if manifest_dir else (config.PROJECT_ROOT / "data" / "processed")
    output_dir = Path(output_dir) if output_dir else (config.PROJECT_ROOT / "models" / "cnn")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"STARTING CNN TRAINING PIPELINE FOR: [{mod_key.upper()} MODALITY]")
    logger.info(f"Architecture: {model_name.upper()} | Input Size: {image_size}x{image_size}x3")
    logger.info(f"Stage 1 Epochs: {epochs_stage1} (LR: {lr}) | Stage 2 Epochs: {epochs_stage2} (LR: {fine_tune_lr})")
    logger.info(f"Hardware: {hardware} | Random Seed: {seed}")
    logger.info("=" * 70)

    # 1. Load Manifests
    train_df, val_df, test_df = load_manifests(manifest_dir=manifest_dir, modality=mod_key)
    logger.info(f"Loaded Datasets: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # 2. Build tf.data Pipelines
    train_ds, val_ds, test_ds = build_tf_datasets(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        modality=mod_key,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=seed
    )

    # 3. Class Weighting Configuration
    class_weights_dict = None
    if use_class_weights or (mod_key == "iv" and use_class_weights):
        class_weights_dict = compute_class_weights(train_df["label"].tolist(), num_classes=7)
        logger.info(f"Class weights enabled: {class_weights_dict}")
    else:
        logger.info("Class weights disabled (uniform sample weighting).")

    # 4. Model Construction (Stage 1: Frozen Backbone)
    model, base_model = build_cnn_model(
        model_name=model_name,
        input_shape=(image_size, image_size, 3),
        num_classes=7,
        dropout_rate=0.3,
        pretrained=True
    )

    # Select Optimizer
    if optimizer_name.lower() == "adamw":
        opt_s1 = optimizers.AdamW(learning_rate=lr)
        opt_s2 = optimizers.AdamW(learning_rate=fine_tune_lr)
    elif optimizer_name.lower() == "sgd":
        opt_s1 = optimizers.SGD(learning_rate=lr, momentum=0.9)
        opt_s2 = optimizers.SGD(learning_rate=fine_tune_lr, momentum=0.9)
    else:
        opt_s1 = optimizers.Adam(learning_rate=lr)
        opt_s2 = optimizers.Adam(learning_rate=fine_tune_lr)

    loss_fn = losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(
        optimizer=opt_s1,
        loss=loss_fn,
        metrics=[metrics.CategoricalAccuracy(name="accuracy"), metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")]
    )

    # 5. Callbacks
    best_model_path = output_dir / f"efficientnetv2s_{mod_key}_best.keras"
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
        mode="min"
    )
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
        mode="min"
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
        mode="min"
    )

    # -------------------------------------------------------------
    # STAGE 1: Train Classification Head
    # -------------------------------------------------------------
    logger.info(">>> COMMENCING STAGE 1: Training Classification Head <<<")
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        class_weight=class_weights_dict,
        callbacks=[cb_checkpoint, cb_early_stop, cb_reduce_lr],
        verbose=1
    )

    # -------------------------------------------------------------
    # STAGE 2: Unfreeze Upper Layers & Fine-Tune
    # -------------------------------------------------------------
    logger.info(f">>> COMMENCING STAGE 2: Fine-Tuning Top {unfreeze_layers} Backbone Layers <<<")
    model, trainable_layers = configure_fine_tuning(model, base_model, unfreeze_layers=unfreeze_layers)
    logger.info(f"Unfrozen {trainable_layers} layers for fine-tuning.")

    model.compile(
        optimizer=opt_s2,
        loss=loss_fn,
        metrics=[metrics.CategoricalAccuracy(name="accuracy"), metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")]
    )

    total_epochs = epochs_stage1 + epochs_stage2
    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=len(history_stage1.epoch),
        epochs=total_epochs,
        class_weight=class_weights_dict,
        callbacks=[cb_checkpoint, cb_early_stop, cb_reduce_lr],
        verbose=1
    )

    # 6. Save Artifacts & Training History
    logger.info("Merging training histories across Stage 1 and Stage 2...")
    combined_history = {}
    for key in history_stage1.history.keys():
        s1_vals = [float(v) for v in history_stage1.history[key]]
        s2_vals = [float(v) for v in history_stage2.history.get(key, [])]
        combined_history[key] = s1_vals + s2_vals

    history_file = output_dir / f"{mod_key}_training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=4)
    logger.info(f"Saved training history to: {history_file}")

    # Class names mapping
    class_names_file = output_dir / f"{mod_key}_class_names.json"
    with open(class_names_file, "w", encoding="utf-8") as f:
        json.dump(config.PV_FAULT_CLASSES, f, indent=4)
    logger.info(f"Saved class names to: {class_names_file}")

    # Model configuration
    model_config = {
        "model_name": model_name,
        "modality": mod_key,
        "image_size": [image_size, image_size],
        "batch_size": batch_size,
        "epochs_stage1": epochs_stage1,
        "epochs_stage2": epochs_stage2,
        "lr_stage1": lr,
        "lr_stage2": fine_tune_lr,
        "optimizer": optimizer_name,
        "unfreeze_layers": unfreeze_layers,
        "class_weights_used": class_weights_dict is not None,
        "checkpoint_metric": "val_loss (min)",
        "random_seed": seed,
        "training_completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    config_file = output_dir / f"{mod_key}_model_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=4)
    logger.info(f"Saved model config to: {config_file}")

    logger.info(f"SUCCESS: Trained {mod_key.upper()} model checkpoint saved to: {best_model_path}")
    return model, combined_history


def parse_args():
    parser = argparse.ArgumentParser(description="PVision AI: Train EfficientNetV2-S for PV Fault Classification")
    parser.add_argument("--modality", type=str, default="gasf", choices=["gasf", "iv"],
                        help="Image modality to train ('gasf' or 'iv')")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to root image directory (default: data/pv_fault)")
    parser.add_argument("--manifest-dir", type=str, default=None,
                        help="Path to manifest CSVs (default: data/processed)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to save models and metrics (default: models/cnn)")
    parser.add_argument("--model-name", type=str, default="efficientnetv2-s",
                        choices=["efficientnetv2-s", "convnext-tiny", "mobilenetv3-large"],
                        help="Architecture backbone")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input square image resolution (default: 224)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size (default: 32)")
    parser.add_argument("--epochs-stage1", type=int, default=10,
                        help="Epochs for training head with frozen backbone (default: 10)")
    parser.add_argument("--epochs-stage2", type=int, default=20,
                        help="Epochs for fine-tuning upper layers (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate for Stage 1 (default: 1e-3)")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4,
                        help="Fine-tuning learning rate for Stage 2 (default: 1e-4)")
    parser.add_argument("--unfreeze-layers", type=int, default=50,
                        help="Number of upper layers to unfreeze in Stage 2 (default: 50)")
    parser.add_argument("--use-class-weights", action="store_true",
                        help="Apply inverse frequency class weights during training")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"],
                        help="Optimizer algorithm (default: adam)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic random seed (default: 42)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_cnn(
        modality=args.modality,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr=args.lr,
        fine_tune_lr=args.fine_tune_lr,
        unfreeze_layers=args.unfreeze_layers,
        use_class_weights=args.use_class_weights,
        optimizer_name=args.optimizer,
        seed=args.seed
    )
