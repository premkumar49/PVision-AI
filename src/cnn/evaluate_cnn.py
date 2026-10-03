"""
PVision AI: CNN Evaluation & Diagnostics Pipeline
Evaluates a trained EfficientNetV2-S model on the unseen test partition.

Calculates:
- Global Accuracy, Precision, Recall, Macro F1-Score
- Per-class Precision, Recall, F1-Score
- Full 7-Class Confusion Matrix
- Training and Validation Loss / Accuracy Learning Curves

IMPORTANT:
Does NOT compute or display fabricated values.
If a trained model (.keras) is not detected, it reports that manual training
in VS Code must be executed first and exits cleanly.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Path resolution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utilities.config import config
from src.utilities.logger import get_logger
from src.cnn.data_loader import load_manifests, build_tf_datasets

logger = get_logger("CNN_Evaluation")


def evaluate_cnn(
    modality: str = "gasf",
    model_path: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    image_size: int = 224,
    batch_size: int = 32
):
    """
    Evaluates a trained CNN model on the test dataset and exports metrics and figures.
    """
    mod_key = "gasf" if "gasf" in modality.lower() else "iv"
    manifest_dir = Path(manifest_dir) if manifest_dir else (config.PROJECT_ROOT / "data" / "processed")
    models_dir = config.PROJECT_ROOT / "models" / "cnn"
    
    if not model_path:
        model_path = models_dir / f"efficientnetv2s_{mod_key}_best.keras"
    else:
        model_path = Path(model_path)

    results_dir = Path(results_dir) if results_dir else (config.PROJECT_ROOT / "results" / "cnn" / mod_key)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"PVision AI: CNN EVALUATION FOR [{mod_key.upper()} MODALITY]")
    logger.info(f"Target Model: {model_path}")
    logger.info(f"Results Directory: {results_dir}")
    logger.info("=" * 70)

    # Check if trained model exists
    if not model_path.exists():
        logger.warning(
            f"\n[NOTICE] Trained model checkpoint not found at: {model_path}\n"
            f"Evaluation cannot be performed because training must be executed manually in VS Code first.\n\n"
            f"To train the model manually in VS Code, execute:\n"
            f"  python src/cnn/train_cnn.py --modality {mod_key} --epochs-stage1 10 --epochs-stage2 20\n\n"
            f"Once training finishes and '{model_path.name}' is created, re-run this evaluation script."
        )
        return None

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow must be installed to load and evaluate the model.")

    logger.info("Loading trained EfficientNetV2-S model...")
    model = tf.keras.models.load_model(str(model_path))

    # Load Manifests & Build Test Pipeline
    _, _, test_df = load_manifests(manifest_dir=manifest_dir, modality=mod_key)
    logger.info(f"Test Set: {len(test_df):,} samples.")

    _, _, test_ds = build_tf_datasets(
        train_df=test_df,  # Dummy pass for loader
        val_df=test_df,
        test_df=test_df,
        modality=mod_key,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=config.RANDOM_SEED
    )

    # Predict Probabilities
    logger.info("Running inference across test partition...")
    y_true = test_df["label"].values
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_names = config.PV_FAULT_CLASSES

    # 1. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )

    logger.info("\n--- TEST EVALUATION SUMMARY ---")
    logger.info(f"Accuracy:        {acc * 100:.2f}%")
    logger.info(f"Macro Precision: {prec_macro * 100:.2f}%")
    logger.info(f"Macro Recall:    {rec_macro * 100:.2f}%")
    logger.info(f"Macro F1-Score:  {f1_macro * 100:.2f}%")

    # 2. Save Classification Report CSV
    clf_rep_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    clf_df = pd.DataFrame(clf_rep_dict).transpose()
    clf_csv_path = results_dir / "classification_report.csv"
    clf_df.to_csv(clf_csv_path)
    logger.info(f"Saved classification report to: {clf_csv_path}")

    # 3. Save Per-Class Metrics CSV
    per_class_df = pd.DataFrame({
        "Class": class_names,
        "Precision": prec_per_class,
        "Recall": rec_per_class,
        "F1_Score": f1_per_class,
        "Support": support_per_class
    })
    per_class_csv = results_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    logger.info(f"Saved per-class metrics to: {per_class_csv}")

    # 4. Generate & Save Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5
    )
    plt.title(f"PV Fault Confusion Matrix - {mod_key.upper()} (EfficientNetV2-S)", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("Actual Class", fontsize=10)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {cm_path}")

    # 5. Class Distribution Plot of Test Predictions
    plt.figure(figsize=(9, 5))
    df_compare = pd.DataFrame({
        "Actual": pd.Series(y_true).value_counts().sort_index(),
        "Predicted": pd.Series(y_pred).value_counts().sort_index()
    }, index=range(len(class_names)))
    df_compare.index = class_names
    df_compare.plot(kind="bar", figsize=(9, 5), color=["#2b5c8f", "#d95f02"], alpha=0.85)
    plt.title(f"Actual vs. Predicted Class Distribution - {mod_key.upper()}", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    dist_path = results_dir / "class_distribution.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()
    logger.info(f"Saved class distribution plot to: {dist_path}")

    # 6. Plot Learning Curves from Training History
    history_file = models_dir / f"{mod_key}_training_history.json"
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            hist = json.load(f)

        epochs = range(1, len(hist.get("loss", [])) + 1)

        # Accuracy Plot
        plt.figure(figsize=(8, 4.5))
        if "accuracy" in hist:
            plt.plot(epochs, hist["accuracy"], "b-o", label="Training Accuracy")
        if "val_accuracy" in hist:
            plt.plot(epochs, hist["val_accuracy"], "r--s", label="Validation Accuracy")
        plt.title(f"Accuracy Curve - {mod_key.upper()} (EfficientNetV2-S)", fontsize=11, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        acc_path = results_dir / "training_accuracy.png"
        plt.savefig(acc_path, dpi=200)
        plt.savefig(results_dir / "validation_accuracy.png", dpi=200)
        plt.close()

        # Loss Plot
        plt.figure(figsize=(8, 4.5))
        if "loss" in hist:
            plt.plot(epochs, hist["loss"], "b-o", label="Training Loss")
        if "val_loss" in hist:
            plt.plot(epochs, hist["val_loss"], "r--s", label="Validation Loss")
        plt.title(f"Loss Curve - {mod_key.upper()} (EfficientNetV2-S)", fontsize=11, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        loss_path = results_dir / "training_loss.png"
        plt.savefig(loss_path, dpi=200)
        plt.savefig(results_dir / "validation_loss.png", dpi=200)
        plt.close()

        logger.info("Saved training and validation learning curve plots.")

    return {
        "modality": mod_key,
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "results_directory": str(results_dir)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="PVision AI: Evaluate Trained EfficientNetV2-S Model")
    parser.add_argument("--modality", type=str, default="gasf", choices=["gasf", "iv"],
                        help="Modality to evaluate ('gasf' or 'iv')")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Explicit path to .keras model checkpoint")
    parser.add_argument("--manifest-dir", type=str, default=None,
                        help="Path to manifest CSVs (default: data/processed)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Output directory for evaluation metrics and charts")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Image resolution (default: 224)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_cnn(
        modality=args.modality,
        model_path=Path(args.model_path) if args.model_path else None,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        image_size=args.image_size,
        batch_size=args.batch_size
    )
