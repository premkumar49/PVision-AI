"""
PVision AI: Artificial Neural Network (ANN) Evaluation Pipeline
Evaluates the trained ANN regression model on the chronological holdout test set (final 15%).
Calculates:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
Generates:
- reports/figures/ann_actual_vs_predicted.png
- reports/figures/ann_residual_plot.png
- reports/figures/ann_training_validation_loss.png
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

# Path resolution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utilities.config import config
from src.utilities.logger import get_logger

logger = get_logger("ANN_Evaluation")


def check_artifacts_availability(
    model_path: Path,
    scaler_path: Path,
    features_path: Path
) -> bool:
    """
    Checks if trained model and preprocessing artifacts exist.
    If missing, informs the user clearly without raising an unhandled exception.
    """
    missing = []
    if not model_path.exists():
        missing.append(f"Model checkpoint: {model_path}")
    if not scaler_path.exists():
        missing.append(f"Feature scaler: {scaler_path}")
    if not features_path.exists():
        missing.append(f"Feature list schema: {features_path}")

    if missing:
        print("\n" + "=" * 65)
        print("           PVISION AI - ANN EVALUATION NOTICE")
        print("=" * 65)
        print("Trained ANN model artifacts are pending training.")
        print("Missing items:")
        for item in missing:
            print(f"  * {item}")
        print("\nPlease execute model training manually in VS Code using:")
        print("  python src/ann/train_ann.py")
        print("=" * 65 + "\n")
        return False
    return True


def load_test_dataset(
    processed_test_path: Path,
    target_col: str = "AC_POWER"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads pre-scaled chronological test dataset if available."""
    features_path = config.PROJECT_ROOT / "models" / "scaler" / "ann_features.json"
    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    if processed_test_path.exists():
        logger.info(f"Loading pre-split chronological test set from: {processed_test_path}")
        df_test = pd.read_csv(processed_test_path)
        X_test = df_test[feature_names].values
        y_test = df_test[target_col].values
        return X_test, y_test, feature_names

    raise FileNotFoundError(
        f"Chronological test dataset not found at: {processed_test_path}. "
        f"Please run 'python src/ann/train_ann.py' first to prepare data partitions."
    )


def generate_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history_path: Path,
    output_dir: Path
):
    """
    Generates high-resolution publication-quality evaluation figures:
    1. Actual vs. Predicted Power Scatter Plot
    2. Residual Plot (Actual - Predicted vs. Predicted)
    3. Training vs. Validation Loss Convergence Curve
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        logger.warning("Matplotlib or Seaborn not available. Skipping plot generation.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Actual vs. Predicted Power
    plt.figure(figsize=(8, 7))
    plt.scatter(y_true, y_pred, alpha=0.35, color="#1f77b4", s=18, edgecolors="none")
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y = x)")
    plt.title("ANN Solar Power Prediction: Actual vs. Predicted AC Power", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Actual AC Power (kW)", fontsize=11)
    plt.ylabel("Predicted AC Power (kW)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    actual_vs_pred_file = output_dir / "ann_actual_vs_predicted.png"
    plt.savefig(actual_vs_pred_file, dpi=300)
    plt.close()
    logger.info(f"Saved Actual vs. Predicted plot to: {actual_vs_pred_file}")

    # 2. Residual Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(8.5, 6))
    plt.scatter(y_pred, residuals, alpha=0.35, color="#2ca02c", s=18, edgecolors="none")
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Reference")
    plt.title("ANN Solar Power Prediction: Residual Analysis", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted AC Power (kW)", fontsize=11)
    plt.ylabel("Residual [Actual - Predicted] (kW)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    residual_file = output_dir / "ann_residual_plot.png"
    plt.savefig(residual_file, dpi=300)
    plt.close()
    logger.info(f"Saved Residual plot to: {residual_file}")

    # 3. Training vs. Validation Loss Convergence Curve
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        loss = hist.get("loss", [])
        val_loss = hist.get("val_loss", [])
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(9, 5.5))
        plt.plot(epochs, loss, label="Training Loss (MSE)", color="#1f77b4", linewidth=2)
        plt.plot(epochs, val_loss, label="Validation Loss (MSE)", color="#ff7f0e", linewidth=2)
        plt.title("ANN Training & Validation Loss Across Epochs", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Mean Squared Error (MSE)", fontsize=11)
        plt.legend(frameon=True)
        plt.tight_layout()
        loss_file = output_dir / "ann_training_validation_loss.png"
        plt.savefig(loss_file, dpi=300)
        plt.close()
        logger.info(f"Saved Training/Validation Loss plot to: {loss_file}")


def evaluate_ann_model(
    model_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
    test_data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Optional[Dict[str, float]]:
    """
    Main evaluation pipeline:
    1. Loads artifacts & model
    2. Runs forward inference on chronological test set
    3. Calculates MAE, RMSE, R²
    4. Generates diagnostic plots
    """
    models_dir = config.PROJECT_ROOT / "models"
    reports_figures_dir = config.PROJECT_ROOT / "reports" / "figures"

    model_p = model_path if model_path else (models_dir / "ann" / "best_ann_model.keras")
    scaler_p = scaler_path if scaler_path else (models_dir / "scaler" / "ann_feature_scaler.pkl")
    features_p = features_path if features_path else (models_dir / "scaler" / "ann_features.json")
    test_p = test_data_path if test_data_path else (config.PROJECT_ROOT / "data" / "processed" / "ann_test_chronological.csv")
    out_dir = output_dir if output_dir else reports_figures_dir
    history_p = models_dir / "ann" / "training_history.json"

    # Verify model availability
    if not check_artifacts_availability(model_p, scaler_p, features_p):
        return None

    try:
        import tensorflow as tf
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError:
        raise ImportError("TensorFlow and Scikit-learn must be installed to evaluate the ANN.")

    # Load Model
    logger.info(f"Loading trained ANN model checkpoint: {model_p}")
    model = tf.keras.models.load_model(str(model_p))

    # Load Test Data
    X_test, y_test, feature_names = load_test_dataset(test_p)
    logger.info(f"Evaluating model on {len(X_test):,} chronological holdout test samples...")

    # Predict
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Calculate Regression Metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    # Print Clean Evaluation Table
    print("\n" + "=" * 50)
    print("           ANN MODEL TEST EVALUATION")
    print("=" * 50)
    print(f"{'Metric':<12} | {'Value'}")
    print("-" * 50)
    print(f"{'MAE':<12} | {mae:10.4f} kW")
    print(f"{'RMSE':<12} | {rmse:10.4f} kW")
    print(f"{'R²':<12} | {r2:10.4f}")
    print("=" * 50 + "\n")

    # Generate Evaluation Figures
    generate_evaluation_plots(
        y_true=y_test,
        y_pred=y_pred,
        history_path=history_p,
        output_dir=out_dir
    )

    # Save metrics JSON
    metrics_dir = config.PROJECT_ROOT / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "ann_evaluation_metrics.json"
    metrics_payload = {
        "model": "PVision_ANN_PowerPredictor",
        "target": "AC_POWER",
        "unit": "kW",
        "test_samples": int(len(y_test)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4)
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    logger.info(f"Saved evaluation metrics to: {metrics_file}")

    return metrics_payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="PVision AI: Evaluate Trained ANN Power Prediction Model"
    )
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to best_ann_model.keras")
    parser.add_argument("--scaler-path", type=str, default=None,
                        help="Path to ann_feature_scaler.pkl")
    parser.add_argument("--features-path", type=str, default=None,
                        help="Path to ann_features.json")
    parser.add_argument("--test-data-path", type=str, default=None,
                        help="Path to test CSV")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_ann_model(
        model_path=Path(args.model_path) if args.model_path else None,
        scaler_path=Path(args.scaler_path) if args.scaler_path else None,
        features_path=Path(args.features_path) if args.features_path else None,
        test_data_path=Path(args.test_data_path) if args.test_data_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
