"""
PVision AI: Artificial Neural Network (ANN) Single-Sample & Batch Prediction Pipeline
Produces AC power generation forecasts from environmental and operational inputs:
- Loads models/ann/best_ann_model.keras
- Loads models/scaler/ann_feature_scaler.pkl
- Loads models/scaler/ann_features.json
- Applies exact scaling transformation
- Returns formatted power output in kW
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

# Path resolution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utilities.config import config
from src.utilities.logger import get_logger

logger = get_logger("ANN_Prediction")


def check_prediction_prerequisites(
    model_path: Path,
    scaler_path: Path,
    features_path: Path
) -> bool:
    """Verifies that trained model and scaler artifacts exist."""
    if not model_path.exists() or not scaler_path.exists() or not features_path.exists():
        print("\n" + "=" * 65)
        print("           PVISION AI - PREDICTION NOTICE")
        print("=" * 65)
        print("ANN model has not been trained yet. Run train_ann.py in VS Code first.")
        print("\nMissing required files:")
        if not model_path.exists():
            print(f"  * Model checkpoint: {model_path}")
        if not scaler_path.exists():
            print(f"  * Feature scaler:   {scaler_path}")
        if not features_path.exists():
            print(f"  * Feature schema:   {features_path}")
        print("=" * 65 + "\n")
        return False
    return True


def prepare_features_vector(
    input_data: Dict[str, float],
    feature_schema: List[str]
) -> np.ndarray:
    """
    Constructs the complete feature vector from provided parameters.
    Automatically derives interaction and cyclical temporal features if raw inputs are provided.
    """
    data = input_data.copy()

    # Automatically derive engineered features if not already provided
    if "HOUR" in data:
        hr = float(data["HOUR"])
        minute = float(data.get("MINUTE", 0.0))
        t_dec = hr + minute / 60.0
        if "TIME_DECIMAL" not in data:
            data["TIME_DECIMAL"] = t_dec
        if "SIN_TIME" not in data:
            data["SIN_TIME"] = np.sin(2 * np.pi * t_dec / 24.0)
        if "COS_TIME" not in data:
            data["COS_TIME"] = np.cos(2 * np.pi * t_dec / 24.0)
        if "DAY_OF_WEEK" not in data:
            data["DAY_OF_WEEK"] = 2.0  # Default Wednesday mid-week

    if "IRRADIATION" in data:
        irrad = float(data["IRRADIATION"])
        if "IS_DAYTIME" not in data:
            data["IS_DAYTIME"] = 1.0 if irrad > 0 else 0.0

    if "MODULE_TEMPERATURE" in data and "AMBIENT_TEMPERATURE" in data:
        m_temp = float(data["MODULE_TEMPERATURE"])
        a_temp = float(data["AMBIENT_TEMPERATURE"])
        if "TEMP_DIFFERENCE" not in data:
            data["TEMP_DIFFERENCE"] = m_temp - a_temp
        if "IRRADIATION" in data and "IRRAD_MODULE_INTERACTION" not in data:
            data["IRRAD_MODULE_INTERACTION"] = float(data["IRRADIATION"]) * m_temp

    # Validate that every expected feature in schema is present
    missing = [f for f in feature_schema if f not in data]
    if missing:
        raise ValueError(f"Missing required predictive features for ANN model: {missing}")

    vector = [float(data[f]) for f in feature_schema]
    return np.array(vector, dtype=np.float32).reshape(1, -1)


def predict_power(
    input_data: Union[Dict[str, float], Path],
    model_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    features_path: Optional[Path] = None
) -> Optional[float]:
    """
    Predicts PV AC power output for a given operational input dictionary or CSV file.
    """
    models_dir = config.PROJECT_ROOT / "models"
    model_p = model_path if model_path else (models_dir / "ann" / "best_ann_model.keras")
    scaler_p = scaler_path if scaler_path else (models_dir / "scaler" / "ann_feature_scaler.pkl")
    features_p = features_path if features_path else (models_dir / "scaler" / "ann_features.json")

    if not check_prediction_prerequisites(model_p, scaler_p, features_p):
        return None

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow must be installed to run ANN inference.")

    # 1. Load Feature Schema
    with open(features_p, "r", encoding="utf-8") as f:
        feature_schema = json.load(f)

    # 2. Load Fitted Scaler
    with open(scaler_p, "rb") as f:
        scaler = pickle.load(f)

    # 3. Load Trained Model Checkpoint
    model = tf.keras.models.load_model(str(model_p))

    # 4. Prepare Features
    if isinstance(input_data, dict):
        raw_vector = prepare_features_vector(input_data, feature_schema)
        scaled_vector = scaler.transform(raw_vector)
        pred_power = float(model.predict(scaled_vector, verbose=0)[0][0])
        # Physical lower bound check: solar power cannot be negative
        pred_power = max(0.0, pred_power)

        print("\n" + "=" * 45)
        print("       ANN POWER PREDICTION RESULT")
        print("=" * 45)
        print("Predicted Power:")
        print(f"{pred_power:.2f} kW")
        print("=" * 45 + "\n")
        return pred_power

    elif isinstance(input_data, (str, Path)) and Path(input_data).exists():
        df_new = pd.read_csv(Path(input_data))
        logger.info(f"Loaded batch input file with {len(df_new)} rows from {input_data}")
        # Verify schema
        missing = [f for f in feature_schema if f not in df_new.columns]
        if missing:
            raise ValueError(f"Batch CSV is missing required features: {missing}")

        X_raw = df_new[feature_schema].values
        X_scaled = scaler.transform(X_raw)
        preds = model.predict(X_scaled, verbose=0).flatten()
        preds = np.maximum(0.0, preds)

        df_out = df_new.copy()
        df_out["PREDICTED_AC_POWER_KW"] = np.round(preds, 2)
        out_csv = config.PROJECT_ROOT / "outputs" / "predictions" / "ann_power_predictions.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_csv, index=False)
        logger.info(f"Saved batch predictions to: {out_csv}")
        return float(np.mean(preds))

    else:
        raise ValueError("Invalid input data. Must be a feature dictionary or path to CSV.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="PVision AI: Predict Solar AC Power Output using Trained ANN"
    )
    parser.add_argument("--ambient-temp", type=float, default=28.5,
                        help="Ambient temperature in °C (default: 28.5)")
    parser.add_argument("--module-temp", type=float, default=42.0,
                        help="Module surface temperature in °C (default: 42.0)")
    parser.add_argument("--irradiation", type=float, default=0.65,
                        help="Solar irradiation level (default: 0.65)")
    parser.add_argument("--hour", type=float, default=12.0,
                        help="Hour of the day 0-23 (default: 12.0)")
    parser.add_argument("--minute", type=float, default=0.0,
                        help="Minute of the hour 0-59 (default: 0.0)")
    parser.add_argument("--day-of-week", type=float, default=2.0,
                        help="Day of week 0-6 (default: 2.0)")
    parser.add_argument("--input-csv", type=str, default=None,
                        help="Path to CSV containing multiple operational records")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input_csv:
        predict_power(input_data=Path(args.input_csv))
    else:
        sample_input = {
            "AMBIENT_TEMPERATURE": args.ambient_temp,
            "MODULE_TEMPERATURE": args.module_temp,
            "IRRADIATION": args.irradiation,
            "HOUR": args.hour,
            "MINUTE": args.minute,
            "DAY_OF_WEEK": args.day_of_week
        }
        predict_power(input_data=sample_input)
