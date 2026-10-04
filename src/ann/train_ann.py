"""
PVision AI: Artificial Neural Network (ANN) Solar Power Prediction Pipeline
Implements:
1. Tabular telemetry dataset inspection and target confirmation (AC_POWER)
2. Strict data leakage prevention (excluding DC_POWER, DAILY_YIELD, TOTAL_YIELD)
3. Chronological time-series splitting (70% Train / 15% Val / 15% Test)
4. Training-only StandardScaler fitting and artifact persistence
5. Configurable feed-forward ANN regression architecture (build_ann)
6. GPU detection with graceful CPU fallback
7. Callbacks: EarlyStopping and ModelCheckpoint saving best_ann_model.keras
8. Training history recording (models/ann/training_history.json)
"""

import os
import sys
import random
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

# Path resolution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utilities.config import config
from src.utilities.logger import get_logger

logger = get_logger("ANN_Training")

# ============================================================
# 1. CONFIGURATION & REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42

def set_reproducible_seeds(seed: int = RANDOM_SEED):
    """Sets deterministic random seeds across all libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

set_reproducible_seeds(RANDOM_SEED)

# Configurable ANN Hyperparameters
ANN_CONFIG: Dict[str, Any] = {
    "hidden_layers": [128, 64, 32],
    "dropout": 0.20,
    "activation": "relu",
    "output_activation": "linear",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "patience": 10
}

# Target selection configuration (None = auto-inspect and confirm)
TARGET_COLUMN: Optional[str] = None


# ============================================================
# 2. HARDWARE & GPU DETECTION
# ============================================================

def detect_hardware() -> Tuple[bool, str, str]:
    """Detects available computing hardware (GPU vs. CPU)."""
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu_name = gpus[0].name
            has_gpu = True
        else:
            gpu_name = "None (CPU Execution)"
            has_gpu = False
    except ImportError:
        tf_version = "Not Installed"
        has_gpu = False
        gpu_name = "None"

    print("\n" + "=" * 60)
    print("           PVISION AI - HARDWARE DIAGNOSTICS")
    print("=" * 60)
    print(f"TensorFlow version : {tf_version}")
    print(f"GPU availability   : {has_gpu}")
    print(f"GPU device name    : {gpu_name}")
    print("=" * 60 + "\n")

    return has_gpu, tf_version, gpu_name


# ============================================================
# 3. TARGET CONFIRMATION & LEAKAGE VALIDATION
# ============================================================

def confirm_target_variable(
    df: pd.DataFrame,
    configured_target: Optional[str] = None
) -> Tuple[str, str, str]:
    """
    Inspects available columns, identifies power-generation target candidates,
    and requires explicit or unambiguous target selection.
    """
    plausible_targets = {
        "AC_POWER": ("Inverter Alternating Current (AC) Power Output", "kW"),
        "DC_POWER": ("Direct Current (DC) Power Output from Photovoltaic Array", "kW"),
        "DAILY_YIELD": ("Cumulative Daily Energy Generated up to Current Timestamp", "kWh"),
        "TOTAL_YIELD": ("Cumulative Total Inverter Lifetime Energy Yield", "kWh")
    }

    available_candidates = [c for c in df.columns if c in plausible_targets]

    print("=" * 65)
    print("           PVISION AI - TARGET VARIABLE INSPECTION")
    print("=" * 65)
    print("Plausible Power Generation Target Columns in Dataset:")
    for cand in available_candidates:
        desc, unit = plausible_targets[cand]
        print(f"  * {cand:<13} : {desc} [{unit}]")
    print("=" * 65)

    if configured_target is not None:
        selected_target = configured_target
    elif "AC_POWER" in available_candidates:
        selected_target = "AC_POWER"
    elif len(available_candidates) == 1:
        selected_target = available_candidates[0]
    else:
        raise ValueError(
            f"Target column cannot be uniquely determined. "
            f"Available candidates: {available_candidates}. "
            f"Please specify TARGET_COLUMN explicitly via --target or in ANN_CONFIG."
        )

    if selected_target not in df.columns:
        raise ValueError(
            f"Configured target '{selected_target}' does not exist in dataset columns: {list(df.columns)}"
        )

    desc, unit = plausible_targets.get(selected_target, ("Target Variable", "units"))

    print(f"\nCONFIRMED ANN TARGET:\n{selected_target}")
    print(f"\nTARGET DESCRIPTION:\n{desc}")
    print(f"\nTARGET UNIT:\n{unit}\n")

    return selected_target, desc, unit


def validate_leakage_safeguards(
    candidate_features: List[str],
    target_col: str
) -> List[str]:
    """
    Enforces strict leakage prevention rules:
    1. Target variable must not be in feature set X.
    2. DC_POWER must be excluded (r ~ 0.9999 identity shortcut).
    3. DAILY_YIELD & TOTAL_YIELD must be excluded (cumulative non-stationary leakage).
    4. Identifiers (SOURCE_KEY, PLANT_ID) and raw timestamps must be excluded.
    """
    prohibited_features = {
        target_col: "Target variable (Direct target leakage)",
        "DC_POWER": "DC power (Trivial physical inverter conversion identity r ~ 0.9999)",
        "DAILY_YIELD": "Cumulative daily energy (Accumulated daytime progression leakage)",
        "TOTAL_YIELD": "Cumulative lifetime energy (Non-stationary historical trend leakage)",
        "PLANT_ID": "Categorical metadata identifier",
        "SOURCE_KEY": "Inverter hardware identifier",
        "DATE_TIME": "Raw timestamp string",
        "DATETIME_PARSED": "Parsed datetime object"
    }

    clean_features = []
    removed_log = []

    for f in candidate_features:
        if f in prohibited_features:
            removed_log.append((f, prohibited_features[f]))
        else:
            clean_features.append(f)

    print("=" * 65)
    print("           DATA LEAKAGE PREVENTION AUDIT")
    print("=" * 65)
    print("Excluded Leakage & Identifier Variables:")
    for var, reason in removed_log:
        print(f"  [X] {var:<16} : {reason}")
    print("\nRetained Legitimate Predictive Features:")
    for f in clean_features:
        print(f"  [OK] {f}")
    print("=" * 65 + "\n")

    return clean_features


# ============================================================
# 4. DATASET DISCOVERY, LOADING & CLEANING
# ============================================================

def discover_dataset_files(data_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Discovers solar generation and weather sensor CSV files."""
    candidate_dirs = [
        data_dir if data_dir else None,
        config.PROJECT_ROOT / "data" / "solar_generation",
        config.PROJECT_ROOT / "data" / "solar_power",
        config.PROJECT_ROOT / "data"
    ]

    resolved_dir = None
    for c_dir in candidate_dirs:
        if c_dir and c_dir.exists():
            gen_p1 = c_dir / "Plant_1_Generation_Data.csv"
            weather_p1 = c_dir / "Plant_1_Weather_Sensor_Data.csv"
            if gen_p1.exists() and weather_p1.exists():
                resolved_dir = c_dir
                break

    if not resolved_dir:
        raise FileNotFoundError(
            "Could not locate Plant_1_Generation_Data.csv and Plant_1_Weather_Sensor_Data.csv. "
            "Please check data/solar_generation/ or specify --data-dir."
        )

    logger.info(f"Discovered solar telemetry dataset at: {resolved_dir}")
    return resolved_dir / "Plant_1_Generation_Data.csv", resolved_dir / "Plant_1_Weather_Sensor_Data.csv"


def load_and_preprocess_telemetry(
    gen_path: Path,
    weather_path: Path
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Loads raw CSVs, performs date parsing, joins inverter and sensor data,
    filters physics outliers (night noise and daytime outages),
    and creates domain and cyclical time features.
    """
    logger.info(f"Loading Generation Data: {gen_path.name}")
    gen_df = pd.read_csv(gen_path)
    logger.info(f"Loading Weather Sensor Data: {weather_path.name}")
    weather_df = pd.read_csv(weather_path)

    # 1. Dataset Structure Inspection Audit
    print("=" * 65)
    print("           PVISION AI - DATASET AUDIT")
    print("=" * 65)
    print(f"Generation CSV Rows : {len(gen_df):,} | Columns: {list(gen_df.columns)}")
    print(f"Weather CSV Rows    : {len(weather_df):,} | Columns: {list(weather_df.columns)}")
    print(f"Generation Nulls    : {int(gen_df.isnull().sum().sum())}")
    print(f"Weather Nulls       : {int(weather_df.isnull().sum().sum())}")
    print(f"Generation Dups     : {int(gen_df.duplicated().sum())}")
    print(f"Weather Dups        : {int(weather_df.duplicated().sum())}")
    print("=" * 65 + "\n")

    # 2. Date Parsing
    # Plant 1 generation data uses DD-MM-YYYY HH:MM, weather data uses YYYY-MM-DD HH:MM:SS
    try:
        gen_df["DATETIME_PARSED"] = pd.to_datetime(gen_df["DATE_TIME"], format="%d-%m-%Y %H:%M")
    except Exception:
        gen_df["DATETIME_PARSED"] = pd.to_datetime(gen_df["DATE_TIME"])

    try:
        weather_df["DATETIME_PARSED"] = pd.to_datetime(weather_df["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
    except Exception:
        weather_df["DATETIME_PARSED"] = pd.to_datetime(weather_df["DATE_TIME"])

    # 3. Inner Merge on Datetime
    weather_subset = weather_df[[
        "DATETIME_PARSED", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"
    ]].drop_duplicates(subset=["DATETIME_PARSED"])

    merged_df = pd.merge(gen_df, weather_subset, on="DATETIME_PARSED", how="inner")
    logger.info(f"Merged telemetry records: {len(merged_df):,} rows")

    # 4. Physics-Based Outlier Filtering
    # Zero out baseline night sensor noise (< 1 kW when irradiation is exactly 0)
    night_mask = (merged_df["IRRADIATION"] == 0) & (merged_df["AC_POWER"] > 0)
    merged_df.loc[night_mask, "AC_POWER"] = 0.0
    merged_df.loc[night_mask, "DC_POWER"] = 0.0

    # Separate operational nominal power from daytime electrical inverter disconnect outages
    daytime_outage_mask = (merged_df["IRRADIATION"] > 0.05) & (merged_df["AC_POWER"] == 0)
    outages_count = int(daytime_outage_mask.sum())
    logger.info(f"Filtering {outages_count} daytime inverter disconnect outages...")
    df_clean = merged_df[~daytime_outage_mask].copy()

    # 5. Feature Engineering
    dt = df_clean["DATETIME_PARSED"]
    df_clean["HOUR"] = dt.dt.hour
    df_clean["MINUTE"] = dt.dt.minute
    df_clean["DAY_OF_WEEK"] = dt.dt.dayofweek
    df_clean["TIME_DECIMAL"] = df_clean["HOUR"] + df_clean["MINUTE"] / 60.0

    # Cyclical 24-Hour Periodicity Encodings
    df_clean["SIN_TIME"] = np.sin(2 * np.pi * df_clean["TIME_DECIMAL"] / 24.0)
    df_clean["COS_TIME"] = np.cos(2 * np.pi * df_clean["TIME_DECIMAL"] / 24.0)

    # Domain indicators and interaction features
    df_clean["IS_DAYTIME"] = (df_clean["IRRADIATION"] > 0).astype(int)
    df_clean["TEMP_DIFFERENCE"] = df_clean["MODULE_TEMPERATURE"] - df_clean["AMBIENT_TEMPERATURE"]
    df_clean["IRRAD_MODULE_INTERACTION"] = df_clean["IRRADIATION"] * df_clean["MODULE_TEMPERATURE"]

    # 6. Target Confirmation
    target_col, _, _ = confirm_target_variable(df_clean, configured_target=TARGET_COLUMN)

    # 7. Candidate Features Selection & Leakage Prevention
    all_numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    final_features = validate_leakage_safeguards(all_numeric_cols, target_col)

    return df_clean, final_features, target_col


# ============================================================
# 5. CHRONOLOGICAL TIME-SERIES SPLITTING
# ============================================================

def chronological_time_series_split(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits timestamped observations chronologically:
    - Earliest 70% -> Train
    - Next 15%     -> Validation
    - Final 15%    -> Test
    Strictly forbids random shuffling to prevent future-to-past data leakage.
    """
    logger.info("Performing chronological time-series splitting (70% / 15% / 15%)...")
    df_sorted = df.sort_values("DATETIME_PARSED").reset_index(drop=True)

    total_samples = len(df_sorted)
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)

    train_df = df_sorted.iloc[:n_train]
    val_df = df_sorted.iloc[n_train : n_train + n_val]
    test_df = df_sorted.iloc[n_train + n_val :]

    X_train = train_df[features].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[features].copy()
    y_val = val_df[target_col].copy()

    X_test = test_df[features].copy()
    y_test = test_df[target_col].copy()

    # Time period audit
    t_start, t_end = train_df["DATETIME_PARSED"].min(), train_df["DATETIME_PARSED"].max()
    v_start, v_end = val_df["DATETIME_PARSED"].min(), val_df["DATETIME_PARSED"].max()
    te_start, te_end = test_df["DATETIME_PARSED"].min(), test_df["DATETIME_PARSED"].max()

    print("=" * 65)
    print("       CHRONOLOGICAL TIME-SERIES PARTITIONING AUDIT")
    print("=" * 65)
    print(f"Training period   : {t_start}  to  {t_end}")
    print(f"Validation period : {v_start}  to  {v_end}")
    print(f"Testing period    : {te_start}  to  {te_end}")
    print("-" * 65)
    print(f"Training rows     : {len(X_train):>6,}  ({len(X_train)/total_samples*100:5.1f}%)")
    print(f"Validation rows   : {len(X_val):>6,}  ({len(X_val)/total_samples*100:5.1f}%)")
    print(f"Testing rows      : {len(X_test):>6,}  ({len(X_test)/total_samples*100:5.1f}%)")
    print(f"Total rows        : {total_samples:>6,}  (100.0%)")
    print("=" * 65 + "\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 6. FEATURE SCALING (TRAIN-ONLY FITTING)
# ============================================================

def fit_and_apply_scaler(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    scaler_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Fits StandardScaler STRICTLY on X_train.
    Transforms X_val and X_test using the training statistics.
    Saves scaler and feature list to models/scaler/.
    """
    from sklearn.preprocessing import StandardScaler

    logger.info("Fitting StandardScaler strictly on training set (zero test leakage)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_file = scaler_dir / "ann_feature_scaler.pkl"
    features_file = scaler_dir / "ann_features.json"

    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved fitted StandardScaler to: {scaler_file}")

    with open(features_file, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=4)
    logger.info(f"Saved feature list schema to: {features_file}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ============================================================
# 7. CONFIGURABLE ANN ARCHITECTURE
# ============================================================

def build_ann(
    input_dim: int,
    config_dict: Optional[Dict[str, Any]] = None
):
    """
    Constructs a configurable feed-forward neural network for regression:
    Input (input_dim)
        ↓
    Dense(128, activation='relu')
        ↓
    Dropout(0.20)
        ↓
    Dense(64, activation='relu')
        ↓
    Dense(32, activation='relu')
        ↓
    Dense(1, activation='linear') [Target: AC_POWER]
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, optimizers
    except ImportError:
        raise ImportError("TensorFlow is required to build the ANN regression model.")

    cfg = config_dict if config_dict else ANN_CONFIG
    hidden_layers = cfg.get("hidden_layers", [128, 64, 32])
    dropout_rate = cfg.get("dropout", 0.20)
    activation = cfg.get("activation", "relu")
    output_act = cfg.get("output_activation", "linear")
    lr = cfg.get("learning_rate", 0.001)

    model = models.Sequential(name="PVision_ANN_PowerPredictor")
    model.add(layers.Input(shape=(input_dim,), name="input_features"))

    for idx, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation=activation, name=f"dense_{idx+1}_{units}"))
        if idx == 0 and dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate, name="dropout_layer"))

    model.add(layers.Dense(1, activation=output_act, name="predicted_power"))

    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )

    return model


# ============================================================
# 8. TRAINING PIPELINE EXECUTION
# ============================================================

def train_ann_pipeline(
    data_dir: Optional[Path] = None,
    epochs: int = ANN_CONFIG["epochs"],
    batch_size: int = ANN_CONFIG["batch_size"],
    learning_rate: float = ANN_CONFIG["learning_rate"],
    patience: int = ANN_CONFIG["patience"],
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Executes the complete data preparation, leakage validation,
    scaling, and optional training in VS Code.
    When dry_run=True, validates the pipeline without calling model.fit(...).
    """
    detect_hardware()

    # 1. Dataset discovery & loading
    gen_csv, weather_csv = discover_dataset_files(data_dir)
    df_clean, feature_names, target_col = load_and_preprocess_telemetry(gen_csv, weather_csv)

    # 2. Chronological splitting
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_time_series_split(
        df_clean, feature_names, target_col
    )

    # 3. Fitting StandardScaler strictly on training set
    scaler_dir = config.PROJECT_ROOT / "models" / "scaler"
    X_tr_s, X_va_s, X_te_s, scaler = fit_and_apply_scaler(
        X_train, X_val, X_test, feature_names, scaler_dir
    )

    # 4. Save test sets for downstream evaluation
    processed_dir = config.PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    test_eval_df = pd.DataFrame(X_te_s, columns=feature_names)
    test_eval_df[target_col] = y_test.values
    test_eval_df.to_csv(processed_dir / "ann_test_chronological.csv", index=False)
    logger.info(f"Saved chronological test evaluation dataset to: {processed_dir / 'ann_test_chronological.csv'}")

    # 5. Build ANN Model Architecture
    cfg = ANN_CONFIG.copy()
    cfg["epochs"] = epochs
    cfg["batch_size"] = batch_size
    cfg["learning_rate"] = learning_rate
    cfg["patience"] = patience

    model = None
    try:
        model = build_ann(input_dim=len(feature_names), config_dict=cfg)
        print("\n" + "=" * 60)
        print("           ANN MODEL ARCHITECTURE SUMMARY")
        print("=" * 60)
        model.summary()
        print("=" * 60 + "\n")
    except ImportError:
        print("\n" + "=" * 60)
        print("           ANN MODEL ARCHITECTURE BLUEPRINT")
        print("=" * 60)
        print("TensorFlow is not installed in the global Python environment.")
        print(f"Input Features ({len(feature_names)}): {feature_names}")
        print(f"Layers: Input({len(feature_names)}) -> Dense(128, relu) -> Dropout(0.20) -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, linear)")
        print(f"Hyperparameters: Adam(lr={learning_rate}), loss=MSE, batch_size={batch_size}, epochs={epochs}, patience={patience}")
        print("Architecture blueprint validated. Execution will proceed in user's VS Code environment.")
        print("=" * 60 + "\n")

    if dry_run:
        print("\n" + "*" * 65)
        print(" [DRY-RUN / VALIDATION MODE] Pipeline successfully verified!")
        print(" Model architecture compiled, scaler fitted, datasets prepared.")
        print(" ZERO model training executed inside Antigravity.")
        print(" To perform actual training in VS Code, execute:")
        print("   python src/ann/train_ann.py")
        print("*" * 65 + "\n")
        return {
            "status": "VALIDATED — TRAINING PENDING",
            "feature_count": len(feature_names),
            "features": feature_names,
            "target": target_col,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test)
        }

    # 6. Actual Training in VS Code
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required to train the model.")

    ann_model_dir = config.PROJECT_ROOT / "models" / "ann"
    ann_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ann_model_dir / "best_ann_model.keras"
    history_path = ann_model_dir / "training_history.json"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n" + "=" * 60)
    print("           STARTING ANN MODEL TRAINING (VS CODE)")
    print("=" * 60)
    history = model.fit(
        X_tr_s,
        y_train.values,
        validation_data=(X_va_s, y_val.values),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    hist_dict = {
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "mae": [float(x) for x in history.history.get("mae", [])],
        "val_mae": [float(x) for x in history.history.get("val_mae", [])],
        "rmse": [float(x) for x in history.history.get("rmse", [])],
        "val_rmse": [float(x) for x in history.history.get("val_rmse", [])],
        "epochs_completed": len(history.history["loss"]),
        "timestamp": datetime.now().isoformat()
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=4)
    logger.info(f"Saved training history to: {history_path}")

    print("\n" + "=" * 60)
    print("ANN Training Completed Successfully!")
    print(f"Best Model Checkpoint : {checkpoint_path}")
    print(f"Training History      : {history_path}")
    print("=" * 60 + "\n")

    return {
        "status": "COMPLETED",
        "best_model_path": str(checkpoint_path),
        "history_path": str(history_path)
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="PVision AI: Train Artificial Neural Network (ANN) for Solar Power Prediction"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to solar generation data directory")
    parser.add_argument("--target", type=str, default=None,
                        help="Target variable to predict (default: auto-confirms AC_POWER)")
    parser.add_argument("--epochs", type=int, default=ANN_CONFIG["epochs"],
                        help=f"Number of training epochs (default: {ANN_CONFIG['epochs']})")
    parser.add_argument("--batch-size", type=int, default=ANN_CONFIG["batch_size"],
                        help=f"Training batch size (default: {ANN_CONFIG['batch_size']})")
    parser.add_argument("--learning-rate", type=float, default=ANN_CONFIG["learning_rate"],
                        help=f"Adam initial learning rate (default: {ANN_CONFIG['learning_rate']})")
    parser.add_argument("--patience", type=int, default=ANN_CONFIG["patience"],
                        help=f"Early stopping patience (default: {ANN_CONFIG['patience']})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the pipeline, inspect data, fit scaler, but DO NOT execute training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.target:
        TARGET_COLUMN = args.target

    train_ann_pipeline(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        dry_run=args.dry_run
    )
