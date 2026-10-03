"""
PVision AI: Central Configuration Module
Handles project paths, dataset schemas, labels, image properties, and constants.
"""

from pathlib import Path
import os

class Config:
    """Central configuration class for PVision AI."""

    # Automatically resolve project root
    # Supports running from PVision_AI or from root workspace (d:\SolarPredict)
    _CURRENT_FILE = Path(__file__).resolve()
    # If this file is in .../PVision_AI/src/utilities/config.py
    SRC_DIR = _CURRENT_FILE.parent.parent
    PROJECT_ROOT = SRC_DIR.parent

    # Check if PVision_AI is a subdirectory of current working directory or vice versa
    if PROJECT_ROOT.name != "PVision_AI" and (PROJECT_ROOT / "PVision_AI").is_dir():
        PROJECT_ROOT = PROJECT_ROOT / "PVision_AI"
        SRC_DIR = PROJECT_ROOT / "src"

    # Core Directories
    DATA_DIR = PROJECT_ROOT / "data"
    PV_FAULT_DATA_DIR = DATA_DIR / "pv_fault"
    SOLAR_GEN_DATA_DIR = DATA_DIR / "solar_generation"

    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    FIGURES_DIR = OUTPUTS_DIR / "figures"
    METRICS_DIR = OUTPUTS_DIR / "metrics"
    PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
    APP_DIR = PROJECT_ROOT / "app"

    # PV Fault Dataset Configuration
    PV_FAULT_CLASSES = [
        "crack",
        "global_aging",
        "hotspot",
        "normal",
        "partial_aging",
        "shading",
        "short_circuit"
    ]
    NUM_CLASSES = len(PV_FAULT_CLASSES)
    CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(PV_FAULT_CLASSES)}
    IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(PV_FAULT_CLASSES)}

    # Image Modalities
    MODALITIES = ["gasf_images", "iv_images"]
    EXPECTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

    # Image Dimensions
    GASF_RAW_SIZE = (256, 256)
    IV_RAW_SIZE = (1343, 808)
    MODEL_INPUT_SIZE = (224, 224)  # Standard input resolution for CNNs

    # Solar Generation Dataset Configuration
    CSV_FILES = {
        "plant1_generation": "Plant_1_Generation_Data.csv",
        "plant1_weather": "Plant_1_Weather_Sensor_Data.csv",
        "plant2_generation": "Plant_2_Generation_Data.csv",
        "plant2_weather": "Plant_2_Weather_Sensor_Data.csv"
    }

    DATETIME_FORMATS = {
        "Plant_1_Generation_Data.csv": "%d-%m-%Y %H:%M",
        "Plant_1_Weather_Sensor_Data.csv": "%Y-%m-%d %H:%M:%S",
        "Plant_2_Generation_Data.csv": "%Y-%m-%d %H:%M:%S",
        "Plant_2_Weather_Sensor_Data.csv": "%Y-%m-%d %H:%M:%S"
    }

    # Features
    GENERATION_FEATURES = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]
    WEATHER_FEATURES = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    TARGET_POWER = "AC_POWER"

    # Reproducibility
    RANDOM_SEED = 42

    @classmethod
    def ensure_directories(cls):
        """Ensure all standard output and project directories exist."""
        for path in [
            cls.DATA_DIR,
            cls.PV_FAULT_DATA_DIR,
            cls.SOLAR_GEN_DATA_DIR,
            cls.NOTEBOOKS_DIR,
            cls.MODELS_DIR,
            cls.OUTPUTS_DIR,
            cls.FIGURES_DIR,
            cls.METRICS_DIR,
            cls.PREDICTIONS_DIR,
            cls.APP_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


config = Config()
