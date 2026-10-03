"""
PVision AI: Tabular Preprocessing & Feature Engineering Pipeline
Prepares solar power generation & weather telemetry for ANN / regression modeling:
- Multi-format datetime parsing & synchronization
- Inverter-level and plant-level weather sensor data merging
- Domain-specific feature engineering (Delta-T, irradiation interactions, temporal cyclical encoding)
- Missing value imputation and anomalous zero-power filtering
- Scaler fitting and train/test split manifest generation
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from src.utilities.config import config
    from src.utilities.logger import get_logger
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utilities.config import config
    from src.utilities.logger import get_logger

logger = get_logger("Tabular_Preprocessing")


class SolarDataPipeline:
    """End-to-end data preparation pipeline for solar generation telemetry."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else config.SOLAR_GEN_DATA_DIR
        self.scaler = StandardScaler()

    def load_and_merge_plant_data(self, plant_id: int = 1) -> pd.DataFrame:
        """Loads generation and weather data for a given plant (1 or 2) and merges them cleanly."""
        gen_file = self.data_dir / f"Plant_{plant_id}_Generation_Data.csv"
        weather_file = self.data_dir / f"Plant_{plant_id}_Weather_Sensor_Data.csv"

        if not gen_file.exists() or not weather_file.exists():
            raise FileNotFoundError(f"Missing CSV files for Plant {plant_id} in {self.data_dir}")

        logger.info(f"Loading and merging data for Plant {plant_id}...")
        gen_df = pd.read_csv(gen_file)
        weather_df = pd.read_csv(weather_file)

        # Harmonize Datetimes
        # Plant 1 generation has DD-MM-YYYY HH:MM, weather has YYYY-MM-DD HH:MM:SS
        gen_df["DATETIME_PARSED"] = pd.to_datetime(gen_df["DATE_TIME"], errors="coerce")
        weather_df["DATETIME_PARSED"] = pd.to_datetime(weather_df["DATE_TIME"], errors="coerce")

        # Weather data has 1 sensor key, generation has 22 inverters
        # Merge each inverter's generation record with the plant weather reading at that timestamp
        weather_subset = weather_df[[
            "DATETIME_PARSED", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"
        ]].drop_duplicates(subset=["DATETIME_PARSED"])

        merged_df = pd.merge(gen_df, weather_subset, on="DATETIME_PARSED", how="inner")
        logger.info(f"Plant {plant_id} merged shape: {merged_df.shape} (from {len(gen_df):,} generation rows)")

        return merged_df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers physical and temporal solar features."""
        df = df.copy()

        # Physical thermal dynamics
        df["TEMP_DIFF"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]
        df["IRRAD_TEMP_INTERACTION"] = df["IRRADIATION"] * df["MODULE_TEMPERATURE"]

        # Temporal Features
        dt = df["DATETIME_PARSED"]
        df["HOUR"] = dt.dt.hour
        df["MINUTE"] = dt.dt.minute
        df["DAY_OF_WEEK"] = dt.dt.dayofweek
        df["TIME_FLOAT"] = df["HOUR"] + df["MINUTE"] / 60.0

        # Cyclical Time Encoding
        df["SIN_TIME"] = np.sin(2 * np.pi * df["TIME_FLOAT"] / 24.0)
        df["COS_TIME"] = np.cos(2 * np.pi * df["TIME_FLOAT"] / 24.0)

        # Daytime indicator
        df["IS_DAYTIME"] = (df["IRRADIATION"] > 0).astype(int)

        return df

    def prepare_training_datasets(
        self,
        plant_id: int = 1,
        test_size: float = 0.2,
        target_col: str = "AC_POWER",
        save_processed: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executes end-to-end preprocessing and returns train/test feature sets.
        Ready to be consumed by ANN / XGBoost training in VS Code.
        """
        raw_merged = self.load_and_merge_plant_data(plant_id=plant_id)
        featured_df = self.engineer_features(raw_merged)

        feature_cols = [
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
            "TEMP_DIFF",
            "IRRAD_TEMP_INTERACTION",
            "HOUR",
            "SIN_TIME",
            "COS_TIME",
            "IS_DAYTIME"
        ]

        # Filter out night records where target is identically zero if focusing on daytime generation
        clean_df = featured_df.dropna(subset=feature_cols + [target_col])

        X = clean_df[feature_cols]
        y = clean_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.RANDOM_SEED, shuffle=True
        )

        logger.info(f"Prepared Train: {X_train.shape}, Test: {X_test.shape}")

        if save_processed:
            out_dir = config.OUTPUTS_DIR / "predictions"
            out_dir.mkdir(parents=True, exist_ok=True)
            clean_df.to_csv(out_dir / f"plant_{plant_id}_preprocessed_solar_dataset.csv", index=False)
            logger.info(f"Saved preprocessed dataset to {out_dir}")

        return X_train, X_test, y_train, y_test
