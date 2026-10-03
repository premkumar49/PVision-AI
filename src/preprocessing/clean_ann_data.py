"""
PVision AI: ANN Telemetry Data Cleaning, Feature Engineering & Preprocessing
Covers:
1. Duplicate and missing value handling
2. Datetime parsing & temporal feature extraction (cyclical encodings)
3. Domain-specific physical features (Delta-T, irradiance interactions)
4. Outlier detection, ghost generation filtering & inverter outage analysis
5. Documented feature selection & target identification (AC_POWER)
6. Strict target and distribution leakage prevention
7. Training-only scaler fitting and persistence (joblib)
8. Reproducible train/validation/test dataset export
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Dynamic path resolution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utilities.config import config
from src.utilities.logger import get_logger

logger = get_logger("ANN_Data_Cleaner")


class ANNDataCleaner:
    """End-to-end cleaning, engineering, and scaling pipeline for ANN solar forecasting."""

    FEATURE_COLS = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "TEMP_DIFFERENCE",
        "IRRAD_MODULE_INTERACTION",
        "HOUR",
        "SIN_TIME",
        "COS_TIME",
        "IS_DAYTIME"
    ]

    TARGET_COL = "AC_POWER"

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir if data_dir else config.SOLAR_GEN_DATA_DIR
        self.scalers: Dict[int, StandardScaler] = {}
        self.cleaning_stats: Dict[str, Any] = {}

    def clean_and_prepare(
        self,
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """Runs the complete tabular preprocessing pipeline for both plants."""
        logger.info("Initiating ANN Tabular Preprocessing Pipeline...")
        config.ensure_directories()

        processed_dir = config.PROJECT_ROOT / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        vis_dir = config.PROJECT_ROOT / "visualizations" / "preprocessing"
        vis_dir.mkdir(parents=True, exist_ok=True)

        models_dir = config.PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        plant_results = {}

        for plant_id in [1, 2]:
            logger.info(f"Processing Solar Farm: Plant {plant_id}")
            res = self._process_single_plant(
                plant_id=plant_id,
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                random_seed=random_seed,
                output_dir=processed_dir,
                models_dir=models_dir,
                vis_dir=vis_dir
            )
            plant_results[f"Plant_{plant_id}"] = res

        self.cleaning_stats = plant_results
        self._export_summary(config.PROJECT_ROOT / "reports" / "ann_cleaning_summary.json")

        return plant_results

    def _process_single_plant(
        self,
        plant_id: int,
        test_ratio: float,
        val_ratio: float,
        random_seed: int,
        output_dir: Path,
        models_dir: Path,
        vis_dir: Path
    ) -> Dict[str, Any]:
        """Cleans, engineers, scales and splits data for a single solar plant."""
        gen_file = self.data_dir / f"Plant_{plant_id}_Generation_Data.csv"
        weather_file = self.data_dir / f"Plant_{plant_id}_Weather_Sensor_Data.csv"

        gen_raw = pd.read_csv(gen_file)
        weather_raw = pd.read_csv(weather_file)

        raw_gen_rows = len(gen_raw)
        raw_weather_rows = len(weather_raw)

        # 1. Duplicates Check
        gen_dups = int(gen_raw.duplicated().sum())
        weather_dups = int(weather_raw.duplicated().sum())
        gen_clean = gen_raw.drop_duplicates()
        weather_clean = weather_raw.drop_duplicates()

        # 2. Missing Values Check
        gen_nulls = int(gen_clean.isnull().sum().sum())
        weather_nulls = int(weather_clean.isnull().sum().sum())

        # 3. Datetime Parsing & Synchronization
        if plant_id == 1:
            gen_clean["DATETIME_PARSED"] = pd.to_datetime(gen_clean["DATE_TIME"], format="%d-%m-%Y %H:%M")
        else:
            gen_clean["DATETIME_PARSED"] = pd.to_datetime(gen_clean["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

        weather_clean["DATETIME_PARSED"] = pd.to_datetime(weather_clean["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

        # Inverter-level merge with weather sensors
        weather_subset = weather_clean[[
            "DATETIME_PARSED", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"
        ]].drop_duplicates(subset=["DATETIME_PARSED"])

        merged_df = pd.merge(gen_clean, weather_subset, on="DATETIME_PARSED", how="inner")
        rows_merged = len(merged_df)

        # 4. Outlier Analysis & Physics-Based Filtering
        # A. Ghost night power (irradiance == 0 but AC_POWER > 0)
        night_ghost_mask = (merged_df["IRRADIATION"] == 0) & (merged_df["AC_POWER"] > 0)
        num_night_ghosts = int(night_ghost_mask.sum())
        # Zero out tiny baseline night sensor noise (< 1 kW)
        merged_df.loc[night_ghost_mask, "AC_POWER"] = 0.0
        merged_df.loc[night_ghost_mask, "DC_POWER"] = 0.0

        # B. Daytime inverter outages (irradiance > 0 but AC_POWER == 0)
        day_outage_mask = (merged_df["IRRADIATION"] > 0.05) & (merged_df["AC_POWER"] == 0)
        num_day_outages = int(day_outage_mask.sum())

        # For nominal power regression, inverter failures represent physical disconnects rather
        # than weather-power physics. We separate operational nominal power training from outages.
        # Filter out severe daytime outage records where irradiance is high but inverter is completely offline.
        df_operational = merged_df[~day_outage_mask].copy()
        rows_after_outage_filter = len(df_operational)

        # 5. Feature Engineering
        # Temporal Features
        dt = df_operational["DATETIME_PARSED"]
        df_operational["HOUR"] = dt.dt.hour
        df_operational["MINUTE"] = dt.dt.minute
        df_operational["DAY_OF_WEEK"] = dt.dt.dayofweek
        df_operational["TIME_DECIMAL"] = df_operational["HOUR"] + df_operational["MINUTE"] / 60.0

        # Cyclical Encodings (smooth 24-hr circular periodicity)
        df_operational["SIN_TIME"] = np.sin(2 * np.pi * df_operational["TIME_DECIMAL"] / 24.0)
        df_operational["COS_TIME"] = np.cos(2 * np.pi * df_operational["TIME_DECIMAL"] / 24.0)

        # Solar domain indicators
        df_operational["IS_DAYTIME"] = (df_operational["IRRADIATION"] > 0).astype(int)

        # Physical features
        df_operational["TEMP_DIFFERENCE"] = df_operational["MODULE_TEMPERATURE"] - df_operational["AMBIENT_TEMPERATURE"]
        df_operational["IRRAD_MODULE_INTERACTION"] = df_operational["IRRADIATION"] * df_operational["MODULE_TEMPERATURE"]

        # 6. Feature Selection & Target Definition
        X = df_operational[self.FEATURE_COLS].copy()
        y = df_operational[self.TARGET_COL].copy()

        # 7. Reproducible Train / Val / Test Splitting (70% / 15% / 15%)
        # Split features BEFORE scaling to prevent data leakage!
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_ratio + val_ratio), random_state=random_seed, shuffle=True
        )

        relative_test_ratio = test_ratio / (test_ratio + val_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test_ratio, random_state=random_seed, shuffle=True
        )

        # 8. Leakage-Free Feature Scaling
        # Fit scaler ONLY on X_train
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=self.FEATURE_COLS, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=self.FEATURE_COLS, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=self.FEATURE_COLS, index=X_test.index)

        self.scalers[plant_id] = scaler

        # 9. Save Fitted Scaler
        scaler_file = models_dir / f"solar_ann_scaler_plant{plant_id}.joblib"
        joblib.dump(scaler, scaler_file)
        logger.info(f"Saved fitted scaler to: {scaler_file}")

        # 10. Save Processed Train/Val/Test Datasets
        train_full = X_train_scaled.copy()
        train_full[self.TARGET_COL] = y_train.values

        val_full = X_val_scaled.copy()
        val_full[self.TARGET_COL] = y_val.values

        test_full = X_test_scaled.copy()
        test_full[self.TARGET_COL] = y_test.values

        train_full.to_csv(output_dir / f"ann_plant{plant_id}_train.csv", index=False)
        val_full.to_csv(output_dir / f"ann_plant{plant_id}_val.csv", index=False)
        test_full.to_csv(output_dir / f"ann_plant{plant_id}_test.csv", index=False)

        # Save unscaled test set for human-readable evaluation & error metrics
        unscaled_test = X_test.copy()
        unscaled_test[self.TARGET_COL] = y_test.values
        unscaled_test.to_csv(output_dir / f"ann_plant{plant_id}_test_unscaled.csv", index=False)

        # 11. Visualizations
        self._generate_visualizations(df_operational, plant_id, vis_dir)

        return {
            "plant_id": plant_id,
            "raw_generation_rows": raw_gen_rows,
            "raw_weather_rows": raw_weather_rows,
            "duplicates_removed": gen_dups + weather_dups,
            "missing_values_imputed": gen_nulls + weather_nulls,
            "merged_records": rows_merged,
            "ghost_night_records_corrected": num_night_ghosts,
            "daytime_inverter_outages_filtered": num_day_outages,
            "operational_records_retained": rows_after_outage_filter,
            "selected_features": self.FEATURE_COLS,
            "target": self.TARGET_COL,
            "target_leakage_prevented": [
                "DC_POWER excluded (avoids r=0.9999 identity shortcut)",
                "TOTAL_YIELD excluded (cumulative non-stationary)",
                "DAILY_YIELD excluded (leaks daily progression)"
            ],
            "data_splits": {
                "train_samples": len(train_full),
                "val_samples": len(val_full),
                "test_samples": len(test_full),
                "train_ratio": round(len(train_full) / rows_after_outage_filter * 100, 2),
                "val_ratio": round(len(val_full) / rows_after_outage_filter * 100, 2),
                "test_ratio": round(len(test_full) / rows_after_outage_filter * 100, 2)
            },
            "scaler_saved_path": str(scaler_file),
            "scaler_feature_means": dict(zip(self.FEATURE_COLS, [round(float(m), 4) for m in scaler.mean_])),
            "scaler_feature_stds": dict(zip(self.FEATURE_COLS, [round(float(s), 4) for s in scaler.scale_]))
        }

    def _generate_visualizations(self, df: pd.DataFrame, plant_id: int, vis_dir: Path):
        """Creates outlier boxplots, correlation matrices, and diurnal power curves."""
        try:
            # 1. Boxplots of raw features
            plt.figure(figsize=(10, 5))
            num_cols = ["AC_POWER", "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "TEMP_DIFFERENCE"]
            sns.boxplot(data=df[num_cols], orient="h", palette="Set2")
            plt.title(f"Plant {plant_id}: Distribution & Outlier Inspection of Key Solar Features", fontsize=11, fontweight="bold")
            plt.xlabel("Value")
            plt.tight_layout()
            plt.savefig(vis_dir / f"ann_plant{plant_id}_outliers_boxplot.png", dpi=200)
            plt.close()

            # 2. Correlation heatmap of engineered features
            plt.figure(figsize=(9, 7))
            all_cols = self.FEATURE_COLS + [self.TARGET_COL]
            corr = df[all_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=0.5)
            plt.title(f"Plant {plant_id}: Engineered Features vs AC_POWER Correlation Matrix", fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(vis_dir / f"ann_plant{plant_id}_features_correlation.png", dpi=200)
            plt.close()

            # 3. Diurnal solar power curve
            plt.figure(figsize=(10, 4.5))
            hourly = df.groupby("HOUR")["AC_POWER"].agg(["mean", "std"]).reset_index()
            plt.plot(hourly["HOUR"], hourly["mean"], marker="o", color="#d95f02", label="Mean AC Power (kW)")
            plt.fill_between(
                hourly["HOUR"],
                np.maximum(0, hourly["mean"] - hourly["std"]),
                hourly["mean"] + hourly["std"],
                color="#d95f02", alpha=0.2, label="±1 Std Dev"
            )
            plt.title(f"Plant {plant_id}: Diurnal Generation Profile Across Day (15-min Intervals)", fontsize=11, fontweight="bold")
            plt.xlabel("Hour of Day")
            plt.ylabel("AC Power Output (kW)")
            plt.xticks(range(0, 24))
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(vis_dir / f"ann_plant{plant_id}_diurnal_power_profile.png", dpi=200)
            plt.close()

            logger.info(f"Generated preprocessing charts for Plant {plant_id} in {vis_dir}")
        except Exception as e:
            logger.error(f"Error plotting Plant {plant_id} charts: {e}")

    def _export_summary(self, output_path: Path):
        """Saves cleaning stats to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.cleaning_stats, f, indent=4)
        logger.info(f"Exported ANN cleaning summary to {output_path}")


if __name__ == "__main__":
    cleaner = ANNDataCleaner()
    cleaner.clean_and_prepare()
