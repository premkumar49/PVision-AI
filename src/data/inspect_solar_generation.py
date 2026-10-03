"""
PVision AI: Solar Generation Dataset Inspector
Performs exhaustive exploratory inspection on solar power generation and weather datasets:
- CSV discovery and schema identification
- Missing values and anomaly detection
- Duplicate row and composite-key duplicate identification
- Numerical vs categorical column classification
- Full descriptive statistics (central tendency, dispersion, skewness, kurtosis)
- Correlation analysis (Pearson & Spearman) within and across merged datasets
- Automated summary export (JSON) and correlation matrix visualizations (PNG)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Handle imports when run directly or as a module
try:
    from src.utilities.config import config
    from src.utilities.logger import get_logger
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utilities.config import config
    from src.utilities.logger import get_logger

logger = get_logger("Solar_Generation_Inspector")


class SolarGenerationInspector:
    """Inspector for Solar Generation and Weather Sensor CSV Datasets."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else config.SOLAR_GEN_DATA_DIR
        self.summary_report: Dict[str, Any] = {}

    def inspect(self) -> Dict[str, Any]:
        """Executes full solar generation CSV inspection workflow."""
        logger.info(f"Starting Solar Generation Dataset Inspection at: {self.data_dir}")
        config.ensure_directories()

        if not self.data_dir.exists():
            logger.error(f"Solar Generation directory does not exist: {self.data_dir}")
            return {"error": f"Directory not found: {self.data_dir}"}

        csv_files = sorted([f for f in self.data_dir.iterdir() if f.is_file() and f.suffix.lower() == ".csv"])
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_dir}")
            return {"error": "No CSV files found"}

        logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")

        file_reports = {}
        dataframes = {}

        for csv_path in csv_files:
            report, df = self._inspect_single_csv(csv_path)
            file_reports[csv_path.name] = report
            dataframes[csv_path.name] = df

        # Merged Cross-Correlation Analysis (Weather + Generation per Plant)
        merged_reports = self._analyze_merged_plants(dataframes)

        self.summary_report = {
            "dataset_directory": str(self.data_dir),
            "csv_files_count": len(csv_files),
            "file_reports": file_reports,
            "merged_plant_analysis": merged_reports
        }

        # Export outputs
        self._export_summary_json()
        self._generate_visualizations(dataframes, merged_reports)
        self._print_console_summary()

        return self.summary_report

    def _inspect_single_csv(self, file_path: Path) -> tuple[Dict[str, Any], pd.DataFrame]:
        """Performs comprehensive inspection of an individual CSV file."""
        logger.info(f"--- Inspecting CSV: {file_path.name} ---")
        df = pd.read_csv(file_path)

        num_rows, num_cols = df.shape
        columns = df.columns.tolist()

        # 1. Missing Value Analysis
        missing_counts = df.isnull().sum()
        missing_report = {
            col: {
                "null_count": int(count),
                "null_percentage": round(float(count / num_rows * 100), 2)
            }
            for col, count in missing_counts.items()
        }
        total_missing = int(missing_counts.sum())

        # 2. Duplicate Row Analysis
        exact_duplicates = int(df.duplicated().sum())

        # Check composite key duplicates if relevant
        composite_key_duplicates = 0
        composite_keys_checked = []
        if "DATE_TIME" in df.columns and "SOURCE_KEY" in df.columns:
            composite_keys_checked = ["DATE_TIME", "SOURCE_KEY"]
            composite_key_duplicates = int(df.duplicated(subset=composite_keys_checked).sum())
        elif "DATE_TIME" in df.columns:
            composite_keys_checked = ["DATE_TIME"]
            composite_key_duplicates = int(df.duplicated(subset=composite_keys_checked).sum())

        # 3. Column Type Classification
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 4. Descriptive Statistics for Numerical Features
        stats_dict = {}
        if numerical_cols:
            desc = df[numerical_cols].describe().T
            for col in numerical_cols:
                series = df[col].dropna()
                stats_dict[col] = {
                    "count": int(desc.loc[col, "count"]),
                    "mean": float(round(desc.loc[col, "mean"], 4)),
                    "std": float(round(desc.loc[col, "std"], 4)),
                    "min": float(round(desc.loc[col, "min"], 4)),
                    "q25": float(round(desc.loc[col, "25%"], 4)),
                    "median": float(round(desc.loc[col, "50%"], 4)),
                    "q75": float(round(desc.loc[col, "75%"], 4)),
                    "max": float(round(desc.loc[col, "max"], 4)),
                    "skewness": float(round(series.skew(), 4)) if len(series) > 2 else 0.0,
                    "kurtosis": float(round(series.kurt(), 4)) if len(series) > 2 else 0.0
                }

        # 5. Categorical Column Profiles
        cat_dict = {}
        for col in categorical_cols:
            cat_dict[col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": df[col].value_counts().head(5).to_dict()
            }

        # 6. Internal Correlation Matrix (Numerical features)
        correlation_matrix = {}
        if len(numerical_cols) > 1:
            corr_df = df[numerical_cols].corr()
            correlation_matrix = {
                c1: {c2: round(float(corr_df.loc[c1, c2]), 4) for c2 in numerical_cols}
                for c1 in numerical_cols
            }

        report = {
            "filename": file_path.name,
            "rows": num_rows,
            "columns": num_cols,
            "column_names": columns,
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "total_missing_values": total_missing,
            "missing_values_by_column": missing_report,
            "exact_duplicate_rows": exact_duplicates,
            "composite_key_checked": composite_keys_checked,
            "composite_key_duplicates": composite_key_duplicates,
            "descriptive_statistics": stats_dict,
            "categorical_summary": cat_dict,
            "correlation_matrix": correlation_matrix
        }

        return report, df

    def _analyze_merged_plants(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyzes timestamp compatibility and cross-correlation between generation and weather."""
        merged_summary = {}

        plant_pairs = [
            ("Plant 1", "Plant_1_Generation_Data.csv", "Plant_1_Weather_Sensor_Data.csv"),
            ("Plant 2", "Plant_2_Generation_Data.csv", "Plant_2_Weather_Sensor_Data.csv")
        ]

        for plant_name, gen_key, weather_key in plant_pairs:
            if gen_key not in dfs or weather_key not in dfs:
                continue

            gen_df = dfs[gen_key].copy()
            weather_df = dfs[weather_key].copy()

            # Parse datetimes
            # Plant 1 generation has format %d-%m-%Y %H:%M, others %Y-%m-%d %H:%M:%S
            gen_df["PARSED_DATETIME"] = pd.to_datetime(gen_df["DATE_TIME"], errors="coerce")
            weather_df["PARSED_DATETIME"] = pd.to_datetime(weather_df["DATE_TIME"], errors="coerce")

            # Aggregate generation across inverters per timestamp to analyze plant-level solar response
            gen_agg = gen_df.groupby("PARSED_DATETIME").agg({
                "DC_POWER": "sum",
                "AC_POWER": "sum",
                "DAILY_YIELD": "sum"
            }).reset_index()

            # Merge on timestamp
            merged = pd.merge(gen_agg, weather_df, on="PARSED_DATETIME", how="inner")

            corr_features = [
                "AC_POWER", "DC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"
            ]
            available_features = [f for f in corr_features if f in merged.columns]

            cross_corr = {}
            if len(available_features) > 1:
                corr_sub = merged[available_features].corr(method="pearson")
                cross_corr = {
                    c1: {c2: round(float(corr_sub.loc[c1, c2]), 4) for c2 in available_features}
                    for c1 in available_features
                }

            merged_summary[plant_name] = {
                "generation_timestamps_count": int(gen_df["PARSED_DATETIME"].nunique()),
                "weather_timestamps_count": int(weather_df["PARSED_DATETIME"].nunique()),
                "matched_timestamps_count": int(len(merged)),
                "unmatched_generation_timestamps": int(gen_df["PARSED_DATETIME"].nunique() - len(merged)),
                "cross_correlation_weather_power": cross_corr
            }

        return merged_summary

    def _export_summary_json(self):
        """Saves inspection findings as a structured JSON artifact."""
        output_file = config.METRICS_DIR / "solar_generation_dataset_summary.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.summary_report, f, indent=4)
            logger.info(f"Saved solar generation dataset summary to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to export summary JSON: {e}")

    def _generate_visualizations(self, dfs: Dict[str, pd.DataFrame], merged_reports: Dict[str, Any]):
        """Generates correlation heatmaps for each plant."""
        plant_pairs = [
            ("Plant 1", "Plant_1_Generation_Data.csv", "Plant_1_Weather_Sensor_Data.csv", "solar_gen_plant1_corr.png"),
            ("Plant 2", "Plant_2_Generation_Data.csv", "Plant_2_Weather_Sensor_Data.csv", "solar_gen_plant2_corr.png")
        ]

        for plant_name, gen_key, weather_key, out_name in plant_pairs:
            if gen_key not in dfs or weather_key not in dfs:
                continue

            gen_df = dfs[gen_key].copy()
            weather_df = dfs[weather_key].copy()

            gen_df["PARSED_DATETIME"] = pd.to_datetime(gen_df["DATE_TIME"], errors="coerce")
            weather_df["PARSED_DATETIME"] = pd.to_datetime(weather_df["DATE_TIME"], errors="coerce")

            gen_agg = gen_df.groupby("PARSED_DATETIME").agg({
                "DC_POWER": "sum",
                "AC_POWER": "sum",
                "DAILY_YIELD": "sum"
            }).reset_index()

            merged = pd.merge(gen_agg, weather_df, on="PARSED_DATETIME", how="inner")
            cols = ["AC_POWER", "DC_POWER", "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"]
            avail = [c for c in cols if c in merged.columns]

            if len(avail) > 1:
                plt.figure(figsize=(8, 6))
                corr_matrix = merged[avail].corr()
                sns.heatmap(corr_matrix, annot=True, cmap="YlOrRd", fmt=".3f", linewidths=0.5, cbar=True)
                plt.title(f"Solar Generation & Weather Correlation Heatmap - {plant_name}", fontsize=12, fontweight="bold")
                plt.tight_layout()
                fig_path = config.FIGURES_DIR / out_name
                try:
                    plt.savefig(fig_path, dpi=200)
                    plt.close()
                    logger.info(f"Saved correlation figure to: {fig_path}")
                except Exception as e:
                    logger.error(f"Failed to save correlation figure: {e}")

    def _print_console_summary(self):
        """Prints a human-readable table report to stdout."""
        print("\n" + "=" * 75)
        print("          SOLAR GENERATION DATASET INSPECTION REPORT")
        print("=" * 75)
        print(f"Path: {self.data_dir}")
        print(f"Total CSV Files Inspected: {len(self.summary_report.get('file_reports', {}))}")
        print("-" * 75)

        for filename, rep in self.summary_report.get("file_reports", {}).items():
            print(f"\n[File: {filename}]")
            print(f"  Rows: {rep['rows']:,} | Columns: {rep['columns']}")
            print(f"  Numerical Columns: {rep['numerical_columns']}")
            print(f"  Categorical Columns: {rep['categorical_columns']}")
            print(f"  Missing Values: {rep['total_missing_values']} | Exact Duplicates: {rep['exact_duplicate_rows']}")
            if rep["composite_key_checked"]:
                print(f"  Composite Key Duplicates ({'+'.join(rep['composite_key_checked'])}): {rep['composite_key_duplicates']}")

            print("\n  Descriptive Statistics (Key Numerical Features):")
            print(f"  {'Feature':<20} | {'Mean':<12} | {'Std':<12} | {'Min':<10} | {'Max':<12} | {'Skew':<8}")
            print("  " + "-" * 82)
            for feat, stats in rep.get("descriptive_statistics", {}).items():
                print(f"  {feat:<20} | {stats['mean']:<12.2f} | {stats['std']:<12.2f} | {stats['min']:<10.2f} | {stats['max']:<12.2f} | {stats['skewness']:<8.2f}")

        merged_info = self.summary_report.get("merged_plant_analysis", {})
        if merged_info:
            print("\n" + "-" * 75)
            print("  Cross-Dataset Merged Correlation (Weather to Power Generation):")
            for plant, pinfo in merged_info.items():
                print(f"\n  [{plant}] Matched Timestamps: {pinfo['matched_timestamps_count']:,}")
                corr_map = pinfo.get("cross_correlation_weather_power", {})
                if "AC_POWER" in corr_map:
                    print("  Correlation with AC_POWER (Target):")
                    for k, v in corr_map["AC_POWER"].items():
                        if k != "AC_POWER":
                            print(f"    - {k:<22}: {v:+.4f}")

        print("\n" + "=" * 75)


if __name__ == "__main__":
    inspector = SolarGenerationInspector()
    inspector.inspect()
