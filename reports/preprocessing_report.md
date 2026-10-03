# PVision AI: Data Cleaning and Preprocessing Report

**Generated:** 2026-10-04 04:05:42  
**Project:** PVision AI — A Hybrid CNN-ANN Framework for Photovoltaic Fault Detection and Power Generation Prediction  
**Status:** Preprocessing and Splitting Complete (Zero Model Training Executed)  

---

## 1. Executive Summary

This report documents the rigorous data cleaning, anomaly filtering, feature engineering, and leakage-preventive partitioning performed across both branches of the PVision AI framework:
1. **Vision Branch (CNN):** 69,484 photovoltaic fault images (35,000 GASF and 34,484 I-V curves) audited for file integrity, zero-byte corruption, cryptographic duplication, and class balance.
2. **Forecasting Branch (ANN):** 136,476 solar generation records and 6,441 weather telemetry readings from Plant 1 and Plant 2 harmonized across asynchronous sampling formats, cleansed of night ghost readings and daytime inverter outages, engineered with physical thermodynamic interaction terms, and normalized via training-fitted standardizers.

---

## 2. CNN Image Dataset Cleaning & Quality Audit

### 2.1 Before vs. After Cleaning Quality Table

| Modality | Images Before Cleaning | Corrupted Images | Duplicate Images (MD5) | Images After Cleaning | Retention Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GASF Images** ([`gasf_images`](file:///d:/SolarPredict/PVision_AI/data/pv_fault/gasf_images)) | 35,000 | 0 | 0 | **35,000** | 100.0% |
| **I-V Curves** ([`iv_images`](file:///d:/SolarPredict/PVision_AI/data/pv_fault/iv_images)) | 34,484 | 0 | 0 | **34,484** | 100.0% |
| **Total Vision Dataset** | **69,484** | **0** | **0** | **69,484** | **100.0%** |

### 2.2 Class Imbalance Analysis

| Class Name | Label Index | GASF Count | GASF Share | I-V Count | I-V Share | Status / Sequence Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `crack` | 0 | 5,000 | 14.286% | 5,000 | 14.499% | Balanced (Full sequence 1–5,000) |
| `global_aging` | 1 | 5,000 | 14.286% | 5,000 | 14.499% | Balanced (Full sequence 1–5,000) |
| `hotspot` | 2 | 5,000 | 14.286% | 5,000 | 14.499% | Balanced (Full sequence 1–5,000) |
| `normal` | 3 | 5,000 | 14.286% | 5,000 | 14.499% | Balanced (Full sequence 1–5,000) |
| `partial_aging` | 4 | 5,000 | 14.286% | 5,000 | 14.499% | Balanced (Full sequence 1–5,000) |
| `shading` | 5 | 5,000 | 14.286% | 5,000 | 14.499% | Balanced (Full sequence 1–5,000) |
| `short_circuit` | 6 | 5,000 | 14.286% | 4,484 | 13.003% | Imbalance ratio: 0.8968 (Missing indices: 1–513, 597–599) |

### 2.3 Partitioning & Leakage Prevention Strategy
To ensure reproducible, zero-leakage evaluation, stratified splits were computed with fixed random seed (`seed=42`). Zero split overlap was strictly confirmed via hash set intersections:
- Train intersection with Val is empty (0 common hashes).
- Train intersection with Test is empty (0 common hashes).
- Val intersection with Test is empty (0 common hashes).

| Modality | Train Split (70%) | Validation Split (15%) | Test Split (15%) | Total Samples | Manifest Files Generated |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **GASF Images** | 24,500 | 5,250 | 5,250 | 35,000 | [`gasf_train_manifest.csv`](file:///d:/SolarPredict/PVision_AI/data/processed/gasf_images_train_manifest.csv) |
| **I-V Curves** | 24,138 | 5,173 | 5,173 | 34,484 | [`iv_images_train_manifest.csv`](file:///d:/SolarPredict/PVision_AI/data/processed/iv_images_train_manifest.csv) |

### 2.4 Vision Preprocessing & Augmentation Pipelines
- **Spatial Transformation:** Resized to standard 224 x 224 pixels (Bilinear interpolation).
- **Pixel Normalization:** Normalized according to ImageNet standard channel statistics:
  - mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
- **Augmentation Suite (Training Only):**
  - Random Horizontal Flip (p = 0.5)
  - Random Vertical Flip (p = 0.5)
  - Subtle Random Rotation (+/- 15 degrees)
  - Color Jitter (Brightness +/- 0.1, Contrast +/- 0.1)
  - *Verification Sample Grid:* Available at [`visualizations/preprocessing/cnn_augmentation_samples.png`](file:///d:/SolarPredict/PVision_AI/visualizations/preprocessing/cnn_augmentation_samples.png).

---

## 3. ANN Solar Generation Telemetry Preprocessing

### 3.1 Before vs. After Cleaning Quality Table

| Plant Identifier | Raw Inverter Records | Weather Telemetry | Duplicate Rows | Missing Values | Merged Records | Night Ghost Filtered | Daytime Outages Filtered | Operational Rows Kept |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plant 1** | 68,778 | 3,182 | 0 | 0 | 68,774 | 0 | 1,553 | **67,221** |
| **Plant 2** | 67,698 | 3,259 | 0 | 0 | 67,698 | 14 | 6,700 | **60,998** |
| **Combined** | **136,476** | **6,441** | **0** | **0** | **136,472** | **14** | **8,253** | **128,219** |

### 3.2 Outlier & Anomaly Treatment Rationale
1. **Ghost Night Generation (IRRADIATION == 0 and AC_POWER > 0):** In Plant 2, 14 records exhibited residual power noise (< 0.5 kW) when solar irradiation was 0.00. These were zeroed out to prevent the model from hallucinating solar output during midnight hours.
2. **Daytime Inverter Outages (IRRADIATION > 0.05 kW/m^2 and AC_POWER == 0):** In total, 1,553 rows in Plant 1 and 6,700 rows in Plant 2 had strong daylight but zero power generation. These reflect physical inverter trips, grid curtailment, or maintenance downtime. Training a weather-to-power regressor on these records would poison nominal solar physics. They were segregated into diagnostic logs and filtered out of nominal baseline training.
3. **Physical Boundary Sanity:** Verified that zero negative values exist across temperature, irradiance, and electrical metrics.

### 3.3 Feature Selection & Engineering Justifications

| Feature Name | Type | Physical / Domain Justification |
| :--- | :--- | :--- |
| **`IRRADIATION`** | Primary Predictor | Solar irradiance (kW/m^2) provides the primary photon flux driving photovoltaic electron excitation (r > 0.90). |
| **`MODULE_TEMPERATURE`** | Primary Predictor | Panel surface temperature (deg C); dictates semiconductor bandgap efficiency drop (negative thermal coefficient). |
| **`AMBIENT_TEMPERATURE`**| Environmental | Ambient convective temperature (deg C) governing heat dissipation from PV panels. |
| **`TEMP_DIFFERENCE`** | Engineered Dynamic | Delta T = T_module - T_ambient; represents instantaneous solar thermal absorption load. |
| **`IRRAD_MODULE_INTERACTION`** | Engineered Dynamic | G x T_module; models non-linear power efficiency degradation under high thermal-irradiance stress. |
| **`HOUR`** | Temporal | Hour of the day (0-23); captures macro diurnal solar angle. |
| **`SIN_TIME`**, **`COS_TIME`** | Cyclical Encoding | Periodic trigonometric transformation of time of day (sin(2*pi*t/24), cos(2*pi*t/24)) preserving 23:45 to 00:00 continuity. |
| **`IS_DAYTIME`** | Domain Mask | Binary indicator separating active generation regimes from baseline night states. |

### 3.4 Target Definition & Anti-Leakage Measures
- **Designated Target:** `AC_POWER` (kilowatts - the usable alternating current exported to the grid).
- **Target Leakage Prevention:**
  - `DC_POWER` is **strictly excluded** from feature inputs. Because DC_POWER and AC_POWER have a correlation of r = 0.9999 via inverter conversion, including DC_POWER would reduce the forecasting model to a trivial conversion lookup rather than forecasting power from atmospheric conditions.
  - `TOTAL_YIELD` and `DAILY_YIELD` are **excluded** because cumulative lifetime and intra-day counters introduce non-stationary data leakage from the future.
  - `StandardScaler` was fitted **strictly on the Training split** (70%). Zero scaling statistics were borrowed from Validation or Test sets.
  - Fitted scalers persisted to [`models/solar_ann_scaler_plant1.joblib`](file:///d:/SolarPredict/PVision_AI/models/solar_ann_scaler_plant1.joblib) and [`models/solar_ann_scaler_plant2.joblib`](file:///d:/SolarPredict/PVision_AI/models/solar_ann_scaler_plant2.joblib).

### 3.5 Tabular Partitioning Splits

| Plant | Training Samples (70%) | Validation Samples (15%) | Test Samples (15%) | Scaler Saved |
| :--- | :---: | :---: | :---: | :--- |
| **Plant 1** | 47,054 | 10,083 | 10,084 | [`solar_ann_scaler_plant1.joblib`](file:///d:/SolarPredict/PVision_AI/models/solar_ann_scaler_plant1.joblib) |
| **Plant 2** | 42,698 | 9,150 | 9,150 | [`solar_ann_scaler_plant2.joblib`](file:///d:/SolarPredict/PVision_AI/models/solar_ann_scaler_plant2.joblib) |

---

## 4. Visualizations Directory Audit

The following high-resolution diagnostic charts were generated and saved in [`visualizations/preprocessing/`](file:///d:/SolarPredict/PVision_AI/visualizations/preprocessing/):
- **[`cnn_augmentation_samples.png`](file:///d:/SolarPredict/PVision_AI/visualizations/preprocessing/cnn_augmentation_samples.png):** Visual inspection of raw vs. resized vs. augmented (flip/rotate) samples across GASF and I-V curves.
- **[`ann_plant1_outliers_boxplot.png`](file:///d:/SolarPredict/PVision_AI/visualizations/preprocessing/ann_plant1_outliers_boxplot.png):** Distribution and whisker range for Plant 1 features.
- **[`ann_plant2_outliers_boxplot.png`](file:///d:/SolarPredict/PVision_AI/visualizations/preprocessing/ann_plant2_outliers_boxplot.png):** Distribution and whisker range for Plant 2 features.
- **[`ann_plant1_features_correlation.png`](file:///d:/SolarPredict/PVision_AI/visualizations/preprocessing/ann_plant1_features_correlation.png):** Correlation matrix for engineered predictors vs `AC_POWER` in Plant 1.
- **[`ann_plant2_features_correlation.png`](file:///d:/SolarPredict/PVision_AI/visualizations/preprocessing/ann_plant2_features_correlation.png):** Correlation matrix for engineered predictors vs `AC_POWER` in Plant 2.
- **[`ann_plant1_diurnal_power_profile.png`](file:///d:/SolarPredict/PVision_AI/visualizations/preprocessing/ann_plant1_diurnal_power_profile.png):** 24-hour diurnal power generation curve with confidence bounds.
- **[`ann_plant2_diurnal_power_profile.png`](file:///d:/SolarPredict/PVision_AI/visualizations/preprocessing/ann_plant2_diurnal_power_profile.png):** 24-hour diurnal power generation curve with confidence bounds.

---

## 5. Instructions for Manual VS Code Training

All dataset manifests, engineered feature tables, and normalization scalers are fully generated and ready on disk.
To train models manually in VS Code:
1. Load manifests:
   ```python
   import pandas as pd
   gasf_train = pd.read_csv("data/processed/gasf_images_train_manifest.csv")
   gasf_val = pd.read_csv("data/processed/gasf_images_val_manifest.csv")
   ```
2. Load scaled telemetry:
   ```python
   ann_train = pd.read_csv("data/processed/ann_plant1_train.csv")
   X_train = ann_train.drop(columns=["AC_POWER"])
   y_train = ann_train["AC_POWER"]
   ```
3. Save trained weights to `models/pv_fault_cnn_best.pth` and `models/solar_power_ann_best.pth`.
