# PVision AI: A Hybrid CNN-ANN Framework for Photovoltaic Fault Detection and Power Generation Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview
**PVision AI** is a state-of-the-art hybrid framework combining Computer Vision and Tabular Machine Learning for photovoltaic (PV) solar farms. It addresses two critical operational challenges:
1. **Automated Fault Detection:** Identifies physical and electrical anomalies on PV modules using 2D image representations (Gramian Angular Summation Fields - GASF and Current-Voltage I-V characteristic curves).
2. **Solar Generation Forecasting:** Predicts continuous power output ($AC\_POWER$) using real-time atmospheric sensor data (solar irradiation, ambient temperature, module surface temperature).
3. **Hybrid Diagnostic Coupling:** Cross-references forecasted generation with detected module faults to quantify efficiency degradation, distinguish weather attenuation from hardware failure, and trigger proactive maintenance.

> [!IMPORTANT]
> **Manual Training Policy:** Models will **NOT** be trained inside automated agent loops. All model training is executed manually by the user within **VS Code**.

---

## 🗂️ Project Directory Structure

```
PVision_AI/
│
├── data/
│   ├── pv_fault/
│   │   ├── gasf_images/         # 35,000 GASF 2D images (256x256 RGB across 7 classes)
│   │   └── iv_images/           # 34,484 I-V characteristic curves (1343x808 RGB across 7 classes)
│   └── solar_generation/
│       ├── Plant_1_Generation_Data.csv      # 68,778 generation records (22 inverters)
│       ├── Plant_1_Weather_Sensor_Data.csv  # 3,182 weather telemetry records
│       ├── Plant_2_Generation_Data.csv      # 67,698 generation records (22 inverters)
│       └── Plant_2_Weather_Sensor_Data.csv  # 3,259 weather telemetry records
│
├── notebooks/
│   ├── 01_pv_fault_dataset_inspection.ipynb     # Interactive image dataset inspection
│   └── 02_solar_generation_dataset_inspection.ipynb  # Interactive telemetry & correlation inspection
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── inspect_pv_fault.py           # PV fault image inspection & audit
│   │   ├── inspect_solar_generation.py   # Solar generation CSV analysis & correlations
│   │   └── dataset_inspector.py          # Unified master dataset inspection CLI
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_preprocessing.py        # Stratified splits, augmentations, transforms
│   │   └── tabular_preprocessing.py      # Datetime harmonization, feature engineering, scaling
│   ├── models/
│   │   ├── __init__.py
│   │   └── architectures.py              # CNN & ANN blueprint specifications
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── hybrid_inference.py           # Hybrid coupling & derating engine
│   └── utilities/
│       ├── __init__.py
│       ├── config.py                     # Central project paths & schema constants
│       └── logger.py                     # Standardized logging
│
├── models/                               # Reserved for trained model weights (.pth / .pkl)
│
├── outputs/
│   ├── figures/                          # Distribution charts & correlation heatmaps
│   ├── metrics/                          # Structured JSON dataset summaries
│   └── predictions/                      # Train/val/test splits & manifests
│
├── app/
│   ├── components.py                     # Streamlit modular UI elements
│   └── styles.py                         # Custom CSS theme
│
├── requirements.txt                      # Project dependencies
├── README.md                             # Project documentation
└── app.py                                # Interactive Streamlit dashboard
```

---

## 🔍 Phase 1 Dataset Inspection Findings

### 1. PV Fault Image Dataset
- **Fault Categories (7 classes):** `crack`, `global_aging`, `hotspot`, `normal`, `partial_aging`, `shading`, `short_circuit`.
- **Modality 1 (GASF Images):**
  - **Resolution:** $256 \times 256$ RGB.
  - **Total Samples:** 35,000 (perfect balance: 5,000 per class).
  - **Corrupted / Zero-byte Files:** 0.
- **Modality 2 (I-V Curves):**
  - **Resolution:** $1343 \times 808$ RGB.
  - **Total Samples:** 34,484.
  - **Missing Sequence Detection:** `short_circuit` has 4,484 images instead of 5,000. Exactly **516 files are missing** (indices 1–513, and 597–599).
  - **Corrupted / Zero-byte Files:** 0.
- **Duplicate Analysis:** Hash scans reveal zero cross-class contamination.

### 2. Solar Generation & Weather Dataset
- **Total Generation Records:** 136,476 (Plant 1: 68,778, Plant 2: 67,698) across 44 inverters.
- **Total Weather Telemetry:** 6,441 records.
- **Missing Values:** **0** across all columns in all CSVs.
- **Duplicate Rows:** **0** exact row duplicates.
- **Datetime Format Discrepancy (Critical Preprocessing Finding):**
  - `Plant_1_Generation_Data.csv`: `%d-%m-%Y %H:%M`
  - `Plant_1_Weather_Sensor_Data.csv`: `%Y-%m-%d %H:%M:%S`
  - `Plant_2` (both files): `%Y-%m-%d %H:%M:%S`
- **Key Physical Correlations:**
  - $AC\_POWER \leftrightarrow IRRADIATION$: Strong positive ($r \approx 0.74 - 0.77$).
  - $AC\_POWER \leftrightarrow MODULE\_TEMPERATURE$: Strong positive ($r \approx 0.78 - 0.81$).
  - High degree of multicollinearity between $DC\_POWER$ and $AC\_POWER$ ($r > 0.99$), confirming inverter efficiency stability.

---

## 🚀 Getting Started

### 1. Installation
Activate your Python virtual environment and install dependencies:
```bash
cd PVision_AI
pip install -r requirements.txt
```

### 2. Run Dataset Inspection
Execute the master inspection runner:
```bash
# Run complete inspection suite
python -m src.data.dataset_inspector --all

# Or run individual inspectors
python -m src.data.inspect_pv_fault
python -m src.data.inspect_solar_generation
```
Outputs generated:
- `outputs/metrics/pv_fault_dataset_summary.json`
- `outputs/metrics/solar_generation_dataset_summary.json`
- `outputs/metrics/pvision_master_inspection_summary.json`
- `outputs/figures/pv_fault_class_distribution.png`
- `outputs/figures/solar_gen_plant1_corr.png`
- `outputs/figures/solar_gen_plant2_corr.png`

### 3. Interactive Jupyter Notebooks in VS Code
Open the inspection notebooks directly in VS Code:
- `notebooks/01_pv_fault_dataset_inspection.ipynb`
- `notebooks/02_solar_generation_dataset_inspection.ipynb`

### 4. Launch the Interactive Dashboard
Launch the PVision AI web application:
```bash
streamlit run app.py
```

---

## 🛠️ Manual Training Guide (VS Code)

To train models manually in VS Code:
1. **Prepare Data Splits & Features:**
   ```python
   from src.preprocessing.image_preprocessing import PVFaultDatasetManifest
   from src.preprocessing.tabular_preprocessing import SolarDataPipeline

   # Create stratified manifests for CNN
   manifest = PVFaultDatasetManifest(modality="gasf_images")
   splits = manifest.create_split_manifest()

   # Create tabular train/test splits for ANN
   pipeline = SolarDataPipeline()
   X_train, X_test, y_train, y_test = pipeline.prepare_training_datasets(plant_id=1)
   ```
2. **Train Models:**
   - Execute training scripts in VS Code using GPU acceleration if available.
   - Save trained checkpoints to `models/` directory:
     - `models/pv_fault_cnn_best.pth`
     - `models/solar_power_ann_best.pth`

---

## 🗺️ Project Roadmap
- [x] **Phase 1: Project Setup & Dataset Inspection (Completed)**
- [ ] **Phase 2: Preprocessing & Manual Model Training in VS Code**
- [ ] **Phase 3: Hybrid Coupling, Evaluation & Web App Deployment**
