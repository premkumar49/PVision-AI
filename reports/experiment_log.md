# PVision AI: Experimentation and Audit Ledger

**Project:** PVision AI — A Hybrid CNN-ANN Framework for Photovoltaic Fault Detection and Power Generation Prediction  
**Environment:** Windows 11 • Python 3.14.7 • PyTorch / Scikit-Learn Stack  
**Policy:** Zero Model Training inside Assistant; All Training Performed Manually by User in VS Code  

---

## Experiment Registry

| Experiment ID | Date / Time | Phase | Focus Domain | Status | Key Output / Metric |
| :---: | :---: | :---: | :--- | :---: | :--- |
| **EXP-001** | 2026-10-04 03:26 | Phase 1 | Project Directory Architecture Setup | PASSED | Standardized folder structure established |
| **EXP-002** | 2026-10-04 03:29 | Phase 1 | Dual Modality PV Fault Image Discovery | PASSED | 69,484 images discovered (35k GASF, 34.4k IV) |
| **EXP-003** | 2026-10-04 03:30 | Phase 1 | Telemetry CSV Schema & Key Inspection | PASSED | 142,917 records checked across Plant 1 & 2 |
| **EXP-004** | 2026-10-04 03:35 | Phase 1 | Initial Cross-Correlation & Inspection Runner | PASSED | $r(AC\_POWER, IRRADIATION) > 0.91$ verified |
| **EXP-005** | 2026-10-04 04:00 | Phase 2 | Full Cryptographic MD5 Deduplication & Integrity | PASSED | 0 corrupted files, 0 duplicates confirmed |
| **EXP-006** | 2026-10-04 04:03 | Phase 2 | Telemetry Anomaly & Ghost Power Identification | PASSED | 14 night ghosts & 8,253 day outages logged |
| **EXP-007** | 2026-10-04 04:05 | Phase 2 | Feature Engineering & Leakage-Free Scaling | PASSED | Cyclical time, $\Delta T$, Scaler fit on 70% Train |
| **EXP-008** | 2026-10-04 04:05 | Phase 2 | Manifest & Artifact Serialization | PASSED | 6 image manifests, 6 tabular CSVs generated |
| **EXP-009** | Planned | Phase 3 | CNN Classification (GASF Modality) | Planned / Not Yet Trained | EfficientNetV2-S Transfer Learning & Fine-Tuning |
| **EXP-010** | Planned | Phase 3 | CNN Classification (I-V Modality) | Planned / Not Yet Trained | EfficientNetV2-S Aspect-Ratio-Preserving Padding |

---

## Detailed Experiment Logs

### EXP-001: Project Directory Architecture Setup
- **Objective:** Establish modular structure for PVision AI separating data, models, preprocessing, outputs, and notebooks.
- **Execution:** Created root `PVision_AI` with `data/pv_fault/`, `data/solar_generation/`, `src/`, `notebooks/`, `outputs/`, `models/`, `app/`, `reports/`, and `visualizations/`.
- **Result:** Successfully validated directory existence and permissions.

### EXP-002: Dual Modality PV Fault Image Discovery
- **Objective:** Audit raw image modalities: GASF (2D time-series transformation) and I-V curves.
- **Observations:**
  - `gasf_images`: 7 classes (`crack`, `global_aging`, `hotspot`, `normal`, `partial_aging`, `shading`, `short_circuit`), exactly 5,000 files per class. Resolution: $256 \times 256$ RGB.
  - `iv_images`: 6 classes with 5,000 files; `short_circuit` has 4,484 files. Resolution: $1343 \times 808$ RGB.
  - Detected 516 missing index sequence numbers in `iv_images/short_circuit` (`1–513`, `597–599`).
- **Conclusion:** Confirmed class distribution and sequence gap for Phase 2 handling.

### EXP-003: Telemetry CSV Schema & Key Inspection
- **Objective:** Verify row counts, column types, missingness, and primary keys in generation and weather CSVs.
- **Results:**
  - Zero missing values in all 4 CSVs.
  - Zero duplicate rows in all 4 CSVs.
  - Identified critical timestamp format discrepancy: Plant 1 Generation uses `%d-%m-%Y %H:%M` while Plant 1 Weather uses `%Y-%m-%d %H:%M:%S`.

### EXP-004: Initial Cross-Correlation & Inspection Runner
- **Objective:** Build unified CLI runner (`dataset_inspector.py`) and compute Pearson correlation matrices between weather sensors and power generation.
- **Results:**
  - $AC\_POWER \leftrightarrow IRRADIATION$: $+0.9959$ (Plant 1), $+0.9111$ (Plant 2).
  - $AC\_POWER \leftrightarrow MODULE\_TEMPERATURE$: $+0.9610$ (Plant 1), $+0.8719$ (Plant 2).
  - Exported `pv_fault_dataset_summary.json` and `solar_generation_dataset_summary.json`.

### EXP-005: Full Cryptographic MD5 Deduplication & Integrity
- **Objective:** Exhaustively scan all 69,484 images for structural corruption (via PIL) and duplicate image files (via cryptographic MD5 hashing).
- **Execution:** Ran complete verification over both modalities (406.96s for GASF, 563.33s for IV).
- **Results:**
  - `gasf_images`: 35,000 valid files, 0 corrupted, 35,000 unique hashes, 0 duplicates.
  - `iv_images`: 34,484 valid files, 0 corrupted, 34,484 unique hashes, 0 duplicates.
  - Total cleaned images: 69,484 (100.0% retention).

### EXP-006: Telemetry Anomaly & Ghost Power Identification
- **Objective:** Identify physically invalid or anomalous operational states in solar generation.
- **Findings:**
  - Negative values: 0 detected across all features.
  - Night Ghost Generation ($IRRADIATION == 0 \land AC\_POWER > 0$): 14 records in Plant 2 ($< 0.5\text{ kW}$). Handled by zeroing out residual baseline sensor noise.
  - Daytime Inverter Outages ($IRRADIATION > 0.05\text{ kW/m}^2 \land AC\_POWER == 0$): 1,553 in Plant 1 and 6,700 in Plant 2. These represent inverter trips or grid curtailment and were segregated from nominal power training to preserve physical weather-generation relationships.

### EXP-007: Feature Engineering & Leakage-Free Scaling
- **Objective:** Construct domain-specific features and implement strict anti-leakage scaling.
- **Engineered Features:**
  - $\Delta T = MODULE\_TEMPERATURE - AMBIENT\_TEMPERATURE$
  - Thermal-Irradiance interaction: $IRRAD \times MODULE\_TEMP$
  - Cyclical time: $\sin(2\pi t / 24)$, $\cos(2\pi t / 24)$
  - Daytime indicator: $IS\_DAYTIME$
- **Target Leakage Prevention:**
  - Removed $DC\_POWER$ ($r = 0.9999$ identity shortcut).
  - Removed $TOTAL\_YIELD$ and $DAILY\_YIELD$ (cumulative future leakage).
  - Fit `StandardScaler` strictly on 70% Training split.
- **Results:** Scalers saved to `models/solar_ann_scaler_plant1.joblib` and `models/solar_ann_scaler_plant2.joblib`.

### EXP-008: Manifest & Artifact Serialization
- **Objective:** Serialize stratified split manifests and preprocessed feature matrices for reproducible training in VS Code.
- **Manifests Generated:**
  - `gasf_images_train_manifest.csv` (24,500), `val` (5,250), `test` (5,250).
  - `iv_images_train_manifest.csv` (24,138), `val` (5,173), `test` (5,173).
  - `ann_plant1_train.csv` (47,054), `val` (10,083), `test` (10,084).
  - `ann_plant2_train.csv` (42,698), `val` (9,150), `test` (9,150).
- **Verification:** Verified zero hash intersection between train, val, and test partitions.

### EXP-009: CNN PV Fault Classification — GASF Modality (EfficientNetV2-S)
- **Experiment ID:** EXP-009
- **Dataset:** PV Fault Image Dataset
- **Modality:** GASF (Gramian Angular Summation Fields)
- **Model Architecture:** EfficientNetV2-S
- **Input Size:** $224 \times 224 \times 3$ (Resized from $256 \times 256$)
- **Transfer Learning:** ImageNet-1k Pretrained Backbone
- **Fine-Tuning Strategy:** Two-Stage
  - Stage 1: Backbone frozen, train custom 7-class head (10 epochs, $\eta = 1 \times 10^{-3}$)
  - Stage 2: Unfreeze top 50 layers, fine-tune (20 epochs, $\eta = 1 \times 10^{-4}$)
- **Batch Size:** 32
- **Epochs:** 30 (10 Stage 1 + 20 Stage 2)
- **Learning Rate:** Initial $1 \times 10^{-3}$, Fine-Tuning $1 \times 10^{-4}$
- **Optimizer:** Adam
- **Hardware:** Auto-detected (GPU with CPU fallback)
- **Callbacks:** EarlyStopping (patience=5), ModelCheckpoint (min val_loss), ReduceLROnPlateau (factor=0.5, patience=2)
- **Status:** Planned / Not Yet Trained (Reserved for manual execution by user in VS Code)
- **Target Weights Output:** `models/cnn/efficientnetv2s_gasf_best.keras`

### EXP-010: CNN PV Fault Classification — I-V Curve Modality (EfficientNetV2-S)
- **Experiment ID:** EXP-010
- **Dataset:** PV Fault Image Dataset
- **Modality:** I-V (Current-Voltage Characteristic Curves)
- **Model Architecture:** EfficientNetV2-S
- **Input Size:** $224 \times 224 \times 3$ (Aspect-ratio-preserving resize and padding from $1343 \times 808$)
- **Transfer Learning:** ImageNet-1k Pretrained Backbone
- **Fine-Tuning Strategy:** Two-Stage
  - Stage 1: Backbone frozen, train custom 7-class head (10 epochs, $\eta = 1 \times 10^{-3}$)
  - Stage 2: Unfreeze top 50 layers, fine-tune (20 epochs, $\eta = 1 \times 10^{-4}$)
- **Batch Size:** 32
- **Epochs:** 30 (10 Stage 1 + 20 Stage 2)
- **Learning Rate:** Initial $1 \times 10^{-3}$, Fine-Tuning $1 \times 10^{-4}$
- **Optimizer:** Adam
- **Class Weights:** Inverse frequency class weighting configured for `short_circuit` (4,484 samples)
- **Hardware:** Auto-detected (GPU with CPU fallback)
- **Callbacks:** EarlyStopping (patience=5), ModelCheckpoint (min val_loss), ReduceLROnPlateau (factor=0.5, patience=2)
- **Status:** Planned / Not Yet Trained (Reserved for manual execution by user in VS Code)
- **Target Weights Output:** `models/cnn/efficientnetv2s_iv_best.keras`

