# PVision AI: Project Technical Report
## A Hybrid CNN-ANN Framework for Photovoltaic Fault Detection and Power Generation Prediction

**Version:** 1.1.0 (Phase 3 Milestone — CNN Pipeline Ready)  
**Date:** October 2026  
**Status:** CNN Architecture & Evaluation Configured • Model Training Reserved for Manual VS Code Execution  

---

## 1. Executive Summary

Solar photovoltaic (PV) installations are susceptible to physical degradation, module micro-cracks, environmental shading, and electrical faults that cause substantial power losses and catastrophic system failures. Concurrently, grid operators require highly reliable forecasts of continuous active power output ($AC\_POWER$) based on real-time atmospheric variables.

**PVision AI** proposes a dual-branch hybrid framework:
1. **Vision Diagnostics Branch (CNN):** Performs 7-class multi-anomaly identification from module imaging representations:
   - Gramian Angular Summation Fields (GASF): 2D spatial-temporal matrices ($256 \times 256$ RGB).
   - Current-Voltage (I-V) characteristic curves: Graphical electrical responses ($1343 \times 808$ RGB).
2. **Forecasting Branch (ANN):** Performs non-linear continuous regression of active power output ($AC\_POWER$) utilizing meteorological telemetry (solar irradiance, panel temperature, ambient temperature, and temporal cyclical features).
3. **Diagnostic Coupling Engine:** Integrates the predicted nominal baseline power with classified physical faults, allowing operators to distinguish environmental attenuation from hardware malfunction and quantify instantaneous efficiency derating.

In strict adherence to project specifications, **no model training was automated inside the assistant environment**. All data cleaning, anomaly mitigation, feature engineering, and leakage-preventive partitioning have been completed and verified.

---

## 2. System Architecture

```
                                  PVision AI SYSTEM WORKFLOW
                                  
     [Module Telemetry / I-V Curves]                     [Meteorological Sensor Data]
                    │                                                  │
                    ▼                                                  ▼
     ┌─────────────────────────────┐                    ┌─────────────────────────────┐
     │  Vision Branch Preprocessing│                    │ Tabular Feature Engineering │
     │  - Resizing (224x224)       │                    │ - Datetime Harmonization    │
     │  - ImageNet Standardization │                    │ - Delta-T & Interaction     │
     │  - Data Augmentation        │                    │ - Sin/Cos Cyclical Time     │
     └──────────────┬──────────────┘                    └──────────────┬──────────────┘
                    │                                                  │
                    ▼                                                  ▼
     ┌─────────────────────────────┐                    ┌─────────────────────────────┐
     │   CNN Classifier (VS Code)  │                    │   ANN Regressor (VS Code)   │
     │   (ResNet-18 / MobileNet)   │                    │   (Multi-Layer Perceptron)  │
     └──────────────┬──────────────┘                    └──────────────┬──────────────┘
                    │                                                  │
                    ▼                                                  ▼
            Diagnosed Fault Class                            Nominal AC Power (kW)
                    │                                                  │
                    └────────────────────┬─────────────────────────────┘
                                         ▼
                         ┌───────────────────────────────┐
                         │    Hybrid Decision Engine     │
                         │   - Fault Impact Derating     │
                         │   - Outage / Soiling Flag     │
                         │   - Expected Efficiency Yield │
                         └───────────────────────────────┘
```

---

## 3. Dataset Audit & Data Quality Findings

### 3.1 PV Fault Image Datasets (Dual Modality)

Full cryptographic (MD5) scans and PIL image integrity verifications were conducted across all **69,484** image files:

| Metric / Dimension | GASF Modality ([`gasf_images`](file:///d:/SolarPredict/PVision_AI/data/pv_fault/gasf_images)) | I-V Curves ([`iv_images`](file:///d:/SolarPredict/PVision_AI/data/pv_fault/iv_images)) | Combined Vision Suite |
| :--- | :---: | :---: | :---: |
| **Total Images Scanned** | 35,000 | 34,484 | **69,484** |
| **Zero-Byte / Corrupted Files** | 0 | 0 | **0** |
| **Cryptographic Duplicates** | 0 | 0 | **0** |
| **Native Image Resolution** | $256 \times 256$ RGB | $1343 \times 808$ RGB | — |
| **Target Preprocessed Resolution** | $224 \times 224$ RGB | $224 \times 224$ RGB | $224 \times 224$ RGB |
| **Remaining Post-Cleaning** | 35,000 | 34,484 | **69,484** |
| **Retention Rate** | 100.0% | 100.0% | **100.0%** |

#### Class Distribution & Sequence Integrity Analysis

1. **GASF Modality:** Perfectly uniform across all 7 classes ($5,000$ samples each, $14.286\%$ class share).
2. **I-V Modality:** 6 classes have $5,000$ samples each ($14.499\%$). The `short_circuit` class contains **4,484 samples** ($13.003\%$, imbalance ratio $0.8968$). Sequence analysis detected a missing sequence gap of **516 files** (indices `1–513` and `597–599`).
3. **Leakage-Free Partitioning:** Stratified splitting was performed with fixed seed (`seed=42`). Intersection between train, validation, and test hashes confirmed strictly zero cross-split contamination.

---

### 3.2 Solar Generation & Weather Telemetry Datasets

Telemetry data from two commercial PV solar power plants covering 34 continuous operational days (May 15, 2020 – June 17, 2020) with 15-minute sampling intervals were audited:

| Data Source | Raw Rows | Columns | Exact Duplicates | Missing Values | Key Diagnostic Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Plant 1 Generation** | 68,778 | 7 | 0 | 0 | 22 inverters; format: `%d-%m-%Y %H:%M` |
| **Plant 1 Weather** | 3,182 | 6 | 0 | 0 | 1 sensor station; format: `%Y-%m-%d %H:%M:%S` |
| **Plant 2 Generation** | 67,698 | 7 | 0 | 0 | 22 inverters; format: `%Y-%m-%d %H:%M:%S` |
| **Plant 2 Weather** | 3,259 | 6 | 0 | 0 | 1 sensor station; format: `%Y-%m-%d %H:%M:%S` |
| **Total Telemetry** | **142,917** | — | **0** | **0** | **100% complete data integrity** |

#### Telemetry Cleaning & Anomaly Resolution
- **Timestamp Harmonization:** Reconciled disparate datetime formats between Plant 1 generation and weather files to enable synchronous inverter-to-weather merges.
- **Nighttime Ghost Power:** Identified 14 records in Plant 2 exhibiting non-zero baseline residual power ($< 0.5\text{ kW}$) when irradiation was strictly zero. These were neutralized to $0.0\text{ kW}$ to prevent spurious night generation.
- **Daytime Inverter Outages:** Identified 1,553 records in Plant 1 and 6,700 records in Plant 2 where strong irradiance ($> 0.05\text{ kW/m}^2$) coincided with zero power output. These reflect physical inverter trips, grid curtailment, or maintenance downtime. They were filtered out of nominal baseline training to prevent poisoning the physical regression surface.
- **Operational Samples Retained:** 67,221 in Plant 1; 60,998 in Plant 2 (Total: 128,219 rows).

---

## 4. Feature Engineering & Selection Justification

| Feature | Category | Rationale & Mathematical Formulation |
| :--- | :--- | :--- |
| **`IRRADIATION`** | Primary Predictor | Solar irradiance ($\text{kW/m}^2$) is the direct photon energy flux driving power conversion ($r > 0.90$). |
| **`MODULE_TEMPERATURE`** | Primary Predictor | Panel surface temperature (°C); governs negative temperature coefficient bandgap losses. |
| **`AMBIENT_TEMPERATURE`**| Environmental | Ambient convective temperature (°C) dictating heat transfer to surrounding air. |
| **`TEMP_DIFFERENCE`** | Engineered Dynamic | $\Delta T = T_{\text{module}} - T_{\text{ambient}}$; quantifies instantaneous solar thermal absorption load. |
| **`IRRAD_MODULE_INTERACTION`** | Engineered Dynamic | $G \times T_{\text{module}}$; models non-linear power efficiency degradation under coupled thermal-irradiance stress. |
| **`HOUR`** | Temporal | Solar elevation proxy ($0–23$). |
| **`SIN_TIME`**, **`COS_TIME`** | Cyclical Encoding | $\sin(2\pi t / 24)$ and $\cos(2\pi t / 24)$; ensures smooth continuity across midnight ($23:45 \to 00:00$). |
| **`IS_DAYTIME`** | Domain Mask | Binary indicator separating active generation regimes from baseline night states. |

### Anti-Leakage Measures
1. **Target Leakage:** `DC_POWER` was strictly excluded ($r = 0.9999$ with `AC_POWER`) to ensure the model learns weather-to-power dynamics rather than inverter rectification. `TOTAL_YIELD` and `DAILY_YIELD` were excluded to prevent future cumulative leakage.
2. **Distribution Leakage:** Standard scalers were fitted **strictly on the Training split** ($70\%$). Val and Test splits were transformed using only training statistics.

---

## 5. Artifacts and Generated Deliverables

### Preprocessed Data & Manifests
- `data/processed/gasf_images_train_manifest.csv` ($24,500$ rows)
- `data/processed/gasf_images_val_manifest.csv` ($5,250$ rows)
- `data/processed/gasf_images_test_manifest.csv` ($5,250$ rows)
- `data/processed/iv_images_train_manifest.csv` ($24,138$ rows)
- `data/processed/iv_images_val_manifest.csv` ($5,173$ rows)
- `data/processed/iv_images_test_manifest.csv` ($5,173$ rows)
- `data/processed/ann_plant1_train.csv`, `val.csv`, `test.csv`
- `data/processed/ann_plant2_train.csv`, `val.csv`, `test.csv`

### Persisted Scalers & Metrics
- `models/solar_ann_scaler_plant1.joblib`
- `models/solar_ann_scaler_plant2.joblib`
- `reports/cnn_cleaning_summary.json`
- `reports/ann_cleaning_summary.json`
- `reports/preprocessing_report.md`

### Visualizations
- `visualizations/preprocessing/cnn_augmentation_samples.png`
- `visualizations/preprocessing/ann_plant1_outliers_boxplot.png`
- `visualizations/preprocessing/ann_plant2_outliers_boxplot.png`
- `visualizations/preprocessing/ann_plant1_features_correlation.png`
- `visualizations/preprocessing/ann_plant2_features_correlation.png`
- `visualizations/preprocessing/ann_plant1_diurnal_power_profile.png`
- `visualizations/preprocessing/ann_plant2_diurnal_power_profile.png`

---

## 6. CNN PV Fault Classification Methodology (Phase 3)

### 6.1 Objective
The vision classification branch provides automated, objective fault diagnosis from 2D representations of PV module health. By identifying physical and electrical degradation early, it prevents catastrophic thermal runaway (e.g. fire hazards from bypass diode failure) and guides cleaning or module replacement schedules.

### 6.2 Datasets & Modalities
The vision branch trains on two independent modalities:
1. **GASF Modality (Gramian Angular Summation Fields):**
   - 35,000 images ($256 \times 256$ RGB).
   - Uniform distribution across all 7 classes ($5,000$ images per class).
   - Captures high-frequency electrical time-series dynamics encoded into 2D polar matrices.
2. **I-V Modality (Current-Voltage Characteristic Curves):**
   - 34,484 images ($1343 \times 808$ RGB).
   - 6 classes contain $5,000$ images; `short_circuit` contains $4,484$ images ($516$ missing sequence files).
   - Graphically depicts the knee point, short-circuit current, and open-circuit voltage response.
   - Modalities are trained and evaluated independently to benchmark spatial vs. graphical feature efficacy.

### 6.3 Seven PV Fault Classes
- `crack`: Silicon crystalline wafer fracture.
- `global_aging`: Uniform degradation of encapsulant, front glass solarization, and series resistance increase.
- `hotspot`: Localized cell overheating under reverse bias.
- `normal`: Optimal operating condition with nominal fill factor.
- `partial_aging`: String-level or sub-module localized wear.
- `shading`: Non-uniform photon blockage (soiling, obstacles, clouds).
- `short_circuit`: Cell junction short-circuit or bypass diode failure.

### 6.4 Preprocessing & Spatial Transformation
- **GASF Preprocessing:** Bilinear resize directly from $256 \times 256$ to $224 \times 224$ RGB.
- **I-V Preprocessing (Aspect-Ratio-Preserving Padding):** Direct non-uniform stretching from $1343 \times 808$ to $224 \times 224$ causes severe geometric distortion of curve slope and knee curvature. An aspect-ratio-preserving resize with centered padding (to $224 \times 224$) is strictly applied to maintain genuine electrical physics.
- **Normalization:** Standard ImageNet channel standardization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Training Augmentation:** Horizontal/vertical flips, subtle rotation ($\pm 15^\circ$), and minor contrast/brightness adjustments applied strictly to the training split.

### 6.5 Stratified Partitioning & Leakage Safeguards
- 70% Training / 15% Validation / 15% Testing.
- Stratified by class labels to preserve class distributions across partitions.
- Exact hash verification confirmed zero split overlap ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$).
- Manifests saved to `data/processed/{modality}_train_manifest.csv`, `val_manifest.csv`, and `test_manifest.csv`.

### 6.6 EfficientNetV2-S Architecture & Transfer Learning
The primary CNN backbone is **EfficientNetV2-S**, chosen for its optimal trade-off between parameter efficiency ($21.5\text{M}$ parameters) and top-1 accuracy via progressive training and Fused-MBConv layers.
- **Pretrained Weights:** ImageNet-1k initialization.
- **Custom Classification Head:**
  - `GlobalAveragePooling2D`
  - `BatchNormalization`
  - `Dropout(rate=0.3)`
  - `Dense(7, activation='softmax')`

### 6.7 Two-Stage Fine-Tuning Strategy
- **Stage 1 (Feature Extraction):**
  - Backbone frozen (`base_model.trainable = False`).
  - Optimizer: Adam ($\eta = 1 \times 10^{-3}$).
  - Epochs: 10 epochs to adapt the classification head to PV fault representations.
- **Stage 2 (Fine-Tuning):**
  - Unfreeze top 50 layers of the EfficientNetV2-S backbone.
  - Optimizer: Adam ($\eta = 1 \times 10^{-4}$, 10x smaller learning rate to avoid destroying pretrained feature extractors).
  - Epochs: 20 epochs with `ReduceLROnPlateau` and `EarlyStopping`.
- **Checkpointing:** Best model saved based on minimum `val_loss`.

### 6.8 Evaluation Methodology
Models will be assessed on the test partition without data fabrication:
- Metrics: Global Accuracy, Macro Precision, Macro Recall, Macro F1-score.
- Per-class metrics: Precision, Recall, and F1-score per fault category.
- Full $7 \times 7$ Confusion Matrix and learning curves exported to `results/cnn/{modality}/`.
- *Numerical results will be entered after the user executes training manually in VS Code.*

---

## 7. Manual VS Code Training & Execution Guide

To train the CNN models manually in VS Code:
```bash
# 1. Train GASF Modality
python src/cnn/train_cnn.py --modality gasf --epochs-stage1 10 --epochs-stage2 20

# 2. Train I-V Modality (with optional class weights)
python src/cnn/train_cnn.py --modality iv --epochs-stage1 10 --epochs-stage2 20 --use-class-weights

# 3. Evaluate Models on Test Data
python src/cnn/evaluate_cnn.py --modality gasf
python src/cnn/evaluate_cnn.py --modality iv

# 4. Predict on a Single Test Image
python src/cnn/predict_cnn.py --image-path data/pv_fault/gasf_images/hotspot/hotspot_00001_GASF.png --modality gasf
```

Model checkpoints will be output to:
- `models/cnn/efficientnetv2s_gasf_best.keras`
- `models/cnn/efficientnetv2s_iv_best.keras`

---

## 8. Phase 4: CNN Feature Extraction Pipeline & Hybrid Integration

### 8.1 Purpose & Role in Hybrid Architecture
The CNN feature extraction pipeline serves as the analytical bridge between raw visual/graphical PV diagnostics and the subsequent modeling stages. Rather than treating the CNN purely as a black-box categorical predictor, the feature extraction framework extracts:
1. **Discrete Fault Diagnostics:** Argmax fault classifications and 7-class posterior confidence scores.
2. **Latent Visual/Physical Embeddings:** Continuous intermediate representations from the penultimate pooling layer that encode high-order non-linear physical characteristics (e.g., knee degradation in I-V curves, texture anomalies in GASF).

### 8.2 Feature Extraction Architecture
Feature extraction is performed without retraining or weight disturbance using a sub-model tapping the intermediate pooling layer:
```text
Input Image (224×224×3)
       │
       ▼
EfficientNetV2-S Backbone (Fused-MBConv / MBConv)
       │
       ▼
Global Average Pooling ('global_avg_pool') ──► [ 1,280-D Feature Vector ]
       │
       ▼
Batch Normalization & Dropout
       │
       ▼
Dense Softmax Head (7 units) ───────────────► [ 7-Class Posterior Probabilities ]
```

### 8.3 Feature Vector Representation & Storage
Extracted representations are stored in `results/cnn/features/`:
- `{modality}_features.npy`: High-performance dense NumPy matrix of shape $(N, 1280)$.
- `{modality}_features.csv`: Tabular metadata containing `image_id`, `true_class`, `predicted_class`, `confidence`, individual class probabilities (`prob_crack` through `prob_short_circuit`), and vector reference pointers.
- `{modality}_feature_metadata.json`: Machine-readable audit trail capturing layer names, dimensions, timestamp, preprocessing parameters, and model lineage.

### 8.4 Feature Visualization (PCA)
To verify class clustering in unsupervised projection space, the pipeline implements 2D Principal Component Analysis (PCA) generating:
- `results/cnn/features/gasf_feature_pca.png`
- `results/cnn/features/iv_feature_pca.png`
*Crucial methodological rule: PCA is used strictly for exploratory 2D geometric visualization. PCA clustering does not substitute for quantitative test set evaluation or guarantee downstream regression improvements.*

### 8.5 Scientific Limitation & Dataset Separation
> [!CAUTION]
> **Prohibition of Artificial 1-to-1 Dataset Pairing:**
> The PV fault image dataset (69,484 GASF and I-V images) and the solar generation time-series dataset (68,778 environmental records) originate from separate installations and physical systems.
> - Under no circumstances will artificial sample-to-sample pairings (e.g., `image_0001` $\to$ `generation_row_0001`) be synthesized.
> - Downstream hybrid ANN integration will operate via scientifically defensible scenario conditioning (e.g., applying fault state penalties or synthetic degradation factors across environmental regimes) rather than arbitrary row-wise merging.

### 8.6 Feature Extraction Execution Guide
Once CNN training is completed:
```bash
# Extract features from GASF test set
python src/cnn/extract_features.py --modality gasf --split test

# Extract features from I-V test set
python src/cnn/extract_features.py --modality iv --split test
```
If the trained model checkpoint is not yet available, the script cleanly informs the user that feature extraction is pending CNN training without failing or raising exceptions.

---

## 9. Phase 5: ANN-Based Solar Power Prediction

### 9.1 Purpose & Role in PVision AI
The Artificial Neural Network (ANN) regression subsystem models the quantitative relationship between meteorological conditions, thermal states, diurnal cycles, and continuous AC electrical power generation ($P_{AC}$, kW). 

**Fundamental Architectural Distinction:**
- **CNN Subsystem:** Operates exclusively on high-resolution image modalities (GASF and I-V curves) to diagnose discrete physical fault categories and extract latent spatial/electrical degradation embeddings.
- **ANN Subsystem:** Operates exclusively on structured tabular meteorological and operational telemetry to predict instantaneous electrical generation.
- *Scientific Clarification:* The CNN does not directly predict numerical power output from images alone. Rather, power prediction is modeled by the ANN from physical environmental telemetry, with CNN fault states providing operational conditioning in later hybrid stages.

### 9.2 Confirmed Target Variable
- **Target:** `AC_POWER` (Inverter Alternating Current Power Output).
- **Unit:** Kilowatts (kW).
- **Selection Rationale:** AC power represents the true commercial energy injected into the electrical grid after inverter conversion.

### 9.3 Predictive Features & Feature Engineering
A 12-dimensional feature vector is selected to capture solar thermodynamics and cyclical diurnal harmonics:
1. `IRRADIATION`: Solar irradiance ($W/m^2$).
2. `AMBIENT_TEMPERATURE`: Ambient dry-bulb air temperature ($^\circ C$).
3. `MODULE_TEMPERATURE`: Module surface temperature ($^\circ C$).
4. `TEMP_DIFFERENCE`: Thermal elevation above ambient ($T_{module} - T_{ambient}$).
5. `IRRAD_MODULE_INTERACTION`: Interaction product ($\text{IRRADIATION} \times T_{module}$).
6. `HOUR`, `MINUTE`, `TIME_DECIMAL`: Continuous daytime progress.
7. `SIN_TIME`, `COS_TIME`: 24-hour circular trigonometric periodicity.
8. `DAY_OF_WEEK`: Calendar day index ($0 - 6$).
9. `IS_DAYTIME`: Binary operational daylight indicator.

### 9.4 Data Leakage Prevention Safeguards
- `AC_POWER`: Removed from feature matrix $X$ (direct target leakage prevention).
- `DC_POWER`: Removed from $X$ (prevents trivial inverter conversion identity shortcut $P_{AC} \approx \eta \cdot P_{DC}$, $r \approx 0.9999$).
- `DAILY_YIELD` & `TOTAL_YIELD`: Removed from $X$ (prevents cumulative time and non-stationary historical leakage).
- `StandardScaler`: Fitted **strictly on the 70% training set**; validation and test sets are transformed without updating scaling statistics.

### 9.5 Chronological Time-Series Splitting
To prevent future-to-past temporal leakage in continuous time series, records are sorted chronologically by `DATETIME_PARSED`:
- **Training Set (Earliest 70%):** 2020-05-15 00:00:00 to 2020-06-08 04:45:00 (48,097 rows).
- **Validation Set (Next 15%):** 2020-06-08 04:45:00 to 2020-06-13 02:00:00 (10,306 rows).
- **Test Set (Final 15% Holdout):** 2020-06-13 02:00:00 to 2020-06-17 23:45:00 (10,308 rows).

### 9.6 ANN Architecture
A configurable feed-forward multi-layer perceptron built with Keras Sequential API:
```text
Input (12 Features)
   │
   ▼
Dense(128, activation='relu')
   │
   ▼
Dropout(rate=0.20)
   │
   ▼
Dense(64, activation='relu')
   │
   ▼
Dense(32, activation='relu')
   │
   ▼
Dense(1, activation='linear') ──► Predicted AC Power (kW)
```

### 9.7 Evaluation Methodology & Metrics
Evaluation will be performed strictly on the unseen 15% chronological test holdout set:
- **Metrics:** Mean Absolute Error (MAE, kW), Root Mean Squared Error (RMSE, kW), Coefficient of Determination ($R^2$).
- **Status:** **PENDING TRAINING** *(To be executed manually in VS Code)*.
- **Diagnostic Plots:** Actual vs. Predicted, Residual Error Distribution, Training/Validation Loss Convergence.

### 9.8 Model Limitations
1. Trained on 34 contiguous summer days (May–June 2020 in India); winter and monsoon patterns require future calibration.
2. Inverter-level telemetry reflects nominal operations and requires hybrid fault-state conditioning to account for physical module defects.


