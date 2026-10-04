# PVision AI: Presentation Deck Blueprint
## A Hybrid CNN-ANN Framework for Photovoltaic Fault Detection and Power Generation Prediction

**Prepared For:** Engineering Review / Academic Defense / Technical Stakeholder Presentation  
**Project Milestone:** Phase 3 Complete (CNN Architecture & Evaluation Pipelines Ready)  

---

### Slide 1: Title Slide
- **Title:** PVision AI: A Hybrid CNN-ANN Framework
- **Subtitle:** Integrated Photovoltaic Fault Detection and Real-Time Power Generation Prediction
- **Presenter:** Technical Engineering Team
- **Core Theme:** Uniting Computer Vision and Tabular Deep Learning for Next-Generation Solar Farm Diagnostics
- **Key Badge:** Two-Stage Transfer Learning with EfficientNetV2-S (Phase 3 Milestone)

---

### Slide 2: Problem Statement & Industrial Motivation
- **The Challenge in Modern Solar Utilities:**
  - *Undetected Faults:* Micro-cracks, hotspots, shading, and aging account for up to 25% annual energy loss and severe fire hazards.
  - *Intermittent Power Generation:* Weather variability (clouds, ambient heat) creates unpredictable active power output ($AC\_POWER$).
  - *Decoupled Systems:* Traditional solar farms monitor weather and inspect panels separately; there is no real-time causal link between a hardware fault and efficiency derating.
- **The Need:** An intelligent system that diagnoses module anomalies from imaging while predicting nominal generation from weather sensors, attributing loss accurately.

---

### Slide 3: The Hybrid CNN-ANN Solution Blueprint
- **Two Modalities • Two Neural Networks • One Decision Engine:**
  - **Vision Branch (CNN):** 7-class fault classifier trained on module imagery (GASF 2D time-series matrices and I-V characteristic curves).
  - **Forecasting Branch (ANN):** Multi-Layer Perceptron trained on irradiance, thermal dynamics, and temporal cyclical features to predict baseline AC power output.
  - **Hybrid Coupling Engine:** Derates expected power when faults are detected and flags sudden inverter curtailment versus weather attenuation.
- **Architectural Principle:** Zero leakage, modular decoupled pipelines, fully reproducible training manifests.

---

### Slide 4: Dual-Modality Vision Dataset (69,484 Images)
- **Modality 1: Gramian Angular Summation Fields (GASF):**
  - Converts sequential 1D I-V electrical readings into $256 \times 256$ RGB spatial correlation matrices.
  - Retains trigonometric angular temporal dependencies.
  - **35,000 images:** Perfectly balanced across 7 classes ($5,000$ per class).
- **Modality 2: Current-Voltage (I-V) Curves:**
  - High-resolution ($1343 \times 808$ RGB) plots of electrical operating curves.
  - Captures knee voltage, short-circuit current drops, and fill factor degradation.
  - **34,484 images:** 6 classes at $5,000$; `short_circuit` at $4,484$.

---

### Slide 5: Photovoltaic Fault Categories (7 Classes)
- **The 7 Physical Operating States:**
  1. `crack`: Crystalline wafer fractures causing disconnected cell segments.
  2. `global_aging`: Uniform degradation of front glass, EVA yellowing, and series resistance.
  3. `hotspot`: Localized cell overheating under reverse bias (severe fire hazard).
  4. `normal`: Healthy operational condition with standard fill factor.
  5. `partial_aging`: String-level or sub-module localized wear.
  6. `shading`: Non-uniform photon blockage from dust, soiling, or nearby obstacles.
  7. `short_circuit`: Cell junction short-circuit or bypass diode failure.
- **Classification Goal:** High sensitivity to critical electrical failure modes (`hotspot`, `short_circuit`, `crack`).

---

### Slide 6: Vision Preprocessing & Aspect-Ratio Preservation
- **GASF Preprocessing Pipeline:**
  - $256 \times 256 \text{ RGB} \to \text{Bilinear Resize to } 224 \times 224 \to \text{EfficientNetV2-S Preprocessing}$.
- **I-V Preprocessing Pipeline (Aspect-Ratio-Preserving Padding):**
  - Native Resolution: $1343 \times 808$ RGB.
  - **Zero Geometric Distortion Rule:** Direct stretching from $1343 \times 808$ to $224 \times 224$ would artificially compress curve curvature and alter knee-voltage slopes.
  - Aspect-ratio-preserving resize with centered padding (to $224 \times 224$) strictly maintains true electrical physics.
- **Pixel Normalization:** Standard ImageNet channel standardization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

### Slide 7: EfficientNetV2-S Architecture & Transfer Learning
- **Why EfficientNetV2-S as the Primary Model?**
  - Progressive training and Fused-MBConv layers in early stages.
  - Fast convergence with $21.5\text{M}$ parameters (significantly lighter than ResNet-50 while outperforming it).
  - Pretrained on ImageNet-1k ($1.28\text{M}$ images) for robust low-level edge and texture feature representations.
- **Custom Classification Head:**
  - `GlobalAveragePooling2D` (spatial dimension collapse).
  - `BatchNormalization` (activation stabilization).
  - `Dropout(0.3)` (regularization against overfitting).
  - `Dense(7, activation='softmax')` (7-class probabilistic output).

---

### Slide 8: Two-Stage Fine-Tuning Methodology
- **Stage 1: Feature Extraction (Head Training):**
  - Backbone frozen (`base_model.trainable = False`).
  - Learning rate: $\eta = 1 \times 10^{-3}$ (Adam optimizer).
  - Rapidly trains the 7-class classification head for 10 epochs.
- **Stage 2: Fine-Tuning (Upper Layers Unfrozen):**
  - Unfreezes the top 50 layers of the EfficientNetV2-S backbone.
  - Learning rate: $\eta = 1 \times 10^{-4}$ ($10\times$ smaller learning rate to preserve core ImageNet features).
  - Trains for 20 epochs with `ReduceLROnPlateau` and `EarlyStopping`.
- **Checkpointing:** Model weights saved based on minimum validation loss (`val_loss`).

---

### Slide 9: Data Splitting, Leakage Prevention & Class Weights
- **Partitioning Protocol (70% Train / 15% Val / 15% Test):**
  - Stratified by class labels to maintain equal representation across splits.
  - **GASF Modality:** $24,500$ Train / $5,250$ Val / $5,250$ Test.
  - **I-V Modality:** $24,138$ Train / $5,173$ Val / $5,173$ Test.
  - **Leakage Safeguards:** Set intersection of cryptographic MD5 hashes confirms zero data leakage across train, val, and test partitions.
- **Class-Weight Mitigation:**
  - I-V modality features 4,484 `short_circuit` images (imbalance ratio $0.8968$).
  - Inverse frequency class weights integrated via `--use-class-weights`.

---

### Slide 10: Training Callbacks & Automated Checkpointing
- **Dynamic Training Callbacks:**
  1. `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`: Prevents overfitting by terminating when validation loss stagnates.
  2. `ModelCheckpoint(monitor='val_loss', save_best_only=True)`: Automatically persists the optimal model weights.
  3. `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)`: Dynamically cuts learning rate when approaching local minima.
- **Saved Model Checkpoints:**
  - `models/cnn/efficientnetv2s_gasf_best.keras`
  - `models/cnn/efficientnetv2s_iv_best.keras`
  - `models/cnn/{modality}_training_history.json`

---

### Slide 11: CNN Evaluation Metrics & Diagnostic Artifacts
- **Comprehensive Evaluation Suite (`evaluate_cnn.py`):**
  - Global Accuracy, Macro-Averaged Precision, Recall, and F1-Score.
  - Per-class Precision, Recall, and F1-Score for each of the 7 fault classes.
  - Full $7 \times 7$ Confusion Matrix heatmap containing all categories.
  - Loss and Accuracy training/validation convergence curves.
- **Diagnostic Export Locations:**
  - `results/cnn/gasf/` (GASF metrics, confusion matrix, and learning curves).
  - `results/cnn/iv/` (I-V metrics, confusion matrix, and learning curves).
  - *No fabricated results:* Values populated upon user's manual training run in VS Code.

---

### Slide 12: Single-Image Inference Pipeline (`predict_cnn.py`)
- **Real-Time Fault Prediction:**
  - Accepts a single input image (.png) from either GASF or I-V modality.
  - Automatically executes modality-appropriate preprocessing.
  - Outputs top predicted class, confidence percentage, and full 7-class probability distribution:
    ```text
    Predicted Class: hotspot
    Confidence: XX.X%

    Class Probabilities:
    crack: XX.X%
    global_aging: XX.X%
    hotspot: XX.X%
    normal: XX.X%
    partial_aging: XX.X%
    shading: XX.X%
    short_circuit: XX.X%
    ```

---

### Slide 13: Model Comparison Preparation (Future Benchmarks)
- **Model Registry Prepared for Phase 4 Benchmarking:**
  1. **EfficientNetV2-S (Primary):** $21.5\text{M}$ parameters (High accuracy / balanced speed).
  2. **ConvNeXt-Tiny (Planned):** $28.6\text{M}$ parameters (Modern pure ConvNet architecture).
  3. **MobileNetV3-Large (Planned):** $5.4\text{M}$ parameters (Ultra-lightweight edge deployment).
- **Architecture Factory:** Ready in `src/cnn/model_factory.py` for comparative analysis.

---

### Slide 14: Phase 3 Milestones & Manual Execution Guide
- **Phase 3 Deliverables Completed:**
  - Modular training script (`src/cnn/train_cnn.py`) with GPU/CPU support and two-stage transfer learning.
  - Evaluation pipeline (`src/cnn/evaluate_cnn.py`) with full diagnostic plots.
  - Single-image inference script (`src/cnn/predict_cnn.py`).
  - Comprehensive documentation (`reports/cnn_report.md`).
- **VS Code Manual Execution Commands:**
  ```bash
  # Train GASF
  python src/cnn/train_cnn.py --modality gasf --epochs-stage1 10 --epochs-stage2 20

  # Train I-V
  python src/cnn/train_cnn.py --modality iv --epochs-stage1 10 --epochs-stage2 20 --use-class-weights

  # Evaluate
  python src/cnn/evaluate_cnn.py --modality gasf
  python src/cnn/evaluate_cnn.py --modality iv
  ```
- **Next Phase:** User executes training in VS Code; reports will be updated with actual empirical scores.

---

### Slide 15: CNN Feature Extraction Architecture (`extract_features.py`)
- **Latent Representation Extraction Flow:**
  ```text
  PV Image (GASF: 256×256 | I-V: 1343×808)
          │
          ▼
  Aspect-Preserving Preprocessing (224×224×3)
          │
          ▼
  EfficientNetV2-S Pretrained Backbone
          │
          ▼
  Global Average Pooling ('global_avg_pool') ──► Latent Feature Vector (1,280-D)
          │
          ▼
  Batch Normalization & Dropout
          │
          ▼
  Dense Head (Softmax 7-Class) ────────────────► Categorical Prediction & 7 Probabilities
  ```
- **Extracted Prediction & Embedding Data:**
  - `predicted_class`: Top predicted fault state among 7 classes.
  - `confidence`: Softmax probability corresponding to the top class.
  - `class_probabilities`: Full posterior distribution across all 7 fault categories.
  - `feature_vector`: High-level continuous embedding vector capturing non-linear curve slopes and spatial texture signatures.
- **Dual-Format Persistence:**
  - NumPy dense matrix: `results/cnn/features/{modality}_features.npy`.
  - Tabular metadata: `results/cnn/features/{modality}_features.csv`.
  - Machine-readable audit: `results/cnn/features/{modality}_feature_metadata.json`.

---

### Slide 16: Hybrid CNN-ANN Concept & Scientific Integrity Boundaries
- **Potential Hybrid Integration Pathways:**
  1. **Discrete Fault Conditioning:** Providing predicted fault labels as categorical indicators to the ANN to shift power baseline.
  2. **Soft Probability Modulation:** Inputting 7-class probability vectors to express diagnostic uncertainty.
  3. **Latent Embedding Conditioning:** Concatenating regularized low-dimensional projections with environmental telemetry.
- **Core Scientific Boundary:**
  - PV images and solar time-series originate from separate physical experimental domains.
  - Zero artificial 1-to-1 pairing (no synthetic `image_001` $\to$ `row_001` joins).
  - Hybrid integration relies on physical scenario analysis rather than arbitrary row-wise merging.

---

### Slide 17: ANN Power Prediction Pipeline (`train_ann.py`)
- **Objective:** Continuous AC power forecasting from structured environmental telemetry.
- **Confirmed Target Variable:** `AC_POWER` (Inverter Alternating Current Power Output in kW).
- **Physical Predictive Features (12 Variables):**
  - Environmental: `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`.
  - Thermodynamic Interactions: `TEMP_DIFFERENCE` ($T_{mod} - T_{amb}$), `IRRAD_MODULE_INTERACTION` ($I \cdot T_{mod}$).
  - Diurnal & Temporal: `HOUR`, `MINUTE`, `TIME_DECIMAL`, `SIN_TIME`, `COS_TIME`, `DAY_OF_WEEK`, `IS_DAYTIME`.
- **Chronological Time-Series Splitting:**
  - 70% Train (earliest: May 15 to June 08) $\to$ $48,097$ samples.
  - 15% Val (intermediate: June 08 to June 13) $\to$ $10,306$ samples.
  - 15% Test (latest: June 13 to June 17) $\to$ $10,308$ samples.
  - Zero random shuffling (prevents temporal future-to-past data leakage).

---

### Slide 18: ANN Regression Architecture & Safeguards
- **Multi-Layer Perceptron (MLP) Topology:**
  ```text
  Input Features (12 units)
          │
          ▼
  Dense Layer 1 (128 units, ReLU)
          │
          ▼
  Dropout Regularization (rate = 0.20)
          │
          ▼
  Dense Layer 2 (64 units, ReLU)
          │
          ▼
  Dense Layer 3 (32 units, ReLU)
          │
          ▼
  Output Layer (1 unit, Linear) ──► Predicted AC Power (kW)
  ```
- **Strict Leakage Prevention Protocol:**
  - `DC_POWER` excluded (avoids $r \approx 0.9999$ identity shortcut).
  - `DAILY_YIELD` & `TOTAL_YIELD` excluded (avoids non-stationary cumulative leakage).
  - `StandardScaler` fitted **strictly on training partition**; test data scaled out-of-sample.

---

### Slide 19: ANN Evaluation Protocol (`evaluate_ann.py`)
- **Chronological Holdout Evaluation:**
  - Evaluated exclusively on the unseen final 15% test set (June 13 to June 17, 2020).
- **Standardized Performance Metrics:**
  - Mean Absolute Error (MAE, kW).
  - Root Mean Squared Error (RMSE, kW).
  - Coefficient of Determination ($R^2$, goodness of fit).
  - *Evaluation Status:* **Results to be populated after VS Code training.**
- **Diagnostic Export Plots (`reports/figures/`):**
  - `ann_actual_vs_predicted.png`: Parity plot with $y=x$ ideal reference.
  - `ann_residual_plot.png`: Error distribution across the operational power range.
  - `ann_training_validation_loss.png`: MSE convergence curve across training epochs.


