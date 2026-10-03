# PVision AI: CNN Photovoltaic Fault Classification Report

**Model:** EfficientNetV2-S (Transfer Learning)  
**Modalities:** Gramian Angular Summation Fields (GASF) & I-V Characteristic Curves  
**Status:** Architecture Configured & Evaluation Pipelines Ready • Awaiting Manual Training in VS Code  

---

## 1. CNN Objective
The objective of the Vision Diagnostics branch in **PVision AI** is to automate the identification and classification of physical, optical, and electrical anomalies on photovoltaic (PV) solar modules. By categorizing anomalies into seven distinct operating states, the system enables predictive maintenance, minimizes downtime, and prevents irreversible damage such as hot-spot-induced thermal fires and micro-crack delamination.

---

## 2. Dataset Description
The vision dataset encompasses **69,484 total images** across two distinct physical representations:
- **GASF Modality:** Gramian Angular Summation Fields transforming high-frequency temporal current-voltage dynamics into 2D quasi-images ($35,000$ samples).
- **I-V Modality:** Direct graphical plots of current-voltage operating curves ($34,484$ samples).
- **Dataset Health:** 0 corrupted images, 0 duplicate files, 100% sample retention across all classes.

---

## 3. GASF Dataset
- **Total Images:** 35,000
- **Native Resolution:** $256 \times 256$ pixels (3-channel RGB PNG)
- **Class Balance:** Perfectly uniform ($5,000$ samples per class across all 7 categories)
- **Representation Type:** 2D Gramian matrices encoding angular cosine sums $\cos(\phi_i + \phi_j)$ of normalized time-series series vectors, preserving temporal correlation and phase information.

---

## 4. I-V Dataset
- **Total Images:** 34,484
- **Native Resolution:** $1343 \times 808$ pixels (3-channel RGB PNG)
- **Class Balance:** 6 classes contain exactly $5,000$ images each. The `short_circuit` class contains **4,484 images** due to a sequence gap of **516 missing files** (indices `1–513` and `597–599`).
- **Imbalance Ratio:** $0.8968$ (addressed via optional class-weight weighting in `train_cnn.py`).
- **Representation Type:** Coordinate axis plots showing the continuous I-V characteristic curve from short-circuit current ($I_{sc}$) to open-circuit voltage ($V_{oc}$).

---

## 5. Fault Classes
The system classifies images into seven mutually exclusive categories:
1. `crack`: Micro-cracks in crystalline silicon cells causing localized electrical disconnects.
2. `global_aging`: Uniform degradation of anti-reflective coating, EVA encapsulant browning, and cell wear across the entire module.
3. `hotspot`: Localized high-temperature heating caused by shaded or defective cells acting as resistive loads.
4. `normal`: Healthy operational condition with standard fill factor and nominal curve curvature.
5. `partial_aging`: Non-uniform cell degradation localized to specific sub-strings.
6. `shading`: Static or dynamic obstacles (soiling, bird droppings, foliage, building shadows) reducing irradiance on partial sub-strings.
7. `short_circuit`: Severe electrical fault where current bypasses the load due to cell junction breakdown or bypass diode failure.

---

## 6. Data Preprocessing
- **GASF Modality:**
  - $256 \times 256 \text{ RGB} \to \text{Bilinear Resize to } 224 \times 224 \to \text{EfficientNetV2 Preprocessing}$.
- **I-V Modality:**
  - $1343 \times 808 \text{ RGB} \to \text{Aspect-Ratio-Preserving Resize & Pad to } 224 \times 224 \to \text{EfficientNetV2 Preprocessing}$.
  - *No Direct Stretching:* Direct non-uniform scaling from $1343 \times 808$ to $224 \times 224$ would distort curve slope and knee curvature. Aspect ratio is preserved with centered padding.

---

## 7. Train/Validation/Test Split
To guarantee unbiased and zero-leakage evaluation, stratified partitions were computed with fixed random seed (`seed=42`):
- **Training Set (70%):** $24,500$ (GASF) / $24,138$ (I-V)
- **Validation Set (15%):** $5,250$ (GASF) / $5,173$ (I-V)
- **Testing Set (15%):** $5,250$ (GASF) / $5,173$ (I-V)
- **Leakage Prevention:** Zero hash intersection verified across all splits. Manifests saved in `data/processed/`.

---

## 8. Data Augmentation
Augmentation is strictly applied to the **training partition only**. Validation and test images are evaluated in their unaltered physical state.
- Random Horizontal Flip ($p = 0.5$)
- Random Vertical Flip ($p = 0.5$)
- Random Subtle Brightness Adjustment ($\pm 8\%$)
- Random Subtle Contrast Adjustment ($0.92 - 1.08$)
- *Note:* Excessive elastic transformations or heavy affine distortions are avoided to preserve electrical physical geometry.

---

## 9. EfficientNetV2-S Architecture
EfficientNetV2-S utilizes Fused-MBConv blocks (combining $3 \times 3$ standard convolutions with expansion) in early stages and standard MBConv blocks with Squeeze-and-Excitation in later stages. This hybrid design yields faster convergence, reduced parameter count ($\approx 21.5\text{M}$ parameters), and higher accuracy compared to ResNet-50 and EfficientNetV1.

---

## 10. Transfer Learning
- Pretrained on ImageNet-1k dataset ($1.28\text{M}$ natural images across $1,000$ classes).
- Initial weights transfer low-level edge, gradient, texture, and corner detectors.
- Standard top classification head is replaced with:
  - `GlobalAveragePooling2D`
  - `BatchNormalization`
  - `Dropout(0.3)`
  - `Dense(7, activation='softmax')`

---

## 11. Fine-Tuning Strategy
Two-stage training methodology:
- **Stage 1 (Feature Extraction):**
  - Backbone is completely frozen (`base_model.trainable = False`).
  - Learning Rate: $\eta = 1 \times 10^{-3}$ (Adam optimizer).
  - Trains only the custom classification head for $10$ epochs.
- **Stage 2 (Fine-Tuning):**
  - Unfreezes the top $50$ layers of EfficientNetV2-S.
  - Learning Rate: $\eta = 1 \times 10^{-4}$ ($10\times$ smaller to prevent catastrophic forgetting).
  - Trains for $20$ epochs with `ReduceLROnPlateau` and `EarlyStopping`.

---

## 12. Training Configuration
- **Batch Size:** $32$
- **Loss Function:** Categorical Cross-Entropy with Label Smoothing ($0.05$)
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$)
- **Input Size:** $224 \times 224 \times 3$
- **Callbacks:**
  - `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
  - `ModelCheckpoint(monitor='val_loss', save_best_only=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)`
- **Class Weights:** Configurable via `--use-class-weights` flag.

---

## 13. Evaluation Metrics
Models will be assessed on the unseen test partition ($15\%$) using:
- **Overall Accuracy:** $\frac{\text{Correct Predictions}}{\text{Total Test Samples}}$
- **Macro-Averaged Precision:** Unweighted mean of class precisions.
- **Macro-Averaged Recall:** Unweighted mean of class recalls.
- **Macro F1-Score:** Harmonic mean of macro precision and recall.
- **Confusion Matrix:** Full $7 \times 7$ grid of true versus predicted counts.

---

## 14. GASF Results
*Note: Numerical metrics will be populated after manual execution of `train_cnn.py` and `evaluate_cnn.py` in VS Code.*

- **Test Accuracy:** [To be updated after training]
- **Macro Precision:** [To be updated after training]
- **Macro Recall:** [To be updated after training]
- **Macro F1-Score:** [To be updated after training]
- **Validation Loss:** [To be updated after training]

---

## 15. I-V Results
*Note: Numerical metrics will be populated after manual execution of `train_cnn.py` and `evaluate_cnn.py` in VS Code.*

- **Test Accuracy:** [To be updated after training]
- **Macro Precision:** [To be updated after training]
- **Macro Recall:** [To be updated after training]
- **Macro F1-Score:** [To be updated after training]
- **Validation Loss:** [To be updated after training]

---

## 16. Confusion Matrix Analysis
- **GASF Confusion Matrix:** [To be updated after training]
- **I-V Confusion Matrix:** [To be updated after training]
- *Anticipated Confusion Analysis:* Evaluating potential classification overlap between `shading` and `hotspot`, or between `partial_aging` and `global_aging`.

---

## 17. Per-Class Performance
| Class Name | GASF Precision | GASF Recall | GASF F1 | I-V Precision | I-V Recall | I-V F1 | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `crack` | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | 750 / 750 |
| `global_aging` | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | 750 / 750 |
| `hotspot` | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | 750 / 750 |
| `normal` | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | 750 / 750 |
| `partial_aging`| [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | 750 / 750 |
| `shading` | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | 750 / 750 |
| `short_circuit`| [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | 750 / 673 |

---

## 18. Model Comparison (Preparation)
The pipeline is designed to benchmark the primary EfficientNetV2-S against:
1. **ConvNeXt-Tiny:** Pure convolutional modernization with $7 \times 7$ depthwise kernels and inverted bottlenecks.
2. **MobileNetV3-Large:** Lightweight edge-optimized architecture with hardware-aware NAS.

| Architecture | Parameters | FLOPs | Top-1 Accuracy (GASF) | Top-1 Accuracy (I-V) | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNetV2-S (Primary)** | $21.5\text{M}$ | $8.4\text{B}$ | [To be updated after training] | [To be updated after training] | [To be updated after training] |
| **ConvNeXt-Tiny (Planned)** | $28.6\text{M}$ | $9.0\text{B}$ | [Planned for Phase 4] | [Planned for Phase 4] | [Planned for Phase 4] |
| **MobileNetV3-Large (Planned)** | $5.4\text{M}$ | $0.4\text{B}$ | [Planned for Phase 4] | [Planned for Phase 4] | [Planned for Phase 4] |

---

## 19. Limitations
1. **Sequence Gap in I-V Modality:** Class `short_circuit` has 516 fewer samples ($4,484$ vs $5,000$). Requires class weighting or focal loss if recall is impaired.
2. **Computational Demands:** Fine-tuning EfficientNetV2-S across $49,000$ training images requires a modern GPU (e.g. NVIDIA RTX 3060/4060 or Google Colab T4/V100).
3. **Decoupled Modality Training:** Models are trained separately for GASF and I-V; cross-modality attention fusion is reserved for future research.

---

## 20. Conclusion
The Phase 3 CNN architecture, data loading with aspect-ratio-preserving padding, two-stage transfer learning pipeline, and evaluation metrics have been fully constructed.
To perform training:
```bash
python src/cnn/train_cnn.py --modality gasf
python src/cnn/train_cnn.py --modality iv
```
Once weights are generated in `models/cnn/`, run:
```bash
python src/cnn/evaluate_cnn.py --modality gasf
python src/cnn/evaluate_cnn.py --modality iv
```
Numerical placeholders in this report will be populated immediately upon receiving the training metrics.
