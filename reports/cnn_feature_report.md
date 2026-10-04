# CNN Feature Extraction Report: PVision AI

## 1. Objective

The primary objective of the CNN feature extraction pipeline is to extract intermediate latent representations (deep visual embeddings) and class posterior probabilities from the trained **EfficientNetV2-S** model. 

In the PVision AI framework, the CNN serves dual purposes:
1. **Direct Categorical Fault Diagnosis:** Generating high-confidence discrete classifications and multi-class probability vectors across 7 operational fault states.
2. **Latent Spatial & Electrical Representation Extraction:** Extracting rich, dense continuous feature vectors from the penultimate pooling layer. These latent features encapsulate spatial visual patterns (in GASF images) and non-linear electrical curvature dynamics (in I-V curves) that cannot be described by raw scalar statistics.

This feature representation can subsequently be evaluated as potential conditioning inputs for the downstream Artificial Neural Network (ANN) power prediction stage, subject to strict scientific validation.

---

## 2. CNN Model

- **Architecture:** EfficientNetV2-S
- **Pretrained Weights:** ImageNet-1k
- **Input Dimensions:** $224 \times 224 \times 3$ RGB
- **Backbone Characteristics:** Progressive learning and Fused-MBConv/MBConv building blocks providing high representational capacity with $21.5\text{M}$ parameters.
- **Classification Head:** 
  - `GlobalAveragePooling2D` (`global_avg_pool`)
  - `BatchNormalization` (`head_bn`)
  - `Dropout(rate=0.3)` (`head_dropout`)
  - `Dense(7, activation='softmax')` (`predictions`)

---

## 3. PV Fault Classes

The feature extraction pipeline operates across all seven distinct photovoltaic fault categories:

| Index | Fault Category | Physical Mechanism |
|---|---|---|
| 0 | `crack` | Silicon crystalline wafer micro-fractures, busbar discontinuity, and mechanical stress |
| 1 | `global_aging` | Uniform EVA encapsulant browning, front glass solarization, and series resistance escalation |
| 2 | `hotspot` | Severe localized cell overheating due to reverse bias breakdown under localized impedance |
| 3 | `normal` | Pristine, un-degraded PV module operation operating at nominal fill factor |
| 4 | `partial_aging` | Sub-module or localized string wear with non-uniform power degradation |
| 5 | `shading` | Non-uniform optical blockage from dust, bird droppings, foliage, or adjacent structural shadows |
| 6 | `short_circuit` | Complete cell junction bypass or diode failure causing localized voltage collapse |

---

## 4. Feature Extraction Method

The feature extraction pipeline preserves the pre-trained and fine-tuned weights without re-training or altering network topology.

```
       [ Input Image ]
       (GASF: 256×256 | I-V: 1343×808)
              │
              ▼
  [ Modality Preprocessing ]
  (GASF: Direct Resize | I-V: Aspect-Preserving Pad)
              │ (224×224×3 Tensor)
              ▼
   [ EfficientNetV2-S Backbone ]
  (Fused-MBConv & MBConv Layers)
              │
              ▼
   [ Feature Extraction Layer ]
      ('global_avg_pool')
              │
              ├──► [ Continuous Feature Vector (D-dim) ] ──► Stored as .npy / .csv
              │
              ▼
    [ BatchNormalization & Dropout ]
              │
              ▼
    [ Dense Softmax Head (7 units) ]
              │
              └──► [ Categorical Prediction & Probabilities ] ──► Stored in Metadata CSV
```

A dual-output inference sub-network is dynamically instantiated:
```python
feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=[model.output, model.get_layer("global_avg_pool").output]
)
```
This architecture computes both the calibrated 7-class prediction distribution and the latent feature embedding in a single forward pass, ensuring zero redundant computational overhead.

---

## 5. Feature Dimension

- **Feature Layer:** `global_avg_pool` (GlobalAveragePooling2D)
- **Feature Dimension:** `[To be determined from trained model]` *(Nominally 1,280 dimensions for standard EfficientNetV2-S)*
- **Data Type:** `float32`

---

## 6. Prediction Information

For every processed image sample, the feature extraction pipeline outputs an immutable metadata record containing:
- `image_id`: Unique file name / identifier (e.g., `hotspot_00001_GASF.png`)
- `true_class`: Ground-truth label derived from verified dataset partition (or `"unlabeled"` if unannotated)
- `predicted_class`: Fault category corresponding to $\operatorname{argmax}(P(y \mid X))$
- `confidence`: Maximum softmax confidence score $\max_c P(y=c \mid X)$
- `prob_crack`: Calibrated posterior probability for `crack`
- `prob_global_aging`: Calibrated posterior probability for `global_aging`
- `prob_hotspot`: Calibrated posterior probability for `hotspot`
- `prob_normal`: Calibrated posterior probability for `normal`
- `prob_partial_aging`: Calibrated posterior probability for `partial_aging`
- `prob_shading`: Calibrated posterior probability for `shading`
- `prob_short_circuit`: Calibrated posterior probability for `short_circuit`
- `feature_vector_reference`: Indexed pointer to the exact row in the corresponding `.npy` feature matrix (e.g., `idx_000042`)

---

## 7. Feature Storage

Extracted features are saved in `results/cnn/features/` with separated modalities to prevent cross-contamination:

1. **NumPy Feature Tensor (`.npy`):**
   - File: `results/cnn/features/{modality}_features.npy`
   - Structure: 2D dense matrix of shape $(N, D)$, where $N$ is the sample count and $D$ is the feature dimension.
   - Purpose: High-speed, zero-copy loading for Scikit-learn, PyTorch, or TensorFlow downstream models.

2. **Prediction & Reference Metadata (`.csv`):**
   - File: `results/cnn/features/{modality}_features.csv`
   - Columns: `image_id`, `true_class`, `predicted_class`, `confidence`, `prob_crack`, `prob_global_aging`, `prob_hotspot`, `prob_normal`, `prob_partial_aging`, `prob_shading`, `prob_short_circuit`, `feature_vector_reference`.
   - Purpose: Transparent tabular joining and error analysis.

3. **Extraction Metadata Record (`.json`):**
   - File: `results/cnn/features/{modality}_feature_metadata.json`
   - Schema: Documents model name, modality, input image dimensions, feature layer name, exact feature dimension, class mappings, preprocessing methodology, source model path, and timestamp.

---

## 8. Feature Visualization

The pipeline includes an automated 2D Principal Component Analysis (PCA) projection module (`visualize_pca`):
- **Axes:**
  - X-axis: Principal Component 1 (annotated with percentage explained variance)
  - Y-axis: Principal Component 2 (annotated with percentage explained variance)
- **Target Export:**
  - `results/cnn/features/gasf_feature_pca.png`
  - `results/cnn/features/iv_feature_pca.png`
- **Scientific Caveat:** PCA projections serve strictly as an unsupervised visual inspection of latent class clustering in 2D space. A clean 2D projection does not prove model generalization, nor does it guarantee that features will boost ANN predictive accuracy.

*Note: Visualizations will be generated when the trained model is placed in `models/cnn/` and extraction is executed.*

---

## 9. Potential CNN-ANN Usage

The extracted CNN representations offer three potential pathways for the downstream hybrid system:

1. **Discrete Fault Conditioning (Baseline):**
   - The predicted fault class (one-hot encoded, 7-D) is passed to the ANN alongside solar irradiance, ambient temperature, and module temperature.
   - Hypothesis: The ANN adjusts its power generation curve dynamically based on the known degradation mode (e.g., lower expected power during `shading` or `short_circuit`).

2. **Probability-Weighted Conditioning:**
   - The full 7-dimensional softmax probability vector is provided to the ANN.
   - Benefit: Retains model uncertainty; borderline cases (e.g., 55% `shading`, 40% `normal`) allow continuous power modulation rather than hard-threshold switching.

3. **Latent Feature Fusion (Dense Embedding):**
   - Dimensionality-reduced CNN features (e.g., top 16 or 32 PCA/Autoencoder components from the 1,280-D vector) are concatenated with the environmental feature vector.
   - Requirement: Requires careful regularization to avoid overfitting the ANN on high-dimensional image embeddings.

---

## 10. Scientific Limitation & Data Integration Constraints

> [!IMPORTANT]
> **Core Constraint Regarding Dataset Pairing:**
> The PV fault image dataset (GASF and I-V curves) and the solar power generation time-series dataset originate from **entirely different physical installations and experimental setups**.
> 
> - **Fault Image Dataset:** Sourced from laboratory / controlled testbed simulations encompassing 69,484 synthetic/measured characteristic curves and Gramian fields.
> - **Solar Generation Dataset:** Sourced from operational utility-scale PV inverters recording ambient weather and power production over continuous chronological timestamps.
> 
> **Scientific Rules:**
> 1. Under no circumstances will artificial, arbitrary one-to-one row pairings be synthesized (e.g., mapping `image_0001` to `generation_row_0001`).
> 2. CNN fault states and power time-series can only be linked through **statistically grounded scenario-based modeling** or **synthetic fault injection experiments** where environmental conditions match the physical operational envelope of the fault.
> 3. CNN features will NOT be blindly appended to the ANN without explicit experimental justification.

---

## 11. Results

**Status:** `[Results pending CNN training]`

Empirical feature dimensions, classification accuracy, confidence distributions, and PCA scatter plots will be populated immediately following the manual execution of CNN training on Kaggle GPU / VS Code.
