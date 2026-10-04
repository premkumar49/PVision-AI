# ANN Power Prediction Report: PVision AI

## 1. Objective

Photovoltaic (PV) power generation exhibits high stochasticity driven by solar irradiance, ambient and module temperatures, cloud transients, and inverter operational dynamics. The primary objective of the Artificial Neural Network (ANN) regression pipeline is to learn non-linear physical energy transfer functions to accurately predict instantaneous AC electrical power output ($P_{AC}$, in kW) from meteorological telemetry and temporal features.

In the unified PVision AI framework:
- The **CNN** handles visual and graphical fault diagnosis from images (GASF and I-V curves).
- The **ANN** models quantitative continuous electrical generation from tabular environmental telemetry.

Accurate power prediction serves as the baseline against which fault-induced power degradation is quantified in utility-scale solar asset monitoring.

---

## 2. Dataset

The regression pipeline is grounded in the empirical solar telemetry dataset from Plant 1 and Plant 2:

### 2.1 File Inventory & Telemetry Specifications
- **Generation File:** `data/solar_generation/Plant_1_Generation_Data.csv`
  - Records: 68,778 rows across 22 grid-tied inverters.
  - Frequency: 15-minute intervals.
  - Columns: `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`.
- **Weather Sensor File:** `data/solar_generation/Plant_1_Weather_Sensor_Data.csv`
  - Records: 3,182 plant-level weather observations.
  - Frequency: 15-minute intervals.
  - Columns: `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`.

### 2.2 Time Span & Synchronization
- **Observed Time Period:** `2020-05-15 00:00:00` to `2020-06-17 23:45:00` (34 contiguous calendar days).
- **Time Parsing & Alignment:** Generation timestamps formatted as `DD-MM-YYYY HH:MM` and weather timestamps formatted as `YYYY-MM-DD HH:MM:SS` were synchronized and parsed into standard ISO datetime indices, producing 68,774 aligned multi-inverter records.

### 2.3 Quality Audit & Cleaning
- **Missing Values:** Zero missing values ($0$ nulls detected across all numeric fields).
- **Duplicate Rows:** Zero duplicate observations ($0$ duplicate rows).
- **Night Noise Correction:** Zeroed out baseline night sensor noise ($P_{AC} > 0$ when $\text{IRRADIATION} = 0$).
- **Daytime Inverter Outage Separation:** Filtered 63 daytime records where $\text{IRRADIATION} > 0.05$ but the inverter was offline ($P_{AC} = 0$), leaving 68,711 operational generation records.

---

## 3. Target Variable

- **Confirmed Target Variable:** `AC_POWER`
- **Target Description:** Inverter Alternating Current (AC) Power Output representing instantaneous usable grid-injected electrical generation.
- **Physical Unit:** Kilowatts (kW).
- **Rationale for Selection:** While `DC_POWER` represents raw array generation before inverter rectification, `AC_POWER` represents the true commercial output of the PV system. Furthermore, `DAILY_YIELD` and `TOTAL_YIELD` represent cumulative energy meters (kWh) rather than instantaneous power capacity (kW).

---

## 4. Feature Engineering

To capture physical and diurnal solar thermodynamics without non-causal data leakage, 12 engineered features were derived:

1. **Direct Environmental Features:**
   - `IRRADIATION`: Solar irradiance recorded at the pyranometer ($W/m^2$ equivalent).
   - `AMBIENT_TEMPERATURE`: Ambient dry-bulb air temperature ($^\circ C$).
   - `MODULE_TEMPERATURE`: Surface temperature of the silicon PV modules ($^\circ C$).
2. **Physical Thermodynamic Interactions:**
   - `TEMP_DIFFERENCE`: Thermal elevation above ambient ($T_{module} - T_{ambient}$), which correlates with efficiency loss under high thermal stress.
   - `IRRAD_MODULE_INTERACTION`: Cross-product term ($\text{IRRADIATION} \times T_{module}$) capturing non-linear photo-thermal coupling.
3. **Temporal & Cyclical Encodings:**
   - `HOUR`: Integer hour ($0 - 23$).
   - `MINUTE`: Observation minute ($0, 15, 30, 45$).
   - `TIME_DECIMAL`: Continuous hour fraction ($\text{HOUR} + \text{MINUTE}/60$).
   - `SIN_TIME`: $\sin(2\pi \cdot \text{TIME\_DECIMAL} / 24)$ smooth diurnal harmonic.
   - `COS_TIME`: $\cos(2\pi \cdot \text{TIME\_DECIMAL} / 24)$ smooth diurnal harmonic.
   - `DAY_OF_WEEK`: Integer day index ($0 - 6$).
   - `IS_DAYTIME`: Binary operational flag ($1$ if $\text{IRRADIATION} > 0$, else $0$).

---

## 5. Leakage Prevention Safeguards

Strict methodological barriers were implemented to prevent distribution and target leakage:

| Variable | Status | Scientific Safeguard Reason |
|---|---|---|
| `AC_POWER` | **Excluded from X** | Target variable (Direct target leakage). |
| `DC_POWER` | **Excluded from X** | Trivial physical inverter conversion identity ($P_{AC} \approx \eta \cdot P_{DC}$, $r \approx 0.9999$). Including it bypasses environmental regression. |
| `DAILY_YIELD` | **Excluded from X** | Cumulative daytime meter that leaks accumulated time and future intraday yield. |
| `TOTAL_YIELD` | **Excluded from X** | Non-stationary lifetime meter that leaks long-term historical sequence order. |
| `PLANT_ID`, `SOURCE_KEY` | **Excluded from X** | Categorical identifiers that prevent generalizable physics modeling. |
| **Chronological Split** | **Enforced** | Sorted strictly by `DATETIME_PARSED`. Random splitting is forbidden. |
| **Scaler Isolation** | **Enforced** | `StandardScaler` is fitted **strictly on the 70% training set**. Validation and test data are only transformed. |

### 5.1 Partition Details (Chronological)
- **Training Set (Earliest 70%):** `2020-05-15 00:00:00` to `2020-06-08 04:45:00` ($48,097$ rows).
- **Validation Set (Next 15%):** `2020-06-08 04:45:00` to `2020-06-13 02:00:00` ($10,306$ rows).
- **Test Set (Final 15% Holdout):** `2020-06-13 02:00:00` to `2020-06-17 23:45:00` ($10,308$ rows).

---

## 6. ANN Architecture

The regression model is implemented as a multi-layer perceptron (MLP) feed-forward network configured via `ANN_CONFIG`:

```text
Input Layer (12 Features)
       │
       ▼
Dense Layer 1: 128 units, ReLU activation
       │
       ▼
Dropout Layer: rate = 0.20
       │
       ▼
Dense Layer 2: 64 units, ReLU activation
       │
       ▼
Dense Layer 3: 32 units, ReLU activation
       │
       ▼
Output Layer: 1 unit, Linear activation (Predicted AC_POWER in kW)
```

### Configurable Hyperparameters:
```python
ANN_CONFIG = {
    "hidden_layers": [128, 64, 32],
    "dropout": 0.20,
    "activation": "relu",
    "output_activation": "linear",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "patience": 10
}
```

---

## 7. Training Protocol

- **Optimizer:** Adam ($\eta = 0.001$).
- **Loss Function:** Mean Squared Error (MSE).
- **Batch Size:** 64.
- **Maximum Epochs:** 100.
- **Early Stopping:** Monitored on `val_loss` with `patience=10` and `restore_best_weights=True`.
- **Model Checkpoint:** Saves optimal weights to `models/ann/best_ann_model.keras`.
- **Reproducibility Seed:** `RANDOM_SEED = 42`.
- **Training Status:** **PENDING TRAINING** *(To be executed manually by the user in VS Code)*.

---

## 8. Evaluation Metrics

Model evaluation will be performed exclusively on the final 15% chronological test holdout partition (`2020-06-13` to `2020-06-17`):

| Metric | Formula | Unit | Status / Value |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$ | kW | **PENDING TRAINING** |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$ | kW | **PENDING TRAINING** |
| **Coefficient of Determination ($R^2$)** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Dimensionless | **PENDING TRAINING** |

*Note: In accordance with project instructions, no metrics have been fabricated.*

---

## 9. Diagnostic Plots

Upon manual execution of `python src/ann/evaluate_ann.py`, the following figures will be generated:
1. `reports/figures/ann_actual_vs_predicted.png`: Scatter plot comparing observed vs. predicted kW with ideal $y=x$ line.
2. `reports/figures/ann_residual_plot.png`: Residual error ($y - \hat{y}$) against predicted kW to diagnose heteroscedasticity.
3. `reports/figures/ann_training_validation_loss.png`: Epoch-by-epoch MSE convergence curves from `training_history.json`.

- **Plot Status:** **PENDING TRAINING**

---

## 10. Limitations & Operational Scope

1. **Temporal Horizon:** The dataset spans 34 contiguous summer days (May–June 2020 in India). Seasonal winter and monsoon dynamics are not represented.
2. **Geographical Scope:** Telemetry represents two utility-scale plants in Gandhinagar, India; localized microclimate transfers require domain recalibration.
3. **Inverter Diversity:** Aggregated inverter telemetry does not capture individual sub-string MPPT tracking variances.
4. **Generalization Boundary:** The ANN models nominal generation physics under clean/baseline conditions. It does not predict fault modes directly from tabular data without hybrid conditioning.
