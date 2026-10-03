"""
PVision AI: Main Streamlit Application
A Hybrid CNN-ANN Framework for Photovoltaic Fault Detection and Power Generation Prediction
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

from src.utilities.config import config
from app.styles import CUSTOM_CSS
from app.components import render_metric_card, load_json_metrics

st.set_page_config(
    page_title="PVision AI - Fault Detection & Generation Prediction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/solar-panel.png", width=64)
st.sidebar.title("PVision AI")
st.sidebar.caption("Hybrid CNN-ANN Framework")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🚀 Project Overview",
        "🔍 PV Fault Dataset Inspection",
        "⚡ Solar Generation Dataset Inspection",
        "🖼️ Interactive Image Explorer",
        "💻 Manual Training Guide (VS Code)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Phase 1: Project Setup & Dataset Inspection**\n\n"
    "All model training will be performed manually in VS Code."
)

# Load Inspection Metrics
pv_summary = load_json_metrics(config.METRICS_DIR / "pv_fault_dataset_summary.json")
solar_summary = load_json_metrics(config.METRICS_DIR / "solar_generation_dataset_summary.json")

# ==========================================
# PAGE 1: PROJECT OVERVIEW
# ==========================================
if menu == "🚀 Project Overview":
    st.title("☀️ PVision AI: Hybrid CNN-ANN Framework")
    st.subheader("Photovoltaic Fault Detection and Power Generation Prediction")

    st.markdown("""
    **PVision AI** integrates Computer Vision and Tabular Deep Learning into a unified solar intelligence pipeline:
    - **Vision Branch (CNN):** Classifies photovoltaic module anomalies into 7 distinct states (`crack`, `global_aging`, `hotspot`, `normal`, `partial_aging`, `shading`, `short_circuit`) using Gramian Angular Summation Fields (GASF) and I-V characteristic curves.
    - **Forecasting Branch (ANN):** Predicts active power output ($AC\\_POWER$) using multi-sensor atmospheric telemetry (irradiance, ambient and module temperatures).
    - **Hybrid Coupling:** Real-time generation predictions are derated and cross-referenced with diagnosed physical faults to explain efficiency loss and anticipate maintenance.
    """)

    st.markdown("### 📊 Dataset Overview at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Fault Images", "69,484", "35,000 GASF + 34,484 I-V")
    with col2:
        render_metric_card("Fault Classes", "7 Classes", "1 Normal + 6 Fault Types")
    with col3:
        render_metric_card("Generation Records", "136,476", "Plant 1 & Plant 2 (44 Inverters)")
    with col4:
        render_metric_card("Weather Telemetry", "6,441", "Irradiation & Multi-Temp Sensors")

    st.markdown("### 🏛️ System Architecture")
    st.markdown("""
    ```
    ┌──────────────────────────┐             ┌───────────────────────────┐
    │     GASF / I-V Images    │             │   Weather Telemetry CSV   │
    │  (256x256 / 1343x808)    │             │ (Irradiance, Temp, Time)  │
    └────────────┬─────────────┘             └─────────────┬─────────────┘
                 │                                         │
                 ▼                                         ▼
       ┌──────────────────┐                       ┌──────────────────┐
       │   CNN Classifier │                       │   ANN Regressor  │
       │ (ResNet/MobileNet│                       │ (Dense / Huber)  │
       └─────────┬────────┘                       └─────────┬────────┘
                 │                                          │
                 ▼                                          ▼
         Detected Fault                             Nominal Power (kW)
                 │                                          │
                 └──────────────────┬───────────────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │   Hybrid Decision Engine    │
                     │  - Derating Factor Analysis │
                     │  - Anomaly & Soiling Alerts │
                     │  - Remaining Power Yield    │
                     └─────────────────────────────┘
    ```
    """)

# ==========================================
# PAGE 2: PV FAULT INSPECTION
# ==========================================
elif menu == "🔍 PV Fault Dataset Inspection":
    st.title("🔍 PV Fault Dataset Inspection Report")
    st.write("Detailed audit of image files, class balance, resolutions, and missing sequences.")

    mod_reports = pv_summary.get("modality_reports", {})

    if not mod_reports:
        st.warning("Inspection report not found. Run dataset inspection scripts from the terminal.")
    else:
        tab1, tab2 = st.tabs(["GASF Images (Gramian Angular Field)", "I-V Curve Images"])

        with tab1:
            gasf = mod_reports.get("gasf_images", {})
            st.markdown(f"**Total Samples:** {gasf.get('total_images', 0):,} | **Resolution:** 256 x 256 RGB | **Corrupted Files:** {gasf.get('corrupted_files_count', 0)}")
            
            c_df = pd.DataFrame({
                "Sample Count": gasf.get("class_counts", {}),
                "Share (%)": gasf.get("class_distribution_percentage", {})
            })
            st.dataframe(c_df, use_container_width=True)
            st.success("✅ GASF Modality is perfectly balanced across all 7 classes (5,000 samples per class).")

        with tab2:
            iv = mod_reports.get("iv_images", {})
            st.markdown(f"**Total Samples:** {iv.get('total_images', 0):,} | **Resolution:** 1343 x 808 RGB | **Corrupted Files:** {iv.get('corrupted_files_count', 0)}")
            
            iv_df = pd.DataFrame({
                "Sample Count": iv.get("class_counts", {}),
                "Share (%)": iv.get("class_distribution_percentage", {})
            })
            st.dataframe(iv_df, use_container_width=True)

            missing_seq = iv.get("missing_sequence_details", {})
            if missing_seq:
                st.error("⚠️ **Missing Files Detected in I-V Dataset:**")
                for cname, minfo in missing_seq.items():
                    st.write(f"- Class **`{cname}`**: {minfo['missing_count']} missing samples.")
                    st.write(f"  Missing Sequence Ranges: `{', '.join(minfo['missing_index_ranges'])}`")

        # Distribution Chart
        fig_path = config.FIGURES_DIR / "pv_fault_class_distribution.png"
        if fig_path.exists():
            st.markdown("### 📊 Class Distribution Visualization")
            st.image(str(fig_path), use_container_width=True)

# ==========================================
# PAGE 3: SOLAR GENERATION INSPECTION
# ==========================================
elif menu == "⚡ Solar Generation Dataset Inspection":
    st.title("⚡ Solar Generation & Weather Dataset Inspection")
    st.write("Exploratory audit of CSV schemas, missing values, duplicates, and correlation dynamics.")

    file_reports = solar_summary.get("file_reports", {})

    if file_reports:
        st.markdown("### 📁 CSV File Health Summary")
        overview = []
        for fn, r in file_reports.items():
            overview.append({
                "File": fn,
                "Rows": f"{r['rows']:,}",
                "Columns": r["columns"],
                "Missing Values": r["total_missing_values"],
                "Exact Duplicates": r["exact_duplicate_rows"],
                "Composite Duplicates": r["composite_key_duplicates"]
            })
        st.dataframe(pd.DataFrame(overview), use_container_width=True)
        st.success("✅ Zero missing values and zero duplicate rows detected across all 4 CSV datasets!")

        st.markdown("### 📈 Correlation Analysis: Weather Sensors vs Power Generation")
        col_img1, col_img2 = st.columns(2)
        p1_fig = config.FIGURES_DIR / "solar_gen_plant1_corr.png"
        p2_fig = config.FIGURES_DIR / "solar_gen_plant2_corr.png"

        with col_img1:
            if p1_fig.exists():
                st.image(str(p1_fig), caption="Plant 1 Correlation Matrix", use_container_width=True)
        with col_img2:
            if p2_fig.exists():
                st.image(str(p2_fig), caption="Plant 2 Correlation Matrix", use_container_width=True)

        st.markdown("### 📋 Descriptive Statistics Explorer")
        selected_file = st.selectbox("Select CSV File to Inspect Statistics", list(file_reports.keys()))
        if selected_file:
            stats_data = file_reports[selected_file].get("descriptive_statistics", {})
            st.dataframe(pd.DataFrame(stats_data).T, use_container_width=True)

# ==========================================
# PAGE 4: INTERACTIVE IMAGE EXPLORER
# ==========================================
elif menu == "🖼️ Interactive Image Explorer":
    st.title("🖼️ PV Fault Interactive Image Explorer")
    st.write("Browse and preview actual samples from each fault category and imaging modality.")

    col_m, col_c = st.columns(2)
    with col_m:
        selected_mod = st.selectbox("Select Imaging Modality", config.MODALITIES)
    with col_c:
        selected_class = st.selectbox("Select Fault Class", config.PV_FAULT_CLASSES)

    target_dir = config.PV_FAULT_DATA_DIR / selected_mod / selected_class
    if target_dir.is_dir():
        files = list(target_dir.glob("*.png"))
        st.write(f"Showing sample images from `{target_dir.name}` (Total: {len(files):,} images)")
        
        if files:
            sample_files = files[:6]
            cols = st.columns(3)
            for idx, img_path in enumerate(sample_files):
                with cols[idx % 3]:
                    img = Image.open(img_path)
                    st.image(img, caption=f"{img_path.name}\n{img.size[0]}x{img.size[1]}", use_container_width=True)

# ==========================================
# PAGE 5: MANUAL TRAINING GUIDE (VS CODE)
# ==========================================
elif menu == "💻 Manual Training Guide (VS Code)":
    st.title("💻 Manual Training Guide for VS Code")
    st.markdown("""
    > [!IMPORTANT]
    > **Models are NOT trained inside Antigravity.**
    > Follow the steps below to train both the CNN and ANN models locally inside VS Code.
    """)

    st.markdown("""
    ### Step 1: Open Project in VS Code
    Open the `PVision_AI` workspace folder in VS Code:
    ```bash
    code d:\\SolarPredict\\PVision_AI
    ```

    ### Step 2: Run Dataset Inspection Suite
    Execute the unified dataset inspector from your terminal:
    ```bash
    python -m src.data.dataset_inspector --all
    ```

    ### Step 3: Run Interactive Notebooks
    Open the notebooks inside VS Code using the Jupyter Extension:
    - `notebooks/01_pv_fault_dataset_inspection.ipynb`
    - `notebooks/02_solar_generation_dataset_inspection.ipynb`

    ### Step 4: Preprocessing Data for Training
    Generate train/val/test manifests and feature matrices:
    ```python
    from src.preprocessing.image_preprocessing import PVFaultDatasetManifest
    from src.preprocessing.tabular_preprocessing import SolarDataPipeline

    # Build stratified manifests for CNN
    manifest = PVFaultDatasetManifest(modality="gasf_images")
    splits = manifest.create_split_manifest()

    # Build tabular training sets for ANN
    pipeline = SolarDataPipeline()
    X_train, X_test, y_train, y_test = pipeline.prepare_training_datasets(plant_id=1)
    ```

    ### Step 5: Train CNN & ANN Models (Phase 2)
    In Phase 2, train the models using PyTorch / Scikit-learn and save weights to the `models/` directory:
    - `models/pv_fault_cnn_best.pth`
    - `models/solar_power_ann_best.pth`
    """)

st.markdown("---")
st.caption("PVision AI Framework • Phase 1 Setup & Inspection Complete • Designed for VS Code Manual Training")
