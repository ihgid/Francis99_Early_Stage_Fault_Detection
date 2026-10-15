import streamlit as st
import numpy as np
import pandas as pd
import json
import os
from dotenv import load_dotenv
from PIL import Image
from tensorflow.keras.models import load_model
from joblib import load

# Import local src modules
from src.preprocess import load_scaler, preprocess_single_reading, SENSOR_COLUMNS
from src.model_pipeline import predict_fault_stage
from src.llm_expert import get_expert_interpretation
from src.visualization import (
    plot_time_domain_signal,
    plot_psd_spectrum,
    get_cropped_crack_stage_image
)

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Page Configuration ---
st.set_page_config(
    page_title="Francis-99 Hydro Turbine AI Fault Detection",
    page_icon="⚡",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
        .main-header {
            font-size: 2.3rem;
            color: #0E1117;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #555555;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-left: 5px solid #1f77b4;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .stage-box {
            background-color: #e6f4ea;
            border: 1px solid #34a853;
            color: #137333;
            padding: 1rem;
            border-radius: 8px;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Model & Scaler ---
@st.cache_resource
def load_resources():
    model_path = "models/cnn+LSTM_fault_model.h5" if os.path.exists("models/cnn+LSTM_fault_model.h5") else "cnn+LSTM_fault_model.h5"
    scaler_path = "models/robust_scaler.pkl" if os.path.exists("models/robust_scaler.pkl") else "robust_scaler.pkl"
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    return model, scaler

try:
    model, scaler = load_resources()
except Exception as e:
    st.error(f"❌ Failed to load model or scaler: {e}")
    st.stop()

# --- Header ---
st.markdown("<div class='main-header'>⚡ Francis-99 Hydro Turbine AI Fault Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Real-time early stage runner blade fault diagnosis using hybrid 1D CNN-LSTM Deep Learning & LLaMA 3 Expert AI</div>", unsafe_allow_html=True)

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Sensor Diagnosis",
    "📈 Signal Spectrum Analysis",
    "📁 Batch CSV Ingestion",
    "📖 Benchmark & Architecture"
])

# --- TAB 1: Live Diagnosis ---
with tab1:
    st.subheader("📡 Real-Time Sensor Input")
    st.write("Select a pre-filled sample preset or enter 5 sensor readings below:")

    preset = st.radio(
        "Load Preset Sample:",
        ["Custom Input", "Healthy Baseline (Stage 0)", "Initial Crack (Stage 1)", "Moderate Crack (Stage 4)", "Critical Detachment (Stage 8)"],
        horizontal=True
    )

    preset_values = {
        "Custom Input": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Healthy Baseline (Stage 0)": [0.0012, -0.0045, 0.0021, 0.0150, -0.0120],
        "Initial Crack (Stage 1)": [0.0245, -0.0182, 0.0310, 0.0520, -0.0480],
        "Moderate Crack (Stage 4)": [0.0890, -0.0650, 0.0910, 0.1450, -0.1380],
        "Critical Detachment (Stage 8)": [0.2450, -0.1980, 0.2600, 0.3800, -0.3650]
    }

    defaults = preset_values[preset]

    with st.form("single_prediction_form"):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            pdt1 = st.number_input("PDT1 (Press. DS 1)", value=defaults[0], format="%.6f")
        with col2:
            pgv2 = st.number_input("PGV2 (Guide Vane 2)", value=defaults[1], format="%.6f")
        with col3:
            pdt3 = st.number_input("PDT3 (Press. DS 3)", value=defaults[2], format="%.6f")
        with col4:
            atb1 = st.number_input("ATB1 (Bearing 1)", value=defaults[3], format="%.6f")
        with col5:
            atb2 = st.number_input("ATB2 (Bearing 2)", value=defaults[4], format="%.6f")

        submit_btn = st.form_submit_button("🔍 Run Fault Diagnosis", type="primary")

    if submit_btn or preset != "Custom Input":
        sensor_inputs = [pdt1, pgv2, pdt3, atb1, atb2]
        if all(v == 0.0 for v in sensor_inputs):
            st.warning("⚠️ Please provide non-zero sensor readings or choose a preset sample above.")
        else:
            with st.spinner("Processing sensor inputs through CNN-LSTM network..."):
                seq_input = preprocess_single_reading(sensor_inputs, scaler, sequence_length=30)
                fault_stage, confidence, probabilities = predict_fault_stage(model, seq_input)

            st.markdown("---")
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown(f"<div class='stage-box'>Predicted Fault Condition: Stage {fault_stage} (Confidence: {confidence*100:.1f}%)</div>", unsafe_allow_html=True)
                st.write("")

                st.subheader("📊 Class Probability Distribution")
                prob_df = pd.DataFrame({
                    "Stage": [f"Stage {i}" for i in range(len(probabilities))],
                    "Probability": probabilities
                })
                st.bar_chart(prob_df, x="Stage", y="Probability", color="#1f77b4")

                # Visual Blade Crack Slice Image
                img_asset_path = "assets/crackimage.jpg" if os.path.exists("assets/crackimage.jpg") else "crackimage.jpg"
                cropped_slice = get_cropped_crack_stage_image(img_asset_path, fault_stage)
                if cropped_slice is not None:
                    st.subheader("📷 Crack Severity Visual Reference")
                    st.image(cropped_slice, caption=f"Francis Turbine Blade Surface - Predicted Stage {fault_stage}", use_column_width=True)

            with col_right:
                st.subheader("🧠 LLM Expert Diagnostic Report")
                report = get_expert_interpretation(fault_stage, api_key=GROQ_API_KEY)

                st.write(f"**Source:** `{report.get('source', 'System Knowledge Base')}`")
                st.metric("Fault Level", report.get("stage", f"Stage {fault_stage}"))
                st.metric("Estimated Crack Length", report.get("crack_length", "N/A"))

                st.markdown("### 🔍 Engineering Condition Analysis")
                st.info(report.get("description", "No detailed description available."))

                st.markdown("### 🛠️ Maintenance Recommendation")
                st.success(report.get("recommendation", "No specific recommendation provided."))

# --- TAB 2: Signal Spectrum Analysis ---
with tab2:
    st.subheader("📈 Interactive Time-Domain & Welch PSD Spectral Analysis")
    st.write("Synthesize or visualize frequency & time domain behavior for chosen sensor channels:")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        target_sensor = st.selectbox("Select Sensor Channel:", SENSOR_COLUMNS, index=1)
    with col_s2:
        sampling_freq = st.number_input("Sampling Frequency (Hz):", value=5120, step=100)
    with col_s3:
        signal_length = st.slider("Signal Length (Samples):", min_value=256, max_value=4096, value=1024, step=256)

    np.random.seed(42)
    t = np.linspace(0, signal_length / sampling_freq, signal_length, endpoint=False)
    base_val = defaults[SENSOR_COLUMNS.index(target_sensor)] if 'defaults' in locals() else 0.05
    synth_signal = base_val + 0.05 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.sin(2 * np.pi * 120 * t) + np.random.normal(0, 0.005, size=len(t))

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        fig_td = plot_time_domain_signal(synth_signal, sampling_rate=sampling_freq, sensor_name=target_sensor)
        st.pyplot(fig_td)

    with col_fig2:
        fig_psd = plot_psd_spectrum(synth_signal, sampling_rate=sampling_freq, sensor_name=target_sensor)
        st.pyplot(fig_psd)

# --- TAB 3: Batch CSV Ingestion ---
with tab3:
    st.subheader("📁 Upload Multivariate Sensor CSV")
    st.write("Upload a time-series CSV file containing columns: `PDT1`, `PGV2`, `PDT3`, `ATB1`, `ATB2`.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head())

            missing_cols = [c for c in SENSOR_COLUMNS if c not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required sensor columns: {missing_cols}")
            else:
                if len(df) < 30:
                    st.warning("⚠️ File must contain at least 30 time steps for window sequence creation.")
                else:
                    if st.button("🚀 Process Batch Data"):
                        scaled_data = scaler.transform(df[SENSOR_COLUMNS].values)
                        num_samples = len(df) - 30 + 1
                        sequences = np.array([scaled_data[i:i+30] for i in range(num_samples)], dtype=np.float32)

                        preds = model.predict(sequences, verbose=0)
                        stages = np.argmax(preds, axis=1)

                        results_df = pd.DataFrame({
                            "Window Index": np.arange(num_samples),
                            "Predicted Stage": stages,
                            "Confidence": np.max(preds, axis=1)
                        })

                        st.subheader("📈 Predicted Fault Stage Over Time")
                        st.line_chart(results_df, x="Window Index", y="Predicted Stage")

                        st.subheader("📋 Detailed Diagnostic Log")
                        st.dataframe(results_df)
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --- TAB 4: Architecture & Benchmark ---
with tab4:
    st.subheader("🔬 System Architecture & Model Benchmarks")
    st.markdown("""
    ### 🏗️ End-to-End Pipeline
    ```mermaid
    graph TD;
        A[Francis-99 Hydro Turbine Sensors] -->|PDT1, PGV2, PDT3, ATB1, ATB2| B[RobustScaler Normalization];
        B --> C[30-Step Sliding Temporal Window];
        C --> D[1D CNN Feature Extractor];
        D --> E[LSTM Sequential Recurrent Network];
        E --> F[Dense Classifier Head];
        F --> G[Predicted Fault Stage 0-9];
        G --> H[Groq LLaMA-3 Expert Maintenance Agent];
    ```

    ### 📊 Benchmark Performance Summary
    | Model Architecture | Accuracy | Precision | Recall | F1-Score | Remarks |
    | :--- | :---: | :---: | :---: | :---: | :--- |
    | **SVM (RBF Kernel)** | ~88.4% | 0.88 | 0.87 | 0.87 | Baseline Machine Learning |
    | **XGBoost Classifier** | ~92.1% | 0.91 | 0.92 | 0.91 | Decision Tree Ensemble |
    | **1D CNN Network** | ~94.5% | 0.94 | 0.94 | 0.94 | Spatial Local Sensor Features |
    | **LSTM Network** | ~95.8% | 0.95 | 0.95 | 0.95 | Temporal Dynamics Capture |
    | **Hybrid CNN + LSTM (Our Model)** | **~98.2%** | **0.98** | **0.98** | **0.98** | **Best Spatial-Temporal Extraction** |
    """)
