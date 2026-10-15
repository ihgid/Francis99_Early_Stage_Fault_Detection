# ⚡ Early-Stage Fault Detection in Francis Turbines using AI/ML

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LLaMA 3](https://img.shields.io/badge/LLM-Groq%20LLaMA--3%2070B-008080?style=flat)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end industrial condition monitoring system designed for early-stage fault detection in **Francis-99 hydro turbine runner blades** using multivariate time-series sensor data, a hybrid **1D CNN-LSTM Deep Learning Architecture**, and an **LLM-powered Expert Decision Support Agent**.

Paper Reference: [EARLY-STAGE FAULT DETECTION IN FRANCIS TURBINES USING AI-ML](https://www.pravakarbogati.com.np/assets/papers/EARLY-STAGE%20FAULT%20DETECTION%20IN%20FRANCIS%20TURBINES%20USING%20AI-ML.pdf)

---

## 📌 Executive Summary

Francis hydro turbines operating under off-design regimes (e.g., Part Load, High Load) experience intense hydraulic instability, pressure pulsations, and acoustic resonance. Over long duty cycles, these cyclic mechanical stresses lead to fatigue cracks near the runner blade trailing edges. If undetected, crack propagation culminates in blade fragment detachment and catastrophic structural failure.

This repository implements a **fact-grounded, end-to-end Machine Learning pipeline** that continuously monitors 5 critical pressure and vibration sensor channels, normalizes noisy industrial signals using `RobustScaler`, constructs 30-step sliding temporal sequences, and classifies turbine health into **10 distinct severity stages** with **~98.2% accuracy**.

---

## 🏗️ End-to-End System Architecture

```mermaid
graph TD;
    subgraph 📡 Data Acquisition
        A1[PDT1: Pressure Downstream 1]
        A2[PGV2: Guide Vane Vibration 2]
        A3[PDT3: Pressure Downstream 3]
        A4[ATB1: Axial Bearing Vibration 1]
        A5[ATB2: Axial Bearing Vibration 2]
    end

    subgraph ⚙️ Preprocessing & Signal Pipeline
        B1[RobustScaler Normalization]
        B2[30-Step Temporal Sliding Window Sequence]
        B3[Fast Fourier Transform & PSD Analysis]
    end

    subgraph 🧠 Hybrid Deep Learning Engine
        C1[1D Convolutional Neural Network - Local Spatial Features]
        C2[LSTM Network - Temporal Dynamics Capture]
        C3[Dense Classification Layer - Softmax Output]
    end

    subgraph 🖥️ Decision Support & Web Dashboard
        D1[Predicted Fault Stage 0-9 & Confidence Score]
        D2[Cropped Blade Crack Image Slice]
        D3[Groq LLaMA-3 AI Expert Maintenance Recommendation]
    end

    A1 & A2 & A3 & A4 & A5 --> B1
    B1 --> B2
    B1 --> B3
    B2 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> D1
    D1 --> D2
    D1 --> D3
```

---

## 📊 Sensor Channels & Fault Stage Mapping

### 📡 Monitored Sensor Channels
| Sensor Code | Description | Physical Metric |
| :--- | :--- | :--- |
| `PDT1` | Pressure Transducer Downstream 1 | Hydraulic Pressure Pulsation ($kPa$) |
| `PGV2` | Pressure Guide Vane Transducer 2 | Guide Vane Dynamic Pressure ($kPa$) |
| `PDT3` | Pressure Transducer Downstream 3 | Draft Tube Hydraulic Pressure ($kPa$) |
| `ATB1` | Axial Turbine Bearing Sensor 1 | Mechanical Vibration ($m/s^2$) |
| `ATB2` | Axial Turbine Bearing Sensor 2 | Bearing Axial Displacement ($mm$) |

### 🔍 Progressive Fault Stages (0 - 9)
| Stage | Condition Status | Approx. Crack Length | Mechanical Severity & Operational Impact |
| :---: | :--- | :---: | :--- |
| **Stage 0** | Healthy Baseline | $0\text{ mm}$ | Normal operation within nominal parameters. |
| **Stage 1** | Initial Surface Crack | $\approx 15\text{ mm}$ | Micro-fissures at trailing edge; slight turbulence. |
| **Stage 2** | Early Crack Growth | $\approx 30\text{ mm}$ | Minor pressure fluctuation increase. |
| **Stage 3** | Substantial Propagation | $\approx 50\text{ mm}$ | Localized stiffness reduction. |
| **Stage 4** | Moderate Structural Defect | $\approx 72\text{ mm}$ | Elevated vibration signatures in guide vane. |
| **Stage 5** | Advanced Fatigue Crack | $\approx 95\text{ mm}$ | High risk of localized flow detachment. |
| **Stage 6** | Critical Blade Crack | $\approx 115\text{ mm}$ | Severe mechanical stress concentration. |
| **Stage 7** | Pre-Detachment Fracture | $\approx 140\text{ mm}$ | Imminent blade fracture; high acoustic pulse. |
| **Stage 8** | Detached Blade Fragment | $\approx 170\text{ mm}$ (Detached) | Fragment released; extreme hydraulic unbalance. |
| **Stage 9** | Enlarged Opening Damage | Fragment Opening | Severe secondary runner damage; emergency halt. |

---

## 📈 Model Benchmark & Comparison

Experiments conducted across multiple machine learning architectures demonstrate that combining spatial feature extraction (1D CNN) with temporal sequence learning (LSTM) achieves superior multi-class fault classification performance.

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Key Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Support Vector Machine (SVM)** | 88.4% | 0.88 | 0.87 | 0.87 | Robust baseline for stationary distributions |
| **XGBoost Classifier** | 92.1% | 0.91 | 0.92 | 0.91 | Effective tabular decision boundary |
| **1D CNN Network** | 94.5% | 0.94 | 0.94 | 0.94 | Captures local multi-sensor spatial patterns |
| **LSTM Recurrent Network** | 95.8% | 0.95 | 0.95 | 0.95 | Tracks temporal sequence dependencies |
| **Hybrid CNN-LSTM (Proposed)** | **98.2%** | **0.98** | **0.98** | **0.98** | **Optimal Spatial-Temporal Signal Feature Fusion** |

---

## 📂 Repository Structure

```
.
├── app.py                     # Streamlit Interactive Web Application
├── cnn+LSTM_fault_model.h5   # Trained Hybrid CNN-LSTM Model Artifact
├── robust_scaler.pkl         # Fitted RobustScaler Normalization Parameters
├── crackimage.jpg            # Visual Reference Grid (5x2 Matrix of Fault Stages)
├── requirements.txt          # Verified Project Dependencies
├── README.md                 # Project Documentation
├── .env.example              # Template for API Key Configurations
├── .gitignore                # Git Exclusions File
├── src/                      # Modular Python Engine
│   ├── __init__.py
│   ├── preprocess.py         # Scaling & Sequence Sliding Window Logic
│   ├── model_pipeline.py     # CNN-LSTM Architecture Definition & Prediction Engine
│   ├── signal_processing.py  # FFT, PSD & Welch Analysis Functions
│   ├── train.py              # End-to-End Retraining Pipeline Script
│   └── llm_expert.py         # Groq LLaMA-3 Expert Advisory Agent
└── tests/                    # Unit Tests & Automated Pipeline Verification
    ├── __init__.py
    └── test_pipeline.py      # Preprocessing & Model Interface Tests
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Francis99_Early_Stage_Fault_Detection.git
cd Francis99_Early_Stage_Fault_Detection
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root folder (or copy `.env.example`):
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(Note: If no Groq API key is provided, the application automatically falls back to an built-in offline expert knowledge base).*

### 5. Launch Web Application
```bash
streamlit run app.py
```

---

## 💡 Web Application Features

1. **🎯 Live Interactive Sensor Inputs & Presets**: Select from pre-configured turbine operational states (Healthy, Stage 1, Stage 4, Stage 8) or manually input 5 sensor readings.
2. **📁 Batch CSV Sequence Ingestion**: Upload time-series CSV sensor recordings to analyze fault progression across windowed time sequences.
3. **📊 Probability Distribution Visualizations**: View stage probability breakdown and prediction confidence metrics.
4. **📷 Blade Surface Reference Cropping**: Automatic dynamic cropping of the runner blade surface slice corresponding to predicted fault severity.
5. **🧠 Groq LLaMA-3 AI Expert Advisory**: Receive maintenance action directives, estimated crack growth, and risk assessments formatted in clean JSON.

---

## 🧪 Running Automated Unit Tests

Run the test suite to verify pipeline functionality:
```bash
python -m unittest discover -s tests
```

---

## 🤝 Citation & Acknowledgments

If you find this research work or implementation useful, please cite:
- **Paper Title**: *Early-Stage Fault Detection in Francis Turbines Using AI-ML*
- **Paper Link**: [Francis Turbine AI-ML Fault Detection PDF](https://www.pravakarbogati.com.np/assets/papers/EARLY-STAGE%20FAULT%20DETECTION%20IN%20FRANCIS%20TURBINES%20USING%20AI-ML.pdf)
- **Dataset Source**: Francis-99 Benchmark Data, Norwegian University of Science and Technology (NTNU).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
