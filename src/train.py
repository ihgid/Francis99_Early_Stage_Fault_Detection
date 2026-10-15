"""
End-to-End Model Training & Evaluation Pipeline for Francis-99 Fault Detection
"""

import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from src.preprocess import SENSOR_COLUMNS, create_sliding_windows
from src.model_pipeline import build_cnn_lstm_model

def train_pipeline(data_path=None, epochs=20, batch_size=64, save_dir="."):
    """
    Train the Hybrid 1D CNN-LSTM model on Francis-99 sensor dataset.
    """
    print("⚡ Starting Francis-99 Training Pipeline...")

    # Generate synthetic/demo dataset if no data path provided
    if data_path is None or not os.path.exists(data_path):
        print("ℹ️ Data path not found. Generating sample synthetic training sequence...")
        np.random.seed(42)
        n_samples = 3000
        n_features = 5
        X_raw = np.random.randn(n_samples, n_features)
        y_raw = np.random.randint(0, 10, size=(n_samples,))
    else:
        df = pd.read_csv(data_path)
        X_raw = df[SENSOR_COLUMNS].values
        y_raw = df['Stage'].values

    # Fit RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)
    scaler_path = os.path.join(save_dir, "robust_scaler.pkl")
    dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")

    # Create Sliding Window Sequences
    sequence_length = 30
    X_seq = create_sliding_windows(X_scaled, window_size=sequence_length)
    y_seq = y_raw[sequence_length - 1:]

    # Train / Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Build and Train Model
    model = build_cnn_lstm_model(input_shape=(sequence_length, len(SENSOR_COLUMNS)), num_classes=10)
    print(model.summary())

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    model_path = os.path.join(save_dir, "cnn+LSTM_fault_model.h5")
    model.save(model_path)
    print(f"✅ Trained model saved to {model_path}")

if __name__ == "__main__":
    train_pipeline()
