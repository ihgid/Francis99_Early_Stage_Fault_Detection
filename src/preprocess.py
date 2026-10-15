import os
import numpy as np
import pandas as pd
from joblib import load

SENSOR_COLUMNS = ['PDT1', 'PGV2', 'PDT3', 'ATB1', 'ATB2']

def load_scaler(scaler_path="models/robust_scaler.pkl"):
    """
    Load pre-fitted RobustScaler.
    """
    if not os.path.exists(scaler_path) and os.path.exists("robust_scaler.pkl"):
        scaler_path = "robust_scaler.pkl"
    return load(scaler_path)

def preprocess_single_reading(sensor_values, scaler, sequence_length=30):
    """
    Preprocess a single 5-sensor vector into scaled 30-timestep sequence array (1, sequence_length, 5).
    """
    raw_input = np.array([sensor_values], dtype=np.float32)
    scaled_input = scaler.transform(raw_input)
    sequence_input = np.tile(scaled_input, (sequence_length, 1)).reshape(1, sequence_length, len(sensor_values)).astype(np.float32)
    return sequence_input

def create_sliding_windows(data, window_size=30, step=1):
    """
    Create sliding window sequence arrays from 2D sensor dataframe/array.
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i : i + window_size])
    return np.array(windows, dtype=np.float32)
