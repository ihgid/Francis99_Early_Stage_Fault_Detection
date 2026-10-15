import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization

def build_cnn_lstm_model(input_shape=(30, 5), num_classes=10):
    """
    Build the Hybrid 1D CNN + LSTM Deep Learning Model Architecture.
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        LSTM(units=64, return_sequences=False),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def predict_fault_stage(model, input_sequence):
    """
    Predict fault stage and class probability distribution.
    """
    probabilities = model.predict(input_sequence, verbose=0)
    fault_stage = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(np.max(probabilities))
    return fault_stage, confidence, probabilities[0]
