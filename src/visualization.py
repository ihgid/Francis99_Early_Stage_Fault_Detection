import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from PIL import Image

def plot_time_domain_signal(signal, sampling_rate=5120, sensor_name="PGV2", title=None):
    """
    Generate a Matplotlib Figure for time-domain signal waveform.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
    time = np.linspace(0, len(signal) / sampling_rate, len(signal), endpoint=False)
    ax.plot(time, signal, color='#1f77b4', linewidth=1.0)
    ax.set_title(title or f"Time-Domain Waveform ({sensor_name})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [V/V]")
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig

def plot_psd_spectrum(signal, sampling_rate=5120, sensor_name="PGV2", title=None):
    """
    Generate a Matplotlib Figure for Power Spectral Density (PSD) using Welch's method.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
    freqs, psd = sig.welch(signal, fs=sampling_rate, window='hann')
    ax.semilogy(freqs, psd, color='#d62728', linewidth=1.2)
    ax.set_title(title or f"Power Spectral Density - Welch Spectrum ({sensor_name})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [$V^2$/Hz]")
    ax.grid(True, which='major', linestyle='-', color='#aaaaaa')
    ax.grid(True, which='minor', linestyle=':', color='#dddddd', alpha=0.5)
    ax.minorticks_on()
    fig.tight_layout()
    return fig

def get_cropped_crack_stage_image(image_path="crackimage.jpg", fault_stage=0):
    """
    Crop and return the runner blade image slice corresponding to fault stage (0 to 9).
    """
    if not os.path.exists(image_path):
        return None

    try:
        img = Image.open(image_path)
        cols, rows = 5, 2
        stage = max(0, min(int(fault_stage), cols * rows - 1))
        cell_w = img.width // cols
        cell_h = img.height // rows
        r = stage // cols
        c = stage % cols
        cropped = img.crop((c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h))
        return cropped
    except Exception:
        return None
