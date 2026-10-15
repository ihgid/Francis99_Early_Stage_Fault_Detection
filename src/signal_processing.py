import numpy as np
import scipy.io as sio
import scipy.signal as sig

def load_mat_signal(mat_filepath, regime='BEP', sensor='PGV3'):
    """
    Load time-domain sensor values from Francis-99 MATLAB (.mat) data format.
    """
    f = sio.loadmat(mat_filepath)
    regime_dict = {'DPL': 0, 'PL': 1, 'BEP': 2, 'HL': 3, 'FL': 4}
    r_idx = regime_dict.get(regime, 2)
    return f['Measurements'][0][r_idx][sensor]['Values'][0, 0].flatten()

def compute_psd(signal, sampling_rate=5120):
    """
    Compute Power Spectral Density (PSD) using Welch's method.
    """
    freqs, psd = sig.welch(signal, fs=sampling_rate, window='hann')
    return freqs, psd

def compute_fft(signal, sampling_rate=5120):
    """
    Compute Fast Fourier Transform (FFT) amplitude spectrum.
    """
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal)) / n
    freqs = np.fft.rfftfreq(n, 1.0 / sampling_rate)
    return freqs, fft_vals
