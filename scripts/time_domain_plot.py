import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

Fs = 5120  # Sampling frequency

def getFile(head, stage):
    filename = 'h{}_{}.mat'.format(head, stage)
    return sio.loadmat(filename)

def getSignal(head, stage, regime, sensor, data):
    f = getFile(head, stage)
    regime_dict = {'DPL': 0, 'PL': 1, 'BEP': 2, 'HL': 3, 'FL': 4}
    return f['Measurements'][0][regime_dict[regime]][sensor][data][0, 0].flatten()

# Parameters
sensor = 'PGV2'
regime = 'BEP'
head = 30
stage_a = 8

# Get signal
amplitude_A = getSignal(head, stage_a, regime, sensor, 'Values')

# Time array
time = np.linspace(0, len(amplitude_A) / Fs, len(amplitude_A), endpoint=False)

# Plot time-domain signal
plt.figure(figsize=(12, 6))
plt.plot(time, amplitude_A, color='blue', linewidth=1.2)
plt.title('Time-Domain Signal ({})'.format(sensor))
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V/V]')
plt.grid(True)
plt.show()
