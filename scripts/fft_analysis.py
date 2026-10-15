import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
import h5py
import csv

Fs = 5120

head = 30
stage = 5

filename = 'h{}_{}.mat'.format(head,stage) 

root = sio.loadmat(filename)

print(root.keys())
# ->  'Measurements', 'Properties'

regime_dict = {'DPL': 0, 'PL': 1, 'BEP': 2, 'HL': 3, 'FL': 4}

channels = ['name', 'PIN1', 'PTC', 'PDT1', 'PDT2', 
		   'PGV1', 'PGV2', 'PGV3', 'PDT3', 'PDT4', 
		   'Patm', 'WTmp', 'rpm', 'Speed', 'Flow', 
		   'Thrust', 'GenTorque', 'FricTorque', 'Pin', 'Pdiff', 
		   'GV', 'AGV', 'ATB1', 'ATB2', 'properties']

regime = 'BEP'
r = regime_dict[regime]

v = 'Values'
p = 'properties'

#To get time-domain values
#print(root['Measurements' (str)][0][regime (int)]['PGV3' (str)][v = 'Values' OR p = 'properties'][0,0].flatten())

print(root['Measurements'][0][r]['PGV3'][v][0,0].flatten())

#get properties of channel
props = [p[0] for p in root['Measurements'][0][r]['PGV3'][p][0,0].dtype.descr]

print(props)

# -> 
props = ['wf_start_time', 'wf_start_offset', 'wf_increment', 
		 'wf_samples', 'BridgeConfiguration', 'BridgeResistance',
		 'NI_ChannelName', 'NI_UnitDescription', 'unit_string',
		 'Offset', 'Scale', 'SensorProducer', 'SerialNumber', 'Type']

#to get vlaue of prop
# eg SerialNumber is 2nd last
prop = props[-2]

print('{}: {}'.format(prop, root['Measurements'][0][r]['PGV3'][p][0,0][prop][0,0].flatten()[0]))



'''
 -- LETS PLOT --

'''

def getFile(head, stage):
	filename = 'h{}_{}'.format(head,stage)
	return sio.loadmat(filename)

def getSignal(head, stage, regime, sensor, data):
	f = getFile(head, stage)
	regime_dict = {'DPL': 0, 'PL': 1, 'BEP': 2, 'HL': 3, 'FL': 4}
	return f['Measurements'][0][regime_dict[regime]][sensor][data][0,0].flatten()

sensor = 'ATB1'
regime = 'BEP'
head = 30
stage_a = 1
stage_b = 8
amplitude_A = getSignal(head, stage_a, regime, sensor, 'Values')
amplitude_B = getSignal(head, stage_b, regime, sensor, 'Values')
time = [i*(1/Fs) for i in range(len(amplitude_A)) ]

# Time domain signal
fig, ((t_plot_A, fft_plot1_A, fft_plot2_A), (t_plot_B, fft_plot1_B, fft_plot2_B)) = plt.subplots(2, 3)
t_plot_A.plot(time, amplitude_A)
t_plot_A.set_title('Time-domain signal of {}'.format(sensor))
t_plot_A.set_xlabel('Time [s]')
t_plot_A.set_ylabel('Amplitude [V/V]')
t_plot_A.axis([min(time), max(time), min(amplitude_A), max(amplitude_A)])
t_plot_A.grid(True)

# Power Spectral Density
f, psd = sig.welch(amplitude_A, Fs, window='hann')
fft_plot1_A.semilogy(f, psd)
fft_plot1_A.set_title('Power spectral density signal of {}'.format(sensor))
fft_plot1_A.set_xlabel('Frequency [Hz]')
fft_plot1_A.set_ylabel('PSD [$V^2$/Hz]') # wrapping string with $ allows superscripts/subscripts etc handy for keeping graph label clean
fft_plot1_A.grid(True, which='major', color='k', linestyle='-') # major axis in black (k) 
fft_plot1_A.grid(True, which='minor', color='r', linestyle='-', alpha=0.2) # minor axis in red (r), alpha for transparency
fft_plot1_A.minorticks_on()

# Power Spectrum
f, ps = sig.welch(amplitude_A, Fs, window='hann', scaling='spectrum')
fft_plot2_A.semilogy(f, np.sqrt(ps))
fft_plot2_A.set_title('Power spectrum of {}'.format(sensor))
fft_plot2_A.set_xlabel('Frequency [Hz]')
fft_plot2_A.set_ylabel('Linear Spectrum [V RMS]')
fft_plot2_A.grid(True, which='major', color='k', linestyle='-')
fft_plot2_A.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
fft_plot2_A.minorticks_on()

t_plot_B.plot(time, amplitude_B)
t_plot_B.set_title('Time-domain signal of {}'.format(sensor))
t_plot_B.set_xlabel('Time [s]')
t_plot_B.set_ylabel('Amplitude [V/V]')
t_plot_B.axis([min(time), max(time), min(amplitude_B), max(amplitude_B)])
t_plot_B.grid(True)

# Power Spectral Density
f, psd = sig.welch(amplitude_B, Fs, window='hann')
fft_plot1_B.semilogy(f, psd)
fft_plot1_B.set_title('Power spectral density signal of {}'.format(sensor))
fft_plot1_B.set_xlabel('Frequency [Hz]')
fft_plot1_B.set_ylabel('PSD [$V^2$/Hz]') # wrapping string with $ allows superscripts/subscripts etc handy for keeping graph label clean
fft_plot1_B.grid(True, which='major', color='k', linestyle='-') # major axis in black (k) 
fft_plot1_B.grid(True, which='minor', color='r', linestyle='-', alpha=0.2) # minor axis in red (r), alpha for transparency
fft_plot1_B.minorticks_on()

# Power Spectrum
f, ps = sig.welch(amplitude_B, Fs, window='hann', scaling='spectrum')
fft_plot2_B.semilogy(f, np.sqrt(ps))
fft_plot2_B.set_title('Power spectrum of {}'.format(sensor))
fft_plot2_B.set_xlabel('Frequency [Hz]')
fft_plot2_B.set_ylabel('Linear Spectrum [V RMS]')
fft_plot2_B.grid(True, which='major', color='k', linestyle='-')
fft_plot2_B.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
fft_plot2_B.minorticks_on()


plt.show()
#BOTH SATAGES ON SAME GRAPH 
# Combined Power Spectrum Plot
plt.figure()

# Compute Power Spectral Density for Stage A
f_a, psd_a = sig.welch(amplitude_A, Fs, window='hann')

# Compute Power Spectral Density for Stage B
f_b, psd_b = sig.welch(amplitude_B, Fs, window='hann')

# Plot PSD for Stage A
plt.semilogy(f_a, psd_a, label='Phase {}'.format(stage_a), color='blue')

# Plot PSD for Stage B
plt.semilogy(f_b, psd_b, label='Phase {}'.format(stage_b), color='orange')

# Add title, labels, and gridlines
plt.title('Power Spectral Density Comparison of {}'.format(sensor))
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [$V^2$/Hz]')
plt.grid(True, which='major', color='k', linestyle='-')
plt.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
plt.minorticks_on()

# Display legend
plt.legend()

# Show the plot
plt.show()