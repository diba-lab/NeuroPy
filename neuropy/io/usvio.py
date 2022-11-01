from scipy.io import wavfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from neuropy.core.signal import Signal
from neuropy.utils.signal_process import filter_sig


## Script here to filter USV file at 500Hz to detect start sync signal.
dir_use = Path('/Users/nkinsky/Documents/UM/Working/Trace_FC/Recording_Rats/Finn/2022_01_20_training/shock_box')
fs, Audiodata = wavfile.read(dir_use / 'T0000002.WAV')
audio_sig = Signal(Audiodata, fs)

bp500 = filter_sig.bandpass(audio_sig, 480, 520, order=2, fs=fs)
_, ax = plt.subplots()
ax.plot(np.linspace(0,240, 240*2000), np.abs(bp500.traces[0][0:(240*fs):int(fs/2000)]))

# basically need to figure out how to take the above and threshold it and find out when it jumps above a certain value for > 0.4sec