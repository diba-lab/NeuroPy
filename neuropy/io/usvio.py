from scipy.io import wavfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from neuropy.core.signal import Signal
from neuropy.utils.signal_process import filter_sig

dir_use = Path('/Users/nkinsky/Documents/UM/Working/Trace_FC/Recording_Rats/Finn/2022_01_20_training/shock_box')


def detect_start_tone(filename, freq_lims: list = [480, 520], start_sec_use: int = 240, thresh=50,
                      plot_check: bool = True):
    """Detect USV start tone - typically a 0.5 sec 500Hz pure tone"""
    fs, Audiodata = wavfile.read(filename)
    audio_sig = Signal(Audiodata, fs)

    bp500 = filter_sig.bandpass(audio_sig, freq_lims[0], freq_lims[1], order=2, fs=fs)

    thresh_bool = np.abs(bp500.traces[0][0:(start_sec_use*fs)]) > thresh

    time_start, time_end = np.where(thresh_bool)[0][[0, -1]]/fs

    if plot_check:
        _, ax = plt.subplots()
        ax.set_title(f'{freq_lims[0]} to {freq_lims[1]} filtered signal')
        ht, = ax.plot(np.linspace(0, start_sec_use, start_sec_use*2000),
                      np.abs(bp500.traces[0][0:(start_sec_use*fs):int(fs/2000)]))
        hthresh = ax.axhline(thresh, color='r')
        htone = ax.axvspan(time_start, time_end, color=[0, 0, 1, 0.3])
        ax.legend([ht, hthresh, htone], ['Signal', 'Threshold', 'Detected Tone'])
        ax.set_xlim([time_start - 3, time_end + 3])
        ax.set_xlabel('Time (sec)')

    return time_start, time_end


if __name__ == "__main__":
    time_start, time_end = detect_start_tone(dir_use / 'T0000002.WAV')
# basically need to figure out how to take the above and threshold it and find out when it jumps above a certain value for > 0.4sec