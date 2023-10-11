import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import scipy.stats as stats

from scipy.io import wavfile  # outdated?
import librosa

from neuropy.core.signal import Signal
from neuropy.core.epoch import Epoch
from neuropy.utils.signal_process import filter_sig, hilbertfast
from neuropy.utils.mathutil import contiguous_regions

dir_use = Path(
    "/Users/nkinsky/Documents/UM/Working/Trace_FC/Recording_Rats/Finn/2022_01_20_training/shock_box"
)


class DeepSqueakIO:
    def __init__(self, filename):
        self.filename = filename
        self.usv_df = None

        self.load_calls(filename)

    def load_calls(self, filename, keys_ignore=["Type"]):
        """Loads calls from DeepSqueak that have been made MATLAB compatible by running the MATLAB function
        Calls2python located in the neuropy.io folder"""

        # Load in file and grab keys and data
        mat_in = loadmat(filename, simplify_cells=True)
        data = mat_in[
            list(mat_in.keys())[
                np.where(["data" in key for key in mat_in.keys()])[0][0]
            ]
        ].T
        keys = mat_in[
            list(mat_in.keys())[
                np.where(["keys" in key for key in mat_in.keys()])[0][0]
            ]
        ]

        df_list = []
        box_keys = ["Begin_Time", "Freq_Min", "Duration", "Freq_Range"]
        for d, key in zip(data, keys):
            if key in keys_ignore:
                continue
            data_array = np.vstack(d).squeeze()
            if key == "Box":
                df_list.append(pd.DataFrame(data_array, columns=box_keys))
            else:
                df_list.append(pd.DataFrame({key: data_array}))

        self.usv_df = pd.concat(df_list, axis=1)


def detect_tone(
    filename,
    freq_lims: list = [480, 520],
    smooth_window: float = 0.05,
    thresh: float = 5,
    tone_length: float = 0.5,
    tone_label:str = "start_tone",
    plot_check: bool = True,
):
    """Detect USV start tone - typically a 0.5 sec 500Hz pure tone"""
    """NRK todo: load in with a downsample. From https://stackoverflow.com/questions/30619740/downsampling-wav-audio-file
    import librosa    
    y, s = librosa.load('test.wav', sr=8000) # Downsample 44.1kHz to 8kHz
    The extra effort to install Librosa is probably worth the peace of mind.

    Pro-tip: when installing Librosa on Anaconda, you need to install ffmpeg as well, so

    pip install librosa
    conda install -c conda-forge ffmpeg"""

    # Load in file after downsampling to 2.5 x top frequency
    sr_ds = int(np.round(np.max(freq_lims)*2.5, -2))  # downsample to 2.5 x top frequency, rounded to 100s
    Audiodata, fs = librosa.load(filename, sr=sr_ds)
    audio_sig = Signal(Audiodata, fs)

    # Bandpass filter within frequency limits and also at half frequency limits to detect any general noise
    # bp = filter_sig.bandpass(audio_sig, freq_lims[0], freq_lims[1], order=2, fs=fs)
    # bp_half = filter_sig.bandpass(audio_sig, int(freq_lims[0]/2), int(freq_lims[1]/2), order=2, fs=fs)

    # Get power in range of interest and background
    power = np.abs(hilbertfast(filter_sig.bandpass(audio_sig.traces[0], lf=freq_lims[0], hf=freq_lims[1], fs=fs)))
    power_half = np.abs(hilbertfast(filter_sig.bandpass(audio_sig.traces[0], lf=freq_lims[0]/2, hf=freq_lims[1]/2, fs=fs)))

    # Zscore, smooth, and threshold power / power_half then look for epochs close to the length of the tone(s)
    power_ratio_sm = pd.Series(stats.zscore(power / power_half)).rolling(int(sr_ds * smooth_window)).mean().values
    tone_cands = contiguous_regions(power_ratio_sm > thresh)
    tone_cands = tone_cands[np.diff(tone_cands).squeeze()/sr_ds > 0.9*tone_length]/sr_ds  # give start and end of tones that meet threshold and time requirements

    if tone_cands.shape[0] == 0:
        print(f'No tones detected at thresh={thresh} and length={tone_length}. Adjust power/time thresholds and re-run')

    if plot_check:
        _, ax = plt.subplots(figsize=(6, 3))
        ax.set_title(f"{freq_lims[0]} Hz to {freq_lims[1]} Hz filtered signal")
        if fs > 2000:  # downsample for plotting to avoid laggy plots
            (ht,) = ax.plot(audio_sig.time[::1000], power_ratio_sm[::1000])
        else:
            (ht,) = ax.plot(audio_sig.time, power_ratio_sm)
        hthresh = ax.axhline(thresh, color="r")

        htone = None
        for tone_times in tone_cands:
            htone = ax.axvspan(tone_times[0], tone_times[1], color=[0, 0, 1, 0.3])

        if htone is not None:
            ax.legend([ht, hthresh, htone], ["Signal", "Threshold", "Detected Tone"])
        else:
            ax.legend(([ht, hthresh], ["Signal", "Threshold"]))
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Smoothed power ratio")

    return Epoch({"start": tone_cands[:, 0], "stop": tone_cands[:, 1], "label": tone_label})


if __name__ == "__main__":
    import matplotlib
    # Debug with different session - this one may have had many false starts! Enter in approximate time of start tone
    # from file too to limit search?  Need to create a csv file with start times for each event!
    # Should be easy!!! Grab start time from csv file and compare to start time from log file.
    matplotlib.use('TkAgg')

    base_dir = Path("/Users/nkinsky/Documents/UM/Working/Trace_FC/Recording_Rats/Finn/2022_01_21_recall1/1_tone_recall")
    detect_tone(sorted(base_dir.glob("**/*.WAV"))[0], freq_lims=(6900, 7100), tone_label="CS+")
    # base_dir = Path(
    #     "/data2/Trace_FC/Recording_Rats/Han/2022_08_03_training/2_training/USVdetections"
    # )
    # DeepSqueakIO(base_dir / "T0000001 2023-09-19  4_23 PM_copy_cell.mat")
    # time_start, time_end = detect_start_tone(dir_use / "T0000002.WAV")
# basically need to figure out how to take the above and threshold it and find out when it jumps above a certain value for > 0.4sec
