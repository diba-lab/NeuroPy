import pandas as pd
from scipy.io import wavfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from neuropy.core.signal import Signal
from neuropy.utils.signal_process import filter_sig

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


def detect_start_tone(
    filename,
    freq_lims: list = [480, 520],
    start_sec_use: int = 240,
    thresh=50,
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
    fs, Audiodata = wavfile.read(filename, mmap=True)
    audio_sig = Signal(Audiodata, fs)

    bp500 = filter_sig.bandpass(audio_sig, freq_lims[0], freq_lims[1], order=2, fs=fs)

    thresh_bool = np.abs(bp500.traces[0][0 : (start_sec_use * fs)]) > thresh

    time_start, time_end = np.where(thresh_bool)[0][[0, -1]] / fs

    if plot_check:
        _, ax = plt.subplots()
        ax.set_title(f"{freq_lims[0]} Hz to {freq_lims[1]} Hz filtered signal")
        (ht,) = ax.plot(
            np.linspace(0, start_sec_use, start_sec_use * 2000),
            np.abs(bp500.traces[0][0 : (start_sec_use * fs) : int(fs / 2000)]),
        )
        hthresh = ax.axhline(thresh, color="r")
        htone = ax.axvspan(time_start, time_end, color=[0, 0, 1, 0.3])
        ax.legend([ht, hthresh, htone], ["Signal", "Threshold", "Detected Tone"])
        ax.set_xlim([time_start - 3, time_end + 3])
        ax.set_xlabel("Time (sec)")

    return time_start, time_end


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')

    base_dir = Path("/Users/nkinsky/Documents/UM/Working/Trace_FC/Recording_Rats/Finn/2022_01_21_recall1/1_tone_recall")
    detect_start_tone(sorted(base_dir.glob("**/*.WAV"))[0], freq_lims=(6800, 7300), start_sec_use=8*60)
    # base_dir = Path(
    #     "/data2/Trace_FC/Recording_Rats/Han/2022_08_03_training/2_training/USVdetections"
    # )
    # DeepSqueakIO(base_dir / "T0000001 2023-09-19  4_23 PM_copy_cell.mat")
    # time_start, time_end = detect_start_tone(dir_use / "T0000002.WAV")
# basically need to figure out how to take the above and threshold it and find out when it jumps above a certain value for > 0.4sec
