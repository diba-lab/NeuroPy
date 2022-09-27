"""The below code works well but only for small files.
See notebooks/Notch_Filtering_template.ipynb for filtering via spyking-circus backbone that works for large files"""

from dataclasses import dataclass
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as filtSig
import scipy.signal as sg
from joblib import Parallel, delayed
from scipy import fftpack, stats
from scipy.fftpack import next_fast_len
from scipy.ndimage import gaussian_filter
from pathlib import Path
import seaborn as sns

from ..plotting import Fig
from .. import core
from ..utils.signal_process import filter_sig
from ..io.binarysignalio import BinarysignalIO


class MiniscopeDenoise(BinarysignalIO):
    """Remove high frequency EWL noise + 60Hz noise"""

    def __init__(
        self,
        binarysig: BinarysignalIO,
        EWLfreq: int or float = 4843,
        EWLbw: int or float = 35,
        EWLharm_freq: int or float = 3 * 4843,
        EWLharm_bw: int or float = 25,
        sixtyhz_top_limit: int = 420,
        sixtyhz_bw=4,
    ):
        self.traces_filt = None
        self.EWL = {
            "w0": EWLfreq,
            "bw": EWLbw,
            "w0_harmonic": EWLharm_freq,
            "bw_harmonic": EWLharm_bw,
        }
        self.sixtyhz = {"top_limit": sixtyhz_top_limit, "bw": sixtyhz_bw}

        self.signal = binarysig.get_signal()
        self.dtype = binarysig.dtype
        self.source_file = Path(binarysig.source_file)

    def __str__(self) -> str:
        return (
            f"EWL frequency: {self.EWL['w0']:0.1f} Hz with {self.ELW['bw']:0.1f} Hz bandwidth \n"
            f"EWL harmonic frequency: {self.EWL['w0_harmonic']:0.1f} Hz with {self.ELW['bw_harmonic']:0.1f} Hz bandwidth \n"
            f"60Hz harmonic top limit: {self.sixtyhz['top_limit']:0.1f} Hz with {self.sixtyhz['bw']:0.1f} Hz bandwidth\n"
        )

    def write_filtered_file(self):
        if self.traces_filt is None:
            print("Data has not yet been filtered - run .denoise first")
            pass

        write_filename = self.source_file.with_name(
            self.source_file.stem + "_noise_filtered.dat"
        )

        if write_filename.exists():
            print(str(write_filename) + " already exists. Delete then try again")
            pass
        else:
            write_data = np.memmap(
                write_filename,
                dtype=self.dtype,
                mode="w+",
                shape=(len(self.signal.traces.reshape(-1))),
            )
            write_data[
                : len(self.signal.traces.reshape(-1))
            ] = self.traces_filt.T.reshape(-1)

            print("Filtered data written to " + str(write_filename))

    def denoise(
        self, type: str in ["EWL", "sixtyhz", "both"] = "both", plot_psd: bool = True
    ):

        # Figure out what to run!
        if type == "EWL":
            runEWL, run60 = True, False
        elif type == "sixtyhz":
            runEWL, run60 = False, True
        else:
            runEWL, run60 = True, True

        # Remove EWL + harmonic noise
        print("Removing EWL noise from traces")
        if runEWL:
            traces_filt = filter_sig.notch(
                self.signal.traces,
                w0=self.EWL["w0"],
                Q=None,
                bw=self.EWL["bw"],
                fs=self.signal.sampling_rate,
            )
            traces_filt = filter_sig.notch(
                traces_filt,
                w0=self.EWL["w0_harmonic"],
                Q=None,
                bw=self.EWL["bw_harmonic"],
                fs=self.signal.sampling_rate,
            )
        else:
            traces_filt = self.signal.traces
        print(f"Removing 60Hz noise up to {self.sixtyhz['top_limit']} Hz from traces")
        if run60:
            traces_filt = filter_sig.notch(
                traces_filt,
                w0=60,
                Q=None,
                bw=self.sixtyhz["bw"],
                fs=self.signal.sampling_rate,
            )
            harmonics = np.arange(180, self.sixtyhz["top_limit"] + 1, 120)
            for harmonic in harmonics:
                traces_filt = filter_sig.notch(
                    traces_filt,
                    w0=harmonic,
                    Q=None,
                    bw=self.sixtyhz["bw"],
                    fs=self.signal.sampling_rate,
                )

        print("Done filtering")

        if plot_psd:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            ch_use = np.round(self.signal.n_channels / 2).astype("int")
            fraw, Pxxraw = sg.welch(
                self.signal.traces[ch_use],
                fs=self.signal.sampling_rate,
                nperseg=self.signal.sampling_rate,
                scaling="spectrum",
            )
            ffilt, Pxxfilt = sg.welch(
                traces_filt[ch_use],
                fs=self.signal.sampling_rate,
                nperseg=self.signal.sampling_rate,
                scaling="spectrum",
            )
            for a in ax:
                (hraw,) = a.plot(fraw, Pxxraw)
                (hfilt,) = a.plot(ffilt, Pxxfilt)
                a.set_xlabel("Hz")
                a.set_ylabel("Power")
                sns.despine(ax=a)
            ax[1].set_xlim([0, 100])
            ax[0].legend((hraw, hfilt), ("Raw", "Filtered"))

        self.traces_filt = traces_filt.astype("int16")
        return traces_filt.astype("int16")
