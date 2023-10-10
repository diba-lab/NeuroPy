"""The class below works well but only for small files.
See notebooks/Miniscope_Denoise.ipynb for filtering via spyking-circus backbone that works for large files"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from pathlib import Path
import seaborn as sns
import scipy.signal as ssignal

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


def plot_miniscope_noise(
    signal,
    ch,
    block_sec=10,
    interval_sec=60,
    remove_disconnects=False,
    disconnect_thresh=20000,
    EWLnoise_range=(4835, 4855),
):
    assert isinstance(signal, core.Signal)

    f_full, Pxx_full, time = [], [], []
    nblocks = np.floor(signal.duration / interval_sec).astype(int)
    for id in range(nblocks):
        block_start = int(interval_sec * id * signal.sampling_rate)
        block_end = int(block_start + signal.sampling_rate * block_sec)
        f, Pxx = sg.welch(
            signal.traces[ch][block_start:block_end],
            fs=signal.sampling_rate,
            nperseg=signal.sampling_rate,
            scaling="spectrum",
        )
        f_full.append(f)
        Pxx_full.append(Pxx)
        time.append(block_start / signal.sampling_rate)

    f_full = np.asarray(f_full)
    Pxx_full = np.asarray(Pxx_full)

    # Quick and dirty method to remove disconnects - threshold excessive high frequency noise
    if remove_disconnects:
        freq_bool = np.bitwise_and(
            f_full[0] > EWLnoise_range[1], f_full[0] < EWLnoise_range[0]
        )
        good_epochs = Pxx_full[:, freq_bool].sum(axis=1) < 20000
        f_full = f_full[good_epochs]
        Pxx_full = Pxx_full[good_epochs]

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, nblocks))
    for fT, PxxT, color in zip(f_full, Pxx_full, colors):
        ax[0][0].plot(fT, PxxT, color=color)
    ax[0][0].set_xlabel("Freq (Hz)")
    ax[0][0].set_ylabel("PSD")

    # noise_limits = [[4835, 4855], [9670, 9700], [14510, 14550], [57, 63]]
    noise_limits = [
        np.array(EWLnoise_range),
        np.array(EWLnoise_range) * 2,
        np.array(EWLnoise_range) * 3,
        np.array([57, 63]),
    ]
    for a, lim in zip(ax.reshape(-1)[1:], noise_limits):
        freq_bool = np.bitwise_and(f > lim[0], f < lim[1])
        if np.any(freq_bool):
            sns.heatmap(Pxx_full[:, freq_bool].T, ax=a)
            a.set_yticks([0, freq_bool.sum()])
            a.set_yticklabels([str(f[freq_bool].min()), str(f[freq_bool].max())])
            a.set_xticks((0, nblocks))
            a.set(xticklabels=("0", str(time[-1])))
            a.set_xlabel("Time (30 sec blocks)")
            a.set_ylabel("Frez (Hz)")
        else:
            a.text(
                0.1,
                0.5,
                f"{lim[0]}-{lim[1]} Hz above Nyquist Frequency",
            )

    fig.suptitle("Miniscope Noise Tracking")

    return f_full, Pxx_full


def check_filter(notch_filter_comb, fs=30000, plot_final_only=True, figf=None):
    """Quick function to plot the filter that will be applied before you run it - if you see any values
    above 0 in the rightmost plot you will probably need to adjust your filters and run two serially."""

    w0all = [notch_filt_int["w0"] for notch_filt_int in notch_filter_comb]
    notch_filt_sort = [notch_filter_comb[ids] for ids in np.argsort(w0all)]
    bcomb, acomb = 1, 1
    for idf, filter in enumerate(notch_filt_sort):
        if filter["Q"] is None:
            assert (
                filter["bw"] is float or int
            ), "If Q is not specified, bw must be provided"
            Quse = np.round(filter["w0"] / filter["bw"])
        else:
            Quse = filter["Q"]
        b, a = ssignal.iirnotch(w0=filter["w0"], Q=Quse, fs=fs)
        bcomb = np.convolve(bcomb, b)
        acomb = np.convolve(acomb, a)
        freq, h = ssignal.freqz(b, a, fs=fs)
        freqcomb, hcomb = ssignal.freqz(bcomb, acomb, fs=fs)
        if not plot_final_only:
            fig, ax = plt.subplots(1, 3, figsize=(10, 3))
            fig.suptitle(f'wo = {filter["w0"]}')
            ax[0].plot(acomb)
            ax[0].plot(a)
            ax[0].set_title("acomb + a")
            ax[1].plot(bcomb)
            ax[1].plot(b)
            ax[1].set_title("bcomb + b")
            ax[2].plot(freq, 20 * np.log10(abs(h)), color="green")
            ax[2].plot(freqcomb, 20 * np.log10(abs(hcomb)), color="red")
            ax[2].set_title("ind + comb filt. response")
    if figf is None:
        figf, axf = plt.subplots(1, 3, figsize=(10, 3))
    else:
        axf = figf.get_axes()
    figf.suptitle("Final Combined Filter")
    axf[0].plot(acomb)
    axf[0].plot(a)
    axf[0].set_title("acomb + a")
    axf[1].plot(bcomb)
    axf[1].plot(b)
    axf[1].set_title("bcomb + b")
    axf[2].plot(freq, 20 * np.log10(abs(h)), color="green")
    axf[2].plot(freqcomb, 20 * np.log10(abs(hcomb)), color="red")
    axf[2].set_title("ind + comb filt. response")
