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
import seaborn as sns
from scipy.interpolate import interp2d

try:
    from ..plotting import Fig
    from .. import core
except ImportError:
    from neuropy.plotting import Fig
    from neuropy import core
from .. import core

import seaborn as sns
from scipy.interpolate import interp2d

try:
    from ..plotting import Fig
    from .. import core
except ImportError:
    from neuropy.plotting import Fig
    from neuropy import core
from .. import core


class filter_sig:
    @staticmethod
    def bandpass(signal, lf, hf, fs=1250, order=3, ax=-1):

        if isinstance(signal, core.Signal):
            y = signal.traces
            nyq = 0.5 * signal.sampling_rate
            b, a = sg.butter(order, [lf / nyq, hf / nyq], btype="bandpass")
            yf = sg.filtfilt(b, a, y, axis=-1)
            yf = core.Signal(
                traces=yf,
                sampling_rate=signal.sampling_rate,
                t_start=signal.t_start,
                channel_id=signal.channel_id,
            )
        else:
            nyq = 0.5 * fs
            b, a = sg.butter(order, [lf / nyq, hf / nyq], btype="bandpass")
            yf = sg.filtfilt(b, a, signal, axis=ax)

        return yf

    @staticmethod
    def highpass(signal, cutoff, fs=1250, order=6, ax=-1):

        if isinstance(signal, core.Signal):
            y = signal.traces
            nyq = 0.5 * signal.sampling_rate
            b, a = sg.butter(order, cutoff / nyq, btype="highpass")
            yf = sg.filtfilt(b, a, y, axis=-1)
            yf = core.Signal(
                traces=yf,
                sampling_rate=signal.sampling_rate,
                t_start=signal.t_start,
                channel_id=signal.channel_id,
            )
        else:
            nyq = 0.5 * fs

            b, a = sg.butter(order, cutoff / nyq, btype="highpass")
            yf = sg.filtfilt(b, a, signal, axis=ax)

        return yf

    @staticmethod
    def lowpass(signal, cutoff, fs=1250, order=6, ax=-1):
        nyq = 0.5 * fs

        b, a = sg.butter(order, cutoff / nyq, btype="lowpass")
        yf = sg.filtfilt(b, a, signal, axis=ax)

        return yf

    @staticmethod
    def notch(
        signal: np.ndarray,
        w0: float or int,
        Q: float or int or None,
        bw: float or int or None = None,
        fs: int = 30000,
        ax: int = -1,
    ):
        """Runs a notch filter on your data. If Q is none, must enter bw (bandwidth) of noise to remove.
        See scipy.signal.iirnotch for more info on parameters."""
        if Q is None:
            assert bw is float or int, "If Q is not specified, bw must be provided"
            Quse = np.round(w0 / bw)
        else:
            Quse = Q
        b, a = sg.iirnotch(w0=w0, Q=Quse, fs=fs)
        try:
            yf = sg.filtfilt(b, a, signal, axis=ax)
        except np.core._exceptions._ArrayMemoryError:
            yf = []
            print(
                "signal array is too large for memory, filtering each channel independently"
            )
            for trace in signal:
                yf.append(sg.filtfilt(b, a, trace, axis=ax).astype("int16"))
            yf = np.asarray(yf, dtype="int16")

        return yf

    @staticmethod
    def delta(signal, fs=1250, order=3, ax=-1):
        return filter_sig.bandpass(signal, lf=0.5, hf=4, fs=fs, order=order, ax=ax)

    @staticmethod
    def theta(signal, fs=1250, order=3, ax=-1):
        return filter_sig.bandpass(signal, lf=4, hf=10, fs=fs, order=order, ax=ax)

    @staticmethod
    def spindle(signal, fs=1250, order=3, ax=-1):
        return filter_sig.bandpass(signal, lf=8, hf=16, fs=fs, order=order, ax=ax)

    @staticmethod
    def slowgamma(signal, fs=1250, order=3, ax=-1):
        return filter_sig.bandpass(signal, lf=25, hf=50, fs=fs, order=order, ax=ax)

    @staticmethod
    def mediumgamma(signal, fs=1250, order=3, ax=-1):
        return filter_sig.bandpass(signal, lf=60, hf=90, fs=fs, order=order, ax=ax)

    @staticmethod
    def fastgamma(signal, fs=1250, order=3, ax=-1):
        return filter_sig.bandpass(signal, lf=100, hf=140, fs=fs, order=order, ax=ax)

    @staticmethod
    def ripple(signal, fs=1250, order=3, ax=-1):
        return filter_sig.bandpass(signal, lf=150, hf=240, fs=fs, order=order, ax=ax)


def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    # print(freqs[:20])
    # freqs1 = np.linspace(0, 2048.0, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back,
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


class WaveletSg(core.Spectrogram):
    def __init__(
        self,
        signal: core.Signal,
        freqs,
        norm_sig=False,
        ncycles=7,
        sigma=None,
    ) -> None:
        """Wavelet spectrogram on core.Signal object

        Parameters
        ----------
        signal : core.Signal
            should have only a single channel
        freqs : np.array
            frequencies of query
        norm_sig : bool, optional
            whether to normalize the signal, by default True
        ncycles : int, optional
            number of cycles for wavelet,higher number gives better frequency resolution at higher frequencies, by default 7 cycles
        sigma : int, optional
            smoothing to apply on spectrum along time axis for each frequency trace, in units of seconds, by default None

        Suggestions/References
        ----------------------

        Wavelet :
            ncycles = 7, [Colgin et al. 2009, Tallon-Baudry et al. 1997]
            ncycles = 3, [MX Cohen, Analyzing neural time series data book, 2014]

        """

        assert signal.n_channels == 1, "signal should have only a single channel"
        trace = signal.traces[0]
        if norm_sig:
            trace = stats.zscore(trace)

        sxx = self._wt(trace, freqs, signal.sampling_rate, ncycles)
        sampling_rate = signal.sampling_rate

        if sigma is not None:
            # TODO it is slow for large array, maybe move to a method and use fastlen padding
            sampling_period = 1 / sampling_rate
            sxx = filtSig.gaussian_filter1d(sxx, sigma=sigma / sampling_period, axis=-1)

        super().__init__(
            traces=sxx, freqs=freqs, sampling_rate=sampling_rate, t_start=signal.t_start
        )

    def _wt(self, signal, freqs, fs, ncycles):
        """wavelet transform"""
        n = len(signal)
        fastn = next_fast_len(n)
        signal = np.pad(signal, (0, fastn - n), "constant", constant_values=0)
        signal = np.tile(signal, (len(freqs), 1))
        conv_val = np.zeros((len(freqs), n), dtype=complex)

        freqs = freqs[:, np.newaxis]
        t_wavelet = np.arange(-4, 4, 1 / fs)[np.newaxis, :]

        sigma = ncycles / (2 * np.pi * freqs)
        A = (sigma * np.sqrt(np.pi)) ** -0.5
        real_part = np.exp(-(t_wavelet ** 2) / (2 * sigma ** 2))
        img_part = np.exp(2j * np.pi * (t_wavelet * freqs))
        wavelets = A * real_part * img_part

        conv_val = sg.fftconvolve(signal, wavelets, mode="same", axes=-1)
        conv_val = np.asarray(conv_val)[:, :n]
        return np.abs(conv_val).astype("float32")


class FourierSg(core.Spectrogram):
    def __init__(
        self,
        signal: core.Signal,
        window=1,
        overlap=0.5,
        norm_sig=True,
        freqs=None,
        multitaper=False,
        sigma=None,
    ) -> None:
        """Forier spectrogram on core.Signal object

        Parameters
        ----------
        signal : core.Signal
            should have only a single channel
        norm_sig : bool, optional
            whether to normalize the signal, by default True
        window : float, optional
            length of each segment in seconds, ignored if using wavelet method, by default 1 s
        overlap : float, optional
            length of overlap between adjacent segments, by default 0.5
        freqs : np.array
            If provided, the spectrogram will use interpolation to evaluate at these frequencies
        multitaper: bool,
            whether to use multitaper for estimation, by default False
        sigma : int, optional
            smoothing to applied on spectrum, in units of seconds, by default 2 s

        NOTE: time is center of windows

        """

        assert signal.n_channels == 1, "signal should have only a single channel"
        trace = signal.traces[0]
        if norm_sig:
            trace = stats.zscore(trace)

        if multitaper:
            sxx, freqs, t = self._ft(
                trace, signal.sampling_rate, window, overlap, mt=True
            )
        else:
            sxx, f, t = self._ft(trace, signal.sampling_rate, window, overlap)

            if freqs is not None:
                func_sxx = interp2d(t, f, sxx)
                sxx = func_sxx(t, freqs)
                f = freqs

        sampling_rate = 1 / (t[1] - t[0])

        if sigma is not None:
            # TODO it is slow for large array, maybe move to a method and use fastlen padding
            sampling_period = 1 / sampling_rate
            sxx = filtSig.gaussian_filter1d(sxx, sigma=sigma / sampling_period, axis=-1)

        super().__init__(
            traces=sxx,
            sampling_rate=sampling_rate,
            freqs=f,
            t_start=signal.t_start + t[0],
        )

    def _ft(self, signal, fs, window, overlap, mt=False):
        """fourier transform"""
        window = int(window * fs)
        overlap = int(overlap * fs)

        f = None
        if mt:
            tapers = sg.windows.dpss(M=window, NW=5, Kmax=6)

            sxx_taper = []
            for taper in tapers:
                f, t, sxx = sg.spectrogram(
                    signal, window=taper, fs=fs, noverlap=overlap
                )
                sxx_taper.append(sxx)
            sxx = np.dstack(sxx_taper).mean(axis=2)

        else:
            f, t, sxx = sg.spectrogram(signal, fs=fs, nperseg=window, noverlap=overlap)

        return sxx, f, t


def hilbertfast(arr, ax=-1):

    """inputs a signal does padding to next power of 2 for faster computation of hilbert transform

    Arguments:
        signal {array} -- [n, dimensional array]

    Returns:
        [type] -- [description]
    """
    signal_length = arr.shape[-1]
    hilbertsig = sg.hilbert(arr, fftpack.next_fast_len(signal_length), axis=ax)

    if np.ndim(arr) > 1:
        hilbertsig = hilbertsig[:, :signal_length]
    else:
        hilbertsig = hilbertsig[:signal_length]

    return hilbertsig


@dataclass
class bicoherence:

    """Generate bicoherence matrix for signal

    Attributes:
    ---------------
        flow: int, low frequency
        fhigh: int, highest frequency
        window: int, segment size
        noverlap:

        bicoher (freq_req x freq_req, array): bicoherence matrix
        freq {array}: frequencies at which bicoherence was calculated
        bispec:
        significance:

    Methods:
    ---------------
    compute
        calculates the bicoherence
    plot
        plots bicoherence matrix in the provided axis

    References:
    -----------------------
    1) Sheremet, A., Burke, S. N., & Maurer, A. P. (2016). Movement enhances the nonlinearity of hippocampal theta. Journal of Neuroscience, 36(15), 4218-4230.
    """

    flow: int = 1
    fhigh: int = 150
    fs: int = 1250
    window: int = 4 * 1250
    overlap: int = 2 * 1250

    def compute(self, signal: np.array):
        """Computes bicoherence

        Parameters
        -----------
            signal: array,
                lfp signal on which bicoherence will be calculated
        """

        # ------ changing dimensions if a single lfp is provided---------
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]

        # ----- subtracting mean and complex spectrogram -------------
        signal = signal - np.mean(signal, axis=-1, keepdims=True)  # zero mean
        f, _, sxx = sg.stft(
            signal,
            window="hann",
            nperseg=self.window,
            noverlap=self.overlap,
            fs=self.fs,
            # mode="complex",
            nfft=fftpack.next_fast_len(self.window),
            # detrend=False,
            # scaling="spectrum",
        )
        sxx = np.require(sxx, dtype=complex)
        sxx = sxx / self.window  # scaling the fourier transform

        # ------ Getting required frequencies and their indices -------------
        freq_ind = np.where((f > self.flow) & (f < self.fhigh))[0]
        freq_req = f[freq_ind]

        """ ===========================================
        bispectrum = |mean( X(f1) * X(f2) * conj(X(f1+f2)) )|
        normalization = sqrt( mean(|X(f1) * X(f2)|^2) * mean(|X(f1+f2)|^2) )

        where,
            X is complex spectrogram
            P is real/absolute square spectrogram
        ================================================="""

        X_f2 = sxx[:, freq_ind, :]  # complex spectrogram of required frequencies
        P_f2 = np.abs(X_f2) ** 2  # absolute square of spectrogram

        def bicoh_product(f_ind):
            X_f1 = sxx[:, f_ind, np.newaxis, :]
            X_f1f2 = sxx[:, freq_ind + f_ind, :]

            # ----- bispectrum triple product --------------
            bispec_freq = np.mean((X_f1 * X_f2) * np.conjugate(X_f1f2), axis=-1)

            # ----- normalization to calculate bicoherence ---------
            P_f1 = np.abs(X_f1) ** 2
            P_f1f2 = np.abs(X_f1f2) ** 2
            norm = np.sqrt(
                np.mean(P_f1, axis=-1)
                * np.mean(P_f2, axis=-1)
                * np.mean(P_f1f2, axis=-1)
            )
            # norm = np.sqrt(
            #     np.mean(np.abs(X_f1 * X_f2) ** 2, axis=-1) * np.mean(P_f1f2, axis=-1)
            # )

            return bispec_freq / norm

        bispec = Parallel(n_jobs=10, require="sharedmem")(
            delayed(bicoh_product)(f_ind) for f_ind in freq_ind
        )

        bispec = np.dstack(bispec)
        bicoher = np.abs(bispec)

        self.bicoher = bicoher.squeeze()
        self.bispec = bispec.squeeze()
        self.freq = freq_req
        self.freq_ind = freq_ind
        self.dof = 2 * sxx.shape[-1]
        self.significance = np.sqrt(6 / self.dof)  # 95 percentile

        return bicoher

    def plot(self, index=None, ax=None, smooth=2, **kwargs):

        if index is None:
            bic = self.bicoher
        else:
            bic = self.bicoher[index].copy()

        lt = np.tril_indices_from(bic, k=-1)
        bic[lt] = 0
        bic[(lt[0], -lt[1])] = 0
        bic[bic < self.significance] = 0

        if smooth is not None:
            bic = gaussian_filter(bic, sigma=smooth)

        if ax is None:
            _, ax = plt.subplots(1, 1)

        bicoh_plt = ax.pcolormesh(self.freq, self.freq, bic, **kwargs)
        ax.set_ylim([0, np.max(self.freq) / 2])
        ax.plot(
            [1, np.max(self.freq / 2)],
            [1, np.max(self.freq) / 2],
            "gray",
        )
        ax.plot(
            [np.max(self.freq) / 2, np.max(self.freq)],
            [np.max(self.freq) / 2, 1],
            "gray",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Frequency (Hz)")
        cb = plt.colorbar(bicoh_plt)
        cb.outline.set_linewidth(0.5)


def power_correlation(signal, fs=1250, window=2, overlap=1, fband=None):
    """Power power correlation between frequencies

    Parameters
    ----------
    signal : [type]
        timeseries for which to calculate
    fs : int, optional
        sampling frequency of the signal, by default 1250
    window : int, optional
        window size for calculating spectrogram, by default 2
    overlap : int, optional
        overlap between adjacent windows, by default 1
    fband : [type], optional
        return correlations between these frequencies only, by default None

    Returns
    -------
    [type]
        [description]
    """
    f, t, sxx = sg.spectrogram(
        signal, fs=fs, nperseg=int(window * fs), noverlap=(overlap * fs)
    )
    if fband is not None:
        assert len(fband) == 2, "fband length should of length of 2"
        f_req_ind = np.where((1 < f) & (f < 100))[0]
        f_req = f[f_req_ind]
        corr_freq = np.corrcoef(sxx[f_req_ind, :])
    else:
        corr_freq = np.corrcoef(sxx)

    np.fill_diagonal(corr_freq, val=0)

    return f_req, corr_freq


@dataclass
class Csd:
    lfp: np.array
    coords: np.array
    chan_label: np.array = None
    fs: int = 1250

    def classic(self):
        coords = self.coords.copy()
        nframes = self.lfp.shape[1]
        csd = -(self.lfp[:-2, :] - 2 * self.lfp[1:-1, :] + self.lfp[2:, :])
        self.csdmap = stats.zscore(csd, axis=None)
        self.csd_coords = coords[1:-1]
        self.time = np.linspace(-1, 1, nframes) * (nframes / self.fs)

    def icsd(self, lfp, coords):
        pass

    def plot(self, ax=None, smooth=3, plotLFP=False, **kwargs):

        if smooth is not None:
            csdmap = gaussian_filter(self.csdmap, sigma=smooth)
        else:
            csdmap = self.csdmap.copy()

        gs = None
        if ax is None:
            figure = Fig()
            _, gs = figure.draw(grid=[1, 1])
            ax = plt.subplot(gs[0])

        ax.pcolormesh(self.time, self.csd_coords, csdmap, **kwargs)
        ax.set_title("Current source density map")
        ax.set_ylabel("y Coordinates")
        ax.set_xlabel("Time (s)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", which="both", length=0)

        if plotLFP:
            lfp = stats.zscore(self.lfp, axis=None) * np.mean(np.diff(self.coords)) / 2
            ax.plot(self.time, lfp.T + self.coords, "gray", lw=1)


def mtspect(signal, nperseg, noverlap, fs=1250):
    window = nperseg
    overlap = noverlap

    tapers = sg.windows.dpss(M=window, NW=5, Kmax=6)

    psd_taper = []
    for taper in tapers:
        f, psd = sg.welch(signal, window=taper, fs=fs, noverlap=overlap)
        psd_taper.append(psd)
    psd = np.asarray(psd_taper).mean(axis=0)

    return f, psd


@dataclass
class PAC:
    """Phase amplitude coupling

    Attributes
    ----------

    Methods
    -------
    compute(lfp)
        calculates phase amplitude coupling
    """

    fphase: tuple = (4, 12)
    famp: tuple = (25, 50)
    binsz: int = 9

    def compute(self, lfp):
        self.angle_bin = np.linspace(0, 360, 360 // self.binsz + 1)
        self.phase_center = self.angle_bin[:-1] + self.binsz / 2
        phase_lfp = stats.zscore(
            filter_sig.bandpass(lfp, lf=self.fphase[0], hf=self.fphase[1])
        )
        amp_lfp = stats.zscore(
            filter_sig.bandpass(lfp, lf=self.famp[0], hf=self.famp[1])
        )

        hil_phaselfp = hilbertfast(phase_lfp)
        hil_amplfp = hilbertfast(amp_lfp)
        amplfp_amp = np.abs(hil_amplfp)

        phaselfp_angle = np.angle(hil_phaselfp, deg=True) + 180

        mean_amplfp = stats.binned_statistic(
            phaselfp_angle, amplfp_amp, bins=self.angle_bin
        )[0]

        self.pac = mean_amplfp / np.sum(mean_amplfp)

    def comodulo(self, lfp, method="tort", njobs=5):
        """comodulogram for frequencies of interest"""

        if method == "tort":
            phase_lfp = filter_sig.bandpass(lfp, lf=4, hf=5)

    def plot(self, ax=None, **kwargs):
        """Bar plot for phase amplitude coupling

        Parameters
        ----------
        ax : axis object, optional
            axis to plot into, by default None
        kwargs : other keyword arguments
            arguments are to plt.bar()

        Returns
        -------
        ax : matplotlib axes
            Axes object with the heatmap
        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        ax.bar(
            np.concatenate((self.phase_center, self.phase_center + 360)),
            np.concatenate((self.pac, self.pac)),
            width=self.binsz,
            **kwargs,
        )
        ax.set_xlabel("Phase (degrees)")
        ax.set_ylabel("Amplitude")

        return ax


@dataclass
class ThetaParams:
    """Estimating various theta oscillation features like phase, asymmetry etc.

    References
    -------
    1) hilbert --> Cole, Scott, and Bradley Voytek. "Cycle-by-cycle analysis of neural oscillations." Journal of neurophysiology (2019)
    2) waveshape --> Belluscio, Mariano A., et al. "Cross-frequency phase–phase coupling between theta and gamma oscillations in the hippocampus." Journal of Neuroscience(2012)
    """

    lfp: np.array
    fs: int = 1250
    method: str = "hilbert"

    def __post_init__(self):
        # --- calculating theta parameters from broadband theta---------
        eegSrate = self.fs

        peak, trough, theta_amp, theta_360, thetalfp = 5 * [None]
        if self.method == "hilbert":
            thetalfp = filter_sig.bandpass(self.lfp, lf=1, hf=25)
            hil_theta = hilbertfast(thetalfp)
            theta_360 = np.angle(hil_theta, deg=True) + 180
            theta_angle = np.abs(np.angle(hil_theta, deg=True))
            trough = sg.find_peaks(theta_angle)[0]
            peak = sg.find_peaks(-theta_angle)[0]
            theta_amp = np.abs(hil_theta) ** 2

        elif self.method == "waveshape":
            thetalfp = filter_sig.bandpass(self.lfp, lf=1, hf=60)
            hil_theta = hilbertfast(thetalfp)
            theta_amp = np.abs(hil_theta) ** 2
            # distance between theta peaks should be >= 80 ms
            distance = int(0.08 * self.fs)

            peak = sg.find_peaks(thetalfp, height=0, distance=distance)[0]
            trough = stats.binned_statistic(
                np.arange(len(thetalfp)), thetalfp, bins=peak, statistic=np.argmin
            )[0]
            trough = peak[:-1] + trough

            def get_desc(arr):
                arr = stats.zscore(arr)
                return np.where(np.diff(np.sign(arr)))[0][0]

            def get_asc(arr):
                arr = stats.zscore(arr)
                return np.where(np.diff(np.sign(arr)))[0][-1]

            zero_up = stats.binned_statistic(
                np.arange(len(thetalfp)), thetalfp, bins=peak, statistic=get_asc
            )[0]

            zero_down = stats.binned_statistic(
                np.arange(len(thetalfp)), thetalfp, bins=peak, statistic=get_desc
            )[0]

            # ---- linear interpolation of angles ---------
            loc = np.concatenate((trough, peak))
            angles = np.concatenate(
                (
                    np.zeros(len(trough)),
                    # 90 * np.ones(len(zero_up)),
                    180 * np.ones(len(peak)),
                    # 270 * np.ones(len(zero_down)),
                )
            )
            sort_ind = np.argsort(loc)
            loc = loc[sort_ind]
            angles = angles[sort_ind]
            theta_angle = np.interp(np.arange(len(self.lfp)), loc, angles)
            angle_descend = np.where(np.diff(theta_angle) < 0)[0]
            theta_angle[angle_descend] = -theta_angle[angle_descend] + 360
            theta_360 = theta_angle

        else:
            print("method not understood")

        if peak[0] < trough[0]:
            peak = peak[1:]
        if trough[-1] > peak[-1]:
            trough = trough[:-1]

        assert len(trough) == len(peak)

        rising_time = (peak[1:] - trough[1:]) / eegSrate
        falling_time = (trough[1:] - peak[:-1]) / eegSrate

        self.amp = theta_amp
        self.angle = theta_360
        self.trough = trough
        self.peak = peak
        self.lfp_filtered = thetalfp
        self.rise_time = rising_time
        self.fall_time = falling_time

    @property
    def rise_mid(self):
        theta_trough = self.trough
        theta_peak = self.peak
        thetalfp = self.lfp_filtered
        rise_midpoints = np.array(
            [
                trough
                + np.argmin(
                    np.abs(
                        thetalfp[trough:peak]
                        - (
                            max(thetalfp[trough:peak])
                            - np.ptp(thetalfp[trough:peak]) / 2
                        )
                    )
                )
                for (trough, peak) in zip(theta_trough, theta_peak)
            ]
        )
        return rise_midpoints

    @property
    def fall_mid(self):
        theta_peak = self.peak
        theta_trough = self.trough
        thetalfp = self.lfp_filtered
        fall_midpoints = np.array(
            [
                peak
                + np.argmin(
                    np.abs(
                        thetalfp[peak:trough]
                        - (
                            max(thetalfp[peak:trough])
                            - np.ptp(thetalfp[peak:trough]) / 2
                        )
                    )
                )
                for (peak, trough) in zip(theta_peak[:-1], theta_trough[1:])
            ]
        )

        return fall_midpoints

    @property
    def peak_width(self):
        return (self.fall_mid - self.rise_mid[:-1]) / self.fs

    @property
    def trough_width(self):
        return (self.rise_mid[1:] - self.fall_mid) / self.fs

    @property
    def asymmetry(self):
        return self.rise_time / (self.rise_time + self.fall_time)

    @property
    def peaktrough(self):
        return self.peak_width / (self.peak_width + self.trough_width)

    def break_by_phase(self, y, binsize=20, slideby=None):
        """Breaks y into theta phase specific components

        Parameters
        ----------
        lfp : array like
            reference lfp from which theta phases are estimated
        y : array like
            timeseries which is broken into components
        binsize : int, optional
            width of each bin in degrees, by default 20
        slideby : int, optional
            slide each bin by this amount in degrees, by default None

        Returns
        -------
        [list]
            list of broken signal into phase components
        """

        assert len(self.lfp) == len(y), "Both signals should be of same length"
        angle_bin = np.arange(0, 360 - binsize, slideby)
        if slideby is None:
            slideby = binsize
            angle_bin = np.arange(0, 360 - binsize, slideby)
        angle_centers = angle_bin + binsize / 2

        y_at_phase = []
        for phase in angle_bin:
            y_at_phase.append(
                y[np.where((self.angle >= phase) & (self.angle < phase + binsize))[0]]
            )

        return y_at_phase, angle_bin, angle_centers

    def sanityCheck(self):
        """Plots raw signal with filtered signal and peak, trough locations with phase

        Returns
        -------
        ax : obj
        """

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 8))

        ax[0].plot(stats.zscore(self.lfp), "k", label="raw")
        ax[0].plot(stats.zscore(self.lfp_filtered), "r", label="filtered")
        ax[0].vlines(
            self.peak,
            ymin=-5,
            ymax=5,
            colors="green",
            linestyles="dashed",
            label="peak",
        )

        ax[0].vlines(
            self.trough,
            ymin=-5,
            ymax=5,
            colors="blue",
            linestyles="dashed",
            label="trough",
        )
        ax[0].set_ylabel("Amplitude")
        ax[0].legend()

        ax[1].plot(self.angle, "k")
        ax[1].set_xlabel("frame number")
        ax[1].set_ylabel("Phase")

        fig.suptitle(f"Theta parameters estimation using {self.method}")

        return ax


def psd_auc(signal: core.Signal, freq_band: tuple, window=10, overlap=5):
    """Calculates area under the power spectrum for a given frequency band

    Parameters
    ----------
    eeg : [array]
        channels x time, has to be two dimensional

    Returns
    -------
    [type]
        [description]
    """

    assert isinstance(
        signal, core.Signal
    ), "signal should be a neuropy.core.Signal object"

    fs = signal.sampling_rate
    aucChans = []
    for sig in signal.traces:

        f, pxx = sg.welch(
            stats.zscore(sig),
            fs=fs,
            nperseg=int(window * fs),
            noverlap=int(overlap * fs),
            axis=-1,
        )
        f_theta = np.where((f > freq_band[0]) & (f < freq_band[1]))[0]
        area_in_freq = np.trapz(pxx[f_theta], x=f[f_theta])

        aucChans.append(area_in_freq)

    return aucChans


def hilbert_amplitude_stat(signals, freq_band, fs, statistic="mean"):
    """Calculates hilbert amplitude statistic over the entire signal

    Parameters
    ----------
    signals : list of signals or np.array
        [description]
    statistic : str, optional
        [description], by default "mean"

    Returns
    -------
    [type]
        [description]
    """

    if statistic == "mean":
        get_stat = lambda x: np.mean(x)
    if statistic == "median":
        get_stat = lambda x: np.median(x)
    if statistic == "std":
        get_stat = lambda x: np.std(x)

    bandpower_stat = np.zeros(len(signals))
    for i, sig in enumerate(signals):
        filtered = filter_sig.bandpass(sig, lf=freq_band[0], hf=freq_band[1], fs=fs)
        amplitude_envelope = np.abs(hilbertfast(filtered))
        bandpower_stat[i] = get_stat(amplitude_envelope)

    return bandpower_stat


def theta_phase_specfic_extraction(signal, y, fs, binsize=20, slideby=None):
    """Breaks y into theta phase specific components

    Parameters
    ----------
    signal : array like
        reference lfp from which theta phases are estimated
    y : array like
        timeseries which is broken into components
    binsize : int, optional
        width of each bin in degrees, by default 20
    slideby : int, optional
        slide each bin by this amount in degrees, by default None

    Returns
    -------
    [list]
        list of broken signal into phase components
    """

    assert len(signal) == len(y), "Both signals should be of same length"
    thetalfp = filter_sig.bandpass(signal, lf=1, hf=25, fs=fs)
    hil_theta = hilbertfast(thetalfp)
    theta_angle = np.angle(hil_theta, deg=True) + 180  # range from 0-360 degree

    if slideby is None:
        slideby = binsize

    # --- sliding windows--------
    angle_bin = np.arange(0, 361)
    slide_angles = np.lib.stride_tricks.sliding_window_view(angle_bin, binsize)[
        ::slideby, :
    ]
    angle_centers = np.mean(slide_angles, axis=1)

    y_at_phase = []
    for phase in slide_angles:
        y_at_phase.append(
            y[np.where((theta_angle >= phase[0]) & (theta_angle <= phase[-1]))[0]]
        )

    return y_at_phase, angle_bin, angle_centers


def irasa(
    data,
    sf=None,
    ch_names=None,
    band=(1, 30),
    hset=np.arange(1.1, 1.95, 0.05),
    return_fit=False,
    win_sec=4,
    kwargs_welch=dict(average="median", window="hamming"),
):

    """
    Separate the aperiodic (= fractal, or 1/f) and oscillatory component of the
    power spectra of EEG data using the IRASA method.
    .. versionadded:: 0.1.7

    Copyright (c) 2018, Raphael Vallat
    All rights reserved.

    Parameters
    ----------
    data : :py:class:`numpy.ndarray` or :py:class:`mne.io.BaseRaw`
        1D or 2D EEG data. Can also be a :py:class:`mne.io.BaseRaw`, in which
        case ``data``, ``sf``, and ``ch_names`` will be automatically
        extracted, and ``data`` will also be converted from Volts (MNE default)
        to micro-Volts (YASA).
    sf : float
        The sampling frequency of data AND the hypnogram.
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...]. If None,
        channels will be labelled ['CHAN001', 'CHAN002', ...].
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    band : tuple or None
        Broad band frequency range.
        Default is 1 to 30 Hz.
    hset : :py:class:`numpy.ndarray`
        Resampling factors used in IRASA calculation. Default is to use a range
        of values from 1.1 to 1.9 with an increment of 0.05.
    return_fit : boolean
        If True (default), fit an exponential function to the aperiodic PSD
        and return the fit parameters (intercept, slope) and :math:`R^2` of
        the fit.
        The aperiodic signal, :math:`L`, is modeled using an exponential
        function in semilog-power space (linear frequencies and log PSD) as:
        .. math:: L = a + \\text{log}(F^b)
        where :math:`a` is the intercept, :math:`b` is the slope, and
        :math:`F` the vector of input frequencies.
    win_sec : int or float
        The length of the sliding window, in seconds, used for the Welch PSD
        calculation. Ideally, this should be at least two times the inverse of
        the lower frequency of interest (e.g. for a lower frequency of interest
        of 0.5 Hz, the window length should be at least 2 * 1 / 0.5 =
        4 seconds).
    kwargs_welch : dict
        Optional keywords arguments that are passed to the
        :py:func:`scipy.signal.welch` function.
    Returns
    -------
    freqs : :py:class:`numpy.ndarray`
        Frequency vector.
    psd_aperiodic : :py:class:`numpy.ndarray`
        The fractal (= aperiodic) component of the PSD.
    psd_oscillatory : :py:class:`numpy.ndarray`
        The oscillatory (= periodic) component of the PSD.
    fit_params : :py:class:`pandas.DataFrame` (optional)
        Dataframe of fit parameters. Only if ``return_fit=True``.
    Notes
    -----
    The Irregular-Resampling Auto-Spectral Analysis (IRASA) method is
    described in Wen & Liu (2016). In a nutshell, the goal is to separate the
    fractal and oscillatory components in the power spectrum of EEG signals.
    The steps are:
    1. Compute the original power spectral density (PSD) using Welch's method.
    2. Resample the EEG data by multiple non-integer factors and their
       reciprocals (:math:`h` and :math:`1/h`).
    3. For every pair of resampled signals, calculate the PSD and take the
       geometric mean of both. In the resulting PSD, the power associated with
       the oscillatory component is redistributed away from its original
       (fundamental and harmonic) frequencies by a frequency offset that varies
       with the resampling factor, whereas the power solely attributed to the
       fractal component remains the same power-law statistical distribution
       independent of the resampling factor.
    4. It follows that taking the median of the PSD of the variously
       resampled signals can extract the power spectrum of the fractal
       component, and the difference between the original power spectrum and
       the extracted fractal spectrum offers an approximate estimate of the
       power spectrum of the oscillatory component.
    Note that an estimate of the original PSD can be calculated by simply
    adding ``psd = psd_aperiodic + psd_oscillatory``.
    For an example of how to use this function, please refer to
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/11_IRASA.ipynb
    References
    ----------
    .. [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
           Components in the Power Spectrum of Neurophysiological Signal.
           Brain Topography, 29(1), 13–26.
           https://doi.org/10.1007/s10548-015-0448-0
    .. [2] https://github.com/fieldtrip/fieldtrip/blob/master/specest/
    .. [3] https://github.com/fooof-tools/fooof
    .. [4] https://www.biorxiv.org/content/10.1101/299859v1
    """
    import fractions

    # Check if input data is a MNE Raw object

    # Safety checks
    assert isinstance(data, np.ndarray), "Data must be a numpy array."
    data = np.atleast_2d(data)
    assert data.ndim == 2, "Data must be of shape (nchan, n_samples)."
    nchan, npts = data.shape
    assert nchan < npts, "Data must be of shape (nchan, n_samples)."
    assert sf is not None, "sf must be specified if passing a numpy array."
    assert isinstance(sf, (int, float))
    if ch_names is None:
        ch_names = ["CHAN" + str(i + 1).zfill(3) for i in range(nchan)]
    else:
        ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
        assert ch_names.ndim == 1, "ch_names must be 1D."
        assert len(ch_names) == nchan, "ch_names must match data.shape[0]."

    # Check the other arguments
    hset = np.asarray(hset)
    assert hset.ndim == 1, "hset must be 1D."
    assert hset.size > 1, "2 or more resampling fators are required."
    hset = np.round(hset, 4)  # avoid float precision error with np.arange.
    band = sorted(band)
    assert band[0] > 0, "first element of band must be > 0."
    assert band[1] < (sf / 2), "second element of band must be < (sf / 2)."
    win = int(win_sec * sf)  # nperseg

    # Calculate the original PSD over the whole data
    freqs, psd = sg.welch(data, sf, nperseg=win, **kwargs_welch)

    # Start the IRASA procedure
    psds = np.zeros((len(hset), *psd.shape))

    for i, h in enumerate(hset):
        # Get the upsampling/downsampling (h, 1/h) factors as integer
        rat = fractions.Fraction(str(h))
        up, down = rat.numerator, rat.denominator
        # Much faster than FFT-based resampling
        data_up = sg.resample_poly(data, up, down, axis=-1)
        data_down = sg.resample_poly(data, down, up, axis=-1)
        # Calculate the PSD using same params as original
        freqs_up, psd_up = sg.welch(data_up, h * sf, nperseg=win, **kwargs_welch)
        freqs_dw, psd_dw = sg.welch(data_down, sf / h, nperseg=win, **kwargs_welch)
        # Geometric mean of h and 1/h
        psds[i, :] = np.sqrt(psd_up * psd_dw)

    # Now we take the median PSD of all the resampling factors, which gives
    # a good estimate of the aperiodic component of the PSD.
    psd_aperiodic = np.median(psds, axis=0)
    print(psd_aperiodic.shape, psd_aperiodic[:2])

    # We can now calculate the oscillations (= periodic) component.
    psd_osc = psd - psd_aperiodic

    # Let's crop to the frequencies defined in band
    mask_freqs = np.ma.masked_outside(freqs, *band).mask
    freqs = freqs[~mask_freqs]
    psd_aperiodic = np.compress(~mask_freqs, psd_aperiodic, axis=-1)
    psd_osc = np.compress(~mask_freqs, psd_osc, axis=-1)

    if return_fit:
        # Aperiodic fit in semilog space for each channel
        from scipy.optimize import curve_fit

        intercepts, slopes, r_squared = [], [], []

        def func(t, a, b):
            # See https://github.com/fooof-tools/fooof
            return a + np.log(t ** b)

        for y in np.atleast_2d(psd_aperiodic):
            y_log = np.log(y)
            # Note that here we define bounds for the slope but not for the
            # intercept.
            popt, pcov = curve_fit(
                func, freqs, y_log, p0=(2, -1), bounds=((-np.inf, -10), (np.inf, 2))
            )
            intercepts.append(popt[0])
            slopes.append(popt[1])
            # Calculate R^2: https://stackoverflow.com/q/19189362/10581531
            residuals = y_log - func(freqs, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
            r_squared.append(1 - (ss_res / ss_tot))

        # Create fit parameters dataframe
        fit_params = {
            "Chan": ch_names,
            "Intercept": intercepts,
            "Slope": slopes,
            "R^2": r_squared,
            "std(osc)": np.std(psd_osc, axis=-1, ddof=1),
        }
        return freqs, psd_aperiodic, psd_osc, pd.DataFrame(fit_params)
    else:
        return freqs, psd_aperiodic, psd_osc


def plot_miniscope_noise(
    signal,
    ch,
    block_sec=10,
    interval_sec=60,
    remove_disconnects=False,
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
        sns.heatmap(Pxx_full[:, freq_bool].T, ax=a)
        a.set_yticks([0, freq_bool.sum()])
        a.set_yticklabels([str(f[freq_bool].min()), str(f[freq_bool].max())])
        a.set_xticks((0, nblocks))
        a.set(xticklabels=("0", str(time[-1])))
        a.set_xlabel("Time (30 sec blocks)")
        a.set_ylabel("Frez (Hz)")

    fig.suptitle("Miniscope Noise Tracking")

    return f_full, Pxx_full


if __name__ == "__main__":
    pass
