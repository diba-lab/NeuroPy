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

from ..plotting import Fig
from .. import core


class filter_sig:
    @staticmethod
    def bandpass(signal, hf, lf, fs=1250, order=3, ax=-1):
        nyq = 0.5 * fs

        b, a = sg.butter(order, [lf / nyq, hf / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=ax)

        return yf

    @staticmethod
    def highpass(signal, cutoff, fs=1250, order=6, ax=-1):
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


class SpectrogramBands:
    def __init__(
        self,
        signal: core.Signal,
        window: float = 1,
        overlap=0.5,
        smooth=None,
        multitaper=False,
        norm_sig=False,
    ):

        assert signal.n_channels == 1, "signal should have only one trace"
        fs = signal.sampling_rate
        window = int(window * fs)
        overlap = int(overlap * fs)

        sig = signal.traces[0]
        if norm_sig:
            sig = stats.zscore(signal.traces[0])

        f = None
        if multitaper:
            tapers = sg.windows.dpss(M=window, NW=5, Kmax=6)

            sxx_taper = []
            for taper in tapers:
                f, t, sxx = sg.spectrogram(sig, window=taper, fs=fs, noverlap=overlap)
                sxx_taper.append(sxx)
            sxx = np.dstack(sxx_taper).mean(axis=2)

        else:
            f, t, sxx = sg.spectrogram(sig, fs=fs, nperseg=window, noverlap=overlap)

        if smooth is not None:
            sxx = filtSig.gaussian_filter1d(sxx, sigma=smooth, axis=-1)

        self.freq = f
        self.time = t + signal.t_start
        self.sxx = sxx
        self.smooth = smooth

    def get_band_power(self, f1=None, f2=None):

        if f1 is None:
            f1 = self.freq[0]

        if f2 is None:
            f2 = self.freq[-1]

        assert f1 >= self.freq[0], "f1 should be greater than lowest frequency"
        assert f2 <= self.freq[-1], "f2 should be lower than highest possible frequency"
        assert f2 > f1, "f2 should be greater than f1"

        ind = np.where((self.freq >= f1) & (self.freq <= f2))[0]
        band_power = np.mean(self.sxx[ind, :], axis=0)
        return band_power

    @property
    def delta(self):
        return self.get_band_power(f1=0.5, f2=4)

    @property
    def deltaplus(self):
        deltaplus_ind = np.where(
            ((self.freq > 0.5) & (self.freq < 4))
            | ((self.freq > 12) & (self.freq < 15))
        )[0]
        deltaplus_sxx = np.mean(self.sxx[deltaplus_ind, :], axis=0)
        return deltaplus_sxx

    @property
    def theta(self):
        return self.get_band_power(f1=5, f2=11)

    @property
    def spindle(self):
        return self.get_band_power(f1=10, f2=20)

    @property
    def gamma(self):
        return self.get_band_power(f1=30, f2=90)

    @property
    def ripple(self):
        return self.get_band_power(f1=140, f2=250)

    @property
    def theta_delta_ratio(self):
        return self.theta / self.delta

    @property
    def theta_deltaplus_ratio(self):
        return self.theta / self.deltaplus

    def plotSpect(self, ax=None, freqRange=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        sxx = self.sxx / np.max(self.sxx)
        sxx = gaussian_filter(sxx, sigma=1)
        vmax = np.max(sxx) / 4
        if freqRange is None:
            freq_indx = np.arange(len(self.freq))
        else:
            freq_indx = np.where(
                (self.freq > freqRange[0]) & (self.freq < freqRange[1])
            )[0]
        ax.pcolormesh(
            self.time,
            self.freq[freq_indx],
            sxx[freq_indx, :],
            cmap="Spectral_r",
            vmax=vmax,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")


@dataclass
class wavelet_decomp:
    lfp: np.array
    freqs: np.array = np.arange(1, 20)
    sampfreq: int = 1250

    def colgin2009(self):
        """colgin


        Returns:
            [type]: [description]

        References
        ------------
        1) Colgin, L. L., Denninger, T., Fyhn, M., Hafting, T., Bonnevie, T., Jensen, O., ... & Moser, E. I. (2009). Frequency of gamma oscillations routes flow of information in the hippocampus. Nature, 462(7271), 353-357.
        2) Tallon-Baudry, C., Bertrand, O., Delpuech, C., & Pernier, J. (1997). Oscillatory γ-band (30–70 Hz) activity induced by a visual search task in humans. Journal of Neuroscience, 17(2), 722-734.
        """
        t_wavelet = np.arange(-4, 4, 1 / self.sampfreq)
        freqs = self.freqs
        n = len(self.lfp)
        fastn = next_fast_len(n)
        signal = np.pad(self.lfp, (0, fastn - n), "constant", constant_values=0)
        # signal = np.tile(np.expand_dims(signal, axis=0), (len(freqs), 1))
        # wavelet_at_freqs = np.zeros((len(freqs), len(t_wavelet)), dtype=complex)
        conv_val = np.zeros((len(freqs), n), dtype=complex)
        # for i, freq in enumerate(freqs):
        def wav_cal(freq):
            sigma = 7 / (2 * np.pi * freq)
            A = (sigma * np.sqrt(np.pi)) ** -0.5
            wavelet_at_freq = (
                A
                * np.exp(-(t_wavelet ** 2) / (2 * sigma ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )

            return sg.fftconvolve(signal, wavelet_at_freq, mode="same", axes=-1)[:n]

        conv_val = Parallel(n_jobs=10)(delayed(wav_cal)(freq) for freq in freqs)
        conv_val = np.asarray(conv_val)

        return np.abs(conv_val) ** 2

    def quyen2008(self):
        """colgin


        Returns:
            [type]: [description]

        References
        ------------
        1) Le Van Quyen, M., Bragin, A., Staba, R., Crépon, B., Wilson, C. L., & Engel, J. (2008). Cell type-specific firing during ripple oscillations in the hippocampal formation of humans. Journal of Neuroscience, 28(24), 6104-6110.
        """
        t_wavelet = np.arange(-4, 4, 1 / self.sampfreq)
        freqs = self.freqs
        signal = self.lfp
        signal = np.tile(np.expand_dims(signal, axis=0), (len(freqs), 1))

        wavelet_at_freqs = np.zeros((len(freqs), len(t_wavelet)))
        for i, freq in enumerate(freqs):
            sigma = 5 / (6 * freq)
            A = np.sqrt(freq)
            wavelet_at_freqs[i, :] = (
                A
                * np.exp(-((t_wavelet) ** 2) / (sigma ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )

        conv_val = sg.fftconvolve(signal, wavelet_at_freqs, mode="same", axes=-1)

        return np.abs(conv_val) ** 2

    def bergel2018(self):
        """colgin


        Returns:
            [type]: [description]

        References:
        ---------------
        1) Bergel, A., Deffieux, T., Demené, C., Tanter, M., & Cohen, I. (2018). Local hippocampal fast gamma rhythms precede brain-wide hyperemic patterns during spontaneous rodent REM sleep. Nature communications, 9(1), 1-12.

        """
        signal = self.lfp
        t_wavelet = np.arange(-4, 4, 1 / self.sampfreq)
        freqs = self.freqs

        wave_spec = []
        for freq in freqs:
            sigma = freq / (2 * np.pi * 7)
            A = (sigma * np.sqrt(np.pi)) ** -0.5
            my_wavelet = (
                A
                * np.exp(-((t_wavelet) ** 2) / (2 * sigma ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )
            # conv_val = np.convolve(signal, my_wavelet, mode="same")
            conv_val = sg.fftconvolve(signal, my_wavelet, mode="same")

            wave_spec.append(conv_val)

        wave_spec = np.abs(np.asarray(wave_spec))
        return wave_spec * np.linspace(1, 150, 100).reshape(-1, 1)

    def torrenceCompo(self):
        # wavelet = _check_parameter_wavelet("morlet")
        # sj = 1 / (wavelet.flambda() * self.freqs)
        # wave, period, scale, coi = wavelet(
        #     self.lfp, 1 / self.sampfreq, pad=1, dj=0.25, s0, j1, mother
        # )
        pass

    def cohen(self, ncycles=3):
        """Implementation of ref. 1 chapter 13


        Returns:
            [type]: [description]

        References:
        ---------------
        1) Cohen, M. X. (2014). Analyzing neural time series data: theory and practice. MIT press.

        """
        signal = self.lfp
        t_wavelet = np.arange(-4, 4, 1 / self.sampfreq)
        freqs = self.freqs

        wave_spec = []
        for freq in freqs:
            s = ncycles / (2 * np.pi * freq)
            A = (s * np.sqrt(np.pi)) ** -0.5
            my_wavelet = (
                A
                * np.exp(-(t_wavelet ** 2) / (2 * s ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )
            # conv_val = np.convolve(signal, my_wavelet, mode="same")
            conv_val = sg.fftconvolve(signal, my_wavelet, mode="same")

            wave_spec.append(conv_val)

        wave_spec = np.abs(np.asarray(wave_spec))
        return wave_spec ** 2


def hilbertfast(signal, ax=-1):

    """inputs a signal does padding to next power of 2 for faster computation of hilbert transform

    Arguments:
        signal {array} -- [n, dimensional array]

    Returns:
        [type] -- [description]
    """
    signal_length = signal.shape[-1]
    hilbertsig = sg.hilbert(signal, fftpack.next_fast_len(signal_length), axis=ax)

    if np.ndim(signal) > 1:
        hilbertsig = hilbertsig[:, :signal_length]
    else:
        hilbertsig = hilbertsig[:signal_length]

    return hilbertsig


def fftnormalized(signal, fs=1250):

    # Number of sample points
    N = len(signal)
    # sample spacing
    T = 1 / fs
    y = signal
    yf = fft(y)
    freq = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    pxx = 2.0 / N * np.abs(yf[0 : N // 2])

    return pxx, freq


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
            plt.plot(self.time, lfp.T + self.coords, "gray")


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


def hilbert_ampltiude_stat(signals, freq_band, fs, statistic="mean"):
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
