from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.linalg as linalg
import scipy.ndimage as filtSig
import scipy.signal as sg
from scipy import stats
from joblib import Parallel, delayed
from numpy import linalg
from numpy.lib.npyio import NpzFile
from scipy import fftpack
from scipy.fft import fft
from scipy.fftpack import next_fast_len
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from plotUtil import Fig
from waveletFunctions import wavelet


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


@dataclass
class spectrogramBands:
    lfp: Any
    sampfreq: float = 1250.0
    window: int = 1  # in seconds
    overlap: float = 0.5  # in seconds
    smooth: int = None
    multitaper: bool = False

    def __post_init__(self):

        window = int(self.window * self.sampfreq)
        overlap = int(self.overlap * self.sampfreq)

        f = None
        if self.multitaper:
            tapers = sg.windows.dpss(M=window, NW=5, Kmax=6)

            sxx_taper = []
            for taper in tapers:
                f, t, sxx = sg.spectrogram(
                    self.lfp, window=taper, fs=self.sampfreq, noverlap=overlap
                )
                sxx_taper.append(sxx)
            sxx = np.dstack(sxx_taper).mean(axis=2)

        else:
            f, t, sxx = sg.spectrogram(
                self.lfp, fs=self.sampfreq, nperseg=window, noverlap=overlap
            )

        delta_ind = np.where((f > 0.5) & (f < 4))[0]
        delta_sxx = np.mean(sxx[delta_ind, :], axis=0)

        deltaplus_ind = np.where(((f > 0.5) & (f < 4)) | ((f > 12) & (f < 15)))[0]
        deltaplus_sxx = np.mean(sxx[deltaplus_ind, :], axis=0)
        # delta_ind = np.where(((f > 0.5) & (f < 16)))[0]
        # delta_smooth = filtSig.gaussian_filter1d(delta_sxx, smooth, axis=0)

        theta_ind = np.where((f > 5) & (f < 11))[0]
        theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
        # theta_smooth = filtSig.gaussian_filter1d(theta_sxx, smooth, axis=0)

        spindle_ind = np.where((f > 10) & (f < 20))[0]
        spindle_sxx = np.mean(sxx[spindle_ind, :], axis=0)
        # spindle_smooth = filtSig.gaussian_filter1d(spindle_sxx, smooth, axis=0)

        gamma_ind = np.where((f > 30) & (f < 90))[0]
        gamma_sxx = np.mean(sxx[gamma_ind, :], axis=0)
        # gamma_smooth = filtSig.gaussian_filter1d(gamma_sxx, smooth, axis=0)

        ripple_ind = np.where((f > 140) & (f < 250))[0]
        ripple_sxx = np.mean(sxx[ripple_ind, :], axis=0)

        if self.smooth is not None:
            smooth = self.smooth
            delta_sxx = filtSig.gaussian_filter1d(delta_sxx, smooth, axis=0)
            deltaplus_sxx = filtSig.gaussian_filter1d(deltaplus_sxx, smooth, axis=0)
            theta_sxx = filtSig.gaussian_filter1d(theta_sxx, smooth, axis=0)
            spindle_sxx = filtSig.gaussian_filter1d(spindle_sxx, smooth, axis=0)
            gamma_sxx = filtSig.gaussian_filter1d(gamma_sxx, smooth, axis=0)
            ripple_sxx = filtSig.gaussian_filter1d(ripple_sxx, smooth, axis=0)

        self.delta = delta_sxx
        self.deltaplus = deltaplus_sxx
        self.theta = theta_sxx
        self.spindle = spindle_sxx
        self.gamma = gamma_sxx
        self.ripple = ripple_sxx
        self.freq = f
        self.time = t
        self.sxx = sxx
        self.theta_delta_ratio = self.theta / self.delta
        self.theta_deltaplus_ratio = self.theta / self.deltaplus

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
        for i, freq in enumerate(freqs):
            sigma = 7 / (2 * np.pi * freq)
            A = (sigma * np.sqrt(np.pi)) ** -0.5
            wavelet_at_freq = (
                A
                * np.exp(-(t_wavelet ** 2) / (2 * sigma ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )

            conv_val[i, :] = sg.fftconvolve(
                signal, wavelet_at_freq, mode="same", axes=-1
            )[:n]

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
        freq_ind = np.where((f > self.flow) & (f < self.fhigh / 2))[0]
        freq_req = f[freq_ind]

        """ ===========================================
        bispectrum = |mean( X(f1) * X(f2) * conj(X(f1+f2)) )|
        normalization = sqrt( mean(|X(f1) * X(f2)|^2) * mean(|X(f1+f2)|^2) )

        where,
            X is complex spectrogram
            P is real/absolute square spectrogram
        ================================================="""

        X_f2 = sxx[:, freq_ind, :]  # complex spectrogram of required frequencies
        # P_f2 = np.abs(X_f2) ** 2  # absolute square of spectrogram

        def bicoh_product(f_ind):
            X_f1 = sxx[:, f_ind, np.newaxis, :]
            X_f1f2 = sxx[:, freq_ind + f_ind, :]

            # ----- bispectrum triple product --------------
            bispec_freq = np.mean((X_f1 * X_f2) * np.conjugate(X_f1f2), axis=-1)

            # ----- normalization to calculate bicoherence ---------
            # P_f1 = np.abs(X_f1) ** 2
            P_f1f2 = np.abs(X_f1f2) ** 2
            # norm = (np.mean(P_f1 * P_f2 * P_f1f2, axis=-1))
            norm = np.sqrt(
                np.mean(np.abs(X_f1 * X_f2) ** 2, axis=-1) * np.mean(P_f1f2, axis=-1)
            )

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

    def plot(self, index=None, ax=None, vmin=0, vmax=0.2, cmap="hot"):

        bic = self.bicoher.copy()
        # bic = bic.T
        lt = np.tril_indices_from(bic, k=-1)
        # bic[lt] = np.nan
        # bic[(lt[0], -lt[1])] = np.nan
        bic[bic < self.significance] = 0
        bic = gaussian_filter(bic, sigma=2)

        if ax is None:
            _, ax = plt.subplots(1, 1)

        bicoh_plt = ax.pcolormesh(
            self.freq,
            self.freq,
            bic,
            cmap=cmap,
            shading="gouraud",
            vmin=0,
            vmax=0.06,
            rasterized=True,
        )

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
        plt.colorbar(bicoh_plt)


def phasePowerCorrelation(signal):
    pass


@dataclass
class Csd:
    lfp: np.array
    coords: np.array
    chan_label: np.array = None

    def classic(self):
        coords = self.coords.copy()
        time = np.linspace(0, 1, self.lfp.shape[1])
        csd = -(self.lfp[:-2, :] - 2 * self.lfp[1:-1, :] + self.lfp[2:, :])
        self.csdmap = csd
        self.coords = coords[1:-1]
        self.time = time

    def icsd(self, lfp, coords):
        pass

    def plot(self, ax=None, cmap="jet", smooth=3):
        csdmap = gaussian_filter(self.csdmap, sigma=smooth)

        gs = None
        if ax is None:
            figure = Fig()
            _, gs = figure.draw(grid=[1, 1])
            ax = plt.subplot(gs[0])
        ax.pcolormesh(
            self.time,
            np.arange(len(self.coords) + 1, 1, -1),
            stats.zscore(csdmap, axis=None),
            cmap=cmap,
            shading="gouraud",
            # vmax=1,
            rasterized=True,
        )
        ax.set_title("Current source density map")
        ax.set_ylabel("y Coordinates")
        ax.set_xlabel("Normalized time")
        ax.set_ylim([0, len(self.coords) + 2])

        lfp = self.lfp[1:-1] - np.min(self.lfp[1:-1])
        lfp = (lfp / np.max(lfp)) * 60
        # lfp = np.flipud(lfp)
        # axlfp = axmap.twinx()
        # axmap.plot(self.time, lfp.T + self.coords, "k", lw=1)
        # axmap.set_ylim(bottom=0)

        # if self.chan_label is not None:
        #     ax[1].scatter(np.ones(len(self.chan_label[1:-1])), self.coords)

        #     for i, txt in enumerate(self.chan_label[1:-1]):
        #         ax[1].annotate(str(txt), (1, self.coords[i]))

        #     ax[1].axis("off")


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
