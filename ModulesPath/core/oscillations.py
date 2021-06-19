from .. import core
import numpy as np
import pandas as pd
from pathlib import Path
from ..utils import mathutil, signal_process
from ..parsePath import Recinfo
from scipy import stats
from ..core import Analogsignal, ProbeGroup


def _detect_freq_band_epochs(
    signal, freq_band, thresh, mindur, maxdur, mergedist, fs, ignore_times=None
):
    """Detects epochs of high power in a given frequency band

    Parameters
    ----------
    thresh : tuple, optional
        low and high threshold for detection
    mindur : float, optional
        minimum duration of epoch
    maxdur : float, optiona
    chans : list
        channels used for epoch detection, if None then chooses best chans
    """

    zscsignal = []
    lf, hf = freq_band
    lowthresh, highthresh = thresh
    for sig in signals:
        yf = filter_sig.bandpass(sig, lf=lf, hf=hf)
        zsc_chan = stats.zscore(np.abs(hilbertfast(yf)))
        zscsignal.append(zsc_chan)

    zscsignal = np.asarray(zscsignal)

    # ---------setting noisy periods zero --------
    if ignore_times is not None:
        assert ignore_times.ndim == 2, "ignore_times should be 2 dimensional array"
        noisy_frames = np.concatenate(
            [
                (np.arange(start, stop) * fs).astype(int)
                for (start, stop) in ignore_times
            ]
        )

        zscsignal[:, noisy_frames] = 0

    # ------hilbert transform --> binarize by > than lowthreshold
    maxPower = np.max(zscsignal, axis=0)
    ThreshSignal = np.where(zscsignal > lowthresh, 1, 0).sum(axis=0)
    ThreshSignal = np.diff(np.where(ThreshSignal > 0, 1, 0))
    start = np.where(ThreshSignal == 1)[0]
    stop = np.where(ThreshSignal == -1)[0]

    # --- getting rid of incomplete epochs at begining or end ---------
    if start[0] > stop[0]:
        stop = stop[1:]
    if start[-1] > stop[-1]:
        start = start[:-1]

    firstPass = np.vstack((start, stop)).T
    print(f"{len(firstPass)} epochs detected initially")

    # --------merging close epochs------------
    min_inter_epoch_samples = mergedist * fs
    secondPass = []
    epoch = firstPass[0]
    for i in range(1, len(firstPass)):
        if firstPass[i, 0] - epoch[1] < min_inter_epoch_samples:
            epoch = [epoch[0], firstPass[i, 1]]
        else:
            secondPass.append(epoch)
            epoch = firstPass[i]
    secondPass.append(epoch)
    secondPass = np.asarray(secondPass)
    print(f"{len(secondPass)} epochs reamining after merging close ones")

    # ------delete epochs with less than threshold power--------
    thirdPass = []
    peakpower, peaktime = [], []

    for i in range(0, len(secondPass)):
        maxValue = max(maxPower[secondPass[i, 0] : secondPass[i, 1]])
        if maxValue > highthresh:
            thirdPass.append(secondPass[i])
            peakpower.append(maxValue)
            peaktime.append(
                secondPass[i, 0]
                + np.argmax(maxPower[secondPass[i, 0] : secondPass[i, 1]])
            )
    thirdPass = np.asarray(thirdPass)
    print(f"{len(thirdPass)} epochs reamining after deleting epochs with weaker power")

    ripple_duration = np.diff(thirdPass, axis=1) / fs
    epochs = pd.DataFrame(
        {
            "start": thirdPass[:, 0],
            "stop": thirdPass[:, 1],
            "peakpower": peakpower,
            "peaktime": np.asarray(peaktime),
            "duration": ripple_duration.squeeze(),
        }
    )

    # ---------delete very short epochs--------
    epochs = epochs[epochs.duration >= mindur]
    print(f"{len(epochs)} epochs reamining after deleting short epochs")

    # ----- delete epochs with unrealistic high power
    # artifactRipples = np.where(peakpower > maxPeakPower)[0]
    # fourthPass = np.delete(thirdPass, artifactRipples, 0)
    # peakpower = np.delete(peakpower, artifactRipples)

    # ---------delete very long epochs---------
    epochs = epochs[epochs.duration <= maxdur]
    print(f"{len(epochs)} epochs reamining after deleting very long epochs")

    # ----- converting to all time stamps to seconds --------
    epochs[["start", "stop", "peakpower", "peaktime"]] /= fs  # seconds

    epochs = epochs.reset_index(drop=True)
    epochs["label"] = ""
    metadata = {
        "params": {
            "lowThres": lowthresh,
            "highThresh": highthresh,
            "freq_band": freq_band,
            "mindur": mindur,
            "maxdur": maxdur,
            "mergedist": mergedist,
        },
    }

    return epochs, metadata


def detect_hpc_slow_wave_epochs():
    """Caculate delta events

    chan --> filter delta --> identify peaks and troughs within sws epochs only --> identifies a slow wave as trough to peak --> thresholds for 100ms minimum duration

    Parameters
    ----------
    chan : int
        channel to be used for detection
    freq_band : tuple, optional
        frequency band in Hz, by default (0.5, 4)
    """

    lfpsRate = self._obj.lfpSrate
    deltachan = self._obj.geteeg(chans=chan)

    # ---- filtering best ripple channel in delta band
    t = np.linspace(0, len(deltachan) / lfpsRate, len(deltachan))
    lf, hf = freq_band
    delta_sig = signal_process.filter_sig.bandpass(deltachan, lf=lf, hf=hf)
    delta = stats.zscore(delta_sig)  # normalization w.r.t session
    delta = -delta  # flipping as this is in sync with cortical slow wave

    # ---- finding peaks and trough for delta oscillations

    up = sg.find_peaks(delta)[0]
    down = sg.find_peaks(-delta)[0]

    if up[0] < down[0]:
        up = up[1:]
    if up[-1] > down[-1]:
        up = up[:-1]

    sigdelta = []
    for i in range(len(down) - 1):
        tbeg = t[down[i]]
        tpeak = t[up[i]]
        tend = t[down[i + 1]]
        peakamp = delta[up[i]]
        endamp = delta[down[i + 1]]
        # ------ thresholds for selecting delta --------
        # if (peakamp > 2 and endamp < 0) or (peakamp > 1 and endamp < -1.5):
        sigdelta.append([peakamp, endamp, tpeak, tbeg, tend])

    sigdelta = np.asarray(sigdelta)
    print(f"{len(sigdelta)} delta detected")

    data = pd.DataFrame(
        {
            "start": sigdelta[:, 3],
            "end": sigdelta[:, 4],
            "peaktime": sigdelta[:, 2],
            "peakamp": sigdelta[:, 0],
            "endamp": sigdelta[:, 1],
        }
    )
    detection_params = {"freq_band": freq_band, "chan": chan}
    hipp_slow_wave = {"events": data, "DetectionParams": detection_params}

    np.save(self.files.events, hipp_slow_wave)
    self._load()


def detect_ripple_epochs(
    signal: Analogsignal,
    chans=None,
    thresh=(1, 5),
    mindur=0.05,
    maxdur=0.450,
    mergedist=0.05,
    ignore_epochs=None,
):
    if chans is None:
        changrps = self._obj.goodchangrp

        selected_chans = []
        for changrp in changrps:
            if changrp:
                lfps = self._obj.geteeg(chans=changrp, timeRange=[0, 3600])
                desc_order = super().get_best_channels(lfps=lfps)
                selected_chans.append(changrp[desc_order[0]])

    else:
        selected_chans = chans

    lfps = self._obj.geteeg(chans=selected_chans)

    epochs, metadata = _detect_freq_band_epochs(
        signal=lfps,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        ignore_times=ignore_epochs,
    )

    metadata["channels"] = selected_chans
    self.epochs = epochs
    self.metadata = metadata
    self.save()


def detect_theta_epochs():
    if chans is None:
        chans = self._obj.goodchans

    lfps = self._obj.time_slice(chans=chans, period=[0, 3600])
    hilbert_amplitudes = signal_process.hilbert_ampltiude_stat(lfps)
    best_chan = chans[np.argmax(hilbert_amplitudes)]

    self.epochs, self.metadata = signal_process.detect_freq_band_epochs(best_chan)


def detect_spindle_epochs(
    chans=None,
    thresh=(1, 5),
    midur=0.4,
    maxdur=1,
    mergedist=0.05,
    ignore_epochs=None,
):
    if chans is None:
        changrps = self._obj.goodchangrp

        selected_chans = []
        for changrp in changrps:
            lfps = self._obj.geteeg(chans=changrp, timeRange=[0, 3600])
            desc_order = super().get_best_channels(lfps=lfps)
            selected_chans.append(changrp[desc_order[0]])
    else:
        selected_chans = chans

    lfps = self._obj.geteeg(chans=selected_chans)

    epochs, metadata = super().detect(
        lfps=lfps,
        thresh=thresh,
        mindur=midur,
        maxdur=maxdur,
        mergedist=mergedist,
        ignore_times=ignore_epochs,
    )

    metadata["channels"] = selected_chans
    self.epochs = epochs
    self.metadata = metadata
    self.save()


def detect_gamma_epochs():
    pass


class Ripple(core.Oscillation, core.Epoch):
    """Ripple class to detect ripple epochs"""

    def __init__(self, signal: Analogsignal, probe: ProbeGroup, filename=None):

        assert isinstance(signal, Analogsignal)
        assert isinstance(probe, ProbeGroup)
        self._obj = signal
        self._probe = probe
        super().__init__(
            freq_band=(150, 250), fs=signal.sampling_rate, filename=filename
        )
        self.load()

    @property
    def bestchans(self):
        if "channels" in self.metadata:
            return self.metadata["channels"]
        else:
            return None

    def plot_summary(self, random=False, shank_id=None):
        """Plots 10 of detected ripples across two randomly selected shanks with their filtered lfp

        Parameters
        ----------
        random : bool, optional
            if True then randomly plots 10 ripples, by default False then it plots 5 weakest and 5 strongest ripples
        """
        fig = plt.figure(num=None, figsize=(10, 6))
        gs = gridspec.GridSpec(2, 10, figure=fig)
        fig.subplots_adjust(hspace=0.5)

        changrp = [shank for shank in self._obj.goodchangrp if shank]
        channels = np.concatenate(np.random.choice(changrp, 2))  # random 2 shanks
        ripples = self.events
        peakpower = self.events.peakNormalizedPower.values
        params = self.params

        # --- sorting ripples by peakpower ------
        sort_ind = np.argsort(peakpower)

        # ---- selecting few ripples to plot -----------
        if random:
            ripple_to_plot = np.random.choice(sort_ind, 10)
        else:
            ripple_to_plot = np.concatenate((sort_ind[:5], sort_ind[-5:]))

        # ------ plotting ripples and filtered lfp -----------
        for ind, ripple in enumerate(ripple_to_plot):

            ax = fig.add_subplot(gs[1, ind])
            self.plot_example(ax=ax, ripple_indx=ripple, pad=0.3, shank_id=shank_id)
            ax.set_title(
                f"zsc = {round(peakpower[ripple],2)} \n {round(ripples.loc[ripple].duration*1000)} ms",
                loc="left",
            )
            ax.axis("off")

        # ------ plotting parameters used during detection ----------
        ax = fig.add_subplot(gs[0, 0])
        ax.text(
            0,
            0.8,
            f" highThresh ={params['highThresh']}\n lowThresh ={params['lowThres']}\n minDuration = {params['minDuration']}\n maxDuration = {params['maxDuration']} \n mergeRipple = {params['mergeDistance']} \n #Ripples = {len(peakpower)}",
        )
        ax.axis("off")

        # ----- plotting channels used for detection --------
        ax = fig.add_subplot(gs[0, 1:4])
        try:
            self._obj.probemap.plot(self.bestchans, ax=ax)
            ax.set_title("selected channel")
        except AttributeError:
            print(
                "No probemap provided - provide to visualize ripple channel location!"
            )

        # ---- peaknormalized power distribution plot ---------
        ax = fig.add_subplot(gs[0, 5])
        histpower, edgespower = np.histogram(peakpower, bins=100)
        ax.plot(edgespower[:-1], histpower, color="#544a4a")
        ax.set_xlabel("Zscore value")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")

        # ----- distribution of ripple duration ---------
        ax = fig.add_subplot(gs[0, 6])
        histdur, edgesdur = np.histogram(ripples.duration * 1000, bins=100)
        ax.plot(edgesdur[:-1], histdur, color="#544a4a")
        ax.set_xlabel("Duration (ms)")
        # ax.set_ylabel("Counts")
        ax.set_yscale("log")

        subname = self._obj.session.subname
        fig.suptitle(f"Ripple detection of {subname}")

    def plot_example(
        self, ax=None, ripple_indx=None, shank_id=None, pad=0.2, color="k"
    ):
        changrp = self._obj.channelgroups
        nShanks = self._obj.nShanks
        if ripple_indx is None:
            ripple_indx = np.random.randint(low=0, high=len(self.events))
        if shank_id is None:
            shank_id = np.random.randint(low=0, high=nShanks)

        ripple_time = self.events.loc[ripple_indx][["start", "end"]].to_list()
        lfp = np.array(self._obj.geteeg(chans=changrp[shank_id], timeRange=ripple_time))
        lfp = lfp / np.max(lfp)  # scaling
        lfp = lfp - lfp[:, 0][:, np.newaxis]  # np.min(lfp, axis=1, keepdims=True)
        pad_vals = np.linspace(0, len(lfp) * pad, len(lfp))[::-1]
        lfp = lfp + pad_vals[:, np.newaxis]

        if ax is None:
            _, ax = plt.subplots(1, 1)

        print(f"Plotting ripple no. {ripple_indx}")
        ax.clear()
        ax.plot(lfp.T, color=color)
        ax.set_yticks(pad_vals)
        ax.set_yticklabels(changrp[shank_id])
        ax.set_xticklabels([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="both", length=0)

        return ax

    def plot_ripples(self, period, ax):
        """Plot ripples between this period on a given axis

        Parameters
        ----------
        period : list
            list of length 2, in seconds
        ax : axis object
            axis
        """

        events = self.events[
            (self.events.start > period[0]) & (self.events.start < period[1])
        ]

        for epoch in events.itertuples():
            color = "#ff928a"
            ax.axvspan(epoch.start, epoch.end, facecolor=color, alpha=0.7)


class Gamma(core.Oscillation, core.Epoch):
    """Events and analysis related to gamma oscillations"""

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        filePrefix = self._obj.files.filePrefix

    def get_peak_intervals(
        self,
        lfp,
        band=(25, 50),
        lowthresh=0,
        highthresh=1,
        minDistance=300,
        minDuration=125,
    ):
        """Returns strong theta lfp. If it has multiple channels, then strong theta periods are calculated from that channel which has highest area under the curve in the theta frequency band. Parameters are applied on z-scored lfp.

        Parameters
        ----------
        lfp : array like, channels x time
            from which strong periods are concatenated and returned
        lowthresh : float, optional
            threshold above which it is considered strong, by default 0 which is mean of the selected channel
        highthresh : float, optional
            [description], by default 0.5
        minDistance : int, optional
            minimum gap between periods before they are merged, by default 300 samples
        minDuration : int, optional
            [description], by default 1250, which means theta period should atleast last for 1 second

        Returns
        -------
        2D array
            start and end frames where events exceeded the set thresholds
        """

        # ---- filtering --> zscore --> threshold --> strong gamma periods ----
        gammalfp = signal_process.filter_sig.bandpass(lfp, lf=band[0], hf=band[1])
        hil_gamma = signal_process.hilbertfast(gammalfp)
        gamma_amp = np.abs(hil_gamma)

        zsc_gamma = stats.zscore(gamma_amp)
        peakevents = mathutil.threshPeriods(
            zsc_gamma,
            lowthresh=lowthresh,
            highthresh=highthresh,
            minDistance=minDistance,
            minDuration=minDuration,
        )

        return peakevents

    def csd(self, period, refchan, chans, band=(25, 50), window=1250):
        """Calculating current source density using laplacian method

        Parameters
        ----------
        period : array
            period over which theta cycles are averaged
        refchan : int or array
            channel whose theta peak will be considered. If array then median of lfp across all channels will be chosen for peak detection
        chans : array
            channels for lfp data
        window : int, optional
            time window around theta peak in number of samples, by default 1250

        Returns:
        ----------
        csd : dataclass,
            a dataclass return from signal_process module
        """
        lfp_period = self._obj.geteeg(chans=chans, timeRange=period)
        lfp_period = signal_process.filter_sig.bandpass(
            lfp_period, lf=band[0], hf=band[1]
        )

        gamma_lfp = self._obj.geteeg(chans=refchan, timeRange=period)
        nChans = lfp_period.shape[0]
        # lfp_period, _, _ = self.getstrongTheta(lfp_period)

        # --- Selecting channel with strongest theta for calculating theta peak-----
        # chan_order = self._getAUC(lfp_period)
        # gamma_lfp = signal_process.filter_sig.bandpass(
        #     lfp_period[chan_order[0], :], lf=5, hf=12, ax=-1)
        gamma_lfp = signal_process.filter_sig.bandpass(
            gamma_lfp, lf=band[0], hf=band[1]
        )
        peak = sg.find_peaks(gamma_lfp)[0]
        # Ignoring first and last second of data
        peak = peak[np.where((peak > 1250) & (peak < len(gamma_lfp) - 1250))[0]]

        # ---- averaging around theta cycle ---------------
        avg_theta = np.zeros((nChans, window))
        for ind in peak:
            avg_theta = avg_theta + lfp_period[:, ind - window // 2 : ind + window // 2]
        avg_theta = avg_theta / len(peak)

        _, ycoord = self._obj.probemap.get(chans=chans)

        csd = signal_process.Csd(lfp=avg_theta, coords=ycoord, chan_label=chans)
        csd.classic()

        return csd
