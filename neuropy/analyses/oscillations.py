import numpy as np
import pandas as pd
from neuropy.utils import mathutil, signal_process
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from neuropy.core import Signal, ProbeGroup, Epoch


def _detect_freq_band_epochs(
    signals,
    freq_band,
    thresh,
    edge_cutoff,
    mindur,
    maxdur,
    mergedist,
    fs,
    sigma,
    ignore_times=None,
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

    lf, hf = freq_band
    dt = 1 / fs
    smooth = lambda x: gaussian_filter1d(x, sigma=sigma / dt, axis=-1)
    lowthresh, highthresh = thresh

    # Because here one shank is selected per shank, based on visualization:
    # mean: very conservative in cases where some shanks may not have that strong ripple
    # max: works well but may have ocassional false positives
    power = np.zeros(signals.shape[1])
    for sig in signals:
        yf = signal_process.filter_sig.bandpass(sig, lf=lf, hf=hf, fs=fs)
        # zsc_chan = smooth(stats.zscore(np.abs(signal_process.hilbertfast(yf))))
        # zscsignal[sig_i] = zsc_chan
        power += np.abs(signal_process.hilbertfast(yf))

    # mean and smooth
    power = smooth(power / signals.shape[0])

    # ---------setting noisy periods zero --------
    if ignore_times is not None:
        assert ignore_times.ndim == 2, "ignore_times should be 2 dimensional array"
        noisy_frames = np.concatenate(
            [
                (np.arange(start * fs, stop * fs)).astype(int)
                for (start, stop) in ignore_times
            ]
        )

        # edge case: remove any frames that might extend past end of recording
        noisy_frames = noisy_frames[noisy_frames < len(power)]
        power[noisy_frames] = 0

    # ---- thresholding and detection ------
    power = stats.zscore(power)
    power_thresh = np.where(power >= edge_cutoff, power, 0)
    peaks, props = sg.find_peaks(
        power_thresh, height=[lowthresh, highthresh], prominence=0
    )
    starts, stops = props["left_bases"], props["right_bases"]
    peaks_power = power_thresh[peaks]

    # ----- merge overlapping epochs ------
    n_epochs = len(starts)
    ind_delete = []
    for i in range(n_epochs - 1):
        if starts[i + 1] - stops[i] < 1e-6:
            # stretch the second epoch to cover the range of both epochs
            starts[i + 1] = min(starts[i], starts[i + 1])
            stops[i + 1] = max(stops[i], stops[i + 1])

            peaks_power[i + 1] = max(peaks_power[i], peaks_power[i + 1])
            peaks[i + 1] = [peaks[i], peaks[i + 1]][
                np.argmax([peaks_power[i], peaks_power[i + 1]])
            ]

            ind_delete.append(i)

    epochs_arr = np.vstack((starts, stops, peaks, peaks_power)).T
    starts, stops, peaks, peaks_power = np.delete(epochs_arr, ind_delete, axis=0).T

    epochs_df = pd.DataFrame(
        dict(
            start=starts, stop=stops, peak_time=peaks, peak_power=peaks_power, label=""
        )
    )
    epochs_df[["start", "stop", "peak_time"]] /= fs  # seconds
    epochs = Epoch(epochs=epochs_df)

    # ------duration thresh---------
    epochs = epochs.duration_slice(min_dur=mindur, max_dur=maxdur)
    print(f"{len(epochs)} epochs reamining with durations within ({mindur},{maxdur})")

    epochs.metadata = {
        "params": {
            "lowThres": lowthresh,
            "highThresh": highthresh,
            "freq_band": freq_band,
            "mindur": mindur,
            "maxdur": maxdur,
            # "mergedist": mergedist,
        },
    }

    return epochs


def detect_hpc_delta_wave_epochs(
    signal: Signal,
    freq_band=(0.5, 4),
    min_dur=0.15,
    max_dur=0.5,
    ignore_epochs: Epoch = None,
):
    """Detect delta waves epochs.

    Method
    -------
    Maingret, Nicolas, Gabrielle Girardeau, Ralitsa Todorova, Marie Goutierre, and Michaël Zugaro. “Hippocampo-Cortical Coupling Mediates Memory Consolidation during Sleep.” Nature Neuroscience 19, no. 7 (July 2016): 959–64. https://doi.org/10.1038/nn.4304.

    -> filtered singal in 0.5-4 Hz (Note: Maingret et al. used 0-6 Hz for cortical LFP)
    -> remove noisy epochs if provided
    -> z-scored the filtered signal, D(t)
    -> flip the sign of signal to be consistent with cortical LFP
    -> calculate derivative, D'(t)
    -> extract upward-downward-upward zero-crossings which correspond to start,peak,stop of delta waves, t_start, t_peak, t_stop
    -> discard sequences below 150ms and above 500ms
    -> Delta waves corresponded to epochs where D(t_peak) > 2, or D(t_peak) > 1 and D(t_stop) < -1.5.

    Parameters
    ----------
    signal : Signal object
        signal trace to be used for detection
    freq_band : tuple, optional
        frequency band in Hz, by default (0.5, 4)
    min_dur: float, optional
        minimum duration for delta waves, by default 0.15 seconds
    max_dur: float, optional
        maximum duration for delta waves, by default 0.5 seconds
    ignore_epochs: core.Epoch, optional
        ignore timepoints within these epochs, primarily used for noisy time periods if known already, by default None

    Returns
    -------
    Epoch
        delta wave epochs. In addition, peak_time, peak_amp_zsc, stop_amp_zsc are also returned as columns
    """

    assert freq_band[1] <= 6, "Upper limit of freq_band can not be above 6 Hz"
    assert signal.n_channels == 1, "Signal should have only 1 channel"

    delta_signal = signal_process.filter_sig.bandpass(
        signal, lf=freq_band[0], hf=freq_band[1]
    ).traces[0]
    time = signal.time

    # ----- remove timepoints provided in ignore_epochs ------
    if ignore_epochs is not None:
        noisy_bool = ignore_epochs.get_indices_for_time(time)
        time = time[~noisy_bool]
        delta_signal = delta_signal[~noisy_bool]

    # ---- normalize and flip the sign to be consistent with cortical lfp ----
    delta_zsc = -1 * stats.zscore(delta_signal)

    # ---- finding peaks and trough for delta oscillations
    delta_zsc_diff = np.diff(delta_zsc).squeeze()
    zero_crossings = np.diff(np.sign(delta_zsc_diff))
    troughs_indx = np.where(zero_crossings > 0)[0]
    peaks_indx = np.where(zero_crossings < 0)[0]

    if peaks_indx[0] < troughs_indx[0]:
        peaks_indx = peaks_indx[1:]

    if peaks_indx[-1] > troughs_indx[-1]:
        peaks_indx = peaks_indx[:-1]

    n_peaks_in_troughs = np.histogram(peaks_indx, troughs_indx)[0]
    assert n_peaks_in_troughs.max() == 1, "Found multiple peaks within troughs"

    troughs_time, peaks_time = time[troughs_indx], time[peaks_indx]
    trough_pairs = np.vstack((troughs_time[:-1], troughs_time[1:])).T
    trough_peak_trough = np.insert(trough_pairs, 1, peaks_time, axis=1)
    duration = np.diff(trough_pairs, axis=1).squeeze()
    peak_amp = delta_zsc[peaks_indx]
    stop_amp = delta_zsc[troughs_indx[1:]]

    # ---- filtering based on duration and z-scored amplitude -------
    good_duration_bool = (duration >= min_dur) & (duration <= max_dur)
    good_amp_bool = (peak_amp > 2) | ((peak_amp > 1.5) & (stop_amp < -1.5))
    good_bool = good_amp_bool & good_duration_bool

    delta_waves_time = trough_peak_trough[good_bool]

    print(f"{delta_waves_time.shape[0]} delta waves detected")

    epochs = pd.DataFrame(
        {
            "start": delta_waves_time[:, 0],
            "stop": delta_waves_time[:, 2],
            "peak_time": delta_waves_time[:, 1],
            "peak_amp_zsc": peak_amp[good_bool],
            "stop_amp_zsc": stop_amp[good_bool],
            "label": "delta_wave",
        }
    )
    params = {"freq_band": freq_band, "channel": signal.channel_id}

    return Epoch(epochs=epochs, metadata=params)


def detect_ripple_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(150, 250),
    thresh=(2.5, None),
    edge_cutoff=0.5,
    mindur=0.05,
    maxdur=0.450,
    mergedist=0.05,
    sigma=0.0125,
    ignore_epochs: Epoch = None,
    ripple_channel: int or list = None,
):
    # TODO chewing artifact frequency (>300 Hz) or emg based rejection of ripple epochs

    if ripple_channel is None:  # auto-detect ripple channel
        if probegroup is None:
            selected_chans = signal.channel_id
            traces = signal.traces

        else:
            if isinstance(probegroup, np.ndarray):
                changrps = np.array(probegroup, dtype="object")
            if isinstance(probegroup, ProbeGroup):
                changrps = probegroup.get_connected_channels(groupby="shank")
                # if changrp:
            selected_chans = []
            for changrp in changrps:
                signal_slice = signal.time_slice(
                    channel_id=changrp.astype("int"),
                    t_start=0,
                    t_stop=np.min((3600, signal.duration)),
                )
                hil_stat = signal_process.hilbert_amplitude_stat(
                    signal_slice.traces,
                    freq_band=freq_band,
                    fs=signal.sampling_rate,
                    statistic="mean",
                )
                selected_chans.append(changrp[np.argmax(hil_stat)])

            traces = signal.time_slice(channel_id=selected_chans).traces
    else:
        assert isinstance(ripple_channel, (list, int))
        selected_chans = (
            [ripple_channel] if isinstance(ripple_channel, int) else ripple_channel
        )
        traces = signal.time_slice(channel_id=selected_chans).traces

    print(f"Selected channels for ripples: {selected_chans}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        edge_cutoff=edge_cutoff,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        sigma=sigma,
        ignore_times=ignore_times,
    )
    epochs = epochs.shift(dt=signal.t_start)
    epochs.metadata = dict(channels=selected_chans)
    return epochs


def detect_sharpwave_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(2, 50),
    thresh=(2.5, None),
    edge_cutoff=0.5,
    mindur=0.05,
    maxdur=0.450,
    mergedist=0.05,
    sigma=0.0125,
    ignore_epochs: Epoch = None,
    sharpwave_channel: int or list = None,
):
    if sharpwave_channel is None:
        if probegroup is None:  # auto-detect sharpwave channel
            selected_chans = signal.channel_id
            traces = signal.traces

        else:
            if isinstance(probegroup, np.ndarray):
                changrps = np.array(probegroup, dtype="object")
            if isinstance(probegroup, ProbeGroup):
                changrps = probegroup.get_connected_channels(groupby="shank")
                # if changrp:
            selected_chans = []
            for changrp in changrps:
                signal_slice = signal.time_slice(
                    channel_id=changrp.astype("int"),
                    t_start=0,
                    t_stop=np.min((3600, signal.duration)),
                )
                hil_stat = signal_process.hilbert_amplitude_stat(
                    signal_slice.traces,
                    freq_band=freq_band,
                    fs=signal.sampling_rate,
                    statistic="mean",
                )
                selected_chans.append(changrp[np.argmax(hil_stat)])

            traces = signal.time_slice(channel_id=selected_chans).traces
    else:
        assert isinstance(sharpwave_channel, (list, int))
        selected_chans = (
            [sharpwave_channel]
            if isinstance(sharpwave_channel, int)
            else sharpwave_channel
        )
        traces = signal.time_slice(channel_id=selected_chans).traces

    print(f"Selected channels for sharp-waves: {selected_chans}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        edge_cutoff=edge_cutoff,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        sigma=sigma,
        ignore_times=ignore_times,
    )
    epochs = epochs.shift(dt=signal.t_start)
    epochs.metadata = dict(channels=selected_chans)
    return epochs


def detect_theta_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(5, 12),
    thresh=(0, 0.5),
    mindur=0.25,
    maxdur=5,
    mergedist=0.5,
    sigma=0.125,
    edge_cutoff=-0.25,
    ignore_epochs: Epoch = None,
):
    if probegroup is None:
        selected_chan = signal.channel_id
        traces = signal.traces
    else:
        if isinstance(probegroup, np.ndarray):
            changrps = np.array(probegroup, dtype="object")
        if isinstance(probegroup, ProbeGroup):
            changrps = probegroup.get_connected_channels(groupby="shank")
        channel_ids = np.concatenate(changrps).astype("int")

        duration = signal.duration
        t1, t2 = signal.t_start, signal.t_start + np.min([duration, 3600])
        signal_slice = signal.time_slice(channel_id=channel_ids, t_start=t1, t_stop=t2)
        hil_stat = signal_process.hilbert_amplitude_stat(
            signal_slice.traces,
            freq_band=freq_band,
            fs=signal.sampling_rate,
            statistic="mean",
        )
        selected_chan = channel_ids[np.argmax(hil_stat)]

        traces = signal.time_slice(channel_id=selected_chan).traces.reshape(1, -1)

    print(f"Best channel for theta: {selected_chan}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        ignore_times=ignore_times,
        sigma=sigma,
        edge_cutoff=edge_cutoff,
    )
    epochs = epochs.shift(dt=signal.t_start)
    epochs.metadata = dict(channels=selected_chan)
    return epochs


def detect_spindle_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(8, 16),
    thresh=(1, 5),
    mindur=0.35,
    maxdur=4,
    mergedist=0.05,
    ignore_epochs: Epoch = None,
    method="hilbert",
):
    if probegroup is None:
        selected_chans = signal.channel_id
        traces = signal.traces

    else:
        if isinstance(probegroup, np.ndarray):
            changrps = np.array(probegroup, dtype="object")
        if isinstance(probegroup, ProbeGroup):
            changrps = probegroup.get_connected_channels(groupby="shank")
            # if changrp:
        selected_chans = []
        for changrp in changrps:
            signal_slice = signal.time_slice(
                channel_id=changrp.astype("int"), t_start=0, t_stop=3600
            )
            hil_stat = signal_process.hilbert_amplitude_stat(
                signal_slice.traces,
                freq_band=freq_band,
                fs=signal.sampling_rate,
                statistic="mean",
            )
            selected_chans.append(changrp[np.argmax(hil_stat)])

        traces = signal.time_slice(channel_id=selected_chans).traces

    print(f"Selected channels for spindles: {selected_chans}")

    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs, metadata = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        ignore_times=ignore_times,
    )
    epochs["start"] = epochs["start"] + signal.t_start
    epochs["stop"] = epochs["stop"] + signal.t_start

    metadata["channels"] = selected_chans
    return Epoch(epochs=epochs, metadata=metadata)


def detect_gamma_epochs():
    pass


class Gamma:
    """Events and analysis related to gamma oscillations"""

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


if __name__ == "__main__":
    from neuropy.io import BinarysignalIO
    from neuropy.core import Shank, Probe, ProbeGroup, Epoch

    eegfile = BinarysignalIO(
        "/data/Working/Trace_FC/Recording_Rats/Finn/2022_01_18_habituation/Finn_habituation2_denoised.eeg",
        n_channels=35,
        sampling_rate=1250,
    )
    signal = eegfile.get_signal()
    prbgrp = ProbeGroup().from_file(
        "/data/Working/Trace_FC/Recording_Rats/Finn/2022_01_18_habituation/Finn_habituation2_denoised.probegroup.npy"
    )
    art_epochs = Epoch(
        epochs=None,
        file="/data/Working/Trace_FC/Recording_Rats/Finn/2022_01_18_habituation/Finn_habituation2_denoised.art_epochs.npy",
    )
    ripple_epochs = detect_ripple_epochs(
        signal, prbgrp, thresh=(2.5, None), ignore_epochs=art_epochs, mindur=0.025
    )
