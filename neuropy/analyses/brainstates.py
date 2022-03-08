from pathlib import Path
from pstats import Stats
from tracemalloc import start
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from ..utils import signal_process, mathutil
from .. import core
from ..plotting import plot_hypnogram


def hmmfit1d(Data):
    # hmm states on 1d data and returns labels with highest mean = highest label
    flag = None
    if np.isnan(Data).any():
        nan_indices = np.where(np.isnan(Data) == 1)[0]
        non_nan_indices = np.where(np.isnan(Data) == 0)[0]
        Data_og = Data
        Data = np.delete(Data, nan_indices)
        hmmlabels = np.nan * np.ones(len(Data_og))
        flag = 1

    Data = (np.asarray(Data)).reshape(-1, 1)
    model = GaussianHMM(n_components=2, n_iter=100).fit(Data)
    hidden_states = model.predict(Data)
    mus = np.squeeze(model.means_)
    sigmas = np.squeeze(np.sqrt(model.covars_))
    transmat = np.array(model.transmat_)

    idx = np.argsort(mus)
    mus = mus[idx]
    sigmas = sigmas[idx]
    transmat = transmat[idx, :][:, idx]

    state_dict = {}
    states = [i for i in range(4)]
    for i in idx:
        state_dict[idx[i]] = states[i]

    relabeled_states = np.asarray([state_dict[h] for h in hidden_states])
    relabeled_states[:2] = [0, 0]
    relabeled_states[-2:] = [0, 0]

    if flag:

        hmmlabels[non_nan_indices] = relabeled_states

    else:
        hmmlabels = relabeled_states

    # state_diff = np.diff(relabeled_states)
    # start = np.where(state_diff == 1)[0]
    # stop = np.where(state_diff == -1)[0]

    # for s, e in zip(start, stop):
    #     if e - s < 50:
    #         relabeled_states[s + 1 : e] = 0
    # print(start_ripple.shape, stop_ripple.shape)
    # states = np.concatenate((start_ripple, stop_ripple), axis=1)

    # relabeled_states = hidden_states
    return hmmlabels


def correlation_emg(
    signal: core.Signal, probe: core.ProbeGroup, window, overlap, n_jobs=8
):
    """Calculating emg

    Parameters
    ----------
    window : int
        window size in seconds
    overlap : float
        overlap between windows in seconds
    n_jobs: int,
        number of cpu/processes to use

    Returns
    -------
    array
        emg calculated at each time window
    """
    print("starting emg calculation")
    highfreq = 600
    lowfreq = 300
    t_start = signal.t_start
    t_stop = signal.t_stop
    sRate = signal.sampling_rate
    n_probes = probe.n_probes
    changrp = np.concatenate(probe.get_connected_channels(groupby="probe"))

    emg_chans = changrp.astype("int")
    n_emg_chans = len(emg_chans)
    # eegdata = signal.time_slice(channel_id=emg_chans)
    total_duration = signal.duration

    timepoints = np.arange(t_start, t_stop - window, window - overlap)

    # ---- Mean correlation across all selected channels calculated in parallel --
    def corrchan(start):
        lfp_req = np.asarray(
            signal.time_slice(
                channel_id=emg_chans, t_start=start, t_stop=start + window
            ).traces
        )
        yf = signal_process.filter_sig.bandpass(
            lfp_req, lf=lowfreq, hf=highfreq, fs=sRate
        )
        ltriang = np.tril_indices(n_emg_chans, k=-1)
        return np.corrcoef(yf)[ltriang].mean()

    corr_per_frame = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(corrchan)(start) for start in timepoints
    )
    emg_lfp = np.asarray(corr_per_frame)

    print("emg calculation done")
    return emg_lfp


def _states2time(label):

    states = np.unique(label)

    all_states = []
    for state in states:

        binary = np.where(label == state, 1, 0)
        binary = np.concatenate(([0], binary, [0]))
        binary_change = np.diff(binary)

        start = np.where(binary_change == 1)[0]
        end = np.where(binary_change == -1)[0]
        start = start[:-1]
        end = end[:-1]
        # duration = end - start
        stateid = state * np.ones(len(start))
        firstPass = np.vstack((start, end, stateid)).T

        all_states.append(firstPass)

    all_states = np.concatenate(all_states)

    return all_states


def detect_brainstates_epochs(
    signal: core.Signal,
    probe: core.ProbeGroup,
    window=1,
    overlap=0.2,
    sigma=3,
    emg_channel=None,
    theta_channel=None,
    delta_channel=None,
    behavior_epochs=None,
    min_dur=6,
    ignore_epochs: core.Epoch = None,
    plot=True,
    plot_filename=None,
):
    """detects sleep states for the recording

    Parameters
    ----------
    chans : int, optional
        channel you want to use for sleep detection, by default None
    window : int, optional
        bin size, by default 1
    overlap : float, optional
        seconds of overlap between adjacent window , by default 0.2
    emgfile : bool, optional
        if True load the emg file in the basepath, by default False
    behavior_epochs : None,
        These are eopchs when the animal was put on a track for behavior.
        Using wavelet these epochs can be further fine tuned for accurate active and quiet period.
    """

    changrp = probe.get_connected_channels(groupby="shank")
    wp = dict(window=window, overlap=overlap)

    smooth_ = lambda arr: gaussian_filter1d(arr, sigma=sigma / (window - overlap))
    print(f"channel for sleep detection: {theta_channel,delta_channel}")

    # ---- theta-delta ratio calculation -----
    if theta_channel is not None:
        theta_signal = signal.time_slice(channel_id=[theta_channel])
        theta_chan_sg = signal_process.FourierSg(theta_signal, norm_sig=True, **wp)
        theta = smooth_(theta_chan_sg.theta)
        band30 = smooth_(theta_chan_sg.get_band_power(1, 30))
        time = theta_chan_sg.time

    if delta_channel is not None:
        delta_signal = signal.time_slice(channel_id=[theta_channel])
        delta_chan_sg = signal_process.FourierSg(delta_signal, norm_sig=True, **wp)
        delta = smooth_(delta_chan_sg.delta)

    print(f"spectral properties calculated")

    theta_delta_ratio = theta / delta

    # ---- emg processing ----
    if emg_channel is None:
        emg = correlation_emg(signal=signal, probe=probe, **wp)
        emg = smooth_(emg)
    else:
        emg = signal.time_slice(emg_channel)

    # ----- set timepoints from ignore_epochs to np.nan ------

    if ignore_epochs is not None:
        noisy = ignore_epochs.as_array()
        noisy_timepoints = []
        for noisy_ind in range(noisy.shape[0]):
            st = noisy[noisy_ind, 0]
            en = noisy[noisy_ind, 1]
            noisy_indices = np.where((time > st) & (time < en))[0]
            noisy_timepoints.extend(noisy_indices)

        band30[noisy_timepoints] = np.nan
        theta_delta_ratio[noisy_timepoints] = np.nan
        emg[noisy_timepoints] = np.nan

    band30_label = hmmfit1d(1 / band30)
    theta_delta_label = hmmfit1d(theta_delta_ratio)
    emg_label = hmmfit1d(emg)

    states = 3 * np.ones(len(band30_label))  # initialize all states to 3 (qyiet awake)
    states[(band30_label == 1) & (emg_label == 1)] = 4  # Wake
    states[(band30_label == 1) & (emg_label == 0) & (theta_delta_label == 0)] = 3  # QW
    states[(band30_label == 1) & (emg_label == 0) & (theta_delta_label == 1)] = 2  # REM
    states[
        (band30_label == 0) & (emg_label == 0) & (theta_delta_label == 0)
    ] = 1  # NREM

    statetime = (_states2time(states)).astype(int)

    epochs = pd.DataFrame(
        {
            "start": time[statetime[:, 0]],
            "stop": time[statetime[:, 1]],
            "duration": time[statetime[:, 1]] - time[statetime[:, 0]],
            "state": statetime[:, 2],
        }
    )
    state_to_label = {1: "nrem", 2: "rem", 3: "quiet", 4: "active"}
    epochs["label"] = epochs["state"].map(state_to_label)
    epochs.drop("state", axis=1, inplace=True)
    metadata = {"window": window, "overlap": overlap}
    epochs = core.Epoch(epochs=epochs)
    epochs = epochs.duration_slice(min_dur=min_dur)
    epochs = epochs.fill_blank("from_left")  # this will also fill ignore_epochs

    # clearing out ignore_epochs
    if ignore_epochs is not None:
        for e in ignore_epochs.as_array():
            epochs = epochs.delete_in_between(e[0], e[1])

    if behavior_epochs is not None:
        for e in behavior_epochs.as_array():
            epochs = epochs.delete_in_between(e[0], e[1])
            theta_in_epoch = theta_signal.time_slice(t_start=e[0], t_stop=e[1])
            # bandpass filter in broad band theta
            theta_bp = signal_process.filter_sig.bandpass(
                theta_in_epoch, lf=1, hf=25
            ).traces[0]
            hilbert_amp = stats.zscore(np.abs(signal_process.hilbertfast(theta_bp)))
            high_theta = mathutil.threshPeriods(
                hilbert_amp,
                lowthresh=0,
                highthresh=0.5,
                minDistance=250,
                minDuration=1250,
            )
            low_theta = np.vstack((high_theta[:-1, 1], high_theta[1:, 0])).T

            if high_theta[0, 0] > e[0]:
                low_theta = np.insert(low_theta, 0, [e[0], high_theta[0, 0]])

            if high_theta[-1, -1] < e[1]:
                low_theta = np.insert(low_theta, -1, [high_theta[-1, -1], e[1]])

            new_epochs = (
                np.vstack((high_theta, low_theta)) / signal.sampling_rate
            ) + e[0]
            new_labels = ["active"] * high_theta.shape[0] + ["quiet"] * low_theta.shape[
                0
            ]
            states_in_epoch = core.Epoch.from_array(
                new_epochs[:, 0], new_epochs[:, 1], new_labels
            )

            # update the epochs to include these new ones
            epochs = epochs + states_in_epoch

    epochs.metadata = metadata

    if plot:
        fig, axs = plt.subplots(5, 1, sharex=True)

        axs[0].plot(time, stats.zscore(delta))
        axs[0].set_ylim([-0.5, 1.5])
        axs[1].plot(time, stats.zscore(theta_delta_ratio))
        axs[1].set_ylim([-0.3, 4])
        axs[2].plot(time, 1 / band30)
        axs[3].plot(time, emg)
        plot_hypnogram(epochs, ax=axs[4])

    return epochs
