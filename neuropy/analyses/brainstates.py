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
from neuropy.core import epoch

from neuropy.core.epoch import Epoch

from ..utils import signal_process, mathutil
from .. import core
from ..plotting import plot_epochs


def hmmfit1d(Data, ret_means=False, **kwargs):
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
    model = GaussianHMM(n_components=2, n_iter=100, **kwargs).fit(Data)
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
    if ret_means:
        return hmmlabels, mus
    else:
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
    # changrp = np.concatenate(probe.get_connected_channels(groupby="probe"))
    probe_df = probe.to_dataframe()
    probe_df = probe_df[probe_df.connected == True]
    emg_chans = probe_df.channel_id.values.astype("int")
    n_emg_chans = len(emg_chans)

    # --- choosing pairs of channels spaced >140 um --------
    x, y = probe_df.x.values.astype("float"), probe_df.y.values.astype("float")
    squared_diff = lambda arr: (arr[:, np.newaxis] - arr[np.newaxis, :]) ** 2
    distance = np.sqrt(squared_diff(x) + squared_diff(y))
    distance_bool = distance > 150

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
        # ltriang = np.tril_indices(n_emg_chans, k=-1)
        ltriang = np.tril(distance_bool, k=-1)
        return np.corrcoef(yf)[ltriang].mean()

    corr_per_frame = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(corrchan)(start) for start in timepoints
    )
    emg_lfp = np.asarray(corr_per_frame)

    print("emg calculation done")
    return emg_lfp


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
    freqs = np.geomspace(0.5, 100)
    spect_kw = dict(window=window, overlap=overlap, freqs=freqs, norm_sig=True)

    smooth_ = lambda arr: gaussian_filter1d(arr, sigma=sigma / (window - overlap))
    print(f"channel for sleep detection: {theta_channel,delta_channel}")

    # ---- theta-delta ratio calculation -----
    if theta_channel is not None:
        theta_signal = signal.time_slice(channel_id=[theta_channel])
        theta_chan_sg = signal_process.FourierSg(theta_signal, **spect_kw)
        theta = smooth_(theta_chan_sg.theta)
        delta_all = smooth_(theta_chan_sg.get_band_power(2, 20))
        time = theta_chan_sg.time
        theta_ratio = stats.zscore(theta / delta_all)

    if delta_channel is not None:
        delta_signal = signal.time_slice(channel_id=[theta_channel])
        delta_chan_sg = signal_process.FourierSg(delta_signal, **spect_kw)
        delta = stats.zscore(smooth_(delta_chan_sg.delta))

    print(f"spectral properties calculated")

    # ---- emg processing ----
    if emg_channel is None:
        emg_kw = dict(window=window, overlap=overlap)
        emg = correlation_emg(signal=signal, probe=probe, **emg_kw)
        emg = smooth_(emg)
    else:
        print("Using emg_channel has not been implemented yet")
        # TODO: if one of the channels provides emg, use that
        # emg = signal.time_slice(emg_channel)

    emg = stats.zscore(emg)
    # ----- set timepoints from ignore_epochs to np.nan ------

    if ignore_epochs is not None:
        noisy = ignore_epochs.as_array()
        noisy_timepoints = []
        for noisy_ind in range(noisy.shape[0]):
            st = noisy[noisy_ind, 0]
            en = noisy[noisy_ind, 1]
            noisy_indices = np.where((time > st) & (time < en))[0]
            noisy_timepoints.extend(noisy_indices)

        delta[noisy_timepoints] = np.nan
        theta_ratio[noisy_timepoints] = np.nan
        emg[noisy_timepoints] = np.nan

    delta_bool = hmmfit1d(delta).astype("bool")
    theta_ratio_bool = hmmfit1d(theta_ratio).astype("bool")
    emg_bool = hmmfit1d(emg).astype("bool")

    # --- states: Active wake (AW), Quiet Wake (QW), REM, NREM
    # initialize all states to Quiet wake
    states = np.array(["QW"] * len(theta_ratio_bool), dtype="U4")
    states[emg_bool & theta_ratio_bool] = "AW"
    states[emg_bool & ~delta_bool & ~theta_ratio_bool] = "QW"
    states[~emg_bool & ~delta_bool & theta_ratio_bool] = "REM"
    states[~emg_bool & delta_bool & ~theta_ratio_bool] = "NREM"

    # ma_bool = emg_bool & delta_bool
    # pad = lambda x: np.pad(x, (1, 1), "constant", constant_values=(0, 0))
    # ma_crossings = np.diff(pad(ma_bool.astype("int")))
    # ma_start = np.where(ma_crossings == 1)[0]
    # ma_stop = np.where(ma_crossings == -1)[0]
    # ma_stop[ma_stop == len(ma_bool)] = len(ma_bool) - 1
    # assert len(ma_start) == len(ma_stop)
    # ma_epochs_arr = np.vstack((ma_start, ma_stop)).T
    # ma_duration = np.diff(ma_epochs_arr, axis=1).squeeze()
    # ma_epochs_arr = ma_epochs_arr[ma_duration < (60 / (window - overlap)), :]
    # ma_indices = np.concatenate([np.arange(e[0], e[1]) for e in ma_epochs_arr])
    # states[ma_indices] = "MA"

    # ---- removing very short epochs -----
    epochs = Epoch.from_string_array(states, t=time)
    metadata = {"window": window, "overlap": overlap}
    epochs = epochs.duration_slice(min_dur=min_dur)
    epochs = epochs.fill_blank("from_left")  # this will also fill ignore_epochs

    # epochs_labels = epochs.labels
    # for i in range(1, len(epochs)):
    #     if (epochs_labels[i] == "REM") and (epochs_labels[i - 1] == "QW"):
    #         epochs_labels[i] = "QW"
    #     if (epochs_labels[i] == "REM") and (epochs_labels[i - 1] == "AW"):
    #         epochs_labels[i] = "AW"

    # epochs = epochs.set_labels(epochs_labels)

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
            new_labels = ["AW"] * high_theta.shape[0] + ["QW"] * low_theta.shape[0]
            states_in_epoch = core.Epoch.from_array(
                new_epochs[:, 0], new_epochs[:, 1], new_labels
            )

            # update the epochs to include these new ones
            epochs = epochs + states_in_epoch

    epochs.metadata = metadata

    if plot:
        _, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True)
        params = [delta, theta_ratio, emg]
        params_names = ["Delta", "Theta ratio", "EMG"]

        for i, (param, name) in enumerate(zip(params, params_names)):
            axs[i].plot(time, param)
            axs[0].set_ylim([-1, 4])
            axs[i].set_title(name, loc="left")

        states_colors = dict(NREM="#e920e2", REM="#f7abf4", QW="#fbc77e", AW="#e28708")
        plot_epochs(
            epochs=epochs,
            ax=axs[3],
            labels_order=["NREM", "REM", "QW", "AW"],
            colors=states_colors,
        )

    return epochs
