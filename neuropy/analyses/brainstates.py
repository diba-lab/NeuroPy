from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as filtSig
import scipy.stats as stats
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed

from ..utils import signal_process
from .. import core


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
    sRate = signal.sampling_rate
    n_probes = probe.n_probes
    changrp = probe.get_connected_channels(groupby="probe")

    # ----selecting a fixed number of shanks from each probe-----
    # max_shanks_probe = [np.min([3, _]) for _ in nShanksProbe]
    # selected_shanks = []
    # for probe in range(nProbes):
    #     shanks_in_probe = [
    #         changrp[_] for _ in np.where(probesid == probe)[0] if changrp[_]
    #     ]
    #     selected_shanks.append(
    #         np.concatenate(
    #             np.random.choice(
    #                 shanks_in_probe, max_shanks_probe[probe], replace=False
    #             )
    #         )
    #     )

    # ---- selecting probe with most number of shanks -------
    # which_probe = np.argmax(nShanksProbe)
    # selected_shanks = np.where(probesid == which_probe)[0]
    # # making sure shanks are not empty
    # selected_shanks = [changrp[_] for _ in selected_shanks if changrp[_]]

    emg_chans = (changrp[0]).astype("int")
    n_emg_chans = len(emg_chans)
    # eegdata = signal.time_slice(channel_id=emg_chans)
    total_duration = signal.duration

    timepoints = np.arange(0, total_duration - window, window - overlap)

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


def _label2states(theta_delta, delta_l, emg_l):

    state = np.zeros(len(theta_delta))
    for i, (ratio, delta, emg) in enumerate(zip(theta_delta, delta_l, emg_l)):

        if ratio == 1 and emg == 1:  # active wake
            state[i] = 4
        elif ratio == 0 and emg == 1:  # quiet wake
            state[i] = 3
        elif ratio == 1 and emg == 0:  # REM
            state[i] = 2
        elif ratio == 0 and emg == 0:  # NREM
            state[i] = 1

    return state


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


def _removetransient(statetime):

    duration = statetime.duration
    start = statetime.start
    end = statetime.stop
    state = statetime.state

    arr = np.zeros((len(start), 4))
    arr[:, 0] = start
    arr[:, 1] = end
    arr[:, 2] = duration
    arr[:, 3] = state

    srt_ind = np.argsort(arr[:, 0])
    arr = arr[srt_ind, :]

    ind = 1
    while ind < len(arr) - 1:
        if (arr[ind, 2] < 50) and (arr[ind - 1, 3] == arr[ind + 1, 3]):
            arr[ind - 1, :] = np.array(
                [
                    arr[ind - 1, 0],
                    arr[ind + 1, 1],
                    arr[ind + 1, 1] - arr[ind - 1, 0],
                    arr[ind - 1, 3],
                ]
            )
            arr = np.delete(arr, [ind, ind + 1], 0)
        else:
            ind += 1

    statetime = pd.DataFrame(
        {
            "start": arr[:, 0],
            "stop": arr[:, 1],
            "duration": arr[:, 2],
            "state": arr[:, 3],
        }
    )

    return statetime


def detect_brainstates_epochs(
    signal: core.Signal,
    probe: core.ProbeGroup,
    window=1,
    overlap=0.2,
    emg: core.Signal = None,
    sigma=10,
    ignore_epochs=None,
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

    """

    if emg is None:
        emg = correlation_emg(
            signal=signal, probe=probe, window=window, overlap=overlap
        )

    emg = filtSig.gaussian_filter1d(emg, sigma=sigma)

    changrp = probe.get_connected_channels(groupby="shank")
    chan = changrp[7][-1]

    print(f"channel for sleep detection: {chan}")

    score_signal = signal.time_slice(channel_id=[chan])
    bands = signal_process.SpectrogramBands(
        score_signal, window=window, overlap=overlap, smooth=sigma, norm_sig=True
    )
    time = bands.time
    deltaplus = bands.deltaplus
    theta = bands.theta
    theta_deltaplus_ratio = theta / deltaplus
    print(f"spectral properties calculated")

    # if (noisy := artifact.time) is not None:
    #     noisy_timepoints = []
    #     for noisy_ind in range(noisy.shape[0]):
    #         st = noisy[noisy_ind, 0]
    #         en = noisy[noisy_ind, 1]
    #         noisy_indices = np.where((time > st) & (time < en))[0]
    #         noisy_timepoints.extend(noisy_indices)

    #     theta_deltaplus_ratio[noisy_timepoints] = np.nan
    #     emg[noisy_timepoints] = np.nan
    #     deltaplus[noisy_timepoints] = np.nan

    deltaplus_label = hmmfit1d(deltaplus)
    theta_deltaplus_label = hmmfit1d(theta_deltaplus_ratio)
    emg_label = hmmfit1d(emg)

    states = _label2states(theta_deltaplus_label, deltaplus_label, emg_label)
    statetime = (_states2time(states)).astype(int)

    statetime = pd.DataFrame(
        {
            "start": time[statetime[:, 0]],
            "stop": time[statetime[:, 1]],
            "duration": time[statetime[:, 1]] - time[statetime[:, 0]],
            "state": statetime[:, 2],
        }
    )

    epochs = _removetransient(statetime)

    state_to_label = {1: "nrem", 2: "rem", 3: "quiet", 4: "active"}
    epochs["label"] = epochs["state"].map(state_to_label)
    epochs.drop("state", axis=1, inplace=True)
    metadata = {"window": window, "overlap": overlap}

    return core.Epoch(epochs=epochs, metadata=metadata)
