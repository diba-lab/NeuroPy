import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from .. import core
from .ccg import correlograms
from typing import List, Union
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


def acg_fit(x, a, b, c, d, e, f, g, h):
    """Approximation function to fit autocorrelograms of spike trains
    Equation obtained from:
    https://cellexplorer.org/pipeline/acg-fit/
    Parameters
    ----------
    x : _type_
        _description_
    a : float
        tau_decay
    b : float
        tau_rise
    c : float
        _description_
    d : float
        _description_
    e : float
        _description_
    f : float
        _description_
    g : float
        _description_
    h : float
        _description_

    Returns
    -------
    _type_
        _description_
    """
    eqtn = (
        c * (np.exp(-(x - f) / a) - d * np.exp(-(x - f) / b))
        + h * np.exp(-(x - f) / g)
        + e
    )
    return np.array([np.max([_, 0]) for _ in eqtn])


def acg_no_burst_fit(x, a, b, c, d, e, f):
    """Approximation function to fit autocorrelograms of spike trains
    Equation obtained from:
    https://cellexplorer.org/pipeline/acg-fit/
    Parameters
    ----------
    x : _type_
        _description_
    a : float
        tau_decay
    b : float
        tau_rise
    c : float
        _description_
    d : float
        _description_
    e : float
        _description_
    f : float
        _description_
    g : float
        _description_
    h : float
        _description_

    Returns
    -------
    _type_
        _description_
    """
    eqtn = c * (np.exp(-(x - f) / a) - d * np.exp(-(x - f) / b)) + e
    return np.array([np.max([_, 0]) for _ in eqtn])


def estimate_neuron_type(
    neurons: Union[core.Neurons, List[core.Neurons]], mua_thresh=1
):
    """Auto label cell type using firing rate, burstiness and waveform shape followed by kmeans clustering.

    Parameters
    ----------
    neurons: core.Neurons
        object containing spiketrains
    mua_thresh: 1<=float<=100
        the percentage refractory period violations

    Reference
    ---------
    Csicsvari, J., Hirase, H., Czurko, A., & Buzsáki, G. (1998). Reliability and state dependence of pyramidal cell–interneuron synapses in the hippocampus: an ensemble approach in the behaving rat. Neuron, 21(1), 179-189.
    """
    if isinstance(neurons, core.Neurons):
        print("It is advisable to use this estimation across multiple neurons object")
        all_neurons = [neurons]
    else:
        assert isinstance(neurons, list), "input can either be list or neurons"
        all_neurons = neurons

    spiketrains = np.concatenate([_.spiketrains for _ in all_neurons])
    nspikes = np.array([len(_) for _ in spiketrains])
    n_neurons = np.array([_.n_neurons for _ in all_neurons])

    # ---- calculate acgs -------
    acgs = np.vstack(
        [
            calculate_neurons_acg(_, bin_size=0.0005, window_size=0.1)
            for _ in all_neurons
        ]
    )
    acgs_max = acgs.max(axis=-1)  # max number of spikes in acg
    acgs = acgs / nspikes.reshape(-1, 1) / 0.0005  # changing to rate
    acgs_nbins = acgs.shape[-1]
    acgs_center_indx = acgs_nbins // 2

    # -- calculate burstiness (mean duration of right ccg)------
    acgs_right = acgs[:, acgs_center_indx + 1 :]
    t_ccg_right = np.arange(acgs_right.shape[1])  # timepoints
    mean_isi = np.sum(acgs_right * t_ccg_right, axis=1) / np.sum(acgs_right, axis=1)
    # mean_isi = np.max(acgs_right, axis=1) / acgs_right[:, -1]
    # mean_isi = np.ma.fix_invalid(mean_isi, fill_value=0)

    # --- calculate frate ------------
    frate = np.asarray([len(_) / np.ptp(_) for _ in spiketrains])

    # ------ calculate shoulder ratio of waveform ----------
    diff_auc = []
    for neur in all_neurons:
        waveform = neurons.waveforms
        # Channels with maximum negative peak are considered peak channels are considered as the peak waveform representing the neuron
        peak_chan_indx = np.argmin(np.min(waveform, axis=2), axis=1)
        waveform = waveform[np.arange(waveform.shape[0]), peak_chan_indx, :]
        # waveform = np.asarray(
        #     [cell[np.argmax(np.ptp(cell, axis=1)), :] for cell in templates]
        # )

        n_t = waveform.shape[1]  # waveform width
        center = np.int(n_t / 2)
        wave_window = int(0.25 * (neur.sampling_rate / 1000))
        from_peak = int(0.18 * (neur.sampling_rate / 1000))
        left_peak = np.trapz(
            waveform[:, center - from_peak - wave_window : center - from_peak], axis=1
        )
        right_peak = np.trapz(
            waveform[:, center + from_peak : center + from_peak + wave_window], axis=1
        )

        diff_auc.append(left_peak - right_peak)
    diff_auc = np.concatenate(diff_auc)

    # ---- refractory contamination ----------
    isi = np.concatenate([_.get_isi(bin_size=0.001, n_bins=100) for _ in all_neurons])
    ref_nspikes = isi[:, :2].sum(axis=1)
    violations = ref_nspikes * 100 / nspikes  # violation percent

    # ---- selection of MUA ----------
    mua_indx = violations > mua_thresh

    # ----- selection LUA -----
    # Excluding units which have very low number of spikes in autocorrelogram making it difficult for exponential fit
    low_indx = acgs_max <= 5

    good_indx = ~np.logical_or(mua_indx, low_indx)

    # ----- fitting acgs and calculating parameters -----
    acgs_good = acgs[good_indx, :]
    acgs_good[:, 100:105] = 0  # making refractory period zero for better fitting
    t = np.arange(0.5, 50.5, 0.5)  # timepoints for half acg

    p_initial = [20, 1, 30, 2, 0.5, 5, 1.5, 0]

    params = []
    for y in acgs_good:
        lb = np.array([1, 0.1, -5, 0, -100, 0, 0.1, 0])
        ub = np.array([500, 50, 500, 25, 70, 20, 5, 100])

        try:
            popt, _ = curve_fit(
                acg_no_burst_fit,
                t,
                y[101:],
                p0=p_initial[:-2],
                bounds=(lb[:-2], ub[:-2]),
            )
        except:
            popt, _ = curve_fit(acg_fit, t, y[101:], p0=p_initial, bounds=(lb, ub))

        params.append(popt)
    tau_decay = np.log10(np.array([_[0] for _ in params]))
    tau_rise = np.array([_[1] for _ in params])

    # ----- cluster classification ------
    features = np.vstack(
        (
            frate[good_indx],
            mean_isi[good_indx],
            tau_decay,
            diff_auc[good_indx],
            tau_rise,
        )
    ).T
    features = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=2, n_init=100).fit(features)
    y_means = kmeans.predict(features)

    # interneuron has higher firing rates
    interneuron_label = np.argmax(kmeans.cluster_centers_[:, 1])
    # interneuron_label = np.argmax(
    #     [features[y_means == 0, 1].mean(), features[y_means == 1, 1].mean()]
    # )
    intneur_indx = np.where(y_means == interneuron_label)[0]
    pyr_indx = np.where(y_means != interneuron_label)[0]

    # ---- labeling -----
    neuron_type = np.ones(np.sum(n_neurons), dtype="U5")
    neuron_type[mua_indx] = "mua"  # multi unit activity
    neuron_type[low_indx] = "lua"  # low unit activity
    neuron_type[np.where(good_indx)[0][intneur_indx]] = "inter"
    neuron_type[np.where(good_indx)[0][pyr_indx]] = "pyr"

    colors = np.ones(np.sum(n_neurons), dtype=object)
    colors[neuron_type == "mua"] = "#bcb8b8"
    colors[neuron_type == "lua"] = "#D50000"
    colors[neuron_type == "inter"] = "#5da42d"
    colors[neuron_type == "pyr"] = "#212121"

    fig = plt.figure()
    ax = fig.add_subplot(121)
    bins = np.arange(0, 100, 1)
    ax.plot(bins[:-1], np.histogram(violations, bins=bins)[0], "gray")
    ax.axvline(mua_thresh, ls="--", color="r")
    ax.set_xlabel("Refractory period violations (%)")
    ax.set_ylabel("No. of neurons")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(122, projection="3d")
    ax.scatter(
        features[:, 1], features[:, 2], features[:, 3], c=list(colors[good_indx]), s=50
    )

    ax.set_xlabel("Firing rate (Hz)")
    ax.set_ylabel("Mean isi (ms)")
    ax.set_zlabel("Difference of \narea under shoulders")

    return np.split(neuron_type, np.cumsum(n_neurons)[:-1])


def calculate_neurons_acg(
    neurons: core.Neurons,
    bin_size=0.001,
    window_size=0.05,
    plot=True,
) -> np.ndarray:
    """Get autocorrelogram

    Parameters
    ----------
    spikes : [type], optional
        [description], by default None
    bin_size : float, optional
        [description], by default 0.001
    window_size : float, optional
        [description], by default 0.05
    """

    spikes = neurons.spiketrains

    correlo = []
    for cell in spikes:
        cell_id = np.zeros(len(cell)).astype(int)
        acg = correlograms(
            cell,
            cell_id,
            sample_rate=neurons.sampling_rate,
            bin_size=bin_size,
            window_size=window_size,
        ).squeeze()

        if acg.size == 0:
            acg = np.zeros(acg.shape[-1])

        correlo.append(acg)

    return np.array(correlo)


def calculate_neurons_ccg(neurons: core.Neurons, bin_size=0.001, window_size=0.25):
    """_summary_

    Parameters
    ----------
    neurons : core.Neurons
        _description_
    bin_size : float, optional
        _description_, by default 0.001
    window_size : float, optional
        _description_, by default 0.25

    Returns
    -------
    ccgs: n_neurons x n_neurons x time
        _description_
    t: 1D array
        time in seconds
    """
    spikes = neurons.spiketrains
    t = np.arange(window_size / bin_size + 1) * bin_size - window_size / 2
    ccgs = np.zeros((len(spikes), len(spikes), len(t))) * np.nan
    spike_ind = np.asarray([_ for _ in range(len(spikes)) if spikes[_].size != 0])
    clus_id = np.concatenate([[_] * len(spikes[_]) for _ in range(len(spikes))]).astype(
        int
    )
    sort_ind = np.argsort(np.concatenate(spikes))
    spikes = np.concatenate(spikes)[sort_ind]
    clus_id = clus_id[sort_ind]
    ccgs_ = correlograms(
        spikes,
        clus_id,
        sample_rate=neurons.sampling_rate,
        bin_size=bin_size,
        window_size=window_size,
    )
    grid = np.ix_(spike_ind, spike_ind, np.arange(len(t)))
    ccgs[grid] = ccgs_

    return ccgs, t


def ccg_temporal_bias(neurons: core.Neurons, com=False, window_size=0.5):
    """Temporal bias calculated using cross-correlograms

    Parameters
    ----------
    neurons : core.Neurons
        object containing spiketrains
    com : bool, optional
        whether to return center of mass instead of raw spike count difference, by default False
    window_size : float, optional
        total window size around 0 of CCG, by default 0.5

    Returns
    -------
    array
        _description_
    """
    ccgs, t = calculate_neurons_ccg(
        neurons=neurons, bin_size=0.001, window_size=window_size
    )
    if com:
        ccg_com = np.sum(ccgs * t[np.newaxis, np.newaxis, :], axis=-1) / np.sum(
            ccgs, axis=-1
        )
        non_redundant_indx = np.tril_indices_from(ccg_com, k=-1)
        temporal_bias = ccg_com[non_redundant_indx]

    else:
        center = ccgs.shape[-1] // 2
        diff = ccgs[:, :, center + 1 :].sum(axis=-1) - ccgs[:, :, : center - 1].sum(
            axis=-1
        )
        non_redundant_indx = np.tril_indices_from(diff, k=-1)
        temporal_bias = diff[non_redundant_indx]

    return temporal_bias


def theta_modulation_index(neurons: core.Neurons, sigma=None, return_acg=False):
    """Theta modulation index based on auto-correlograms

    Parameters
    ----------
    neurons : core.Neurons
        neurons for which to calculate the index

    Returns
    -------
    tmi
        index for each neuron within neurons

    #TODO finding the second peak/trough and take a window around it for tmi ???
    """
    acg = calculate_neurons_acg(neurons, bin_size=0.001, window_size=0.5)
    if sigma is not None:
        acg = gaussian_filter1d(acg, axis=1, sigma=sigma)

    acg_right = acg[:, acg.shape[1] // 2 :]

    trough, peak = (acg_right[:, 50:70]).mean(axis=1), (acg_right[:, 80:160]).mean(
        axis=1
    )
    tmi = (peak - trough) / (peak + trough)

    if return_acg:
        return tmi, acg
    else:
        return tmi
