import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .. import core
from .ccg import correlograms
from typing import List, Union
from scipy.ndimage import gaussian_filter1d


def estimate_neuron_type(neurons: Union[core.Neurons, List[core.Neurons]]):
    """Auto label cell type using firing rate, burstiness and waveform shape followed by kmeans clustering.

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

    features, ref_period_ratios, n_neurons = [], [], []
    for neurons in all_neurons:
        n_neurons.append(neurons.n_neurons)
        spikes = neurons.spiketrains
        sampling_rate = neurons.sampling_rate
        ccgs = calculate_neurons_acg(neurons, bin_size=0.001, window_size=0.05)
        ccg_width = ccgs.shape[-1]
        ccg_center_ind = int(ccg_width / 2)

        # -- calculate burstiness (mean duration of right ccg)------
        ccg_right = ccgs[:, ccg_center_ind + 1 :]
        t_ccg_right = np.arange(ccg_right.shape[1])  # timepoints
        mean_isi = np.sum(ccg_right * t_ccg_right, axis=1) / np.sum(ccg_right, axis=1)

        # --- calculate frate ------------
        frate = np.asarray([len(cell) / np.ptp(cell) for cell in spikes])

        # ------ calculate peak ratio of waveform ----------
        waveform = neurons.waveforms
        # waveform = np.asarray(
        #     [cell[np.argmax(np.ptp(cell, axis=1)), :] for cell in templates]
        # )

        n_t = waveform.shape[1]  # waveform width
        center = np.int(n_t / 2)
        wave_window = int(0.25 * (sampling_rate / 1000))
        from_peak = int(0.18 * (sampling_rate / 1000))
        left_peak = np.trapz(
            waveform[:, center - from_peak - wave_window : center - from_peak], axis=1
        )
        right_peak = np.trapz(
            waveform[:, center + from_peak : center + from_peak + wave_window], axis=1
        )

        diff_auc = left_peak - right_peak

        # ---- refractory contamination ----------
        isi = [np.diff(_) for _ in spikes]
        isi_bin = np.arange(0, 0.1, 0.001)
        isi_hist = np.asarray([np.histogram(_, bins=isi_bin)[0] for _ in isi])
        n_spikes_ref = np.sum(isi_hist[:, :2], axis=1) + 1e-16

        ref_period_ratios.append(n_spikes_ref / np.max(isi_hist, axis=1))

        features.append(np.vstack((frate, mean_isi, diff_auc)))

    neuron_type = np.ones(np.sum(n_neurons), dtype="U5")

    # ---- auto selection of MUA ----------
    ref_period_ratios = np.concatenate(ref_period_ratios)
    ref_period_thresh = np.std(ref_period_ratios)
    mua_id = np.where(ref_period_ratios > ref_period_thresh)[0]
    good_id = np.where(ref_period_ratios <= ref_period_thresh)[0]

    # print(ref_period_ratios)
    features = np.hstack(features).T
    features_good = features[good_id, :]
    features_good = StandardScaler().fit_transform(features_good)
    kmeans = KMeans(n_clusters=2).fit(features_good)
    y_means = kmeans.predict(features_good)

    # interneuron has higher firing rates
    interneuron_label = np.argmax(kmeans.cluster_centers_[:, 0])
    intneur_id = np.where(y_means == interneuron_label)[0]
    pyr_id = np.where(y_means != interneuron_label)[0]

    # intneur_id = np.setdiff1d(intneur_id, mua_id)
    # pyr_id = np.setdiff1d(pyr_id, mua_id)
    neuron_type[mua_id] = "mua"
    neuron_type[good_id[intneur_id]] = "inter"
    neuron_type[good_id[pyr_id]] = "pyr"

    colors = np.ones(np.sum(n_neurons), dtype=object)
    colors[mua_id] = "#bcb8b8"
    colors[good_id[intneur_id]] = "#5da42d"
    colors[good_id[pyr_id]] = "#222020"

    fig = plt.figure()
    ax = fig.add_subplot(121)
    bins = np.arange(0, 1, 0.01)
    ax.plot(bins[:-1], np.histogram(ref_period_ratios, bins=bins)[0], "gray")
    ax.axvline(ref_period_thresh, ls="--", color="r")
    ax.set_xlabel("Refractory period ratio (#spikes in 2ms)/max_peak_isi)")
    ax.set_ylabel("No. of neurons")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(122, projection="3d")
    ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=colors, s=50)

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


def calculate_neurons_ccg(self, ids):
    spikes = self.get_spiketrains(ids)
    ccgs = np.zeros((len(spikes), len(spikes), 251)) * np.nan
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
        sample_rate=self._obj.sampfreq,
        bin_size=0.001,
        window_size=0.25,
    )
    grid = np.ix_(spike_ind, spike_ind, np.arange(251))
    ccgs[grid] = ccgs_

    center = int(ccgs.shape[-1] / 2) - 1
    diff = ccgs[:, :, center + 2 :].sum(axis=-1) - ccgs[:, :, : center - 2].sum(axis=-1)
    non_redundant_indx = np.tril_indices_from(diff, k=-1)

    return diff[non_redundant_indx]


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
