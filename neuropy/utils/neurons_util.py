import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .. import core
from .ccg import correlograms


def estimate_neuron_type(neurons: core.Neurons):
    """Auto label cell type using firing rate, burstiness and waveform shape followed by kmeans clustering.

    Reference
    ---------
    Csicsvari, J., Hirase, H., Czurko, A., & Buzsáki, G. (1998). Reliability and state dependence of pyramidal cell–interneuron synapses in the hippocampus: an ensemble approach in the behaving rat. Neuron, 21(1), 179-189.
    """

    n_neurons = neurons.n_neurons
    neuron_type = np.ones(n_neurons, dtype="U5")
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
    ref_period_ratio = (np.max(isi_hist, axis=1) / n_spikes_ref) * 100
    mua_id = np.where(ref_period_ratio < 400)[0]
    # good_cells = np.where(ref_period_ratio >= 400)[0]

    neuron_type[mua_id] = "mua"

    param1 = frate
    param2 = mean_isi
    param3 = diff_auc

    features = np.vstack((param1, param2, param3)).T
    features = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=2).fit(features)
    y_means = kmeans.predict(features)

    interneuron_label = np.argmax(kmeans.cluster_centers_[:, 0])
    intneur_id = np.where(y_means == interneuron_label)[0]
    pyr_id = np.where(y_means != interneuron_label)[0]

    intneur_id = np.setdiff1d(intneur_id, mua_id)
    pyr_id = np.setdiff1d(pyr_id, mua_id)
    neuron_type[intneur_id] = "inter"
    neuron_type[pyr_id] = "pyr"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        param1[mua_id],
        param2[mua_id],
        param3[mua_id],
        c="#bcb8b8",
        s=50,
        label="mua",
    )

    ax.scatter(
        param1[pyr_id],
        param2[pyr_id],
        param3[pyr_id],
        c="#222020",
        s=50,
        label="pyr",
    )

    ax.scatter(
        param1[intneur_id],
        param2[intneur_id],
        param3[intneur_id],
        c="#5da42d",
        s=50,
        label="int",
    )
    ax.legend()
    ax.set_xlabel("Firing rate (Hz)")
    ax.set_ylabel("Mean isi (ms)")
    ax.set_zlabel("Difference of \narea under shoulders")

    return neuron_type


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
