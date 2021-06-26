import numpy as np
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
    spikes = self.times
    self.info["celltype"] = None
    ccgs = calculate_neurons_acg(spikes=spikes, bin_size=0.001, window_size=0.05)
    ccg_width = ccgs.shape[-1]
    ccg_center_ind = int(ccg_width / 2)

    # -- calculate burstiness (mean duration of right ccg)------
    ccg_right = ccgs[:, ccg_center_ind + 1 :]
    t_ccg_right = np.arange(ccg_right.shape[1])  # timepoints
    mean_isi = np.sum(ccg_right * t_ccg_right, axis=1) / np.sum(ccg_right, axis=1)

    # --- calculate frate ------------
    frate = np.asarray([len(cell) / np.ptp(cell) for cell in spikes])

    # ------ calculate peak ratio of waveform ----------
    templates = self.templates
    waveform = np.asarray(
        [cell[np.argmax(np.ptp(cell, axis=1)), :] for cell in templates]
    )
    n_t = waveform.shape[1]  # waveform width
    center = np.int(n_t / 2)
    wave_window = int(0.25 * (self._obj.sampfreq / 1000))
    from_peak = int(0.18 * (self._obj.sampfreq / 1000))
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
    mua_cells = np.where(ref_period_ratio < 400)[0]
    good_cells = np.where(ref_period_ratio >= 400)[0]

    self.info.loc[mua_cells, "celltype"] = "mua"

    param1 = frate[good_cells]
    param2 = mean_isi[good_cells]
    param3 = diff_auc[good_cells]

    features = np.vstack((param1, param2, param3)).T
    features = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=2).fit(features)
    y_means = kmeans.predict(features)

    interneuron_label = np.argmax(kmeans.cluster_centers_[:, 0])
    intneur_id = np.where(y_means == interneuron_label)[0]
    pyr_id = np.where(y_means != interneuron_label)[0]
    self.info.loc[good_cells[intneur_id], "celltype"] = "intneur"
    self.info.loc[good_cells[pyr_id], "celltype"] = "pyr"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        frate[mua_cells],
        mean_isi[mua_cells],
        diff_auc[mua_cells],
        c=self.colors["mua"],
        s=50,
        label="mua",
    )

    ax.scatter(
        param1[pyr_id],
        param2[pyr_id],
        param3[pyr_id],
        c=self.colors["pyr"],
        s=50,
        label="pyr",
    )

    ax.scatter(
        param1[intneur_id],
        param2[intneur_id],
        param3[intneur_id],
        c=self.colors["intneur"],
        s=50,
        label="int",
    )
    ax.legend()
    ax.set_xlabel("Firing rate (Hz)")
    ax.set_ylabel("Mean isi (ms)")
    ax.set_zlabel("Difference of \narea under shoulders")

    data = np.load(self.files.spikes, allow_pickle=True).item()
    data["info"] = self.info


def calculate_neurons_acg(
    self, spikes=None, bin_size=0.001, window_size=0.05
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

    if isinstance(spikes, np.ndarray):
        spikes = [spikes]
    nCells = len(spikes)

    correlo = []
    for cell in spikes:
        cell_id = np.zeros(len(cell)).astype(int)
        acg = correlograms(
            cell,
            cell_id,
            sample_rate=self._obj.sampfreq,
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
