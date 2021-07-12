import matplotlib.pyplot as plt
import matplotlib as mpl
from .. import core
import numpy as np


def plot_raster(
    neurons: core.Neurons,
    ax=None,
    sort_by_frate=False,
    color=None,
    marker="|",
    markersize=2,
    add_vert_jitter=False,
):
    """creates raster plot using spiktrains in neurons

    Parameters
    ----------
    neurons : list, optional
        Each array within list represents spike times of that unit, by default None
    ax : obj, optional
        axis to plot onto, by default None
    period : array like, optional
        only plot raster for spikes within this period, by default None
    sort_by_frate : bool, optional
        If true then sorts spikes by the number of spikes (frate), by default False
    tstart : int, optional
        positions the x-axis labels to start from this, by default 0
    color : [type], optional
        color for raster plots, by default None
    marker : str, optional
        marker style, by default "|"
    markersize : int, optional
        size of marker, by default 2
    add_vert_jitter: boolean, optional
        adds vertical jitter to help visualize super dense spiking, not standardly used for rasters...
    """
    if ax is None:
        fig, ax = plt.subplots()

    n_neurons = neurons.n_neurons

    if color is None:
        color = ["#2d3143"] * n_neurons
    elif isinstance(color, str):
        try:
            cmap = mpl.cm.get_cmap(color)
            color = [cmap(_ / n_neurons) for _ in range(n_neurons)]
        except:
            color = [color] * n_neurons

    for ind, spiketrain in enumerate(neurons.spiketrains):
        if add_vert_jitter:
            jitter_add = np.random.randn(len(spiketrain)) * 0.1
            alpha_use = 0.25
        else:
            jitter_add, alpha_use = 0, 0.5
        ax.plot(
            spiketrain,
            (ind + 1) * np.ones(len(spiketrain)) + jitter_add,
            marker,
            markersize=markersize,
            color=color[ind],
            alpha=alpha_use,
        )

    ax.set_xlim([neurons.t_start, neurons.t_stop])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Units")

    return ax


def plot_mua(mua: core.Mua, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(mua.time, mua.spike_counts, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike counts")


def plot_placefields():
    pass


def plot_spectrogram():
    pass


def plot_ccg(self, clus_use, type="all", bin_size=0.001, window_size=0.05, ax=None):

    """Plot CCG for clusters in clus_use (list, max length = 2). Supply only one cluster in clus_use for ACG only.
    type: 'all' or 'ccg_only'.
    ax (optional): if supplied len(ax) must be 1 for type='ccg_only' or nclus^2 for type 'all'"""

    def ccg_spike_assemble(clus_use):
        """Assemble an array of sorted spike times and cluIDs for the input cluster ids the list clus_use """
        spikes_all, clus_all = [], []
        [
            (
                spikes_all.append(self.times[idc]),
                clus_all.append(np.ones_like(self.times[idc]) * idc),
            )
            for idc in clus_use
        ]
        spikes_all, clus_all = np.concatenate(spikes_all), np.concatenate(clus_all)
        spikes_sorted, clus_sorted = (
            spikes_all[spikes_all.argsort()],
            clus_all[spikes_all.argsort()],
        )

        return spikes_sorted, clus_sorted.astype("int")

    spikes_sorted, clus_sorted = ccg_spike_assemble(clus_use)
    ccgs = correlograms(
        spikes_sorted,
        clus_sorted,
        sample_rate=self._obj.sampfreq,
        bin_size=bin_size,
        window_size=window_size,
    )

    if type == "ccgs_only":
        ccgs = ccgs[0, 1, :].reshape(1, 1, -1)

    if ax is None:
        fig, ax = plt.subplots(ccgs.shape[0], ccgs.shape[1])

    winsize_bins = 2 * int(0.5 * window_size / bin_size) + 1
    bins = np.linspace(0, 1, winsize_bins)
    for a, ccg in zip(ax.reshape(-1), ccgs.reshape(-1, ccgs.shape[2])):
        a.bar(bins, ccg, width=1 / (winsize_bins - 1))
        a.set_xticks([0, 1])
        a.set_xticklabels(np.ones((2,)) * np.round(window_size / 2, 2))
        a.set_xlabel("Time (s)")
        a.set_ylabel("Spike Count")
        pretty_plot(a)

    return ax
