import matplotlib.pyplot as plt
import matplotlib as mpl
from .. import core
import numpy as np
from neuropy.utils.colors_util import ColorsUtil

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
    sort_by_frate : bool, optional
        If true then sorts spikes by the number of spikes (frate), by default False
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


def plot_correlograms(
        neurons : core.Neurons,
        neuron_inds,
        type="all", #all, acg, ccg
        sample_rate=1.0,
        bin_size=0.001,
        window_size=0.05,
        ax=None,
        use_cupy=False,
        ref_p=False,
        ref_t=0.002
):

    # Allow integer inputs (assumes ACG)
    if isinstance(neuron_inds, int):
        neuron_inds = [neuron_inds]
    # Remove duplicates because we get ACGs already
    else:
        neuron_inds = list(dict.fromkeys(neuron_inds))

    n_neurons = len(neuron_inds)

    # Get spike correlations
    ccgs = correlations.spike_correlations(
        neurons,
        neuron_inds=neuron_inds,
        sample_rate=sample_rate,
        bin_size=bin_size,
        window_size=window_size,
        use_cupy=use_cupy)


    # Check the shape of ccgs to determine if it’s an ACG or CCG
    is_acg = ccgs.shape[:2] == (1, 1) # Shape is (1, 1, N) for ACGs

    # Adjust for CCG or ACG only
    if type == "acg":
        ccgs = np.diagonal(ccgs,axis1=0,axis2=1).T.reshape(1,n_neurons,-1)
    if type == "ccg" and not is_acg:
        ccgs = ccgs[0, 1, :][None, None, :]

    # Determine subplot dimensions based on the correlogram shape
    if ax is None:
        fig, ax = plt.subplots(ccgs.shape[0], ccgs.shape[1], squeeze=False)  # Always create 2D array of axes

    # Make sure `ax` is a 2D array for reshaping
    if isinstance(ax, plt.Axes):
        ax = np.array([ax])  # Convert single Axes object to array for consistency

    # Calculate bins for the histogram
    winsize_bins = 2 * int(0.5 * window_size / bin_size) + 1

    # Plot correlograms using pertinent library
    bins = np.linspace(-window_size / 2, window_size / 2, winsize_bins)

    # Plot
    if use_cupy:
        for a, ccg in zip(ax.reshape(-1), ccgs.reshape(-1, ccgs.shape[2])):
            ccg_cpu = ccg.get()

            a.bar(bins, ccg_cpu, width=bins[1] - bins[0])

            if ref_p:
                a.axvline(-ref_t, color='blue', linestyle='--', linewidth=1,
                          label='Refractory Period Boundary')  # Line at -1 ms
                a.axvline(ref_t, color='blue', linestyle='--', linewidth=1)  # Line at +1 ms

            a.set_xticks([-window_size / 2, 0, window_size / 2])
            a.set_xlabel("Time (s)")
            a.set_ylabel("Spike Count")
    else:
        for a, ccg in zip(ax.reshape(-1), ccgs.reshape(-1, ccgs.shape[2])):
            a.bar(bins, ccg, width=bins[1] - bins[0])

            if ref_p:
                a.axvline(-ref_t, color='blue', linestyle='--', linewidth=1,
                          label='Refractory Period Boundary')  # Line at -1 ms
                a.axvline(ref_t, color='blue', linestyle='--', linewidth=1)  # Line at +1 ms

            a.set_xticks([-window_size / 2, 0, window_size / 2])
            a.set_xlabel("Time (s)")
            a.set_ylabel("Spike Count")
    plt.tight_layout()

    return ax


def plot_ccg(self, clus_use, type="all", bin_size=0.001, window_size=0.05, ax=None):

    """Plot CCG for clusters in clus_use (list, max length = 2). Supply only one cluster in clus_use for ACG only.
    type: 'all' or 'ccg_only'.
    ax (optional): if supplied len(ax) must be 1 for type='ccg_only' or nclus^2 for type 'all'"""

    def ccg_spike_assemble(clus_use):
        """Assemble an array of sorted spike times and cluIDs for the input cluster ids the list clus_use"""
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
        a.bar(bins, ccg, width=bins[1] - bins[0])

        if is_acg:
            a.axvline(-0.002, color='blue', linestyle='--', linewidth=1,
                      label='Refractory Period Boundary')  # Line at -1 ms
            a.axvline(0.002, color='blue', linestyle='--', linewidth=1)  # Line at +1 ms

        a.set_xticks([-window_size / 2, 0, window_size / 2])

        a.set_xlabel("Time (s)")
        a.set_ylabel("Spike Count")
        pretty_plot(a)

    return ax


def plot_firing_rate(
    neurons: core.Neurons,
    bin_size=60,
    stacked=False,
    normalize=False,
    sortby="frate",
    cmap="tab20c",
):

    binned_neurons = neurons.get_binned_spiketrains(bin_size=bin_size)
    n_neurons = neurons.n_neurons
    time = binned_neurons.time
    mean_frate = neurons.firing_rate
    frate = binned_neurons.spike_counts / bin_size

    if sortby == "frate":
        sort_ind = np.argsort(mean_frate)
        frate = frate[sort_ind, :]

    cmap = mpl.cm.get_cmap(cmap)
    if stacked:
        fig, ax = plt.subplots(ncols=1, nrows=n_neurons, sharex=True)

        for i, f in enumerate(frate):
            ax[i].plot(time, f, color=cmap(i / n_neurons))
            ax[i].spines["right"].set_visible(False)
            ax[i].spines["top"].set_visible(False)
            # ax[i].spines["left"].set_visible(False)

        ax[i].set_ylabel("Firing rate")
        ax[i].set_xlabel("Time (s)")
        # ax[i].spines["left"].set_visible(True)
    else:
        fig, ax = plt.subplots()
        for i, f in enumerate(frate):
            ax.plot(time, f, color=cmap(i / n_neurons))
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")

        ax.set_ylabel("Firing rate")
        ax.set_xlabel("Time (s)")

    fig.suptitle(f"{neurons.n_neurons} Neurons")


def plot_waveforms(neurons: core.Neurons, sort_order=None, color="#afadac"):
    """Plot waveforms in the neurons object

    Parameters
    ----------
    neurons : core.Neurons
        [description]
    sort_order : array, optional
        sorting order for the neurons, by default None
    color : str, optional
        [description], by default "#afadac"

    Returns
    -------
    ax
    """
    waves = neurons.waveforms
    # waves = np.where(waves != 0, waves, np.nan)

    if sort_order is not None:
        assert (
            len(sort_order) == neurons.n_neurons
        ), "sort_order should match the number of neurons"
        waves = waves[sort_order, :, :]

    waves = waves.transpose(1, 0, 2).reshape(waves.shape[1], -1)
    waves = waves + np.linspace(0, 1000, waves.shape[0]).reshape(-1, 1)

    _, ax = plt.subplots()

    ax.plot(waves.T, color=color, alpha=0.5)

    return ax


