import matplotlib.pyplot as plt
from .. import core
import numpy as np


def plot_raster(
    neurons: core.Neurons,
    ax=None,
    period=None,
    sort_by_frate=False,
    tstart=0,
    color=None,
    marker="|",
    markersize=2,
    add_vert_jitter=False,
):
    """creates raster plot using spiktrains in neurons

    Parameters
    ----------
    spikes : list, optional
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

    if spikes is None:
        pyr = self.pyr
        intneur = self.intneur
        mua = self.mua
        spikes = mua + pyr + intneur

        color = (
            [self.colors["mua"]] * len(mua)
            + [self.colors["pyr"]] * len(pyr)
            + [self.colors["intneur"]] * len(intneur)
        )

        # --- mimics legend for labeling unit category ---------
        y = 0.5
        for cell in self.colors:
            ax.annotate(
                cell,
                xy=(0.9, y),
                xycoords="figure fraction",
                color=self.colors[cell],
            )
            y -= 0.05
    else:
        assert isinstance(spikes, list), "Please provide a list of arrays"
        if color is None:
            color = ["#2d3143"] * len(spikes)
        elif isinstance(color, str):

            try:
                cmap = mpl.cm.get_cmap(color)
                color = [cmap(_ / len(spikes)) for _ in range(len(spikes))]
            except:
                color = [color] * len(spikes)

    # print(f"Plotting {len(spikes)} cells")
    frate = [len(cell) for cell in spikes]  # number of spikes ~= frate

    if period is not None:
        period_duration = np.diff(period)
        spikes = [
            cell[np.where((cell > period[0]) & (cell < period[1]))[0]]
            for cell in spikes
        ]
        frate = np.asarray([len(cell) / period_duration for cell in spikes]).squeeze()

    if sort_by_frate:
        sort_frate_indices = np.argsort(frate)
        spikes = [spikes[indx] for indx in sort_frate_indices]

    for cell, spk in enumerate(spikes):
        if add_vert_jitter:
            jitter_add = np.random.randn(len(spk)) * 0.1
            alpha_use = 0.25
        else:
            jitter_add, alpha_use = 0, 0.5
        ax.plot(
            spk - tstart,
            (cell + 1) * np.ones(len(spk)) + jitter_add,
            marker,
            markersize=markersize,
            color=color[cell],
            alpha=alpha_use,
        )

    ax.set_ylim([0.5, len(spikes) + 0.5])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Units")

    return ax


def plot_placefields():
    pass


def plot_spectrogram():
    pass


def plot_ccg(ccgs, bin_size, ax=None):

    if ax is None:
        fig, ax = plt.subplots(ccgs.shape[0], ccgs.shape[1])

    window_size = 2 * ccgs.shape[1]

    winsize_bins = 2 * int(0.5 * window_size / bin_size) + 1
    bins = np.linspace(0, 1, winsize_bins)
    for a, ccg in zip(ax.reshape(-1), ccgs.reshape(-1, ccgs.shape[2])):
        a.bar(bins, ccg, width=1 / (winsize_bins - 1))
        a.set_xticks([0, 1])
        a.set_xticklabels(np.ones((2,)) * np.round(window_size / 2, 2))
        a.set_xlabel("Time (s)")
        a.set_ylabel("Spike Count")
    return ax