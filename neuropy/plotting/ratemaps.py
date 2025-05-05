import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .. import core
from ..utils import mathutil
from .figure import Fig


def plot_ratemap(
    ratemap: core.Ratemap,
    normalize_xbin=False,
    ax=None,
    pad=2,
    normalize_tuning_curve=False,
    cross_norm=None,
    sortby=None,
    cmap="tab20b",
):
    """Plot 1D place fields stacked

    Parameters
    ----------
    ax : [type], optional
        [description], by default None
    speed_thresh : bool, optional
        [description], by default False
    pad : int, optional
        [description], by default 2
    normalize : bool, optional
        [description], by default False
    cross_norm : np.array, optional
        Nx2 numpy array including xmin and xptp per neuron, by default None.
    sortby : array, optional
        [description], by default True
    cmap : str, optional
        [description], by default "tab20b"

    Returns
    -------
    [type]
        [description]
    """
    cmap = mpl.cm.get_cmap(cmap)

    tuning_curves = ratemap.tuning_curves
    n_neurons = ratemap.n_neurons
    # bin_cntr = ratemap.xbin_centers
    bin_cntr = ratemap.x_coords()
    if normalize_xbin:
        bin_cntr = (bin_cntr - np.min(bin_cntr)) / np.ptp(bin_cntr)

    if ax is None:
        fig = Fig(nrows=1, ncols=1, size=(4.5, 11))
        ax = fig.subplot(fig.gs[0])

    if normalize_tuning_curve:
        if isinstance(cross_norm, np.ndarray):
            # Create xmin array and set it to be broadcastable during
            xmin = cross_norm[:,0]
            xptp = cross_norm[:,1] - cross_norm[:,0]

            tuning_curves = mathutil.min_max_external_scaler(tuning_curves, xmin, xptp)
        else:
            tuning_curves = mathutil.min_max_scaler(tuning_curves)
        pad = 1

    if sortby is None:
        sort_ind = np.argsort(np.argmax(tuning_curves, axis=1))
    elif isinstance(sortby, (list, np.ndarray)):
        sort_ind = sortby
    else:
        sort_ind = np.arange(n_neurons)

    for i, neuron_ind in enumerate(sort_ind):
        color = cmap(i / len(sort_ind))

        ax.fill_between(
            bin_cntr,
            i * pad,
            i * pad + tuning_curves[neuron_ind],
            color=color,
            ec=None,
            alpha=0.7,
            zorder=i + 1,
        )
        ax.plot(
            bin_cntr,
            i * pad + tuning_curves[neuron_ind],
            color=color,
            alpha=1,
            lw=0.6,
        )

    ax.set_yticks(list(range(len(sort_ind))))
    ax.set_yticklabels(list(ratemap.neuron_ids[sort_ind]))
    ax.set_xlabel("Position")
    ax.spines["left"].set_visible(False)
    if normalize_xbin:
        ax.set_xlim([0, 1])
    ax.tick_params("y", length=0)

    # Set y axis as the neuron id
    ax.set_ylim([0, len(sort_ind)])

    # Set y-axis as firing rate if cross norm is provided and normalized tuning curve is false
    if cross_norm is not None and normalize_tuning_curve is False:
        ax.set_ylim(cross_norm[0])
        ax.set_yticks(np.linspace(*cross_norm[0], num=3))
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(*cross_norm[0], num=3)])
        ax.tick_params("y", length=4)

    return ax


def plot_raw(self, ax=None, subplots=(8, 9)):
    """Plot spike location on animal's path

    Parameters
    ----------
    speed_thresh : bool, optional
        [description], by default False
    ax : [type], optional
        [description], by default None
    subplots : tuple, optional
        [description], by default (8, 9)
    """

    mapinfo = self.ratemaps
    nCells = len(mapinfo["pos"])

    def plot_(cell, ax):
        if subplots is None:
            ax.clear()
        ax.plot(self.x, self.t, color="gray", alpha=0.6)
        ax.plot(mapinfo["pos"][cell], mapinfo["spikes"][cell], ".", color="#ff5f5c")
        ax.set_title(
            " ".join(filter(None, ("Cell", str(cell), self.run_dir.capitalize())))
        )
        ax.invert_yaxis()
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Time (s)")

    if ax is None:

        if subplots is None:
            _, gs = Fig().draw(grid=(1, 1), size=(6, 8))
            ax = plt.subplot(gs[0])
            widgets.interact(
                plot_,
                cell=widgets.IntSlider(
                    min=0,
                    max=nCells - 1,
                    step=1,
                    description="Cell ID:",
                ),
                ax=widgets.fixed(ax),
            )
        else:
            _, gs = Fig().draw(grid=subplots, size=(10, 11))
            for cell in range(nCells):
                ax = plt.subplot(gs[cell])
                ax.set_yticks([])
                plot_(cell, ax)
