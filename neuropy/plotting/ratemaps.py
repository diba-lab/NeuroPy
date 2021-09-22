import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .. import core


def plot_ratemaps(
    ratemap: core.Ratemap, ax=None, pad=2, normalize=False, sortby=None, cmap="tab20b"
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
    sortby : bool, optional
        [description], by default True
    cmap : str, optional
        [description], by default "tab20b"

    Returns
    -------
    [type]
        [description]
    """
    cmap = mpl.cm.get_cmap(cmap)

    ratemaps = ratemap.tuning_curves
    nCells = len(ratemaps)
    bin_cntr = self.bin[:-1] + np.diff(self.bin).mean() / 2
    bin_cntr = (bin_cntr - np.min(bin_cntr)) / np.ptp(bin_cntr)

    if ax is None:
        _, gs = Fig().draw(grid=(1, 1), size=(4.5, 11))
        ax = plt.subplot(gs[0])

    if normalize:
        ratemaps = [_ / np.max(_) for _ in ratemaps]
        pad = 1

    if sortby is None:
        sort_ind = np.argsort(np.argmax(np.asarray(ratemaps), axis=1))
    elif isinstance(sortby, (list, np.ndarray)):
        sort_ind = sortby
    else:
        sort_ind = np.arange(len(ratemaps))
    for cellid, cell in enumerate(sort_ind):
        color = cmap(cellid / len(sort_ind))

        ax.fill_between(
            bin_cntr,
            cellid * pad,
            cellid * pad + ratemaps[cell],
            color=color,
            ec=None,
            alpha=0.5,
            zorder=cellid + 1,
        )
        ax.plot(
            bin_cntr,
            cellid * pad + ratemaps[cell],
            color=color,
            alpha=0.7,
        )

    ax.set_yticks(list(range(len(sort_ind))))
    ax.set_yticklabels(list(sort_ind))
    ax.set_xlabel("Position")
    ax.spines["left"].set_visible(False)
    ax.set_xlim([0, 1])
    ax.tick_params("y", length=0)
    ax.set_ylim([0, len(sort_ind)])
    if self.run_dir is not None:
        ax.set_title(self.run_dir.capitalize() + " Runs only")

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
