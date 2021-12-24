from ipywidgets import widgets
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from neuropy.core.neuron_identities import NeuronExtendedIdentityTuple
from .. import core
from neuropy.utils import mathutil
from .figure import Fig

## TODO: refactor plot_ratemap_1D and plot_ratemap_2D to a single flat function (if that's appropriate).
## TODO: refactor plot_ratemap_1D and plot_ratemap_2D to take a **kwargs and apply optional defaults (find previous code where I did that using the | and dict conversion. In my 3D code.


def plot_single_tuning_curve_2D(xbin, ybin, pfmap, occupancy, neuron_extended_id: NeuronExtendedIdentityTuple=None, drop_below_threshold: float=0.0000001, ax=None):
    """Plots a single tuning curve Heatmap

    Args:
        xbin ([type]): [description]
        ybin ([type]): [description]
        pfmap ([type]): [description]
        occupancy ([type]): [description]
        drop_below_threshold (float, optional): [description]. Defaults to 0.0000001.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if ax is None:
        ax = plt.gca()
        
    curr_pfmap = np.array(pfmap.copy()) / np.nanmax(pfmap)
    if drop_below_threshold is not None:
        curr_pfmap[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy
    
    # curr_pfmap = np.rot90(np.fliplr(curr_pfmap)) ## Bug was introduced here! At least with pcolorfast, this order of operations is wrong!
    curr_pfmap = np.rot90(curr_pfmap)
    curr_pfmap = np.fliplr(curr_pfmap)
    # # curr_pfmap = curr_pfmap / np.nanmax(curr_pfmap) # for when the pfmap already had its transpose taken


    # mesh_X, mesh_Y = np.meshgrid(xbin, ybin)
    # ax.pcolormesh(mesh_X, mesh_Y, curr_pfmap, cmap='jet', vmin=0, edgecolors='k', linewidths=0.1)
    # ax.pcolormesh(mesh_X, mesh_Y, curr_pfmap, cmap='jet', vmin=0)

    im = ax.pcolorfast(
        xbin,
        ybin,
        curr_pfmap,
        cmap="jet", vmin=0.0
    )
            
    # ax.vlines(200, 'ymin'=0, 'ymax'=1, 'r')
    # ax.set_xticks([25, 50])
    # ax.vline(50, 'r')
    # ax.vlines([50], 0, 1, transform=ax.get_xaxis_transform(), colors='r')
    # ax.vlines([50], 0, 1, colors='r')

    # im = ax.pcolorfast(
    #     xbin,
    #     ybin,
    #     curr_pfmap,
    #     cmap="jet",
    #     vmin=0,
    # )
    # im = ax.pcolorfast(
    #     self.xbin,
    #     self.ratemap.ybin,
    #     np.rot90(np.fliplr(pfmap)) / np.nanmax(pfmap),
    #     cmap="jet",
    #     vmin=0,
    # )  # rot90(flipud... is necessary to match plotRaw configuration.
    # im = ax.pcolor(
    #     xbin,
    #     ybin,
    #     np.rot90(np.fliplr(pfmap)) / np.nanmax(pfmap),
    #     cmap="jet",
    #     vmin=0,
    # )
    
    ax.axis("off")
    extended_id_string = f'(shank {neuron_extended_id.shank}, cluster {neuron_extended_id.cluster})'
    ax.set_title(
            f"Cell {neuron_extended_id.id} - {extended_id_string} \n{round(np.nanmax(pfmap),2)} Hz"
    ) # f"Cell {ratemap.neuron_ids[cell]} - {ratemap.get_extended_neuron_id_string(neuron_i=cell)} \n{round(np.nanmax(pfmap),2)} Hz"
    
    return im
    
    

# all extracted from the 2D figures
def plot_ratemap_2D(ratemap: core.Ratemap, computation_config=None, subplots=(10, 8), figsize=(6, 10), fignum=None, enable_spike_overlay=True, drop_below_threshold: float=0.0000001):
    """Plots heatmaps of placefields with peak firing rate

    Parameters
    ----------
    speed_thresh : bool, optional
        [description], by default False
    subplots : tuple, optional
        number of cells within each figure window. If cells exceed the number of subplots, then cells are plotted in successive figure windows of same size, by default (10, 8)
    fignum : int, optional
        figure number to start from, by default None
    """

    map_use, thresh = ratemap.tuning_curves, computation_config.speed_thresh

    nCells = len(map_use)
    nfigures = nCells // np.prod(subplots) + 1

    if fignum is None:
        if f := plt.get_fignums():
            fignum = f[-1] + 1
        else:
            fignum = 1

    figures, gs = [], []
    for fig_ind in range(nfigures):
        fig = plt.figure(fignum + fig_ind, figsize=figsize, clear=True)
        gs.append(GridSpec(subplots[0], subplots[1], figure=fig))
        fig.subplots_adjust(hspace=0.2)
        
        title_string = f'2D Placemaps Placemaps ({len(ratemap.neuron_ids)} good cells)'
        if computation_config is not None:
            if computation_config.speed_thresh is not None:
                title_string = f'{title_string} (speed_threshold = {str(computation_config.speed_thresh)})'
            
        fig.suptitle(title_string)
        figures.append(fig)

    for cell, pfmap in enumerate(map_use):
        ind = cell // np.prod(subplots)
        subplot_ind = cell % np.prod(subplots)
        ax1 = figures[ind].add_subplot(gs[ind][subplot_ind])
        
        
        
        im = plot_single_tuning_curve_2D(ratemap.xbin, ratemap.ybin, pfmap, ratemap.occupancy, neuron_extended_id=ratemap.neuron_extended_ids[cell], drop_below_threshold=drop_below_threshold, ax=ax1)
        
        # if enable_spike_overlay:
        #     ax1.scatter(self.spk_pos[cell][0], self.spk_pos[cell][1], s=1, c='white', alpha=0.3, marker=',')
        #     # ax1.scatter(self.spk_pos[cell][1], self.spk_pos[cell][0], s=1, c='white', alpha=0.3, marker=',')
        
        
        # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
        # cbar = fig.colorbar(im, cax=cbar_ax)
        # cbar.set_label("firing rate (Hz)")
        
    return figures, gs
    

def plot_ratemap_1D(ratemap: core.Ratemap, normalize_xbin=False, ax=None, pad=2, normalize_tuning_curve=False, sortby=None, cmap="tab20b"):
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

    tuning_curves = ratemap.tuning_curves
    n_neurons = ratemap.n_neurons
    bin_cntr = ratemap.xbin_centers
    if normalize_xbin:
        bin_cntr = (bin_cntr - np.min(bin_cntr)) / np.ptp(bin_cntr)

    if ax is None:
        _, gs = Fig().draw(grid=(1, 1), size=(5.5, 11))
        ax = plt.subplot(gs[0])

    if normalize_tuning_curve:
        tuning_curves = mathutil.min_max_scaler(tuning_curves)
        pad = 1

    if sortby is None:
        # sort by the location of the placefield's maximum
        sort_ind = np.argsort(np.argmax(tuning_curves, axis=1))
    elif isinstance(sortby, (list, np.ndarray)):
        # use the provided sort indicies
        sort_ind = sortby
    else:
        sort_ind = np.arange(n_neurons)

    ## TODO: actually sort the ratemap object's neuron_ids and tuning_curves by the sort_ind
    # sorted_neuron_ids = ratemap.neuron_ids[sort_ind]
    
    sorted_neuron_ids = np.take_along_axis(np.array(ratemap.neuron_ids), sort_ind, axis=0)
    
    sorted_alt_tuple_neuron_ids = ratemap.neuron_extended_ids.copy()
    # sorted_alt_tuple_neuron_ids = ratemap.metadata['tuple_neuron_ids'].copy()
    sorted_alt_tuple_neuron_ids = [sorted_alt_tuple_neuron_ids[a_sort_idx] for a_sort_idx in sort_ind]
    
    # sorted_tuning_curves = tuning_curves[sorted_neuron_ids, :]
    # sorted_neuron_id_labels = [f'Cell[{a_neuron_id}]' for a_neuron_id in sorted_neuron_ids]
    sorted_neuron_id_labels = [f'C[{sorted_neuron_ids[i]}]({sorted_alt_tuple_neuron_ids[i].shank}|{sorted_alt_tuple_neuron_ids[i].cluster})' for i in np.arange(len(sorted_neuron_ids))]
    
    colors_array = np.zeros((4, n_neurons))
    for i, neuron_ind in enumerate(sort_ind):
        color = cmap(i / len(sort_ind))
        colors_array[:, i] = color
        curr_neuron_id = sorted_neuron_ids[i]

        ax.fill_between(
            bin_cntr,
            i * pad,
            i * pad + tuning_curves[neuron_ind],
            color=color,
            ec=None,
            alpha=0.5,
            zorder=i + 1,
        )
        ax.plot(
            bin_cntr,
            i * pad + tuning_curves[neuron_ind],
            color=color,
            alpha=0.7,
        )
        ax.set_title('Cell[{}]'.format(curr_neuron_id)) # this doesn't appear to be visible, so what is it used for?

    # ax.set_yticks(list(range(len(sort_ind)) + 0.5))
    ax.set_yticks(list(np.arange(len(sort_ind)) + 0.5))
    # ax.set_yticklabels(list(sort_ind))
    ax.set_yticklabels(list(sorted_neuron_id_labels))
    
    ax.set_xlabel("Position")
    ax.spines["left"].set_visible(False)
    if normalize_xbin:
        ax.set_xlim([0, 1])
    ax.tick_params("y", length=0)
    ax.set_ylim([0, len(sort_ind)])
    # if self.run_dir is not None:
    #     ax.set_title(self.run_dir.capitalize() + " Runs only")

    return ax, sort_ind, colors_array


def plot_raw(ratemap: core.Ratemap, t, x, run_dir, ax=None, subplots=(8, 9)):
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

    # mapinfo = self.ratemaps
    mapinfo = ratemap
    nCells = len(mapinfo["pos"])

    def plot_(cell, ax):
        if subplots is None:
            ax.clear()
        ax.plot(x, t, color="gray", alpha=0.6)
        ax.plot(mapinfo["pos"][cell], mapinfo["spikes"][cell], ".", color="#ff5f5c")
        ax.set_title(
            " ".join(filter(None, ("Cell", str(cell), run_dir.capitalize())))
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
