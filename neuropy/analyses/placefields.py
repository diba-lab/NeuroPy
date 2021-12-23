from copy import deepcopy
from dataclasses import dataclass

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.image import NonUniformImage
from scipy.ndimage import gaussian_filter, gaussian_filter1d, interpolation
# from neuropy.analyses.pho_custom_placefields import PfND
from neuropy.core.epoch import Epoch
from neuropy.core.neurons import Neurons
from neuropy.core.position import Position
from neuropy.core.ratemap import Ratemap
from neuropy.core.signal import Signal

from neuropy.plotting.figure import pretty_plot

# from .. import core
# import neuropy.core as core
from neuropy.utils.signal_process import ThetaParams
from .. import plotting
from neuropy.utils.mixins.print_helpers import SimplePrintable, OrderedMeta


class PlacefieldComputationParameters(SimplePrintable, metaclass=OrderedMeta):
    """A simple wrapper object for parameters used in placefield calcuations"""
    decimal_point_character=","
    param_sep_char='-'
    variable_names=['speed_thresh', 'grid_bin', 'smooth', 'frate_thresh']
    variable_inline_names=['speedThresh', 'gridBin', 'smooth', 'frateThresh']
    variable_inline_names=['speedThresh', 'gridBin', 'smooth', 'frateThresh']
    
    def __init__(self, speed_thresh=3, grid_bin=2, smooth=2, frate_thresh=1):
        self.speed_thresh = speed_thresh
        if not isinstance(grid_bin, (tuple, list)):
            grid_bin = (grid_bin, grid_bin) # make it into a 2 element tuple

        self.grid_bin = grid_bin
        if not isinstance(smooth, (tuple, list)):
            smooth = (smooth, smooth) # make it into a 2 element tuple
        self.smooth = smooth
        self.frate_thresh = frate_thresh
    
    
    @property
    def grid_bin_1D(self):
        """The grid_bin_1D property."""
        if np.isscalar(self.grid_bin):
            return self.grid_bin
        else:
            return self.grid_bin[0]

    @property
    def smooth_1D(self):
        """The smooth_1D property."""
        if np.isscalar(self.smooth):
            return self.smooth
        else:
            return self.smooth[0]


    def str_for_filename(self, is_2D):
        if is_2D:
            return '-'.join([f"speedThresh_{self.speed_thresh:.2f}", f"gridBin_{self.grid_bin[0]:.2f}_{self.grid_bin[1]:.2f}", f"smooth_{self.smooth[0]:.2f}_{self.smooth[1]:.2f}", f"frateThresh_{self.frate_thresh:.2f}"])
            # return "speedThresh_{:.2f}-gridBin_{:.2f}_{:.2f}-smooth_{:.2f}_{:.2f}-frateThresh_{:.2f}".format(self.speed_thresh, self.grid_bin[0], self.grid_bin[1], self.smooth[0], self.smooth[1], self.frate_thresh)
            # return f"speedThresh_{self.speed_thresh:.2f}-gridBin_{self.grid_bin[0]:.2f}_{self.grid_bin[1]:.2f}-smooth_{self.smooth[0]:.2f}_{self.smooth[1]:.2f}-frateThresh_{self.frate_thresh:.2f}"
        else:
            return '-'.join([f"speedThresh_{self.speed_thresh:.2f}", f"gridBin_{self.grid_bin_1D:.2f}", f"smooth_{self.smooth_1D:.2f}", f"frateThresh_{self.frate_thresh:.2f}"])
            # return f"speedThresh_{self.speed_thresh:.2f}-gridBin_{self.grid_bin_1D:.2f}-smooth_{self.smooth_1D:.2f}-frateThresh_{self.frate_thresh:.2f}"
        
    def str_for_display(self, is_2D):
        """ For rendering in a title, etc """
        if is_2D:
            return f"(speedThresh_{self.speed_thresh:.2f}, gridBin_{self.grid_bin[0]:.2f}_{self.grid_bin[1]:.2f}, smooth_{self.smooth[0]:.2f}_{self.smooth[1]:.2f}, frateThresh_{self.frate_thresh:.2f})"
        else:
            return f"(speedThresh_{self.speed_thresh:.2f}, gridBin_{self.grid_bin_1D:.2f}, smooth_{self.smooth_1D:.2f}, frateThresh_{self.frate_thresh:.2f})"



def _filter_by_frate(tuning_maps, frate_thresh, debug=False):
    # ---- cells with peak frate abouve thresh ------
    n_neurons = len(tuning_maps)
    thresh_neurons_indx = [
        neuron_indx
        for neuron_indx in range(n_neurons)
        if np.nanmax(tuning_maps[neuron_indx]) > frate_thresh
    ]
    if debug:
        print('_filter_by_frate(...):')
        print('\t frate_thresh: {}'.format(frate_thresh))
        print('\t n_neurons: {}'.format(n_neurons))
        print('\t thresh_neurons_indx: {}'.format(thresh_neurons_indx))
    # filter_function: just indexes its passed list argument by thresh_neurons_indx (including only neurons that meet the thresholding criteria)
    filter_function = lambda list_: [list_[_] for _ in thresh_neurons_indx]
    # there is only one tuning_map per neuron that means the thresh_neurons_indx:
    filtered_tuning_maps = np.asarray(filter_function(tuning_maps))
    return filtered_tuning_maps, filter_function 

def _bin_pos_nD(x: np.ndarray, y: np.ndarray, num_bins=None, bin_size=None):
    """ Spatially bins the provided x and y vectors into position bins based on either the specified num_bins or the specified bin_size
    Usage:
        ## Binning with Fixed Number of Bins:    
        xbin, ybin, bin_info = _bin_pos(pos_df.x.to_numpy(), pos_df.y.to_numpy(), bin_size=active_config.computation_config.grid_bin) # bin_size mode
        print(bin_info)
        ## Binning with Fixed Bin Sizes:
        xbin, ybin, bin_info = _bin_pos(pos_df.x.to_numpy(), pos_df.y.to_numpy(), num_bins=num_bins) # num_bins mode
        print(bin_info)
    """
    assert (num_bins is None) or (bin_size is None), 'You cannot constrain both num_bins AND bin_size. Specify only one or the other.'
    assert (num_bins is not None) or (bin_size is not None), 'You must specify either the num_bins XOR the bin_size.'
    
    bin_info_out_dict = dict()
    
    if num_bins is not None:
        ## Binning with Fixed Number of Bins:
        mode = 'num_bins'
        if np.isscalar(num_bins):
            num_bins = [num_bins]
        
        xnum_bins = num_bins[0]
        xbin, xstep = np.linspace(np.nanmin(x), np.nanmax(x), num=xnum_bins, retstep=True)  # binning of x position

        if y is not None:
            ynum_bins = num_bins[1]
            ybin, ystep = np.linspace(np.nanmin(y), np.nanmax(y), num=ynum_bins, retstep=True)  # binning of y position       
            
    elif bin_size is not None:
        ## Binning with Fixed Bin Sizes:
        mode = 'bin_size'
        if np.isscalar(bin_size):
            print(f'np.isscalar(bin_size): {bin_size}')
            bin_size = [bin_size]
            
        xstep = bin_size[0]
        xbin = np.arange(np.nanmin(x), (np.nanmax(x) + xstep), xstep)  # binning of x position
        xnum_bins = len(xbin)

        if y is not None:
            ystep = bin_size[1]
            ybin = np.arange(np.nanmin(y), (np.nanmax(y) + ystep), ystep)  # binning of y position
            ynum_bins = len(ybin)
            
    # print('xbin: {}'.format(xbin))
    # print('ybin: {}'.format(ybin))
    bin_info_out_dict = {'mode':mode, 'xstep':xstep, 'xnum_bins':xnum_bins}
    if y is not None:
        # if at least 2D output, add the y-axis properties to the info dictionary
        bin_info_out_dict['ystep'], bin_info_out_dict['ynum_bins']  = ystep, ynum_bins
    else:
        ybin = None
        
    return xbin, ybin, bin_info_out_dict # {'mode':mode, 'xstep':xstep, 'ystep':ystep, 'xnum_bins':xnum_bins, 'ynum_bins':ynum_bins}

def _normalized_occupancy(raw_occupancy, dt=None, position_srate=None):
    # raw occupancy is defined in terms of the number of samples that fall into each bin.
    # if position_srate is not None:
    #     dt = 1.0 / float(position_srate)
    #  seconds_occupancy is the number of seconds spent in each bin. This is computed by multiplying the raw occupancy (in # samples) by the duration of each sample.
    # seconds_occupancy = raw_occupancy * dt  # converting to seconds
    seconds_occupancy = raw_occupancy / (float(position_srate) + 1e-16) # converting to seconds
    # seconds_occupancy = occupancy / (position_srate + 1e-16)  # converting to seconds
    # normalized occupancy gives the ratio of samples that feel in each bin. ALL BINS ADD UP TO ONE.
    normalized_occupancy = raw_occupancy / np.nansum(raw_occupancy) # the normalized occupancy determines the relative number of samples spent in each bin

    return seconds_occupancy, normalized_occupancy



class PfnConfigMixin:
    def str_for_filename(self, is_2D=True):
        return self.config.str_for_filename(is_2D)

    
class PfnDMixin(SimplePrintable):
    @property
    def spk_pos(self):
        return self.ratemap_spiketrains_pos
    
    @property
    def spk_t(self):
        return self.ratemap_spiketrains
    
    @property
    def cell_ids(self):
        return self.ratemap.neuron_ids


class PfnDPlottingMixin(PfnDMixin):
    # Extracted fro the 1D figures:
    def plot_ratemaps(self, ax=None, pad=2, normalize=True, sortby=None, cmap="tab20b"):
        """ Note that normalize is required to fit all of the plots on this kind of stacked figure. """
        # returns: ax , sort_ind, colors
        return plotting.plot_ratemap(self.ratemap, ax=ax, pad=pad, normalize_tuning_curve=normalize, sortby=sortby, cmap=cmap)
    
    # all extracted from the 2D figures
    def plotMap(self, subplots=(10, 8), figsize=(6, 10), fignum=None, enable_spike_overlay=True):
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

        map_use, thresh = self.ratemap.tuning_curves, self.speed_thresh

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
            
            title_string = f'2D Placemaps Placemaps ({len(self.ratemap.neuron_ids)} good cells)'
            if thresh is not None:
                title_string = f'{title_string} (speed_threshold = {str(thresh)})'
                
            fig.suptitle(title_string)
            figures.append(fig)

        mesh_X, mesh_Y = np.meshgrid(self.ratemap.xbin, self.ratemap.ybin)

        for cell, pfmap in enumerate(map_use):
            ind = cell // np.prod(subplots)
            subplot_ind = cell % np.prod(subplots)
 
            
            # Working:
            curr_pfmap = np.array(pfmap) / np.nanmax(pfmap)
            # curr_pfmap = np.rot90(np.fliplr(curr_pfmap)) ## Bug was introduced here! At least with pcolorfast, this order of operations is wrong!
            curr_pfmap = np.rot90(curr_pfmap)
            curr_pfmap = np.fliplr(curr_pfmap)
            # # curr_pfmap = curr_pfmap / np.nanmax(curr_pfmap) # for when the pfmap already had its transpose taken
            ax1 = figures[ind].add_subplot(gs[ind][subplot_ind])
            # ax1.pcolormesh(mesh_X, mesh_Y, curr_pfmap, cmap='jet', vmin=0, edgecolors='k', linewidths=0.1)
            # ax1.pcolormesh(mesh_X, mesh_Y, curr_pfmap, cmap='jet', vmin=0)
            
            im = ax1.pcolorfast(
                self.ratemap.xbin,
                self.ratemap.ybin,
                curr_pfmap,
                cmap="jet", vmin=0.0
            )
                    
            
            # ax1.vlines(200, 'ymin'=0, 'ymax'=1, 'r')
            # ax1.set_xticks([25, 50])
            # ax1.vline(50, 'r')
            # ax1.vlines([50], 0, 1, transform=ax1.get_xaxis_transform(), colors='r')
            # ax1.vlines([50], 0, 1, colors='r')
                

            # im = ax1.pcolorfast(
            #     self.ratemap.xbin,
            #     self.ratemap.ybin,
            #     curr_pfmap,
            #     cmap="jet",
            #     vmin=0,
            # )
            # im = ax1.pcolorfast(
            #     self.ratemap.xbin,
            #     self.ratemap.ybin,
            #     np.rot90(np.fliplr(pfmap)) / np.nanmax(pfmap),
            #     cmap="jet",
            #     vmin=0,
            # )  # rot90(flipud... is necessary to match plotRaw configuration.
            # im = ax1.pcolor(
            #     self.ratemap.xbin,
            #     self.ratemap.ybin,
            #     np.rot90(np.fliplr(pfmap)) / np.nanmax(pfmap),
            #     cmap="jet",
            #     vmin=0,
            # )
            
            # ax1.scatter(self.spk_pos[ind]) # tODO: add spikes
            # max_frate =
            
            # if enable_spike_overlay:
            #     ax1.scatter(self.spk_pos[cell][0], self.spk_pos[cell][1], s=1, c='white', alpha=0.3, marker=',')
            #     # ax1.scatter(self.spk_pos[cell][1], self.spk_pos[cell][0], s=1, c='white', alpha=0.3, marker=',')
            
            curr_cell_alt_id = self.ratemap.tuple_neuron_ids[cell]
            curr_cell_shank = curr_cell_alt_id[0]
            curr_cell_cluster = curr_cell_alt_id[1]
            
            ax1.axis("off")
            ax1.set_title(
                f"Cell {self.ratemap.neuron_ids[cell]} - (shank {curr_cell_shank}, cluster {curr_cell_cluster}) \n{round(np.nanmax(pfmap),2)} Hz"
            )

            # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
            # cbar = fig.colorbar(im, cax=cbar_ax)
            # cbar.set_label("firing rate (Hz)")
            
        return figures, gs

    def plot_raw(self, subplots=(10, 8), fignum=None, alpha=0.5, label_cells=False, ax=None, clus_use=None):
        if self.ndim < 2:
            ## TODO: Pf1D Temporary Workaround:
            return plotting.plot_raw(self.ratemap, self.t, self.x, 'BOTH', ax=ax, subplots=subplots)
        else:        
            if ax is None:
                fig = plt.figure(fignum, figsize=(6, 10))
                gs = GridSpec(subplots[0], subplots[1], figure=fig)
                # fig.subplots_adjust(hspace=0.4)
            else:
                assert len(ax) == len(
                    clus_use
                ), "Number of axes must match number of clusters to plot"
                fig = ax[0].get_figure()

            # spk_pos_use = self.spk_pos
            spk_pos_use = self.ratemap_spiketrains_pos

            if clus_use is not None:
                spk_pos_tmp = spk_pos_use
                spk_pos_use = []
                [spk_pos_use.append(spk_pos_tmp[a]) for a in clus_use]

            for cell, (spk_x, spk_y) in enumerate(spk_pos_use):
                if ax is None:
                    ax1 = fig.add_subplot(gs[cell])
                else:
                    ax1 = ax[cell]
                ax1.plot(self.x, self.y, color="#d3c5c5")
                ax1.plot(spk_x, spk_y, '.', markersize=0.8, color=[1, 0, 0, alpha])
                ax1.axis("off")
                if label_cells:
                    # Put info on title
                    info = self.cell_ids[cell]
                    ax1.set_title(f"Cell {info}")

            fig.suptitle(
                f"Place maps for cells with their peak firing rate (frate thresh={self.frate_thresh},speed_thresh={self.speed_thresh})"
            )
            
            

    # def plotRaw_v_time_1D_ONLY(self, cellind, speed_thresh=False, alpha=0.5, ax=None):
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1, sharex=True)
    #         fig.set_size_inches([23, 9.7])
            
    #     if ax is not list:
    #         ax = [ax]

    #     # plot trajectories

            
    #     for a, pos, ylabel in zip(
    #         ax, [self.x], ["X position (cm)"]
    #     ):
    #         a.plot(self.t, pos)
    #         a.set_xlabel("Time (seconds)")
    #         a.set_ylabel(ylabel)
    #         pretty_plot(a)

    #     # Grab correct spike times/positions
    #     if speed_thresh:
    #         spk_pos_, spk_t_ = self.run_spk_pos, self.run_spk_t
    #     else:
    #         spk_pos_, spk_t_ = self.spk_pos, self.spk_t

    #     # plot spikes on trajectory
    #     for a, pos in zip(ax, [spk_pos_[cellind]]):
    #         a.plot(spk_t_[cellind], pos, ".", color=[0, 0, 0.8, alpha])

    #     # Put info on title
    #     ax[0].set_title(
    #         "Cell "
    #         + str(self.cell_ids[cellind])
    #         + ":, speed_thresh="
    #         + str(self.speed_thresh)
    #     )
        
        
    def plotRaw_v_time(self, cellind, speed_thresh=False, alpha=0.5, ax=None):
        """ Updated to work with both 1D and 2D Placefields """   
        if ax is None:
            fig, ax = plt.subplots(self.ndim, 1, sharex=True)
            fig.set_size_inches([23, 9.7])
        
        if np.isscalar(ax):
            ax = [ax]
            
        # plot trajectories
        if self.ndim < 2:
            variable_array = [self.x]
            label_array = ["X position (cm)"]
        else:
            variable_array = [self.x, self.y]
            label_array = ["X position (cm)", "Y position (cm)"]
        for a, pos, ylabel in zip(ax, variable_array, label_array):
            a.plot(self.t, pos)
            a.set_xlabel("Time (seconds)")
            a.set_ylabel(ylabel)
            pretty_plot(a)

        # Grab correct spike times/positions
        if speed_thresh:
            spk_pos_, spk_t_ = self.run_spk_pos, self.run_spk_t
        else:
            spk_pos_, spk_t_ = self.spk_pos, self.spk_t

        # plot spikes on trajectory
        for a, pos in zip(ax, spk_pos_[cellind]):
            a.plot(spk_t_[cellind], pos, ".", color=[0, 0, 0.8, alpha])

        # Put info on title
        ax[0].set_title(
            "Cell "
            + str(self.cell_ids[cellind])
            + ":, speed_thresh="
            + str(self.speed_thresh)
        )
        return ax

    def plot_all(self, cellind, speed_thresh=True, alpha=0.4, fig=None):
        if fig is None:
            fig_use = plt.figure(figsize=[28.25, 11.75])
        else:
            fig_use = fig
        gs = GridSpec(2, 4, figure=fig_use)
        ax2d = fig_use.add_subplot(gs[0, 0])
        axccg = np.asarray(fig_use.add_subplot(gs[1, 0]))
        axx = fig_use.add_subplot(gs[0, 1:])
        axy = fig_use.add_subplot(gs[1, 1:], sharex=axx)

        self.plot_raw(speed_thresh=speed_thresh, clus_use=[cellind], ax=[ax2d])
        self.plotRaw_v_time(
            cellind, speed_thresh=speed_thresh, ax=[axx, axy], alpha=alpha
        )
        self._obj.spikes.plot_ccg(clus_use=[cellind], type="acg", ax=axccg)

        return fig_use






class Pf1D(PfnConfigMixin, PfnDMixin):
    
    @staticmethod
    def _compute_occupancy(x, xbin, position_srate, smooth):
        # --- occupancy map calculation -----------
        # NRK todo: might need to normalize occupancy so sum adds up to 1
        raw_occupancy, xedges = np.histogram(x, bins=xbin)
        if ((smooth is not None) and (smooth > 0.0)):
            raw_occupancy = gaussian_filter1d(raw_occupancy, sigma=smooth)
        # # raw occupancy is defined in terms of the number of samples that fall into each bin.
        seconds_occupancy, normalized_occupancy = _normalized_occupancy(raw_occupancy, position_srate=position_srate)
        return seconds_occupancy, xedges
     
    @staticmethod   
    def _compute_tuning_map(spk_x, xbin, occupancy, smooth):
        tuning_map = np.histogram(spk_x, bins=xbin)[0]
        if ((smooth is not None) and (smooth > 0.0)):
            tuning_map = gaussian_filter1d(tuning_map, sigma=smooth)
        tuning_map = tuning_map / occupancy
        return tuning_map
    
    def str_for_filename(self, prefix_string=''):
        return '-'.join(['pf1D', f'{prefix_string}{self.config.str_for_filename(False)}'])
    
    
    def __init__(
        self,
        neurons: Neurons,
        position: Position,
        epochs: Epoch = None,
        frate_thresh=1,
        speed_thresh=5,
        grid_bin=1,
        smooth=1,
    ):
        """computes 1d place field using linearized coordinates. It always computes two place maps with and
        without speed thresholds.

        Parameters
        ----------
        track_name : str
            name of track
        direction : forward, backward or None
            direction of running, by default None which means direction is ignored
        grid_bin : int
            bin size of position bining, by default 5
        speed_thresh : int
            speed threshold for calculating place field
        """

        assert position.ndim == 1, "Only 1 dimensional position are acceptable"
        # save the config that was used to perform the computations
        self.config = PlacefieldComputationParameters(speed_thresh=speed_thresh, grid_bin=grid_bin, smooth=smooth, frate_thresh=frate_thresh)

        spiketrains = neurons.spiketrains
        neuron_ids = neurons.neuron_ids
        n_neurons = neurons.n_neurons
        position_srate = position.sampling_rate
        self.x = position.x
        self.speed = position.speed
        if ((smooth is not None) and (smooth > 0.0)):
            self.speed = gaussian_filter1d(self.speed, sigma=20)
        self.t = position.time
        t_start = position.t_start
        t_stop = position.t_stop

        # xbin = np.arange(min(self.x), max(self.x), grid_bin)  # binning of x position
        xbin, bin_info = _bin_pos_1D(self.x, bin_size=grid_bin) # bin_size mode

        spk_pos, spk_t, tuning_curve = [], [], []

        # ------ if direction then restrict to those epochs --------
        if epochs is not None:
            assert isinstance(epochs, Epoch), "epochs should be Epoch object"
            # print(f" using {run_dir} running only")
            spks = [
                np.concatenate(
                    [
                        spktrn[(spktrn > epc.start) & (spktrn < epc.stop)]
                        for epc in epochs.to_dataframe().itertuples()
                    ]
                )
                for spktrn in spiketrains
            ]
            # changing x, speed, time to only run epochs so occupancy map is consistent with that
            indx = np.concatenate(
                [
                    np.where((self.t >= epc.start) & (self.t <= epc.stop))[0]
                    for epc in epochs.to_dataframe().itertuples()
                ]
            )
            self.x = self.x[indx] # (52121,)
            self.speed = self.speed[indx] # (52121,)
            self.t = self.t[indx] # (52121,)
            
            
            occupancy, xedges = Pf1D._compute_occupancy(self.x, xbin, position_srate, smooth)

            for cell in spks:
                spk_spd = np.interp(cell, self.t, self.speed)
                spk_x = np.interp(cell, self.t, self.x)

                spk_pos.append(spk_x)
                spk_t.append(cell)

                # tuning curve calculation
                tuning_curve.append(Pf1D._compute_tuning_map(spk_x, xbin, occupancy, smooth))

        else:
            # --- speed thresh occupancy----

            spks = [
                spktrn[(spktrn > t_start) & (spktrn < t_stop)] for spktrn in spiketrains
            ]
            indx = np.where(self.speed >= speed_thresh)[0]
            self.x, self.speed, self.t = self.x[indx], self.speed[indx], self.t[indx]

            occupancy, xedges = Pf1D._compute_occupancy(self.x, xbin, position_srate, smooth)
            
            # occupancy = np.histogram(self.x, bins=xbin)[0] / position_srate + 1e-16
            # occupancy = gaussian_filter1d(occupancy, sigma=smooth)

            for cell in spks:
                spk_spd = np.interp(cell, self.t, self.speed)
                spk_x = np.interp(cell, self.t, self.x)

                # speed threshold
                spd_ind = np.where(spk_spd > speed_thresh)[0]
                spk_pos.append(spk_x[spd_ind])
                spk_t.append(cell[spd_ind])

                # tuning curve calculation
                tuning_curve.append(Pf1D._compute_tuning_map(spk_x, xbin, occupancy, smooth))

        # ---- cells with peak frate abouve thresh ------
        thresh_neurons_indx = [
            neuron_indx
            for neuron_indx in range(n_neurons)
            if np.nanmax(tuning_curve[neuron_indx]) > frate_thresh
        ]

        get_elem = lambda list_: [list_[_] for _ in thresh_neurons_indx]

        tuning_curve = get_elem(tuning_curve)
        tuning_curve = np.asarray(tuning_curve)
        self.ratemap = Ratemap(
            tuning_curve, xbin=xbin, neuron_ids=get_elem(neuron_ids)
        )
        self.ratemap_spiketrains = get_elem(spk_t)
        self.ratemap_spiketrains_pos = get_elem(spk_pos)
        self.occupancy = occupancy
        self.frate_thresh = frate_thresh
        self.speed_thresh = speed_thresh

    def estimate_theta_phases(self, signal: Signal):
        """Calculates phase of spikes computed for placefields

        Parameters
        ----------
        theta_chan : int
            lfp channel to use for calculating theta phases
        """
        assert signal.n_channels == 1, "signal should have only a single trace"
        sig_t = signal.time
        thetaparam = ThetaParams(signal.traces, fs=signal.sampling_rate)

        phase = []
        for spiketrain in self.ratemap_spkitrains:
            phase.append(np.interp(spiketrain, sig_t, thetaparam.angle))

        self.ratemap_spiketrains_phases = phase

    def plot_with_phase(self, ax=None, normalize=True, stack=True, cmap="tab20b", subplots=(5, 8)):
        cmap = mpl.cm.get_cmap(cmap)

        mapinfo = self.ratemaps

        ratemaps = mapinfo["ratemaps"]
        if normalize:
            ratemaps = [map_ / np.max(map_) for map_ in ratemaps]
        phases = mapinfo["phases"]
        position = mapinfo["pos"]
        nCells = len(ratemaps)
        bin_cntr = self.bin[:-1] + np.diff(self.bin).mean() / 2

        def plot_(cell, ax, axphase):
            color = cmap(cell / nCells)
            if subplots is None:
                ax.clear()
                axphase.clear()
            ax.fill_between(bin_cntr, 0, ratemaps[cell], color=color, alpha=0.3)
            ax.plot(bin_cntr, ratemaps[cell], color=color, alpha=0.2)
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Normalized frate")
            ax.set_title(
                " ".join(filter(None, ("Cell", str(cell), self.run_dir.capitalize())))
            )
            if normalize:
                ax.set_ylim([0, 1])
            axphase.scatter(position[cell], phases[cell], c="k", s=0.6)
            if stack:  # double up y-axis as is convention for phase precession plots
                axphase.scatter(position[cell], phases[cell] + 360, c="k", s=0.6)
            axphase.set_ylabel(r"$\theta$ Phase")

        if ax is None:

            if subplots is None:
                _, gs = plotting.Fig().draw(grid=(1, 1), size=(10, 5))
                ax = plt.subplot(gs[0])
                ax.spines["right"].set_visible(True)
                axphase = ax.twinx()
                widgets.interact(
                    plot_,
                    cell=widgets.IntSlider(
                        min=0,
                        max=nCells - 1,
                        step=1,
                        description="Cell ID:",
                    ),
                    ax=widgets.fixed(ax),
                    axphase=widgets.fixed(axphase),
                )
            else:
                _, gs = plotting.Fig().draw(grid=subplots, size=(15, 10))
                for cell in range(nCells):
                    ax = plt.subplot(gs[cell])
                    axphase = ax.twinx()
                    plot_(cell, ax, axphase)

        return ax

    def plot_ratemaps(self, ax=None, pad=2, normalize=False, sortby=None, cmap="tab20b"):
        # returns: ax , sort_ind, colors
        raise NotImplementedError # this isn't supposed to be used anymore!
        # return plotting.plot_ratemap(self.ratemap, normalize_tuning_curve=True)

    def plot_raw(self, ax=None, subplots=(8, 9)):
        raise NotImplementedError # this isn't supposed to be used anymore!
        # return plotting.plot_raw(self.ratemap, self.t, self.x, 'BOTH', ax=ax, subplots=subplots)

    
        


class Pf2D(PfnConfigMixin, PfnDPlottingMixin):

    @staticmethod
    def _compute_occupancy(x, y, xbin, ybin, position_srate, smooth, should_return_raw_occupancy=False):
        # --- occupancy map calculation -----------
        # NRK todo: might need to normalize occupancy so sum adds up to 1
        # Please note that the histogram does not follow the Cartesian convention where x values are on the abscissa and y values on the ordinate axis. Rather, x is histogrammed along the first dimension of the array (vertical), and y along the second dimension of the array (horizontal).
        raw_occupancy, xedges, yedges = np.histogram2d(x, y, bins=(xbin, ybin))
        # occupancy = occupancy.T # transpose the occupancy before applying other operations
        # raw_occupancy = raw_occupancy / position_srate + 10e-16  # converting to seconds
        if ((smooth is not None) and ((smooth[0] > 0.0) & (smooth[1] > 0.0))): 
            raw_occupancy = gaussian_filter(raw_occupancy, sigma=(smooth[1], smooth[0])) # 2d gaussian filter
        # Histogram does not follow Cartesian convention (see Notes),
        # therefore transpose occupancy for visualization purposes.
        # raw occupancy is defined in terms of the number of samples that fall into each bin.
        if should_return_raw_occupancy:
            return raw_occupancy, xedges, yedges
        else:   
            seconds_occupancy, normalized_occupancy = _normalized_occupancy(raw_occupancy, position_srate=position_srate)
            return seconds_occupancy, xedges, yedges


        # return seconds_occupancy, xedges, yedges
     
    @staticmethod   
    def _compute_tuning_map(spk_x, spk_y, xbin, ybin, occupancy, smooth, should_return_raw_tuning_map=False):
        # raw_tuning_map: is the number of spike counts in each bin for this unit
        raw_tuning_map = np.histogram2d(spk_x, spk_y, bins=(xbin, ybin))[0]
        if ((smooth is not None) and ((smooth[0] > 0.0) & (smooth[1] > 0.0))):
            raw_tuning_map = gaussian_filter(raw_tuning_map, sigma=(smooth[1], smooth[0])) # need to flip smooth because the x and y are transposed
        if should_return_raw_tuning_map:
            return raw_tuning_map
        else:
            occupancy[occupancy == 0.0] = np.nan # pre-set the zero occupancy locations to NaN to avoid a warning in the next step. They'll be replaced with zero aftwards anyway
            occupancy_weighted_tuning_map = raw_tuning_map / occupancy # dividing by positions with zero occupancy result in a warning and the result being set to NaN. Set to 0.0 instead.
            occupancy_weighted_tuning_map = np.nan_to_num(occupancy_weighted_tuning_map, copy=True, nan=0.0) # set any NaN values to 0.0, as this is the correct weighted occupancy
            occupancy[np.isnan(occupancy)] = 0.0 # restore these entries back to zero
            return occupancy_weighted_tuning_map

    def str_for_filename(self, prefix_string=''):
        return '-'.join(['pf2D', f'{prefix_string}{self.config.str_for_filename(True)}'])
    
    def str_for_display(self, prefix_string=''):
        return '-'.join(['pf2D', f'{prefix_string}{self.config.str_for_display(True)}', f'cell_{curr_cell_id:02d}'])
    
    
    def __init__(
        self,
        neurons: Neurons,
        position: Position,
        epochs: Epoch = None,
        frate_thresh=1,
        speed_thresh=5,
        grid_bin=(1,1),
        smooth=(1,1),
    ):
        """computes 2d place field using (x,y) coordinates. It always computes two place maps with and
        without speed thresholds.

        Parameters
        ----------
        track_name : str
            name of track
        direction : forward, backward or None
            direction of running, by default None which means direction is ignored
        grid_bin : int
            bin size of position bining, by default 5
        speed_thresh : int
            speed threshold for calculating place field
        """
        
    
        # save the config that was used to perform the computations
        self.config = PlacefieldComputationParameters(speed_thresh=speed_thresh, grid_bin=grid_bin, smooth=smooth, frate_thresh=frate_thresh)
        # assert position.ndim < 2, "Only 2+ dimensional position are acceptable"
        spiketrains = neurons.spiketrains
        neuron_ids = neurons.neuron_ids
        n_neurons = neurons.n_neurons
        position_srate = position.sampling_rate
        
        self.x = position.x
        self.y = position.y
        self.t = position.time
        t_start = position.t_start
        t_stop = position.t_stop

        ## Binning with Fixed Number of Bins:    
        xbin, ybin, bin_info = _bin_pos_nD(self.x, self.y, bin_size=grid_bin) # bin_size mode
        # xbin = np.arange(min(self.x), max(self.x) + grid_bin[0], grid_bin[0])  # binning of x position
        # ybin = np.arange(min(self.y), max(self.y) + grid_bin[1], grid_bin[1])  # binning of y position

        # plot with:
            # X, Y = np.meshgrid(xbin, ybin) 
            # plt.pcolor(X, Y, occupancy)
            
        # diff_posx = np.diff(self.x)
        # diff_posy = np.diff(self.y)
        # self.speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2) / (1 / position_srate)
        self.speed = position.speed
        if ((smooth is not None) and (smooth[0] > 0.0)):
            self.speed = gaussian_filter1d(self.speed, sigma=smooth[0])
        
        spk_pos, spk_t, tuning_maps = [], [], []

        # ------ if direction then restrict to those epochs --------
        if epochs is not None:
            assert isinstance(epochs, Epoch), "epochs should be Epoch object"
            # print(f" using {run_dir} running only")
            spks = [
                np.concatenate(
                    [
                        spktrn[(spktrn > epc.start) & (spktrn < epc.stop)]
                        for epc in epochs.to_dataframe().itertuples()
                    ]
                )
                for spktrn in spiketrains
            ]
            # changing x, speed, time to only run epochs so occupancy map is consistent with that
            indx = np.concatenate(
                [
                    np.where((self.t > epc.start) & (self.t < epc.stop))[0]
                    for epc in epochs.to_dataframe().itertuples()
                ]
            ) 
            self.x = self.x[indx]
            self.y = self.y[indx]
            self.speed = self.speed[indx]
            self.t = self.t[indx]

            # --- occupancy map calculation -----------
            # NRK todo: might need to normalize occupancy so sum adds up to 1
            # occupancy = np.histogram2d(self.x, self.y, bins=(xbin, ybin))[0]
            # occupancy = occupancy / position_srate + 10e-16  # converting to seconds
            # occupancy = gaussian_filter(occupancy, sigma=smooth) # 2d gaussian filter
            occupancy, xedges, yedges = Pf2D._compute_occupancy(self.x, self.y, xbin, ybin, position_srate, smooth)
            # plot with:
            # X, Y = np.meshgrid(xedges, yedges) 
            # plt.pcolor(X, Y, occupancy)
            
            # re-interpolate given the updated spks
            for cell in spks:
                spk_spd = np.interp(cell, self.t, self.speed)
                spk_x = np.interp(cell, self.t, self.x)
                spk_y = np.interp(cell, self.t, self.y)
                spk_pos.append([spk_x, spk_y])
                spk_t.append(cell)
                # tuning curve calculation:               
                tuning_maps.append(Pf2D._compute_tuning_map(spk_x, spk_y, xbin, ybin, occupancy, smooth))

        else:
            # --- speed thresh occupancy----

            spks = [
                spktrn[(spktrn > t_start) & (spktrn < t_stop)] for spktrn in spiketrains
            ]
            dt = self.t[1] - self.t[0]
            indx = np.where(self.speed / dt > speed_thresh)[0]
            self.x, self.y, self.speed, self.t = self.x[indx], self.y[indx], self.speed[indx], self.t[indx]
            
            # --- occupancy map calculation -----------
            occupancy, xedges, yedges = Pf2D._compute_occupancy(self.x, self.y, xbin, ybin, position_srate, smooth)
            
            
            # re-interpolate here too:
            for cell in spks:
                spk_spd = np.interp(cell, self.t, self.speed)
                spk_x = np.interp(cell, self.t, self.x)
                spk_y = np.interp(cell, self.t, self.y)

                # speed threshold
                spd_ind = np.where(spk_spd > speed_thresh)[0]
                spk_pos.append([spk_x[spd_ind], spk_y[spd_ind]])
                spk_t.append(cell[spd_ind])

                # tuning curve calculation:
                tuning_maps.append(Pf2D._compute_tuning_map(spk_x, spk_y, xbin, ybin, occupancy, smooth))
                

        # ---- cells with peak frate abouve thresh ------
        # thresh_neurons_indx = [
        #     neuron_indx
        #     for neuron_indx in range(n_neurons)
        #     if np.nanmax(tuning_maps[neuron_indx]) > frate_thresh
        # ]

        # get_elem = lambda list_: [list_[_] for _ in thresh_neurons_indx]
        # there is only one tuning_map per neuron that means the thresh_neurons_indx:
        # tuning_maps = get_elem(tuning_maps)
        # tuning_maps = np.asarray(tuning_maps)
        
        filtered_tuning_maps, filter_function = _filter_by_frate(tuning_maps.copy(), frate_thresh)

        self.ratemap = Ratemap(
            filtered_tuning_maps, xbin=xbin, ybin=ybin, neuron_ids=filter_function(neuron_ids)
        )
        self.ratemap_spiketrains = filter_function(spk_t)
        self.ratemap_spiketrains_pos = filter_function(spk_pos)
        self.occupancy = occupancy
        self.frate_thresh = frate_thresh
        self.speed_thresh = speed_thresh

   


