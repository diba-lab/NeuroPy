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
from neuropy.utils.misc import is_iterable

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


    

    def plot_raw(self, subplots=(10, 8), fignum=None, alpha=0.5, label_cells=False, ax=None, clus_use=None):
        """ Plots the Placefield raw spiking activity for all cells"""
        if self.ndim < 2:
            ## TODO: Pf1D Temporary Workaround:
            return plotting.plot_raw(self.ratemap, self.t, self.x, 'BOTH', ax=ax, subplots=subplots)
        else:        
            if ax is None:
                fig = plt.figure(fignum, figsize=(12, 20))
                gs = GridSpec(subplots[0], subplots[1], figure=fig)
                # fig.subplots_adjust(hspace=0.4)
            else:
                assert len(ax) == len(clus_use), "Number of axes must match number of clusters to plot"
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
                ax1.plot(self.x, self.y, color="#d3c5c5") # Plot the animal's position. This will be the same for all cells
                ax1.plot(spk_x, spk_y, '.', markersize=0.8, color=[1, 0, 0, alpha]) # plot the cell-specific spike locations
                ax1.axis("off")
                if label_cells:
                    # Put cell info (id, etc) on title
                    info = self.cell_ids[cell]
                    ax1.set_title(f"Cell {info}")

            fig.suptitle(f"Place maps for cells with their peak firing rate (frate thresh={self.frate_thresh},speed_thresh={self.speed_thresh})")
            return fig
            
        
    def plotRaw_v_time(self, cellind, speed_thresh=False, alpha=0.5, ax=None):
        """ Updated to work with both 1D and 2D Placefields """   
        if ax is None:
            fig, ax = plt.subplots(self.ndim, 1, sharex=True)
            fig.set_size_inches([23, 9.7])
        
        if not is_iterable(ax):
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
    def _compute_firing_map(spk_x, xbin, smooth):
        firing_map = np.histogram(spk_x, bins=xbin)[0]
        if ((smooth is not None) and (smooth > 0.0)):
            firing_map = gaussian_filter1d(firing_map, sigma=smooth)
        return firing_map
    
    @staticmethod   
    def _compute_tuning_map(spk_x, xbin, occupancy, smooth, should_also_return_intermediate_firing_map=False):
        firing_map = Pf1D._compute_firing_map(spk_x, xbin, smooth)
        tuning_map = firing_map / occupancy
        if should_also_return_intermediate_firing_map:
            return tuning_map, firing_map
        else:
            return tuning_map
    
    def __init__(self, neurons: Neurons, position: Position, epochs: Epoch = None, frate_thresh=1, speed_thresh=5, grid_bin=1, smooth=1, ):
        raise DeprecationWarning

    # ## TO REFACTOR
    # def estimate_theta_phases(self, signal: Signal):
    #     """Calculates phase of spikes computed for placefields

    #     Parameters
    #     ----------
    #     theta_chan : int
    #         lfp channel to use for calculating theta phases
    #     """
    #     assert signal.n_channels == 1, "signal should have only a single trace"
    #     sig_t = signal.time
    #     thetaparam = ThetaParams(signal.traces, fs=signal.sampling_rate)

    #     phase = []
    #     for spiketrain in self.ratemap_spkitrains:
    #         phase.append(np.interp(spiketrain, sig_t, thetaparam.angle))

    #     self.ratemap_spiketrains_phases = phase

    # def plot_with_phase(self, ax=None, normalize=True, stack=True, cmap="tab20b", subplots=(5, 8)):
    #     cmap = mpl.cm.get_cmap(cmap)

    #     mapinfo = self.ratemaps

    #     ratemaps = mapinfo["ratemaps"]
    #     if normalize:
    #         ratemaps = [map_ / np.max(map_) for map_ in ratemaps]
    #     phases = mapinfo["phases"]
    #     position = mapinfo["pos"]
    #     nCells = len(ratemaps)
    #     bin_cntr = self.bin[:-1] + np.diff(self.bin).mean() / 2

    #     def plot_(cell, ax, axphase):
    #         color = cmap(cell / nCells)
    #         if subplots is None:
    #             ax.clear()
    #             axphase.clear()
    #         ax.fill_between(bin_cntr, 0, ratemaps[cell], color=color, alpha=0.3)
    #         ax.plot(bin_cntr, ratemaps[cell], color=color, alpha=0.2)
    #         ax.set_xlabel("Position (cm)")
    #         ax.set_ylabel("Normalized frate")
    #         ax.set_title(
    #             " ".join(filter(None, ("Cell", str(cell), self.run_dir.capitalize())))
    #         )
    #         if normalize:
    #             ax.set_ylim([0, 1])
    #         axphase.scatter(position[cell], phases[cell], c="k", s=0.6)
    #         if stack:  # double up y-axis as is convention for phase precession plots
    #             axphase.scatter(position[cell], phases[cell] + 360, c="k", s=0.6)
    #         axphase.set_ylabel(r"$\theta$ Phase")

    #     if ax is None:

    #         if subplots is None:
    #             _, gs = plotting.Fig().draw(grid=(1, 1), size=(10, 5))
    #             ax = plt.subplot(gs[0])
    #             ax.spines["right"].set_visible(True)
    #             axphase = ax.twinx()
    #             widgets.interact(
    #                 plot_,
    #                 cell=widgets.IntSlider(
    #                     min=0,
    #                     max=nCells - 1,
    #                     step=1,
    #                     description="Cell ID:",
    #                 ),
    #                 ax=widgets.fixed(ax),
    #                 axphase=widgets.fixed(axphase),
    #             )
    #         else:
    #             _, gs = plotting.Fig().draw(grid=subplots, size=(15, 10))
    #             for cell in range(nCells):
    #                 ax = plt.subplot(gs[cell])
    #                 axphase = ax.twinx()
    #                 plot_(cell, ax, axphase)

    #     return ax


    
        


class Pf2D(PfnConfigMixin, PfnDMixin):

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
    def _compute_firing_map(spk_x, spk_y, xbin, ybin, smooth):
        # firing_map: is the number of spike counts in each bin for this unit
        firing_map = np.histogram2d(spk_x, spk_y, bins=(xbin, ybin))[0]
        if ((smooth is not None) and ((smooth[0] > 0.0) & (smooth[1] > 0.0))):
            firing_map = gaussian_filter(firing_map, sigma=(smooth[1], smooth[0])) # need to flip smooth because the x and y are transposed
        return firing_map
    
    @staticmethod   
    def _compute_tuning_map(spk_x, spk_y, xbin, ybin, occupancy, smooth, should_also_return_intermediate_firing_map=False):
        # raw_tuning_map: is the number of spike counts in each bin for this unit
        firing_map = Pf2D._compute_firing_map(spk_x, spk_y, xbin, ybin, smooth)
        occupancy[occupancy == 0.0] = np.nan # pre-set the zero occupancy locations to NaN to avoid a warning in the next step. They'll be replaced with zero aftwards anyway
        occupancy_weighted_tuning_map = firing_map / occupancy # dividing by positions with zero occupancy result in a warning and the result being set to NaN. Set to 0.0 instead.
        occupancy_weighted_tuning_map = np.nan_to_num(occupancy_weighted_tuning_map, copy=True, nan=0.0) # set any NaN values to 0.0, as this is the correct weighted occupancy
        occupancy[np.isnan(occupancy)] = 0.0 # restore these entries back to zero
        
        if should_also_return_intermediate_firing_map:
            return occupancy_weighted_tuning_map, firing_map
        else:
            return occupancy_weighted_tuning_map

    def __init__(self, neurons: Neurons, position: Position, epochs: Epoch = None, frate_thresh=1, speed_thresh=5, grid_bin=(1,1), smooth=(1,1), ):
        raise DeprecationWarning

