from dataclasses import dataclass

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from neuropy.plotting.figure import pretty_plot

from .. import core
from neuropy.utils.signal_process import ThetaParams
from .. import plotting
from neuropy.utils.mixins.print_helpers import SimplePrintable, OrderedMeta


class PlacefieldComputationParameters(SimplePrintable, metaclass=OrderedMeta):
    """A simple wrapper object for parameters used in placefield calcuations"""
    decimal_point_character=","
    param_sep_char='-'
    def __init__(self, speed_thresh=3, grid_bin=2, smooth=2, frate_thresh=1):
        self.speed_thresh = speed_thresh
        self.grid_bin = grid_bin
        self.smooth = smooth
        self.frate_thresh = frate_thresh
        
    def str_for_filename(self):
        return "speedThresh_{:.2f}-gridBin_{:.2f}-smooth_{:.2f}-frateThresh_{:.2f}".format(self.speed_thresh, self.grid_bin, self.smooth, self.frate_thresh)
        
        
def perform_compute_placefields(active_epoch_session_Neurons, active_epoch_pos, computation_config: PlacefieldComputationParameters, active_epoch_placefields1D=None, active_epoch_placefields2D=None, should_force_recompute_placefields=True):
    """ Computes both 1D and 2D placefields.
    active_epoch_session_Neurons: 
    active_epoch_pos: 
    active_epoch_placefields1D (Pf1D, optional) & active_epoch_placefields2D (Pf2D, optional): allow you to pass already computed Pf1D and Pf2D objects from previous runs and it won't recompute them so long as should_force_recompute_placefields=False, which is useful in interactive Notebooks/scripts
    Usage:
        active_epoch_placefields1D, active_epoch_placefields2D = perform_compute_placefields(active_epoch_session_Neurons, active_epoch_pos, active_epoch_placefields1D, active_epoch_placefields2D, active_config.computation_config, should_force_recompute_placefields=True)
    """
    ## Linearized (1D) Position Placefields:
    if ((active_epoch_placefields1D is None) or should_force_recompute_placefields):
        print('Recomputing active_epoch_placefields...')
        active_epoch_placefields1D = Pf1D(neurons=active_epoch_session_Neurons, position=active_epoch_pos.linear_pos_obj,
                                          speed_thresh=computation_config.speed_thresh, grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
        print('\t done.')
    else:
        print('active_epoch_placefields1D already exists, reusing it')

    ## 2D Position Placemaps:
    if ((active_epoch_placefields2D is None) or should_force_recompute_placefields):
        print('Recomputing active_epoch_placefields2D...')
        active_epoch_placefields2D = Pf2D(neurons=active_epoch_session_Neurons, position=active_epoch_pos,
                                          speed_thresh=computation_config.speed_thresh, grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
        print('\t done.')
    else:
        print('active_epoch_placefields2D already exists, reusing it')
    
    return active_epoch_placefields1D, active_epoch_placefields2D


def plot_all_placefields(active_epoch_placefields1D, active_epoch_placefields2D, active_config):
    """ 
    active_epoch_placefields1D: (Pf1D)
    active_epoch_placefields2D: (Pf2D)
    active_config:
    Usage:
        ax_pf_1D, occupancy_fig, active_pf_2D_figures = plot_all_placefields(active_epoch_placefields1D, active_epoch_placefields2D, active_config)
    """
    active_epoch_name = active_config.active_epochs.name
    ## Linearized (1D) Position Placefields:
    if active_epoch_placefields1D is not None:
        ax_pf_1D = active_epoch_placefields1D.plot_ratemaps()
        active_pf_1D_identifier_string = '1D Placefields - {}'.format(active_epoch_name)
        plt.title(active_pf_1D_identifier_string)
        active_pf_1D_output_filename = '{}.pdf'.format(active_pf_1D_identifier_string)
        active_pf_1D_output_filepath = active_config.plotting_config.active_output_parent_dir.joinpath(active_pf_1D_output_filename)
        print('Saving 1D Placefield image out to "{}"...'.format(active_pf_1D_output_filepath))
        plt.savefig(active_pf_1D_output_filepath)
        print('done.')
    else:
        print('plot_all_placefields(...): active_epoch_placefields1D does not exist. Skipping it.')
        ax_pf_1D = None

    ## 2D Position Placemaps:
    if active_epoch_placefields2D is not None:
        active_pf_occupancy_2D_identifier_string = '2D Occupancy - {}'.format(active_epoch_name)
        occupancy_fig = plt.figure()
        occupancy_ax = occupancy_fig.gca()
        im = occupancy_ax.pcolorfast(
            active_epoch_placefields2D.ratemap.xbin_centers,
            active_epoch_placefields2D.ratemap.ybin_centers,
            np.rot90(np.fliplr(active_epoch_placefields2D.occupancy)) / np.max(active_epoch_placefields2D.occupancy),
            cmap="jet",
            vmin=0,
        )  # rot90(flipud... is necessary to match plotRaw configuration.
        plt.title(active_pf_occupancy_2D_identifier_string)
        plt.show()
        # Save ocupancy figure out to disk:
        active_pf_occupancy_2D_output_filename = '{}.pdf'.format(active_pf_occupancy_2D_identifier_string)
        active_pf_occupancy_2D_output_filepath = active_config.plotting_config.active_output_parent_dir.joinpath(active_pf_occupancy_2D_output_filename)
        print('Saving 2D Placefield image out to "{}"...'.format(active_pf_occupancy_2D_output_filepath))
        occupancy_fig.savefig(active_pf_occupancy_2D_output_filepath)
        print('\t done.')
        ## 2D Tuning Curves Figure:
        active_pf_2D_figures, active_pf_2D_gs = active_epoch_placefields2D.plotMap(subplots=(7, 7),figsize=(10, 10))
        # active_epoch_placefields2D.plotRaw()
        active_pf_2D_identifier_string = '2D Placefields - {}'.format(active_epoch_name)
        # plt.title(active_pf_2D_identifier_string)
        active_pf_2D_output_filename = '{}.pdf'.format(active_pf_2D_identifier_string)
        active_pf_2D_output_filepath = active_config.plotting_config.active_output_parent_dir.joinpath(active_pf_2D_output_filename)
        print('Saving 2D Placefield image out to "{}"...'.format(active_pf_2D_output_filepath))
        for aFig in active_pf_2D_figures:
            aFig.savefig(active_pf_2D_output_filepath)
        print('\t done.')
    else:
        print('plot_all_placefields(...): active_epoch_placefields2D does not exist. Skipping it.')
        occupancy_fig = None
        active_pf_2D_figures = None
    
    return ax_pf_1D, occupancy_fig, active_pf_2D_figures

class PfnConfigMixin:
    @property
    def str_for_filename(self):
        return self.config.str_for_filename()

    
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
    


class Pf1D(PfnConfigMixin, PfnDMixin):
    def __init__(
        self,
        neurons: core.Neurons,
        position: core.Position,
        epochs: core.Epoch = None,
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
        self.speed = gaussian_filter1d(self.speed, sigma=20)
        self.t = position.time
        t_start = position.t_start
        t_stop = position.t_stop

        xbin = np.arange(min(self.x), max(self.x), grid_bin)  # binning of x position

        spk_pos, spk_t, tuning_curve = [], [], []

        # ------ if direction then restrict to those epochs --------
        if epochs is not None:
            assert isinstance(epochs, core.Epoch), "epochs should be core.Epoch object"
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
            self.speed = self.speed[indx]
            self.t = self.t[indx]
            occupancy = np.histogram(self.x, bins=xbin)[0] / position_srate + 1e-16
            occupancy = gaussian_filter1d(occupancy, sigma=smooth)

            for cell in spks:
                spk_spd = np.interp(cell, self.t, self.speed)
                spk_x = np.interp(cell, self.t, self.x)

                spk_pos.append(spk_x)
                spk_t.append(cell)

                # tuning curve calculation
                tuning_curve.append(
                    gaussian_filter1d(np.histogram(spk_x, bins=xbin)[0], sigma=smooth)
                    / occupancy
                )

        else:
            # --- speed thresh occupancy----

            spks = [
                spktrn[(spktrn > t_start) & (spktrn < t_stop)] for spktrn in spiketrains
            ]
            indx = np.where(self.speed >= speed_thresh)[0]
            self.x, self.speed, self.t = self.x[indx], self.speed[indx], self.t[indx]

            occupancy = np.histogram(self.x, bins=xbin)[0] / position_srate + 1e-16
            occupancy = gaussian_filter1d(occupancy, sigma=smooth)

            for cell in spks:
                spk_spd = np.interp(cell, self.t, self.speed)
                spk_x = np.interp(cell, self.t, self.x)

                # speed threshold
                spd_ind = np.where(spk_spd > speed_thresh)[0]
                spk_pos.append(spk_x[spd_ind])
                spk_t.append(cell[spd_ind])

                # tuning curve calculation
                tuning_curve.append(
                    gaussian_filter1d(np.histogram(spk_x, bins=xbin)[0], sigma=smooth)
                    / occupancy
                )

        # ---- cells with peak frate abouve thresh ------
        thresh_neurons_indx = [
            neuron_indx
            for neuron_indx in range(n_neurons)
            if np.max(tuning_curve[neuron_indx]) > frate_thresh
        ]

        get_elem = lambda list_: [list_[_] for _ in thresh_neurons_indx]

        tuning_curve = get_elem(tuning_curve)
        tuning_curve = np.asarray(tuning_curve)
        self.ratemap = core.Ratemap(
            tuning_curve, xbin=xbin, neuron_ids=get_elem(neuron_ids)
        )
        self.ratemap_spiketrains = get_elem(spk_t)
        self.ratemap_spiketrains_pos = get_elem(spk_pos)
        self.occupancy = occupancy
        self.frate_thresh = frate_thresh
        self.speed_thresh = speed_thresh



    
    def estimate_theta_phases(self, signal: core.Signal):
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

    def plot_with_phase(
        self, ax=None, normalize=True, stack=True, cmap="tab20b", subplots=(5, 8)
    ):
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
        return plotting.plot_ratemap(self.ratemap, normalize_tuning_curve=True)

    def plot_raw(self, ax=None, subplots=(8, 9)):
        return plotting.plot_raw(self.ratemap, self.t, self.x, 'BOTH', ax=ax, subplots=subplots)
        # return plotting.plot_raw(self, ax=ax, subplots=subplots)

    def plotRaw_v_time(self, cellind, speed_thresh=False, alpha=0.5, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, sharex=True)
            fig.set_size_inches([23, 9.7])
            
            
        if ax is not list:
            ax = [ax]

        # plot trajectories
        for a, pos, ylabel in zip(
            ax, [self.x], ["X position (cm)"]
        ):
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
        for a, pos in zip(ax, [spk_pos_[cellind]]):
            a.plot(spk_t_[cellind], pos, ".", color=[1, 0, 0, alpha])

        # Put info on title
        ax[0].set_title(
            "Cell "
            + str(self.cell_ids[cellind])
            + ":, speed_thresh="
            + str(self.speed_thresh)
        )
        


class Pf2D(PfnConfigMixin, PfnDMixin):
    def __init__(
        self,
        neurons: core.Neurons,
        position: core.Position,
        epochs: core.Epoch = None,
        frate_thresh=1,
        speed_thresh=5,
        grid_bin=1,
        smooth=1,
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

        xbin = np.arange(min(self.x), max(self.x) + grid_bin, grid_bin)  # binning of x position
        ybin = np.arange(min(self.y), max(self.y) + grid_bin, grid_bin)  # binning of y position

        diff_posx = np.diff(self.x)
        diff_posy = np.diff(self.y)
        self.speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2) / (1 / position_srate)
        self.speed = gaussian_filter1d(self.speed, sigma=smooth)
        
        spk_pos, spk_t, tuning_maps = [], [], []

        # ------ if direction then restrict to those epochs --------
        if epochs is not None:
            assert isinstance(epochs, core.Epoch), "epochs should be core.Epoch object"
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
            occupancy = np.histogram2d(self.x, self.y, bins=(xbin, ybin))[0]
            occupancy = occupancy / position_srate + 10e-16  # converting to seconds
            occupancy = gaussian_filter(occupancy, sigma=smooth) # 2d gaussian filter
            

            for cell in spks:
                spk_spd = np.interp(cell, self.t, self.speed)
                spk_x = np.interp(cell, self.t, self.x)
                spk_y = np.interp(cell, self.t, self.y)
                spk_pos.append([spk_x, spk_y])
                spk_t.append(cell)

                # tuning curve calculation:               
                tuning_maps.append(
                    gaussian_filter(np.histogram2d(spk_x, spk_y, bins=(xbin, ybin))[0], sigma=smooth)
                    / occupancy
                )

        else:
            # --- speed thresh occupancy----

            spks = [
                spktrn[(spktrn > t_start) & (spktrn < t_stop)] for spktrn in spiketrains
            ]
            dt = self.t[1] - self.t[0]
            indx = np.where(self.speed / dt > speed_thresh)[0]
            self.x, self.y, self.speed, self.t = self.x[indx], self.y[indx], self.speed[indx], self.t[indx]
            
            # --- occupancy map calculation -----------
            # NRK todo: might need to normalize occupancy so sum adds up to 1
            occupancy = np.histogram2d(self.x, self.y, bins=(xbin, ybin))[0]
            occupancy = occupancy / position_srate + 10e-16  # converting to seconds
            occupancy = gaussian_filter(occupancy, sigma=smooth) # 2d gaussian filter
            
            for cell in spks:
                spk_spd = np.interp(cell, self.t, self.speed)
                spk_x = np.interp(cell, self.t, self.x)
                spk_y = np.interp(cell, self.t, self.y)

                # speed threshold
                spd_ind = np.where(spk_spd > speed_thresh)[0]
                spk_pos.append([spk_x[spd_ind], spk_y[spd_ind]])
                spk_t.append(cell[spd_ind])

                # tuning curve calculation:
                tuning_maps.append(
                    gaussian_filter(np.histogram2d(spk_x, spk_y, bins=(xbin, ybin))[0], sigma=smooth)
                    / occupancy
                )
                

        # ---- cells with peak frate abouve thresh ------
        thresh_neurons_indx = [
            neuron_indx
            for neuron_indx in range(n_neurons)
            if np.max(tuning_maps[neuron_indx]) > frate_thresh
        ]

        get_elem = lambda list_: [list_[_] for _ in thresh_neurons_indx]

        tuning_maps = get_elem(tuning_maps)
        tuning_maps = np.asarray(tuning_maps)
        self.ratemap = core.Ratemap(
            tuning_maps, xbin=xbin, ybin=ybin, neuron_ids=get_elem(neuron_ids)
        )
        self.ratemap_spiketrains = get_elem(spk_t)
        self.ratemap_spiketrains_pos = get_elem(spk_pos)
        self.occupancy = occupancy
        self.frate_thresh = frate_thresh
        self.speed_thresh = speed_thresh

   

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
            fig.subplots_adjust(hspace=0.4)
            fig.suptitle(
                "Place maps with peak firing rate (speed_threshold = "
                + str(thresh)
                + ")"
            )
            figures.append(fig)

        for cell, pfmap in enumerate(map_use):
            ind = cell // np.prod(subplots)
            subplot_ind = cell % np.prod(subplots)
            ax1 = figures[ind].add_subplot(gs[ind][subplot_ind])
            im = ax1.pcolorfast(
                self.ratemap.xbin,
                self.ratemap.ybin,
                np.rot90(np.fliplr(pfmap)) / np.max(pfmap),
                cmap="jet",
                vmin=0,
            )  # rot90(flipud... is necessary to match plotRaw configuration.
            # ax1.scatter(self.spk_pos[ind]) # tODO: add spikes
            # max_frate =
            
            if enable_spike_overlay:
                ax1.scatter(self.spk_pos[ind][0], self.spk_pos[ind][1], s=1, c='white', alpha=0.3, marker=',')
            
            ax1.axis("off")
            ax1.set_title(
                f"Cell {self.ratemap.neuron_ids[cell]} \n{round(np.nanmax(pfmap),2)} Hz"
            )

            # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
            # cbar = fig.colorbar(im, cax=cbar_ax)
            # cbar.set_label("firing rate (Hz)")
            
        return figures, gs

    def plotRaw(self,
        subplots=(10, 8),
        fignum=None,
        alpha=0.5,
        label_cells=False,
        ax=None,
        clus_use=None,
    ):
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

    def plotRaw_v_time(self, cellind, speed_thresh=False, alpha=0.5, ax=None):
        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True)
            fig.set_size_inches([23, 9.7])

        # plot trajectories
        for a, pos, ylabel in zip(
            ax, [self.x, self.y], ["X position (cm)", "Y position (cm)"]
        ):
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
            a.plot(spk_t_[cellind], pos, ".", color=[1, 0, 0, alpha])

        # Put info on title
        ax[0].set_title(
            "Cell "
            + str(self.cell_ids[cellind])
            + ":, speed_thresh="
            + str(self.speed_thresh)
        )

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

        self.plotRaw(speed_thresh=speed_thresh, clus_use=[cellind], ax=[ax2d])
        self.plotRaw_v_time(
            cellind, speed_thresh=speed_thresh, ax=[axx, axy], alpha=alpha
        )
        self._obj.spikes.plot_ccg(clus_use=[cellind], type="acg", ax=axccg)

        return fig_use





# class PF1d(SimplePrintable):
#     def compute(
#         self,
#         track_name,
#         frate_thresh=1,
#         run_dir=None,
#         grid_bin=5,
#         speed_thresh=0,
#         smooth=1,
#     ):
#         """computes 1d place field using linearized coordinates. It always computes two place maps with and
#         without speed thresholds.

#         Parameters
#         ----------
#         track_name : str
#             name of track
#         direction : forward, backward or None
#             direction of running, by default None which means direction is ignored
#         grid_bin : int
#             bin size of position bining, by default 5
#         speed_thresh : int
#             speed threshold for calculating place field
#         """

#         tracks = Track(self._obj)
#         assert track_name in tracks.names, f"{track_name} doesn't exist"
#         spikes = Spikes(self._obj)
#         trackingSRate = ExtractPosition(self._obj).tracking_sRate

#         maze = tracks.data[track_name]
#         assert (
#             "linear" in maze
#         ), f"Track {track_name} doesn't have linearized coordinates. First run tracks.linearize_position(track_name='{track_name}')"

#         x = maze.linear
#         speed = maze.speed
#         t = maze.time
#         period = [np.min(t), np.max(t)]
#         spks = spikes.pyr
#         cell_ids = spikes.pyrid
#         xbin = np.arange(min(x), max(x), grid_bin)  # binning of x position
#         nCells = len(spks)

#         spk_pos, spk_t, ratemap = [], [], []

#         # ------ if direction then restrict to those epochs --------
#         if run_dir in ["forward", "backward"]:
#             print(f" using {run_dir} running only")
#             run_epochs = tracks.get_laps(track_name)
#             run_epochs = run_epochs[run_epochs["direction"] == run_dir]
#             spks = [
#                 np.concatenate(
#                     [
#                         cell[(cell > epc.start) & (cell < epc.end)]
#                         for epc in run_epochs.itertuples()
#                     ]
#                 )
#                 for cell in spks
#             ]
#             # changing x, speed, time to only run epochs so occupancy map is consistent with that
#             indx = np.concatenate(
#                 [
#                     np.where((t > epc.start) & (t < epc.end))[0]
#                     for epc in run_epochs.itertuples()
#                 ]
#             )
#             x = x[indx]
#             speed = speed[indx]
#             t = t[indx]
#             occupancy = np.histogram(x, bins=xbin)[0] / trackingSRate + 1e-16
#             occupancy = gaussian_filter1d(occupancy, sigma=smooth)

#             for cell in spks:
#                 spk_spd = np.interp(cell, t, speed)
#                 spk_x = np.interp(cell, t, x)

#                 spk_pos.append(spk_x)
#                 spk_t.append(cell)

#                 # ratemap calculation
#                 ratemap.append(
#                     gaussian_filter1d(np.histogram(spk_x, bins=xbin)[0], sigma=smooth)
#                     / occupancy
#                 )

#         else:
#             # --- speed thresh occupancy----

#             spks = [cell[(cell > period[0]) & (cell < period[1])] for cell in spks]
#             indx = np.where(speed >= speed_thresh)[0]
#             x = x[indx]
#             speed = speed[indx]
#             t = t[indx]

#             occupancy = np.histogram(x, bins=xbin)[0] / trackingSRate + 1e-16
#             occupancy = gaussian_filter1d(occupancy, sigma=smooth)

#             for cell in spks:
#                 spk_spd = np.interp(cell, t, speed)
#                 spk_x = np.interp(cell, t, x)

#                 # speed threshold
#                 spd_ind = np.where(spk_spd > speed_thresh)[0]
#                 spk_pos.append(spk_x[spd_ind])
#                 spk_t.append(cell[spd_ind])

#                 # ratemap calculation
#                 ratemap.append(
#                     gaussian_filter1d(np.histogram(spk_x, bins=xbin)[0], sigma=smooth)
#                     / occupancy
#                 )

#         # ---- cells with peak frate abouve thresh ------
#         good_cells_indx = [
#             cell_indx
#             for cell_indx in range(nCells)
#             if np.max(ratemap[cell_indx]) > frate_thresh
#         ]

#         get_elem = lambda list_: [list_[_] for _ in good_cells_indx]

#         self.ratemap = get_elem(ratemap)
#         self.spk_t = get_elem(spk_t)
#         self.spk_pos = get_elem(spk_pos)
#         self.cell_ids = cell_ids[good_cells_indx]
#         self.occupancy = occupancy
#         self.speed = maze.speed
#         self.x = maze.linear
#         self.t = maze.time
#         self.bin = xbin
#         self.track_name = track_name
#         self.period = period
#         self.run_dir = run_dir
#         self.frate_thresh = frate_thresh
#         self.speed_thresh = speed_thresh

#     def lap_by_lap(self):
#         """lap by lap place field (very preliminary)
#         NOTE: it can be added to compute step, if instead of taking track_name, a time window is taken as input


#         Returns
#         -------
#         [type]
#             [description]
#         """

#         assert (
#             self.run_dir is not None
#         ), "Please compute using run_dir to forward or backward "

#         track = Track(self._obj)
#         track_srate = ExtractPosition(self._obj).tracking_sRate
#         cell_t = self.spk_t
#         cell_pos = self.spk_pos
#         track_data = track.data[self.track_name]
#         laps = track.laps[self.track_name]
#         laps = laps[laps["direction"] == self.run_dir]

#         # --- lap by lap occupancy ------
#         occupancy = []
#         for lap in laps.itertuples():
#             lap_pos = track_data[
#                 (track_data.time > lap.start) & (track_data.time < lap.end)
#             ].linear
#             occ_ = np.histogram(lap_pos, bins=self.bin)[0] / track_srate + 1e-16
#             occupancy.append(occ_)

#         all_cells = []
#         for t, pos in zip(cell_t, cell_pos):
#             lap_ratemap = []
#             for lap in laps.itertuples():
#                 ind = np.where((t > lap.start) & (t < lap.end))[0]
#                 epoch_pos = pos[ind]
#                 lap_ratemap.append(np.histogram(epoch_pos, bins=self.bin)[0])
#             lap_ratemap = np.asarray(lap_ratemap)
#             all_cells.append(lap_ratemap)

#         return all_cells



# ## Old:
# class PF2d(SimplePrintable):
#     # def __init__(self, basepath, **kwargs):
#     #     if isinstance(basepath, Recinfo):
#     #         self._obj = basepath
#     #     else:
#     #         self._obj = Recinfo(basepath)

#     def __init__(self, period, spikes, cell_ids, x, y, t, trackingRate, gridbin=10, speed_thresh=5, frate_thresh=1, smooth=2):
#         self.perform_compute(period, spikes, cell_ids, x, y, t, trackingRate, gridbin, speed_thresh, frate_thresh, smooth)            
            
#     def compute(self, period, spikes=None, gridbin=10, speed_thresh=5, frate_thresh=1, smooth=2):
#         """Calculates 2D placefields

#         Parameters
#         ----------
#         period : list/array
#             in seconds, time period between which placefields are calculated
#         gridbin : int, optional
#             bin size of grid in centimeters, by default 10
#         speed_thresh : int, optional
#             speed threshold in cm/s, by default 10 cm/s

#         Returns
#         -------
#         [type]
#             [description]
#         """
#         assert len(period) == 2, "period should have length 2"
#         position = ExtractPosition(self._obj)
#         # ------ Cell selection ---------
#         if spikes is None:
#             spike_info = Spikes(self._obj)
#             spikes = spike_info.pyr
#             cell_ids = spike_info.pyrid
#         else:
#             cell_ids = np.arange(len(spikes))

#         # ----- Position---------
#         xcoord = position.x
#         ycoord = position.y
#         time = position.t
#         trackingRate = position.tracking_sRate

#         ind_maze = np.where((time > period[0]) & (time < period[1]))
#         x = xcoord[ind_maze]
#         y = ycoord[ind_maze]
#         t = time[ind_maze]
        
#         return self.perform_compute(period, spikes, cell_ids, x, y, t, trackingRate, gridbin=gridbin, speed_thresh=speed_thresh, frate_thresh=frate_thresh, smooth=smooth)        
        
#     # x, y, t, trackingRate
#     def perform_compute(self, period, spikes, cell_ids, x, y, t, trackingRate, gridbin=10, speed_thresh=5, frate_thresh=1, smooth=2):
#         nCells = len(spikes)
        
#         x_grid = np.arange(min(x), max(x) + gridbin, gridbin)
#         y_grid = np.arange(min(y), max(y) + gridbin, gridbin)
#         # x_, y_ = np.meshgrid(x_grid, y_grid)

#         diff_posx = np.diff(x)
#         diff_posy = np.diff(y)

#         speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2) / (1 / trackingRate)
#         speed = gaussian_filter1d(speed, sigma=smooth)

#         dt = t[1] - t[0]
#         running = np.where(speed / dt > speed_thresh)[0]

#         x_thresh = x[running]
#         y_thresh = y[running]
#         t_thresh = t[running]

#         def make_pfs(t_, x_, y_, spkAll_, occupancy_, speed_thresh_, maze_, x_grid_, y_grid_):
#             maps, spk_pos, spk_t = [], [], []
#             for cell in spkAll_:
#                 # assemble spikes and position data
#                 spk_maze = cell[np.where((cell > maze_[0]) & (cell < maze_[1]))]
#                 spk_speed = np.interp(spk_maze, t_[1:], speed)
#                 spk_y = np.interp(spk_maze, t_, y_)
#                 spk_x = np.interp(spk_maze, t_, x_)

#                 # speed threshold
#                 spd_ind = np.where(spk_speed > speed_thresh_)
#                 # spk_spd = spk_speed[spd_ind]
#                 spk_x = spk_x[spd_ind]
#                 spk_y = spk_y[spd_ind]

#                 # Calculate maps
#                 spk_map = np.histogram2d(spk_x, spk_y, bins=(x_grid_, y_grid_))[0]
#                 spk_map = gaussian_filter(spk_map, sigma=smooth)
#                 maps.append(spk_map / occupancy_)

#                 spk_t.append(spk_maze[spd_ind])
#                 spk_pos.append([spk_x, spk_y])

#             return maps, spk_pos, spk_t

#         # --- occupancy map calculation -----------
#         # NRK todo: might need to normalize occupancy so sum adds up to 1
#         occupancy = np.histogram2d(x_thresh, y_thresh, bins=(x_grid, y_grid))[0]
#         occupancy = occupancy / trackingRate + 10e-16  # converting to seconds
#         occupancy = gaussian_filter(occupancy, sigma=2) # 2d gaussian filter

#         maps, spk_pos, spk_t = make_pfs(t, x, y, spikes, occupancy, speed_thresh, period, x_grid, y_grid)

#         # ---- cells with peak frate abouve thresh ------
#         good_cells_indx = [
#             cell_indx
#             for cell_indx in range(nCells)
#             if np.max(maps[cell_indx]) > frate_thresh
#         ]

#         get_elem = lambda list_: [list_[_] for _ in good_cells_indx]

#         # tuning_curve  ->  spk_pos
#         # neuron_indx   ->  cell_indx
#         # thresh_neurons_indx   ->  good_cells_indx
#         self.spk_pos = get_elem(spk_pos)
#         self.spk_t = get_elem(spk_t)
#         self.ratemaps = get_elem(maps)
#         self.cell_ids = cell_ids[good_cells_indx]
#         self.occupancy = occupancy
#         self.speed = speed
#         self.x = x
#         self.y = y
#         self.t = t
#         self.xgrid = x_grid
#         self.ygrid = y_grid
#         self.gridbin = gridbin
#         self.speed_thresh = speed_thresh
#         self.period = period
#         self.frate_thresh = frate_thresh
#         self.mesh = np.meshgrid(
#             self.xgrid[:-1] + self.gridbin / 2,
#             self.ygrid[:-1] + self.gridbin / 2,
#         )
#         ngrid_centers_x = self.mesh[0].size
#         ngrid_centers_y = self.mesh[1].size
#         x_center = np.reshape(self.mesh[0], [ngrid_centers_x, 1], order="F")
#         y_center = np.reshape(self.mesh[1], [ngrid_centers_y, 1], order="F")
#         xy_center = np.hstack((x_center, y_center))
#         self.gridcenter = xy_center.T

#     def plotMap(self, subplots=(7, 4), fignum=None):
#         """Plots heatmaps of placefields with peak firing rate

#         Parameters
#         ----------
#         speed_thresh : bool, optional
#             [description], by default False
#         subplots : tuple, optional
#             number of cells within each figure window. If cells exceed the number of subplots, then cells are plotted in successive figure windows of same size, by default (10, 8)
#         fignum : int, optional
#             figure number to start from, by default None
#         """

#         map_use, thresh = self.ratemaps, self.speed_thresh

#         nCells = len(map_use)
#         nfigures = nCells // np.prod(subplots) + 1

#         if fignum is None:
#             if f := plt.get_fignums():
#                 fignum = f[-1] + 1
#             else:
#                 fignum = 1

#         figures, gs = [], []
#         for fig_ind in range(nfigures):
#             fig = plt.figure(fignum + fig_ind, figsize=(6, 10), clear=True)
#             gs.append(GridSpec(subplots[0], subplots[1], figure=fig))
#             fig.subplots_adjust(hspace=0.4)
#             fig.suptitle(
#                 "Place maps with peak firing rate (speed_threshold = "
#                 + str(thresh)
#                 + ")"
#             )
#             figures.append(fig)

#         for cell, pfmap in enumerate(map_use):
#             ind = cell // np.prod(subplots)
#             subplot_ind = cell % np.prod(subplots)
#             ax1 = figures[ind].add_subplot(gs[ind][subplot_ind])
#             im = ax1.pcolorfast(
#                 self.xgrid,
#                 self.ygrid,
#                 np.rot90(np.fliplr(pfmap)) / np.max(pfmap),
#                 cmap="jet",
#                 vmin=0,
#             )  # rot90(flipud... is necessary to match plotRaw configuration.
#             # max_frate =
#             ax1.axis("off")
#             ax1.set_title(
#                 f"Cell {self.cell_ids[cell]} \n{round(np.nanmax(pfmap),2)} Hz"
#             )

#             # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
#             # cbar = fig.colorbar(im, cax=cbar_ax)
#             # cbar.set_label("firing rate (Hz)")

#     def plotRaw(self,
#         subplots=(10, 8),
#         fignum=None,
#         alpha=0.5,
#         label_cells=False,
#         ax=None,
#         clus_use=None,
#     ):
#         if ax is None:
#             fig = plt.figure(fignum, figsize=(6, 10))
#             gs = GridSpec(subplots[0], subplots[1], figure=fig)
#             # fig.subplots_adjust(hspace=0.4)
#         else:
#             assert len(ax) == len(
#                 clus_use
#             ), "Number of axes must match number of clusters to plot"
#             fig = ax[0].get_figure()

#         spk_pos_use = self.spk_pos

#         if clus_use is not None:
#             spk_pos_tmp = spk_pos_use
#             spk_pos_use = []
#             [spk_pos_use.append(spk_pos_tmp[a]) for a in clus_use]

#         for cell, (spk_x, spk_y) in enumerate(spk_pos_use):
#             if ax is None:
#                 ax1 = fig.add_subplot(gs[cell])
#             else:
#                 ax1 = ax[cell]
#             ax1.plot(self.x, self.y, color="#d3c5c5")
#             ax1.plot(spk_x, spk_y, ".r", markersize=0.8, color=[1, 0, 0, alpha])
#             ax1.axis("off")
#             if label_cells:
#                 # Put info on title
#                 info = self.cell_ids[cell]
#                 ax1.set_title(f"Cell {info}")

#         fig.suptitle(
#             f"Place maps for cells with their peak firing rate (frate thresh={self.peak_frate},speed_thresh={self.speed_thresh})"
#         )

#     def plotRaw_v_time(self, cellind, speed_thresh=False, alpha=0.5, ax=None):
#         if ax is None:
#             fig, ax = plt.subplots(2, 1, sharex=True)
#             fig.set_size_inches([23, 9.7])

#         # plot trajectories
#         for a, pos, ylabel in zip(
#             ax, [self.x, self.y], ["X position (cm)", "Y position (cm)"]
#         ):
#             a.plot(self.t, pos)
#             a.set_xlabel("Time (seconds)")
#             a.set_ylabel(ylabel)
#             pretty_plot(a)

#         # Grab correct spike times/positions
#         if speed_thresh:
#             spk_pos_, spk_t_ = self.run_spk_pos, self.run_spk_t
#         else:
#             spk_pos_, spk_t_ = self.spk_pos, self.spk_t

#         # plot spikes on trajectory
#         for a, pos in zip(ax, spk_pos_[cellind]):
#             a.plot(spk_t_[cellind], pos, "r.", color=[1, 0, 0, alpha])

#         # Put info on title
#         ipbool = self._obj.spikes.pyrid[cellind] == self._obj.spikes.info.index
#         info = self._obj.spikes.info.iloc[ipbool]
#         ax[0].set_title(
#             "Cell "
#             + str(info["id"])
#             + ": q = "
#             + str(info["q"])
#             + ", speed_thresh="
#             + str(self.speed_thresh)
#         )

#     def plot_all(self, cellind, speed_thresh=True, alpha=0.4, fig=None):
#         if fig is None:
#             fig_use = plt.figure(figsize=[28.25, 11.75])
#         else:
#             fig_use = fig
#         gs = GridSpec(2, 4, figure=fig_use)
#         ax2d = fig_use.add_subplot(gs[0, 0])
#         axccg = np.asarray(fig_use.add_subplot(gs[1, 0]))
#         axx = fig_use.add_subplot(gs[0, 1:])
#         axy = fig_use.add_subplot(gs[1, 1:], sharex=axx)

#         self.plotRaw(speed_thresh=speed_thresh, clus_use=[cellind], ax=[ax2d])
#         self.plotRaw_v_time(
#             cellind, speed_thresh=speed_thresh, ax=[axx, axy], alpha=alpha
#         )
#         self._obj.spikes.plot_ccg(clus_use=[cellind], type="acg", ax=axccg)

#         return fig_use
