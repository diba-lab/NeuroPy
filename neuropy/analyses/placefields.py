from dataclasses import dataclass

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths
from copy import deepcopy
import seaborn as sns

from neuropy import core
from neuropy.utils.signal_process import ThetaParams
from neuropy import plotting
from neuropy.utils.mathutil import contiguous_regions
from neuropy.externals.peak_prominence2d import getProminence


class Pf1Dsplit():
    """Class used to split up Pf1D object by blocks to assess reliability"""
    def __init__(
            self,
            neurons: core.Neurons,
            position: core.Position,
            epochs: core.Epoch = None,
            frate_thresh=1.0,
            speed_thresh=3,
            grid_bin=5,
            sigma=1,
            t_interval_split=60,
    ):
        self.t_start = position.t_start
        self.t_stop = position.t_stop

        # Get epochs for each split of the session
        blocks1, blocks2 = self.get_split_session_blocks(t_interval=t_interval_split)

        # Merge speed_thresh and blocks1 (speed_thresh is ignored in Pf1D class if epochs is provided)
        abv_thresh_epochs = core.Epoch.from_boolean_array(position.speed > speed_thresh, position.time)
        blocks1 = blocks1.intersection(abv_thresh_epochs, res=1/position.sampling_rate)
        blocks2 = blocks2.intersection(abv_thresh_epochs, res=1 / position.sampling_rate)

        # Last merge any other epochs provided
        if epochs is not None:
            blocks1 = blocks1.intersection(epochs, res=1/position.sampling_rate)
            blocks2 = blocks2.intersection(epochs, res=1/position.sampling_rate)

        # Create Pf1D object for each block
        self.pf1 = Pf1D(neurons, position, blocks1, frate_thresh, speed_thresh, grid_bin, sigma)
        self.pf2 = Pf1D(neurons, position, blocks2, frate_thresh, speed_thresh, grid_bin, sigma)

    def get_split_session_blocks(self, t_interval):
        """Calculate within session correlations for placefields calculated in 'time_interval_sec' blocks.

        :param t_interval: block size to use. Default (60) will break up session into 60 second blocks
        for calculating odd vs. even minute placefields. Using session midpoint will calculate 1st v 2nd half."""

        # Break out blocks
        blocks = core.Epoch(pd.DataFrame({"start": np.arange(self.t_start, self.t_stop - t_interval, t_interval),
                                          "stop": np.arange(self.t_start + t_interval, self.t_stop, t_interval),
                                          "label": ""}))
        blocks1 = blocks[::2]
        blocks1.set_labels("even")
        blocks2 = blocks[1::2]
        blocks2.set_labels("odd")

        return blocks1, blocks2

    def get_correlations(self, sigma_bin):
        """Calculate between block correlations after smoothing with sigma"""

        tuning_curves1 = self.pf1.smooth_tuning_curves(sigma_bin)
        tuning_curves2 = self.pf2.smooth_tuning_curves(sigma_bin)

        corrs = []
        for tc1, tc2 in zip(tuning_curves1, tuning_curves2):
            corrs.append(np.corrcoef(tc1, tc2)[0, 1])

        return np.array(corrs)


class Pf1D(core.Ratemap):
    def __init__(
        self,
        neurons: core.Neurons,
        position: core.Position,
        epochs: core.Epoch = None,
        frate_thresh=1.0,
        speed_thresh=3,
        grid_bin=5,
        sigma=0,
        sigma_pos=0.1,
    ):
        """computes 1d place field using linearized coordinates. It always computes two place maps with and
        without speed thresholds.
        Parameters
        ----------
        neurons : core.Neurons
            neurons obj containing spiketrains and related info
        position: core.Position
            1D position
        grid_bin : int
            bin size of position binning, by default 5 cm
        epochs : core.Epoch,
            restrict calculation to these epochs, default None
        frate_thresh : float,
            peak firing rate should be above this value, default 1 Hz
        speed_thresh : float
            speed threshold for calculating place field, by default None
        sigma : float
            standard deviation for smoothing occupancy and spikecounts in each position bin,
            in units of cm, PRIOR to calculating binned tuning curves. default 0 cm
            NOTE that smoothing before creating tuning-curves is not standard, kept for legacy purposes.
        sigma_pos: float
            smoothing kernel for smoothing position (and therefore speed) before speed thresholding and calculating
            occupancy.
            Recommended for high sample rates to remove artificially high speeds due to division by a very small
            denominator (1 / sample_rate).
        NOTE: speed_thresh is ignored if epochs is provided
        """

        assert position.ndim == 1, "Only 1 dimensional position are acceptable"
        neuron_ids = neurons.neuron_ids
        position_srate = position.sampling_rate
        if sigma_pos > 0:
            position = position.get_smoothed(sigma_pos)
        x = position.x
        speed = position.speed
        t = position.time
        t_start = position.t_start
        t_stop = position.t_stop

        smooth_ = lambda f: gaussian_filter1d(
            f, sigma / grid_bin, axis=-1
        ) if sigma > 0 else f  # divide by grid_bin to account for discrete spacing

        xbin = np.arange(np.nanmin(x), np.nanmax(x) + grid_bin, grid_bin)

        if epochs is not None:
            assert isinstance(epochs, core.Epoch), "epochs should be core.Epoch object"

            spiketrains = [
                np.concatenate(
                    [
                        spktrn[(spktrn >= epc.start) & (spktrn <= epc.stop)]
                        for epc in epochs.to_dataframe().itertuples()
                    ]
                )
                for spktrn in neurons.spiketrains
            ]
            # changing x, speed, time to only run epochs so occupancy map is consistent
            indx = np.concatenate(
                [
                    np.where((t >= epc.start) & (t <= epc.stop))[0]
                    for epc in epochs.to_dataframe().itertuples()
                ]
            )

            speed_thresh = None
            print("Note: speed_thresh is ignored when epochs is provided")
        else:
            spiketrains = neurons.time_slice(t_start, t_stop).spiketrains
            indx = np.where(speed >= speed_thresh)[0]

        # to avoid interpolation error, speed and position estimation for spiketrains should use time
        # and speed of entire position (not only on threshold crossing time points)
        x_thresh = x[indx]

        spk_pos, spk_t, spkcounts = [], [], []
        for spktrn in spiketrains:
            spk_spd = np.interp(spktrn, t, speed)
            spk_x = np.interp(spktrn, t, x)
            if speed_thresh is not None:
                indices = np.where(spk_spd >= speed_thresh)[0]
                spk_x = spk_x[indices]
                spktrn = spktrn[indices]

            spk_pos.append(spk_x)
            spk_t.append(spktrn)
            spkcounts.append(np.histogram(spk_x, bins=xbin)[0])

        spkcounts = smooth_(np.asarray(spkcounts))
        occupancy = np.histogram(x_thresh, bins=xbin)[0] / position_srate + 1e-16
        occupancy = smooth_(occupancy)
        tuning_curve = spkcounts / occupancy.reshape(1, -1)

        # ---- neurons with peak firing rate above thresh ------
        frate_thresh_indx = np.where(np.max(tuning_curve, axis=1) >= frate_thresh)[0]
        tuning_curve = tuning_curve[frate_thresh_indx, :]
        neuron_ids = neuron_ids[frate_thresh_indx]
        spk_t = [spk_t[_] for _ in frate_thresh_indx]
        spk_pos = [spk_pos[_] for _ in frate_thresh_indx]

        super().__init__(
            tuning_curves=tuning_curve, coords=xbin[:-1], neuron_ids=neuron_ids
        )
        self.ratemap_spiketrains = spk_t
        self.ratemap_spiketrains_pos = spk_pos
        self.occupancy = occupancy
        self.frate_thresh = frate_thresh
        self.speed_thresh = speed_thresh
        self.speed = speed
        self.t = t
        self.x = x
        self.t_start = t_start
        self.t_stop = t_stop
        self.sigma = sigma
        self.sigma_pos = sigma_pos

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
        for spiketrain in self.ratemap_spiketrains:
            phase.append(np.interp(spiketrain, sig_t, thetaparam.angle.squeeze()))

        self.ratemap_spiketrains_phases = phase

    def plot_with_phase(
        self, sigma=0, ax=None, normalize=True, stack=True, cmap="tab20b", subplots=(5, 8)
    ):
        cmap = mpl.cm.get_cmap(cmap)

        # mapinfo = self.ratemaps

        # ratemaps = mapinfo["ratemaps"]
        # ratemaps = self.ratemap_spiketrains
        ratemaps = self.tuning_curves
        if sigma > 0:
            ratemaps = gaussian_filter1d(ratemaps, sigma=sigma, axis=1)
        if normalize:
            # ratemaps = [map_ / np.max(map_) for map_ in ratemaps]
            ratemaps = [map_ / np.max(map_) if len(map_) > 0 else np.array([]) for map_ in ratemaps]
        # phases = mapinfo["phases"]
        # position = mapinfo["pos"]
        phases = self.ratemap_spiketrains_phases
        position = self.ratemap_spiketrains_pos
        nCells = len(ratemaps)
        bin_cntr = self.x_coords() + np.diff(self.x_coords()).mean() / 2

        def plot_(cell, ax, axphase):
            color = cmap(cell / nCells)
            if subplots is None:
                ax.clear()
                axphase.clear()
            ax.fill_between(bin_cntr, 0, ratemaps[cell], color=color, alpha=0.3)
            ax.plot(bin_cntr, ratemaps[cell], color=color, alpha=0.2)
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Normalized frate") if normalize else ax.set_ylabel("frate")
            ax.set_title(f"Cell id {self.neuron_ids[cell]}")
            # ax.set_title(
            #     " ".join(filter(None, ("Cell", str(cell), self.run_dir.capitalize())))
            # )
            if normalize:
                ax.set_ylim([0, 1])
            axphase.scatter(position[cell], phases[cell], c="k", s=0.6)
            if stack:  # double up y-axis as is convention for phase precession plots
                axphase.scatter(position[cell], phases[cell] + 360, c="k", s=0.6)
            axphase.set_ylabel(r"$\theta$ Phase")

        if ax is None:
            if subplots is None:
                Fig = plotting.Fig(nrows=1, ncols=1, size=(8, 3))
                ax = plt.subplot(Fig.gs[0])
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
                Fig = plotting.Fig(nrows=subplots[1], ncols=subplots[0],
                                     size=(15, 10))
                for cell in range(nCells):
                    ax = plt.subplot(Fig.gs[cell])
                    axphase = ax.twinx()
                    plot_(cell, ax, axphase)
        else: #assumes sending in one cell
            axphase = ax.twinx()
            cell = 0
            plot_(cell, ax, axphase)

        return ax

    def plot_ratemaps(self, **kwargs):
        return plotting.plot_ratemap(self, **kwargs)

    def plot_rasters(self, jitter=0, plot_time=False, scale=None, sort=True, ax=None):
        """Plot ratemap as a raster for each neuron

        Parameters
        ----------
        jitter: float, offset each neuron's spikes and get an estimate of spiking density

        plot_time: bool, True = show timing of each spike on y-axis

        scale: None = keep in native coords (input), 'tuning_curve' = scale to match tuning curve

        sort: True = sort by peak location, False = don't sort
        """

        assert isinstance(ax, plt.Axes) or (ax is None)
        assert (scale == "tuning_curve") or (scale is None)
        if ax is None:
            _, ax = plt.subplots()

        order = self.get_sort_order(by="index") if sort else np.arange(self.n_neurons)
        spiketrains_pos = [self.ratemap_spiketrains_pos[i] for i in order]
        spiketrains_t = [self.ratemap_spiketrains[i] for i in order]

        scale_factor = 1
        if scale == "tuning_curve":
            ncm = np.ptp(self.coords)
            nbins = self.tuning_curves.shape[1]
            scale_factor = (nbins - 0) / ncm

        for i, (spk_pos, spk_t) in enumerate(zip(spiketrains_pos, spiketrains_t)):
            if plot_time:
                ypos = (spk_t - self.t_start) / ((self.t_stop - self.t_start) * 1.1) + i - 0.45  # spike time
                ypos_traj = (self.t - self.t_start) / ((self.t_stop - self.t_start) * 1.1) + i - 0.45  # trajectory time
                ax.plot(self.x * scale_factor - 0.5, ypos_traj, "-", color=[0, 0, 1, 0.3])
                # NRK Todo: allow plotting by lap.
            else:
                ypos = i * np.ones_like(spk_pos) + np.random.randn(spk_pos.shape[0]) * jitter
            ax.plot(spk_pos * scale_factor - 0.5, ypos, "k.", markersize=2)

    def plot_ratemap_w_raster(self, ind=None, id=None, ax=None, **kwargs):

        # Get neuron index
        assert (ind == None) != (id == None), "Exactly one of 'inds' and 'ids' must be a list or array"
        if ind is None:
            ind = np.where(id == self.neuron_ids)[0][0]

        # Slice desired neuron's placefield
        pfuse = self.neuron_slice([ind])

        # Create axes
        if ax is None:
            _, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True,
                                 height_ratios=[3, 2])

        # Plot tuning curve
        pfuse.plot_ratemaps(ax=ax[0], **kwargs)

        # Plot raster below
        pfuse.plot_rasters(plot_time=True, ax=ax[1])

        return ax

    def plot_raw_ratemaps_laps(self, ax=None, subplots=(8, 9)):
        return plotting.plot_raw_ratemaps()

    def neuron_slice(self, inds=None, ids=None):
        """Slice out neurons"""
        assert (inds is None) != (ids is None), "Exactly one of 'inds' and 'ids' must be a list or array"
        if ids is not None:
            inds = [np.where(idd == self.neuron_ids)[0][0] for idd in np.atleast_1d(ids)]
        inds = np.sort(np.atleast_1d(inds))

        # Make a copy and slice
        pfslice = deepcopy(self)
        pfslice.tuning_curves = self.tuning_curves[inds]
        pfslice.neuron_ids = self.neuron_ids[inds]
        pfslice.ratemap_spiketrains = [self.ratemap_spiketrains[ind] for ind in inds]
        pfslice.ratemap_spiketrains_pos = [self.ratemap_spiketrains_pos[ind] for ind in inds]

        return pfslice

    def get_pf_data(self, sigma=1.5, plot=False, **kwargs):
        """Gets all pf data: peak heights, peak prominences, peak centers, peak edges, and peak widths
        :param: sigma: smoothing kernel width, default = 1.5
        :param: plot: bool, True = plot all placefields with peaks and widths overlaid, default = False
        :param: **kwargs: inputs to .get_pf_peaks or .get_pf_widths

        :return: pf_stats_df: pd.DataFrame with all place field stats"""

        # First parse kwargs out
        peaks_keys = ["step", "centroid_num_to_center", "verbose"]
        kwargs_peaks = {key: value for key, value in kwargs.items() if key in peaks_keys}
        kwargs_widths = {key: value for key, value in kwargs.items() if key in ["height_thresh"]}

        # Now loop through each neuron and calculate peak / width information
        pf_stats_list = []
        ind = 0
        for nid in tqdm(self.neuron_ids):
            heights, prominences, centers, tuning_curve = self.get_pf_peaks(cell_id=nid, sigma=sigma, **kwargs_peaks)
            widths, edges = self.get_pf_widths(tuning_curve.squeeze(), heights, prominences, centers, plot=plot,
                                               **kwargs_widths)
            for idp, (height, prom, cent, width, edge) in enumerate(zip(heights, prominences, centers, widths, edges)):
                pf_stats_list.append(pd.DataFrame({"cell_id": nid, "peak_no": idp, "height": height,
                                                   "prominence": prom, "center_bin": cent, "width_bin": width,
                                                   "left_edge": edge[0], "right_edge": edge[1]}, index=[ind]))
                ind += 1

        return pd.concat(pf_stats_list, axis=0)

    def get_pf_peaks(self, cell_ind=None, cell_id=None, sigma=1.5,
                     step=0.1, centroid_num_to_center=1, verbose=False, **kwargs):
        """Gets pf peaks using `peak_prominence2d.getProminence external package
        (https://github.com/Xunius/python_peak_promience2d)

        :param: cell_ind, cell_id: cell index or id to calculate
        :param: sigma: gaussian smoothing kernel size to use
        :param: step: step-size for peak_prominence2d.getProminence iterations, default=0.1
        :param: centroid_num_to_center: input to peak_prominence2d.getProminence, default=1
        :param: verbose: input to peak_prominence2d.getProminence, default=False
        :param: **kwargs: other inputs to peak_prominence2d.getProminence

        :return: heights, prominences, centers, tuning_curve
            np.ndarrays of height, prominence, and center for each field + smoothed tuning curve"""
        pf_use = self.neuron_slice(inds=cell_ind, ids=cell_id)
        tuning_curve = pf_use.tuning_curves
        if sigma > 0:
            tuning_curve = gaussian_filter1d(pf_use.tuning_curves, sigma=sigma, axis=1)
        peaks, idmap, promap, parentmap = getProminence(np.repeat(tuning_curve, 20, axis=0),
                                                        step=step, centroid_num_to_center=centroid_num_to_center,
                                                        verbose=verbose, **kwargs)

        centers, heights, prominences = [], [], []
        for ii, vv in peaks.items():
            xii, yii = vv['center']
            centers.append(xii)


            z2ii = vv['height']
            heights.append(z2ii)

            pro = vv['prominence']
            prominences.append(pro)

        return np.array(heights), np.array(prominences), np.array(centers), tuning_curve

    def get_pf_widths(self, tuning_curve, heights, prominences, centers, height_thresh=0.5, plot=False, ax=None):
        """Gets placefield widths after obtaining peak height, location, and prominence data using .get_pf_peaks

        :param: tuning_curve: smoothed tuning curve, output from .get_pf_peaks
        :param: heights, prominences, centers: outputs from .get_pf_peaks
        :param: height thresh: float between 0 and 1, height threshold at which to calculate pf width.
                1 = at peak, 0 = at base
        :param: plot: plots identified, heights, prominences, and widths, default = False
        :param: ax: axes to plot into if plot=True

        :return: widths, edges: 1d and 2d np.ndarrays of widths, and left/right edges for each field.
                 one edge = np.nan means the edge of the field lies outside of the data limits
                            at that height_thresh, width from other edge to track limit is still reported"""

        assert tuning_curve.ndim == 1, "Tuning curve must be 1-dimensional"
        edges, widths = [], []

        if plot:
            if ax is None:
                _, ax = plt.subplots()

        track_width = tuning_curve.size
        for height, pro, center in zip(heights, prominences, centers):
            # identify regions above height threshold
            abv_thresh_regions = contiguous_regions(tuning_curve - (height - pro * (1 - height_thresh)) > 0)

            # In case multiple peaks are above this threshold, grab only the indices which contain the peak center

            try:
                left_ind, right_ind = abv_thresh_regions[
                    np.array([(center > lims[0]) & (center < lims[1]) for lims in abv_thresh_regions])].squeeze()

                # Interpolate the exact crossing point
                left_edge, right_edge = np.nan, np.nan
                if left_ind > 0:
                    left_edge = np.interp(0, tuning_curve[[left_ind - 1, left_ind]] - (height - pro * (1 - height_thresh)),
                                          [left_ind - 1, left_ind])
                if right_ind < track_width:
                    right_edge = np.interp(0, tuning_curve[[right_ind, right_ind - 1]] - (height - pro * (1 - height_thresh)),
                                           [right_ind, right_ind - 1])

                if np.isnan(left_edge):
                    width_use = right_edge
                elif np.isnan(right_edge):
                    width_use = track_width - left_edge
                else:
                    width_use = right_edge - left_edge
            except ValueError:  # If width is less than one bin at that height threshold, make everything a nan
                left_edge, right_edge, width_use = np.nan, np.nan, np.nan

            widths.append(width_use)
            edges.append(np.array([left_edge, right_edge]))

        ### TODO: cleanup and remove any small peaks that are entirely within the width of another peak
        # or just merge peaks that are within a given distance of one another?  Not many!

        # Plot tuning curve
        if plot:

            self.plot_pf_peaks_and_widths(tuning_curve, widths, edges, heights, prominences, centers,
                                         height_thresh, track_width=track_width, ax=ax)

        return np.array(widths), np.array(edges)

    @staticmethod
    def plot_pf_peaks_and_widths(tuning_curve, widths, edges, heights, prominences, centers,
                                height_thresh, track_width=None, ax=None):

        # Create axes
        if ax is None:
            _, ax = plt.subplots()

        if track_width is None:
            track_width = tuning_curve.size

        # Plot tuning curve with peak, prominence, and width all shown
        ax.plot(tuning_curve, ".-")
        for width, edge, height, pro, center in zip(widths, edges, heights, prominences, centers):
            ax.plot([center, center], [height - pro, height], 'k:')
            if ~np.isnan(width):
                left_edge_use = 0 if np.isnan(edge[0]) else edge[0]
                right_edge_use = track_width if np.isnan(edge[1]) else edge[1]
                ax.plot([left_edge_use, right_edge_use],
                        [height - pro * (1 - height_thresh), height - pro * (1 - height_thresh)],
                        'r')

    # def get_pf_widths(self, rel_height_thresh: float = 0.5, dist_thresh: float = 10,
    #                   width: float = 3, height: float = 1, prominence: float = 1, smooth_sigma=1,
    #                   keep: str in ['all', 'peak_only'] = 'peak_only', sanity_check_plot: bool = False):
    #
    #     pf_tc_smooth = gaussian_filter1d(self.tuning_curves, sigma=smooth_sigma, axis=1) \
    #         if (smooth_sigma > 0) else self.tuning_curves
    #
    #     widths = []
    #     for idt, (tc, peak_loc) in enumerate(zip(pf_tc_smooth, self.peak_locations())):
    #         peaks, _ = find_peaks(tc, distance=dist_thresh, rel_height=rel_height_thresh, width=width, height=height,
    #                               prominence=prominence)
    #         biggest = np.argmax(tc)
    #         if keep == 'peak_only':
    #             try:
    #                 peaks = [peaks.flat[np.abs(peaks - biggest).argmin()]]
    #             except ValueError:
    #                 peaks = peaks
    #         width_results = peak_widths(tc, peaks=peaks, rel_height=rel_height_thresh)
    #         widths.append(width_results[0].squeeze() if width_results[0].size > 0 else np.array(np.nan))
    #
    #     if sanity_check_plot:  # plot last neuron
    #         _, ax = plt.subplots(2, 1, height_ratios=[4, 1], sharex=True)
    #         ax[0].plot(tc)
    #         ax[0].plot(peaks, tc[peaks], 'r.')
    #         ax[0].plot(peak_loc, tc[peak_loc], 'g*')
    #         ax[0].plot(biggest, tc[biggest], 'co', markerfacecolor=None)
    #         ax[0].hlines(*width_results[1:], 'r')
    #         ax[0].set_ylabel('Firing rate (Hz)')
    #
    #         self.plot_ratemaps_raster(jitter=0.1, scale='tuning_curve', sort=False, ax=ax[1])
    #         ax[1].set_xlabel(f"Bin # ({self.x_binsize} cm bins)")
    #         ax[1].set_ylim([idt - 0.5, idt + 0.5])
    #         ax[1].set_yticks([])
    #         sns.despine(left=True, ax=ax[1])
    #
    #     return widths

class Pf2D:
    def __init__(
        self,
        neurons: core.Neurons,
        position: core.Position,
        epochs: core.Epoch = None,
        frate_thresh=1.0,
        speed_thresh=3,
        grid_bin=1,
        sigma=1,
    ):
        """Calculates 2D placefields
        Parameters
        ----------
        period : list/array
            in seconds, time period between which placefields are calculated
        gridbin : int, optional
            bin size of grid in centimeters, by default 10
        speed_thresh : int, optional
            speed threshold in cm/s, by default 10 cm/s
        Returns
        -------
        [type]
            [description]
        """
        assert position.ndim > 1, "Position is not 2D"
        period = [position.t_start, position.t_stop]
        #smooth_ = lambda f: gaussian_filter1d(
        #    f, sigma / grid_bin, axis=-1
        #)  # divide by grid_bin to account for discrete spacing
        smooth_ = lambda f: gaussian_filter(f, sigma=(sigma / grid_bin, 
                                                      sigma / grid_bin))

        spikes = neurons.time_slice(*period).spiketrains
        cell_ids = neurons.neuron_ids
        nCells = len(spikes)

        # ----- Position---------
        xcoord = position.x
        ycoord = position.z #default for optitrack input
        time = position.time
        trackingRate = position.sampling_rate

        ind_maze = np.where((time > period[0]) & (time < period[1]))
        x = xcoord[ind_maze]
        y = ycoord[ind_maze]
        t = time[ind_maze]

        x_grid = np.arange(np.nanmin(x), np.nanmax(x) + grid_bin, grid_bin)
        y_grid = np.arange(np.nanmin(y), np.nanmax(y) + grid_bin, grid_bin)
        # x_, y_ = np.meshgrid(x_grid, y_grid)

        diff_posx = np.diff(x)
        diff_posy = np.diff(y)

        speed = np.sqrt(diff_posx**2 + diff_posy**2) / (1 / trackingRate)

        #speed = smooth_(speed)

        dt = t[1] - t[0]
        running = np.where(speed / dt > speed_thresh)[0]

        x_thresh = x[running]
        y_thresh = y[running]
        t_thresh = t[running]

        def make_pfs(
            t_, x_, y_, spkAll_, occupancy_, speed_thresh_, maze_, x_grid_, y_grid_
        ):
            maps, spk_pos, spk_t = [], [], []
            for cell in spkAll_:
                # assemble spikes and position data
                spk_maze = cell[np.where((cell > maze_[0]) & (cell < maze_[1]))]
                spk_speed = np.interp(spk_maze, t_[1:], speed)
                spk_y = np.interp(spk_maze, t_, y_)
                spk_x = np.interp(spk_maze, t_, x_)

                # speed threshold
                spd_ind = np.where(spk_speed > speed_thresh_)
                # spk_spd = spk_speed[spd_ind]
                spk_x = spk_x[spd_ind]
                spk_y = spk_y[spd_ind]

                # Calculate maps
                spk_map = np.histogram2d(spk_x, spk_y, bins=(x_grid_, y_grid_))[0]
                spk_map = smooth_(spk_map)
                maps.append(spk_map / occupancy_)

                spk_t.append(spk_maze[spd_ind])
                spk_pos.append([spk_x, spk_y])

            return maps, spk_pos, spk_t

        # --- occupancy map calculation -----------
        # NRK todo: might need to normalize occupancy so sum adds up to 1
        occupancy = np.histogram2d(x_thresh, y_thresh, bins=(x_grid, y_grid))[0]
        occupancy = occupancy / trackingRate + 10e-16  # converting to seconds
        occupancy = smooth_(occupancy)

        maps, spk_pos, spk_t = make_pfs(
            t, x, y, spikes, occupancy, speed_thresh, period, x_grid, y_grid
        )

        # ---- cells with peak frate abouve thresh ------
        good_cells_indx = [
            cell_indx
            for cell_indx in range(nCells)
            if np.max(maps[cell_indx]) > frate_thresh
        ]

        get_elem = lambda list_: [list_[_] for _ in good_cells_indx]

        self.spk_pos = get_elem(spk_pos)
        self.spk_t = get_elem(spk_t)
        self.ratemaps = get_elem(maps)
        self.cell_ids = cell_ids[good_cells_indx]
        self.occupancy = occupancy
        self.speed = speed
        self.x = x
        self.y = y
        self.t = t
        self.xgrid = x_grid
        self.ygrid = y_grid
        self.gridbin = grid_bin
        self.speed_thresh = speed_thresh
        self.period = period
        self.frate_thresh = frate_thresh
        self.mesh = np.meshgrid(
            self.xgrid[:-1] + self.gridbin / 2,
            self.ygrid[:-1] + self.gridbin / 2,
        )
        ngrid_centers_x = self.mesh[0].size
        ngrid_centers_y = self.mesh[1].size
        x_center = np.reshape(self.mesh[0], [ngrid_centers_x, 1], order="F")
        y_center = np.reshape(self.mesh[1], [ngrid_centers_y, 1], order="F")
        xy_center = np.hstack((x_center, y_center))
        self.gridcenter = xy_center.T

    def plotMap(self, subplots=(7, 4), fignum=None):
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

        map_use, thresh = self.ratemaps, self.speed_thresh

        nCells = len(map_use)
        nfigures = nCells // np.prod(subplots) + 1

        if fignum is None:
            if f := plt.get_fignums():
                fignum = f[-1] + 1
            else:
                fignum = 1

        figures, gs = [], []
        for fig_ind in range(nfigures):
            fig = plt.figure(fignum + fig_ind, figsize=(6, 10), clear=True)
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
                self.xgrid,
                self.ygrid,
                np.rot90(np.fliplr(pfmap)) / np.max(pfmap),
                cmap="jet",
                vmin=0,
            )  # rot90(flipud... is necessary to match plotRaw configuration.
            # max_frate =
            ax1.axis("off")
            ax1.set_title(
                f"Cell {self.cell_ids[cell]} \n{round(np.nanmax(pfmap),2)} Hz"
            )

            # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
            # cbar = fig.colorbar(im, cax=cbar_ax)
            # cbar.set_label("firing rate (Hz)")

    def plotRaw(
        self,
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

        spk_pos_use = self.spk_pos

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
            ax1.plot(spk_x, spk_y, ".r", markersize=0.8)  #, color=[1, 0, 0, alpha])
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
            fig.set_size_inches([10,7]) #([23, 9.7])

        # plot trajectories
        for a, pos, ylabel in zip(
            ax, [self.x, self.y], ["X position (cm)", "Y position (cm)"]
        ):
            a.plot(self.t, pos)
            a.set_xlabel("Time (seconds)")
            a.set_ylabel(ylabel)
            # pretty_plot(a)

        # Grab correct spike times/positions
        if speed_thresh:
            spk_pos_, spk_t_ = self.run_spk_pos, self.run_spk_t
        else:
            spk_pos_, spk_t_ = self.spk_pos, self.spk_t

        # plot spikes on trajectory
        for a, pos in zip(ax, spk_pos_[cellind]):
            a.plot(spk_t_[cellind], pos, "r.", color=[1, 0, 0, alpha])

        # Put info on title
        ipbool = self._obj.spikes.pyrid[cellind] == self._obj.spikes.info.index
        info = self._obj.spikes.info.iloc[ipbool]
        ax[0].set_title(
            "Cell "
            + str(info["id"])
            + ": q = "
            + str(info["q"])
            + ", speed_thresh="
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

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    import DataPaths.subjects as subjects
    from neuropy.io import BinarysignalIO
    sessions = subjects.remaze_sess()[1:]  # RatSDay2NSD does not have remaze position info
    sess = sessions[0]
    maze = sess.paradigm["maze"].flatten()
    remaze = sess.paradigm["re-maze"].flatten()
    neurons = sess.neurons_stable.get_neuron_type("pyr")
    kw = dict(frate_thresh=0, grid_bin=5)
    signal = sess.theta

    pfremaze = Pf1D(neurons, position=sess.remaze, **kw)

    pfmaze = Pf1D(neurons, position=sess.maze, **kw)

    eegtheta_file = sorted(sess.recinfo.dat_filename.parent.glob("*_thetachan.eeg"))[0]
    sess.thetachan_eeg = BinarysignalIO(eegtheta_file, n_channels=1, sampling_rate=sess.recinfo.eeg_sampling_rate)
    print(
        f"eeg file min = {sess.thetachan_eeg.n_frames / 1250 / 60:.3f}, last spike time = {neurons.get_all_spikes()[-1] / 60:.3f}")

    theta_sig = sess.thetachan_eeg.get_signal()

    pfmaze.estimate_theta_phases(theta_sig.time_slice(t_start=pfmaze.t_start, t_stop=pfmaze.t_stop))
    pfmaze.neuron_slice(inds=range(40)).plot_with_phase()

