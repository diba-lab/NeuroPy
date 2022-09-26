from typing import Union
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.special import factorial
from tqdm import tqdm

from neuropy.analyses.placefields import PfND

from .. import core
from neuropy.utils import mathutil

from neuropy.utils.mixins.binning_helpers import BinningContainer # for epochs_spkcount getting the correct time bins
from neuropy.utils.mixins.binning_helpers import build_spanning_grid_matrix # for Decode2d reverse transformations from flat points


def epochs_spkcount(neurons: Union[core.Neurons, pd.DataFrame], epochs: Union[core.Epoch, pd.DataFrame], bin_size=0.01, slideby=None, export_time_bins:bool=False, included_neuron_ids=None, debug_print:bool=False):
    """Binning events and calculating spike counts

    Args:
        neurons (Union[core.Neurons, pd.DataFrame]): _description_
        epochs (Union[core.Epoch, pd.DataFrame]): _description_
        bin_size (float, optional): _description_. Defaults to 0.01.
        slideby (_type_, optional): _description_. Defaults to None.
        export_time_bins (bool, optional): If True returns a list of the actual time bin centers for each epoch in time_bins. Defaults to False.
        included_neuron_ids (bool, optional): Only relevent if using a spikes_df for the neurons input. Ensures there is one spiketrain built for each neuron in included_neuron_ids, even if there are no spikes.
        debug_print (bool, optional): _description_. Defaults to False.

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        list: spkcount - one for each epoch in filter_epochs
        list: nbins - A count of the number of time bins that compose each decoding epoch e.g. nbins: [7 2 7 1 5 2 7 6 8 5 8 4 1 3 5 6 6 6 3 3 4 3 6 7 2 6 4 1 7 7 5 6 4 8 8 5 2 5 5 8]
        list: time_bin_containers_list - None unless export_time_bins is True. 
        
    Usage:
    
        spkcount, nbins, time_bin_containers_list = 
    """
    
    # Handle extracting the spiketrains, which are a list with one entry for each neuron and each list containing the timestamps of the spike event
    if isinstance(neurons, core.Neurons):
        spiketrains = neurons.spiketrains
    elif isinstance(neurons, pd.DataFrame):
        # a spikes_df is passed in, build the spiketrains
        spikes_df = neurons
        spiketrains = spikes_df.spikes.get_unit_spiketrains(included_neuron_ids=included_neuron_ids)
    else:
        raise NotImplementedError

    # Handle either core.Epoch or pd.DataFrame objects:
    if isinstance(epochs, core.Epoch):
        epoch_df = epochs.to_dataframe()
        n_epochs = epochs.n_epochs
        
    elif isinstance(epochs, pd.DataFrame):
        epoch_df = epochs
        n_epochs = np.shape(epoch_df)[0] # there is one row per epoch
    else:
        raise NotImplementedError
    
    spkcount = []
    if export_time_bins:
        time_bin_containers_list = []
    else:
        time_bin_containers_list = None

    nbins = np.zeros(n_epochs, dtype="int")

    window_shape  = int(bin_size * 1000) # Ah, forces integer binsizes!
    if slideby is None:
        slideby = bin_size
        
    if debug_print:
        print(f'window_shape: {window_shape}, slideby: {slideby}')

    # ----- little faster but requires epochs to be non-overlapping ------
    # bins_epochs = []
    # for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
    #     bins = np.arange(epoch.start, epoch.stop, bin_size)
    #     nbins[i] = len(bins) - 1
    #     bins_epochs.extend(bins)
    # spkcount = np.asarray(
    #     [np.histogram(_, bins=bins_epochs)[0] for _ in spiketrains]
    # )

    # deleting unwanted columns that represent time between events
    # cumsum_nbins = np.cumsum(nbins)
    # del_columns = cumsum_nbins[:-1] + np.arange(len(cumsum_nbins) - 1)
    # spkcount = np.delete(spkcount, del_columns.astype(int), axis=1)

    for i, epoch in enumerate(epoch_df.itertuples()):
        # first dividing in 1ms
        bins = np.arange(epoch.start, epoch.stop, 0.001)
        spkcount_ = np.asarray(
            [np.histogram(_, bins=bins)[0] for _ in spiketrains]
        )
        if debug_print:
            print(f'i: {i}, epoch: [{epoch.start}, {epoch.stop}], bins: {np.shape(bins)}, np.shape(spkcount_): {np.shape(spkcount_)}')
        slide_view = np.lib.stride_tricks.sliding_window_view(spkcount_, window_shape, axis=1)[:, :: int(slideby * 1000), :].sum(axis=2)

        nbins[i] = slide_view.shape[1]
        if export_time_bins:
            if debug_print:
                print(f'nbins[i]: {nbins[i]}') # nbins: 20716
            
            reduced_slide_by_amount = int(slideby * 1000)
            reduced_time_bin_edges = bins[:: reduced_slide_by_amount] # equivalent to bins[np.arange(0, num_bad_time_bins, reduced_slide_by_amount, dtype=int)]
            # reduced_time_bins: only the FULL number of bin *edges*
            # reduced_time_bins # array([22.26, 22.36, 22.46, ..., 2093.66, 2093.76, 2093.86])
            bin_container = BinningContainer(edges=reduced_time_bin_edges)
            reduced_time_bin_centers = bin_container.centers
            
            # reduced_time_bin_centers = get_bin_centers(reduced_time_bin_edges) # get the centers of each bin. The length should be the same as nbins
            if debug_print:
                num_bad_time_bins = len(bins)
                print(f'num_bad_time_bins: {num_bad_time_bins}')
                print(f'reduced_slide_by_amount: {reduced_slide_by_amount}')
                print(f'reduced_time_bin_edges.shape: {reduced_time_bin_edges.shape}') # reduced_time_bin_edges.shape: (20717,)
                print(f'reduced_time_bin_centers.shape: {reduced_time_bin_centers.shape}') # reduced_time_bin_centers.shape: (20716,)

            assert len(reduced_time_bin_centers) == nbins[i], f"The length of the produced reduced_time_bin_centers and the nbins[i] should be the same, but len(reduced_time_bin_centers): {len(reduced_time_bin_centers)} and nbins[i]: {nbins[i]}!"
            # time_bin_centers_list.append(reduced_time_bin_centers)
            time_bin_containers_list.append(bin_container)
            
        spkcount.append(slide_view)

    return spkcount, nbins, time_bin_containers_list


class Decode1d:
    n_jobs = 8

    def __init__(self, neurons: core.Neurons, ratemap: core.Ratemap, epochs: core.Epoch=None, time_bin_size=0.5, slideby=None):
        self.ratemap = ratemap
        self._events = None
        self.posterior = None
        self.neurons = neurons
        self.time_bin_size = time_bin_size
        self.decodingtime = None
        self.time_bin_centers = None
        
        self.decoded_position = None
        self.epochs = epochs
        self.slideby = slideby
        self.score = None
        self.shuffle_score = None

        self._estimate()

    def _decoder(self, spkcount, ratemaps):
        """
        ===========================
        Probability is calculated using this formula
        prob = (1 / nspike!)* ((tau * frate)^nspike) * exp(-tau * frate)
        where,
            tau = binsize
        ===========================
        """
        tau = self.time_bin_size
        nCells = spkcount.shape[0]
        cell_prob = np.zeros((ratemaps.shape[1], spkcount.shape[1], nCells))
        for cell in range(nCells):
            cell_spkcnt = spkcount[cell, :][np.newaxis, :]
            cell_ratemap = ratemaps[cell, :][:, np.newaxis]

            coeff = 1 / (factorial(cell_spkcnt))
            # broadcasting
            cell_prob[:, :, cell] = (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (
                np.exp(-tau * cell_ratemap)
            )

        posterior = np.prod(cell_prob, axis=2)
        posterior /= np.sum(posterior, axis=0)

        return posterior

    def _estimate(self):
        """Estimates position within each"""

        tuning_curves = self.ratemap.tuning_curves
        bincntr = self.ratemap.xbin_centers

        if self.epochs is not None:
            spkcount, nbins, time_bin_centers_list = epochs_spkcount(self.neurons, self.epochs, self.time_bin_size, self.slideby)
            posterior = self._decoder(np.hstack(spkcount), tuning_curves)
            decodedPos = bincntr[np.argmax(posterior, axis=0)]
            cum_nbins = np.cumsum(nbins)[:-1]

            self.decodingtime = None # time bins are invalid for this mode
            self.time_bin_centers = None

            self.decoded_position = np.hsplit(decodedPos, cum_nbins)
            self.posterior = np.hsplit(posterior, cum_nbins)
            self.spkcount = spkcount
            self.nbins_epochs = nbins
            self.score, _ = self.score_posterior(self.posterior)

        else:
            flat_filtered_neurons = self.neurons.get_binned_spiketrains(bin_size=self.time_bin_size)
            spkcount = flat_filtered_neurons.spike_counts
            neuropy_decoder_time_bins = flat_filtered_neurons.time
            self.decodingtime = neuropy_decoder_time_bins # get the time_bins (bin edges)
            self.time_bin_centers = self.decodingtime[:-1] + np.diff(self.decodingtime) / 2.0
            # spkcount = self.neurons.get_binned_spiketrains(bin_size=self.bin_size).spike_counts

            self.posterior = self._decoder(spkcount, tuning_curves)
            self.decoded_position = bincntr[np.argmax(self.posterior, axis=0)]
            self.score = None

    def calculate_shuffle_score(self, n_iter=100, method="column"):
        """Shuffling and decoding epochs"""

        # print(f"Using {kind} shuffle")

        if method == "neuron_id":
            posterior, score = [], []
            for i in range(n_iter):
                tuning_curves = self.ratemap.tuning_curves.copy()
                np.random.shuffle(tuning_curves)
                post_ = self._decoder(np.hstack(self.spkcount), tuning_curves)
                cum_nbins = np.cumsum(self.nbins_epochs)[::-1]
                posterior.extend(np.hsplit(post_, cum_nbins))

            score = self.score_posterior(posterior)[0]
            score = score.reshape(n_iter, len(self.spkcount))

        if method == "column":

            def col_shuffle(mat):
                shift = np.random.randint(1, mat.shape[1], mat.shape[1])
                direction = np.random.choice([-1, 1], size=mat.shape[1])
                shift = shift * direction

                mat = np.array([np.roll(mat[:, i], sh) for i, sh in enumerate(shift)])
                return mat.T

            score = []
            for i in tqdm(range(n_iter)):
                evt_shuff = [col_shuffle(arr) for arr in self.posterior]
                score.append(self._score_events(evt_shuff)[0])

        # score = np.concatenate(score)
        self.shuffle_score = np.array(score)

    def score_posterior(self, p):
        """Scoring of epochs

        Returns
        -------
        [type]
            [description]

        References
        ----------
        1) Kloosterman et al. 2012
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(mathutil.radon_transform)(epoch) for epoch in p
        )
        score = [res[0] for res in results]
        slope = [res[1] for res in results]

        return np.asarray(score), np.asarray(slope)

    @property
    def p_value(self):
        shuff_score = self.shuffle_score
        n_iter = shuff_score.shape[0]
        diff_score = shuff_score - np.tile(self.score, (n_iter, 1))
        chance = np.where(diff_score > 0, 1, 0).sum(axis=0)
        return (chance + 1) / (n_iter + 1)

    def plot_in_bokeh(self):
        pass

    def plot_replay_epochs(self, pval=0.05, speed_thresh=True, cmap="hot"):
        pval_events = self.p_val_events
        replay_ind = np.where(pval_events < pval)[0]
        posterior = [self.posterior[_] for _ in replay_ind]
        sort_ind = np.argsort(self.score[replay_ind])[::-1]
        posterior = [posterior[_] for _ in sort_ind]
        events = self.events.iloc[replay_ind].reset_index(drop=True)
        events["score"] = self.score[replay_ind]
        events["slope"] = self.slope[replay_ind]
        events.sort_values(by=["score"], inplace=True, ascending=False)

        spikes = Spikes(self._obj)
        spks = spikes.pyr
        pf1d_obj = self.ratemaps

        mapinfo = pf1d_obj.ratemaps
        ratemaps = np.asarray(mapinfo["ratemaps"])

        # ----- removing cells that fire < 1 HZ --------
        good_cells = np.where(np.max(ratemaps, axis=1) > 1)[0]
        spks = [spks[_] for _ in good_cells]
        ratemaps = ratemaps[good_cells, :]

        # --- sorting the cells according to pf location -------
        sort_ind = np.argsort(np.argmax(ratemaps, axis=1))
        spks = [spks[_] for _ in sort_ind]
        ratemaps = ratemaps[sort_ind, :]

        figure = Fig()
        fig, gs = figure.draw(grid=(6, 12), hspace=0.34)

        for i, epoch in enumerate(events.itertuples()):
            gs_ = figure.subplot2grid(gs[i], grid=(2, 1), hspace=0.1)
            ax = plt.subplot(gs_[0])
            spikes.plot_raster(
                spks, ax=ax, period=[epoch.start, epoch.end], tstart=epoch.start
            )
            ax.set_title(
                f"Score = {np.round(epoch.score,2)},\n Slope = {np.round(epoch.slope,2)}",
                loc="left",
            )
            ax.set_xlabel("")
            ax.tick_params(length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            axdec = plt.subplot(gs_[1], sharex=ax)
            axdec.pcolormesh(
                np.arange(posterior[i].shape[1] + 1) * self.binsize,
                self.ratemaps.bin - np.min(self.ratemaps.bin),
                posterior[i],
                cmap=cmap,
                vmin=0,
                vmax=0.5,
            )
            axdec.set_ylabel("Position")

            if i % 12:
                ax.set_ylabel("")
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(axdec.get_yticklabels(), visible=False)
                axdec.set_ylabel("")

            if i > (5 * 6 - 1):
                axdec.set_xlabel("Time (ms)")


class Decode2d:
    """ 2D Decoder 
    
    """
    def __init__(self, pf2d_obj: PfND):
        assert isinstance(pf2d_obj, PfND)
        self.pf = pf2d_obj
        self.ratemap = self.pf.ratemap

        self._all_positions_matrix = None
        self._original_data_shape = None
        self._flat_all_positions_matrix = None
        
        self.time_bin_size = None
        self.decodingtime = None
        self.time_bin_centers = None
        
        self.actualbin = None
        self.posterior = None
        self.actualpos = None
        self.decoded_position = None

    def _decoder(self, spkcount, ratemaps):
        """
        ===========================
        Probability is calculated using this formula
        prob = (1 / nspike!)* ((tau * frate)^nspike) * exp(-tau * frate)
        where,
            tau = binsize
        ===========================
        """
        tau = self.time_bin_size
        nCells = spkcount.shape[0]
        # nSpikes = spkcount.shape[1] 
        # nFlatPositionBins = ratemaps.shape[1]
        cell_prob = np.zeros((ratemaps.shape[1], spkcount.shape[1], nCells))
        for cell in range(nCells):
            cell_spkcnt = spkcount[cell, :][np.newaxis, :]
            cell_ratemap = ratemaps[cell, :][:, np.newaxis]

            coeff = 1 / (factorial(cell_spkcnt))
            # broadcasting
            cell_prob[:, :, cell] = (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (
                np.exp(-tau * cell_ratemap)
            )

        posterior = np.prod(cell_prob, axis=2)
        posterior /= np.sum(posterior, axis=0)

        return posterior
    
    def estimate_behavior(self, spikes_df, t_start_end, time_bin_size=0.25, smooth=1, plot=True):
        """ 
        Updates:
            ._all_positions_matrix
            ._original_data_shape
            ._flat_all_positions_matrix
            .bin_size
            .decodingtime
            .time_bin_centers
            .actualbin
            .posterior
            .actualpos
            .decodedPos
        """
        ratemap_cell_ids = self.pf.cell_ids
        # spks = Spikes(self._obj).get_cells(ids=ratemap_cell_ids)
        spk_dfs = spikes_df.spikes.get_split_by_unit(included_neuron_ids=ratemap_cell_ids)
        spk_times = [cell_df[spikes_df.spikes.time_variable_name].to_numpy() for cell_df in spk_dfs]
        
        # ratemaps = self.pf.ratemap
        # tuning_curves = self.pf.ratemap.tuning_curves
        tuning_curves = self.ratemap.tuning_curves
        
        speed = self.pf.speed
        xgrid = self.pf.xbin
        ygrid = self.pf.ybin
        # gridbin = self.pf.gridbin
        
        # gridbin = (self.pf.bin_info['xstep'], self.pf.bin_info['ystep'])
        gridbin_x = self.pf.bin_info['xstep']
        gridbin_y = self.pf.bin_info['ystep']
        
        # gridcenter = self.pf.gridcenter
        # gridcenter = self.pf.gridcenter
        self._all_positions_matrix, self._flat_all_positions_matrix, self._original_data_shape = build_spanning_grid_matrix(x_values=self.pf.xbin_centers, y_values=self.pf.ybin_centers, debug_print=False)
        # len(self._flat_all_positions_matrix) # 1066
        
        # --- average position in each time bin and which gridbin it belongs to ----
        t = self.pf.t
        x = self.pf.x
        y = self.pf.y
        assert t_start_end is not None and isinstance(t_start_end, tuple)
        # t_start_end = self.pf.period
        tmz = np.arange(t_start_end[0], t_start_end[1], time_bin_size)
        self.time_bin_size = time_bin_size
        self.decodingtime = tmz # time_bin_edges
        self.time_bin_centers = tmz[:-1] + np.diff(tmz) / 2.0
        
        actualposx = stats.binned_statistic(t, values=x, bins=tmz)[0]
        actualposy = stats.binned_statistic(t, values=y, bins=tmz)[0]
        actualpos = np.vstack((actualposx, actualposy))
        self.actualpos = actualpos

        actualbin_x = xgrid[np.digitize(actualposx, bins=xgrid) - 1] + gridbin_x / 2
        actualbin_y = ygrid[np.digitize(actualposy, bins=ygrid) - 1] + gridbin_y / 2
        self.actualbin = np.vstack((actualbin_x, actualbin_y))

        # ---- spike counts and linearize 2d ratemaps -------
        spkcount = np.asarray([np.histogram(cell, bins=tmz)[0] for cell in spk_times])
        spkcount = gaussian_filter1d(spkcount, sigma=3, axis=1)
        # ratemaps = np.asarray([ratemap.flatten() for ratemap in ratemaps])
        tuning_curves = np.asarray([ratemap.flatten() for ratemap in tuning_curves]) # note .flatten() returns a deepcopy, np.ravel(a) returns a shallow copy

        print(f'tuning_curves.shape: {np.shape(tuning_curves)}')
        print(f'spkcount.shape: {np.shape(spkcount)}')
        
        nCells = spkcount.shape[0]
        nTimeBins = spkcount.shape[1]
        nFlatPositionBins = tuning_curves.shape[1]
        print(f'\nnCells: {nCells}, nTimeBins: {nTimeBins}, nFlatPositionBins: {nFlatPositionBins}') # nCells: 66, nTimeBins: 3529, nFlatPositionBins: 1066
        
        self.posterior = self._decoder(spkcount=spkcount, ratemaps=tuning_curves) # self.posterior.shape: (nFlatPositionBins, nTimeBins)
        print(f'self.posterior.shape: {np.shape(self.posterior)}') # self.posterior.shape: (1066, 3529)
        
        # Compute the decoded position from the posterior:
        _test_most_likely_position_flat_idxs = np.argmax(self.posterior, axis=0)
        # _test_most_likely_position_flat_idxs.shape # (3529,)
        _test_most_likely_positions = np.array([self._flat_all_positions_matrix[a_pos_idx] for a_pos_idx in _test_most_likely_position_flat_idxs])
        # _test_most_likely_positions.shape # (3529, 2)
        self.decoded_position = _test_most_likely_positions
        # _test_most_likely_position = np.argmax(self.posterior, axis=0)
        # print(f'_test_most_likely_position: {_test_most_likely_position}')        
        # self.decodedPos = gridcenter[:, _test_most_likely_position]
        
        # if plot:
        #     _, gs = Fig().draw(grid=(4, 4), size=(15, 6))
        #     axposx = plt.subplot(gs[0, :3])
        #     axposx.plot(self.actualbin[0, :], "k")
        #     axposx.set_ylabel("Actual position")

        #     axdecx = plt.subplot(gs[1, :3], sharex=axposx)
        #     axdecx.plot(self.decodedPos[0, :], "gray")
        #     axdecx.set_ylabel("Decoded position")

        #     axposy = plt.subplot(gs[2, :3], sharex=axposx)
        #     axposy.plot(self.actualpos_gridcntr[1, :], "k")
        #     axposy.set_ylabel("Actual position")

        #     axdecy = plt.subplot(gs[3, :3], sharex=axposx)
        #     axdecy.plot(
        #         # self.decodedPos,
        #         self.decodedPos[1, :],
        #         "gray",
        #     )
        #     axdecy.set_ylabel("Decoded position")

    def decode_events(self, binsize=0.02, slideby=0.005):
        """Decodes position within events which are set using self.events

        Parameters
        ----------
        binsize : float, seconds, optional
            size of binning withing each events, by default 0.02
        slideby : float, seconds optional
            sliding by this much, by default 0.005

        Returns
        -------
        [type]
            [description]
        """

        events = self.events
        ratemap_cell_ids = self.pf.cell_ids
        spks = Spikes(self._obj).get_cells(ids=ratemap_cell_ids)
        nCells = len(spks)
        print(f"Number of cells/ratemaps in pf2d: {nCells}")

        ratemaps = self.pf.ratemaps
        gridcenter = self.pf.gridcenter

        nbins, spkcount, time_bin_centers_list = epochs_spkcount(binsize, slideby, events, spks)

        # ---- linearize 2d ratemaps -------
        ratemaps = np.asarray([ratemap.flatten() for ratemap in ratemaps])

        posterior = self._decoder(spkcount=spkcount, ratemaps=ratemaps)
        decodedPos = gridcenter[:, np.argmax(posterior, axis=0)]

        # --- splitting concatenated time bins into separate arrays ------
        cum_nbins = np.cumsum(nbins)[:-1]
        self.posterior = np.hsplit(posterior, cum_nbins)
        self.decoded_position = np.hsplit(decodedPos, cum_nbins)

        return decodedPos, posterior

    def plot(self):

        # decodedPos = gaussian_filter1d(self.decodedPos, sigma=1, axis=1)
        decodedPos = self.decoded_position
        posterior = self.posterior
        decodingtime = self.decodingtime[1:]
        actualPos = self.actualPos
        speed = self.speed
        error = np.sqrt(np.sum((decodedPos - actualPos) ** 2, axis=0))

        plt.clf()
        fig = plt.figure(1, figsize=(10, 15))
        gs = gridspec.GridSpec(3, 6, figure=fig)
        fig.subplots_adjust(hspace=0.3)

        ax = fig.add_subplot(gs[0, :])
        # ax.pcolormesh(decodingtime, np.arange(npos), posterior, cmap="binary")
        ax.plot(decodingtime, actualPos[0, :], "#4FC3F7")
        ax.plot(decodingtime, decodedPos[0, :], "#F48FB1")
        ax.set_ylabel("X coord")
        ax.set_title("Bayesian position estimation (only pyr cells)")

        ax = fig.add_subplot(gs[1, :], sharex=ax)
        # ax.pcolormesh(decodingtime, np.arange(npos), posterior, cmap="binary")
        ax.plot(decodingtime, actualPos[1, :], "#4FC3F7")
        ax.plot(decodingtime, decodedPos[1, :], "#F48FB1")
        ax.set_ylabel("Y coord")
        ax.set_title("Bayesian position estimation (only pyr cells)")

        ax = fig.add_subplot(gs[2, :], sharex=ax)
        # ax.pcolormesh(decodingtime, np.arange(npos), posterior, cmap="binary")
        ax.plot(decodingtime, speed, "k")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("speed (cm/s)")
        # ax.set_title("Bayesian position estimation (only pyr cells)")
        ax.set_ylim([0, 120])
        ax.spines["right"].set_visible(True)

        axerror = ax.twinx()
        axerror.plot(decodingtime, gaussian_filter1d(error, sigma=1), "#05d69e")
        axerror.set_ylabel("error (cm)")
