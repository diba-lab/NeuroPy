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

from .. import core
from .. import plotting
from ..utils import mathutil


def radon_transform(arr, nlines=10000, dt=1, dx=1, neighbours=1):
    """Line fitting algorithm primarily used in decoding algorithm, a variant of radon transform, algorithm based on Kloosterman et al. 2012

    Parameters
    ----------
    arr : 2d array
        time axis is represented by columns, position axis is represented by rows
    dt : float
        time binsize in seconds, only used for velocity/intercept calculation
    dx : float
        position binsize in cm, only used for velocity/intercept calculation
    neighbours : int,
        probability in each bin is replaced by sum of itself and these many 'neighbours' column wise, default 1 neighbour

    NOTE: when returning velcoity the sign is flipped to match with position going from bottom to up

    Returns
    -------
    score:
        sum of values (posterior) under the best fit line
    velocity:
        speed of replay in cm/s
    intercept:
        intercept of best fit line

    References
    ----------
    1) Kloosterman et al. 2012
    """
    t = np.arange(arr.shape[1])
    nt = len(t)
    tmid = (nt + 1) / 2 - 1

    pos = np.arange(arr.shape[0])
    npos = len(pos)
    pmid = (npos + 1) / 2 - 1

    # using convolution to sum neighbours
    arr = np.apply_along_axis(
        np.convolve, axis=0, arr=arr, v=np.ones(2 * neighbours + 1)
    )

    # exclude stationary events by choosing phi little below 90 degree
    # NOTE: angle of line is given by (90-phi), refer Kloosterman 2012
    phi = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=nlines)
    diag_len = np.sqrt((nt - 1) ** 2 + (npos - 1) ** 2)
    rho = np.random.uniform(low=-diag_len / 2, high=diag_len / 2, size=nlines)

    rho_mat = np.tile(rho, (nt, 1)).T
    phi_mat = np.tile(phi, (nt, 1)).T
    t_mat = np.tile(t, (nlines, 1))
    posterior = np.zeros((nlines, nt))

    y_line = ((rho_mat - (t_mat - tmid) * np.cos(phi_mat)) / np.sin(phi_mat)) + pmid
    y_line = np.rint(y_line).astype("int")

    # if line falls outside of array in a given bin, replace that with median posterior value of that bin across all positions
    t_out = np.where((y_line < 0) | (y_line > npos - 1))
    t_in = np.where((y_line >= 0) & (y_line <= npos - 1))
    posterior[t_out] = np.median(arr[:, t_out[1]], axis=0)
    posterior[t_in] = arr[y_line[t_in], t_in[1]]

    old_settings = np.seterr(all="ignore")
    posterior_mean = np.nanmean(posterior, axis=1)

    best_line = np.argmax(posterior_mean)
    score = posterior_mean[best_line]
    best_phi, best_rho = phi[best_line], rho[best_line]
    time_mid, pos_mid = nt * dt / 2, npos * dx / 2

    velocity = dx / (dt * np.tan(best_phi))
    intercept = (
        (dx * time_mid) / (dt * np.tan(best_phi))
        + (best_rho / np.sin(best_phi)) * dx
        + pos_mid
    )
    np.seterr(**old_settings)

    return score, -velocity, intercept


def epochs_spkcount(
    neurons: core.Neurons, epochs: core.Epoch, bin_size=0.01, slideby=None
):
    # ---- Binning events and calculating spike counts --------
    spkcount = []
    nbins = np.zeros(epochs.n_epochs, dtype="int")

    if slideby is None:
        slideby = bin_size
    # ----- little faster but requires epochs to be non-overlapping ------
    # bins_epochs = []
    # for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
    #     bins = np.arange(epoch.start, epoch.stop, bin_size)
    #     nbins[i] = len(bins) - 1
    #     bins_epochs.extend(bins)
    # spkcount = np.asarray(
    #     [np.histogram(_, bins=bins_epochs)[0] for _ in neurons.spiketrains]
    # )

    # deleting unwanted columns that represent time between events
    # cumsum_nbins = np.cumsum(nbins)
    # del_columns = cumsum_nbins[:-1] + np.arange(len(cumsum_nbins) - 1)
    # spkcount = np.delete(spkcount, del_columns.astype(int), axis=1)

    for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
        # first dividing in 1ms
        bins = np.arange(epoch.start, epoch.stop, 0.001)
        spkcount_ = np.asarray(
            [np.histogram(_, bins=bins)[0] for _ in neurons.spiketrains]
        )

        # if signficant portion at end of epoch is not included then append zeros
        # if (frac := epoch.duration / bin_size % 1) > 0.7:
        #     extra_columns = int(100 * (1 - frac))
        #     spkcount_ = np.hstack(
        #         (spkcount_, np.zeros((neurons.n_neurons, extra_columns)))
        #     )

        slide_view = np.lib.stride_tricks.sliding_window_view(
            spkcount_, int(bin_size * 1000), axis=1
        )[:, :: int(slideby * 1000), :].sum(axis=2)

        nbins[i] = slide_view.shape[1]
        spkcount.append(slide_view)

    return spkcount, nbins


class Decode1d:
    n_jobs = 8

    def __init__(
        self,
        neurons: core.Neurons,
        ratemap: core.Ratemap,
        epochs: core.Epoch = None,
        bin_size=0.5,
        slideby=None,
    ):
        self.ratemap = ratemap
        self._events = None
        self.posterior = None
        self.neurons = neurons
        self.bin_size = bin_size
        self.pos_bin_size = ratemap.xbin_size
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
            NOTE: multiplying with (1/n_positions) is not necessary as it gets cancelled out eventually
        ===========================
        """
        tau = self.bin_size
        n_positions, n_time_bins = ratemaps.shape[1], spkcount.shape[1]

        prob = np.zeros((n_positions, n_time_bins))
        for i in range(n_positions):
            frate = (tau * ratemaps[:, i, np.newaxis]) ** spkcount
            spkcount_factorial = factorial(spkcount)
            exp_frate = np.exp(-tau * ratemaps[:, i, np.newaxis])
            prob[i, :] = np.prod((frate / spkcount_factorial) * exp_frate, axis=0)

        old_settings = np.seterr(all="ignore")
        prob /= np.sum(prob, axis=0, keepdims=True)
        np.seterr(**old_settings)

        return prob

    def _estimate(self):

        """Estimates position within each"""

        tuning_curves = self.ratemap.tuning_curves
        bincntr = self.ratemap.xbin_centers

        if self.epochs is not None:

            spkcount, nbins = epochs_spkcount(
                self.neurons, self.epochs, self.bin_size, self.slideby
            )
            posterior = self._decoder(np.hstack(spkcount), tuning_curves)
            decodedPos = bincntr[np.argmax(posterior, axis=0)]
            cum_nbins = np.cumsum(nbins)[:-1]

            self.decoded_position = np.hsplit(decodedPos, cum_nbins)
            self.posterior = np.hsplit(posterior, cum_nbins)
            self.spkcount = spkcount
            self.nbins_epochs = nbins
            self.score, self.velocity, self.intercept = self.score_posterior(
                self.posterior
            )

        else:
            spkcount = self.neurons.get_binned_spiketrains(
                bin_size=self.bin_size
            ).spike_counts

            self.posterior = self._decoder(spkcount, tuning_curves)
            self.decoded_position = bincntr[np.argmax(self.posterior, axis=0)]
            self.score = None

    def calculate_shuffle_score(self, n_iter=100, method="column"):
        """Shuffling and decoding epochs"""

        # print(f"Using {kind} shuffle")

        if method == "neuron_id":
            score = []
            for i in tqdm(range(n_iter)):
                tuning_curves = self.ratemap.tuning_curves.copy()
                np.random.shuffle(tuning_curves)
                post_ = self._decoder(np.hstack(self.spkcount), tuning_curves)
                cum_nbins = np.cumsum(self.nbins_epochs)[:-1]
                score.append(self.score_posterior(np.hsplit(post_, cum_nbins))[0])
            score = np.asarray(score)
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
            delayed(radon_transform)(
                epoch, nlines=5000, dt=self.bin_size, dx=self.pos_bin_size, neighbours=2
            )
            for epoch in p
        )
        score, velocity, intercept = np.asarray(results).T

        return score, velocity, intercept

    @property
    def p_value(self):
        shuff_score = self.shuffle_score
        n_iter = shuff_score.shape[0]
        diff_score = shuff_score - np.tile(self.score, (n_iter, 1))
        chance = np.where(diff_score > 0, 1, 0).sum(axis=0)
        return (chance + 1) / (n_iter + 1)

    def plot_in_bokeh(self):
        pass

    def plot_summary(self):
        n_posteriors = len(self.posterior)
        posterior_ind = np.random.default_rng().integers(0, n_posteriors, 5)
        arrs = [self.posterior[i] for i in posterior_ind]

        _, axs = plt.subplots(2, 5, sharey="row", sharex="col")
        zsc_tuning = stats.zscore(self.ratemap.tuning_curves, axis=1)
        sort_ind = np.argsort(np.argmax(zsc_tuning, axis=1))

        for i, arr in enumerate(arrs):

            t_start = self.epochs[posterior_ind[i]].flatten()[0]
            velocity, intercept = (
                self.velocity[posterior_ind[i]],
                self.intercept[posterior_ind[i]],
            )
            arr = np.apply_along_axis(
                np.convolve, axis=0, arr=arr, v=np.ones(2 * 2 + 1)
            )
            t = np.arange(arr.shape[1]) * self.bin_size + t_start
            pos = np.arange(arr.shape[0]) * 2

            axs[0, i].pcolormesh(t, pos, arr, cmap="hot")
            axs[0, i].plot(t, velocity * (t - t_start) + intercept, color="w")
            axs[0, i].set_ylim([pos.min(), pos.max()])

            plotting.plot_raster(
                self.neurons[sort_ind].time_slice(t_start=t[0], t_stop=t[-1]),
                ax=axs[1, i],
                color="k",
            )

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
    def __init__(self):

        assert isinstance(pf2d_obj, PF2d)
        self._obj = pf2d_obj._obj
        self.pf2d = pf2d_obj

    def estimate_behavior(self, binsize=0.25, smooth=1, plot=True):

        ratemap_cell_ids = self.pf2d.cell_ids
        spks = Spikes(self._obj).get_cells(ids=ratemap_cell_ids)
        ratemaps = self.pf2d.ratemaps
        speed = self.pf2d.speed
        xgrid = self.pf2d.xgrid
        ygrid = self.pf2d.ygrid
        gridbin = self.pf2d.gridbin
        gridcenter = self.pf2d.gridcenter

        # --- average position in each time bin and which gridbin it belongs to ----
        t = self.pf2d.t
        x = self.pf2d.x
        y = self.pf2d.y
        period = self.pf2d.period
        tmz = np.arange(period[0], period[1], binsize)
        actualposx = stats.binned_statistic(t, values=x, bins=tmz)[0]
        actualposy = stats.binned_statistic(t, values=y, bins=tmz)[0]
        actualpos = np.vstack((actualposx, actualposy))

        actualbin_x = xgrid[np.digitize(actualposx, bins=xgrid) - 1] + gridbin / 2
        actualbin_y = ygrid[np.digitize(actualposy, bins=ygrid) - 1] + gridbin / 2
        self.actualbin = np.vstack((actualbin_x, actualbin_y))

        # ---- spike counts and linearize 2d ratemaps -------
        spkcount = np.asarray([np.histogram(cell, bins=tmz)[0] for cell in spks])
        spkcount = gaussian_filter1d(spkcount, sigma=3, axis=1)
        ratemaps = np.asarray([ratemap.flatten() for ratemap in ratemaps])

        self.posterior = self._decoder(spkcount=spkcount, ratemaps=ratemaps)
        self.decodedPos = gridcenter[:, np.argmax(self.posterior, axis=0)]
        self.decodingtime = tmz
        self.actualpos = actualpos

        if plot:
            _, gs = Fig().draw(grid=(4, 4), size=(15, 6))
            axposx = plt.subplot(gs[0, :3])
            axposx.plot(self.actualbin[0, :], "k")
            axposx.set_ylabel("Actual position")

            axdecx = plt.subplot(gs[1, :3], sharex=axposx)
            axdecx.plot(self.decodedPos[0, :], "gray")
            axdecx.set_ylabel("Decoded position")

            axposy = plt.subplot(gs[2, :3], sharex=axposx)
            axposy.plot(self.actualpos_gridcntr[1, :], "k")
            axposy.set_ylabel("Actual position")

            axdecy = plt.subplot(gs[3, :3], sharex=axposx)
            axdecy.plot(
                # self.decodedPos,
                self.decodedPos[1, :],
                "gray",
            )
            axdecy.set_ylabel("Decoded position")

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
        ratemap_cell_ids = self.pf2d.cell_ids
        spks = Spikes(self._obj).get_cells(ids=ratemap_cell_ids)
        nCells = len(spks)
        print(f"Number of cells/ratemaps in pf2d: {nCells}")

        ratemaps = self.pf2d.ratemaps
        gridcenter = self.pf2d.gridcenter

        nbins, spkcount = epochs_spkcount(binsize, slideby, events, spks)

        # ---- linearize 2d ratemaps -------
        ratemaps = np.asarray([ratemap.flatten() for ratemap in ratemaps])

        posterior = self._decoder(spkcount=spkcount, ratemaps=ratemaps)
        decodedPos = gridcenter[:, np.argmax(posterior, axis=0)]

        # --- splitting concatenated time bins into separate arrays ------
        cum_nbins = np.cumsum(nbins)[:-1]
        self.posterior = np.hsplit(posterior, cum_nbins)
        self.decodedPos = np.hsplit(decodedPos, cum_nbins)

        return decodedPos, posterior

    def plot(self):

        # decodedPos = gaussian_filter1d(self.decodedPos, sigma=1, axis=1)
        decodedPos = self.decodedPos
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
