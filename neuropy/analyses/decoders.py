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
        np.convolve, axis=0, arr=arr, v=np.ones(2 * neighbours + 1), mode="same"
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

    # converts to real world values
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
        decode_margin=15,
        nlines=5000,
    ):
        """1D decoding using ratemaps

        Parameters
        ----------
        neurons : core.Neurons
            neurons object containing spiketrains
        ratemap : core.Ratemap
            ratemap containing tuning curves
        epochs : core.Epoch, optional
            if provided then decode within these epochs only,if None then uses entire duration of neurons, by default None
        bin_size : float, optional
            bining size to calculate spike counts, by default 0.5
        slideby : float, optional
            slide the bining window by this amount, by default None
        decode_margin : int, optional
            in cm, likelihood of position is within this distance, used only if epochs are provided, , by default 15
        nlines : int, optional
            number of lines to fit, used only if replay trajectories are decoded within epochs, by default 5000
        """
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
        self.decode_margin = decode_margin
        self.nlines = nlines

        self._estimate()

    def _decoder(self, spkcount, ratemaps):
        """
        ===========================
        Probability is calculated using this formula
        prob = ((frate)^nspike) * exp(-tau * frate)
        where,
            tau = binsize

        ===========================
        """
        tau = self.bin_size
        n_positions, n_time_bins = ratemaps.shape[1], spkcount.shape[1]

        prob = np.zeros((n_positions, n_time_bins))
        for i in range(n_positions):
            frate = (ratemaps[:, i, np.newaxis]) ** spkcount
            exp_frate = np.exp(-tau * np.sum(ratemaps[:, i]))
            prob[i, :] = np.prod(frate, axis=0) * exp_frate

        old_settings = np.seterr(all="ignore")
        prob /= np.sum(prob, axis=0, keepdims=True)
        np.seterr(**old_settings)

        return prob

    def _estimate(self):

        """Estimates position within each bin"""

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

    def calculate_shuffle_score(self, n_iter=100, method="neuron_id"):
        """Shuffling and decoding epochs"""

        # print(f"Using {kind} shuffle")

        if method == "neuron_id":
            score = []
            for i in tqdm(range(n_iter)):
                shuffled_tc = self.ratemap.tuning_curves.copy()
                np.random.default_rng().shuffle(shuffled_tc)
                post_ = self._decoder(np.hstack(self.spkcount), shuffled_tc)
                cum_nbins = np.cumsum(self.nbins_epochs)[:-1]
                score.append(self.score_posterior(np.hsplit(post_, cum_nbins))[0])
            score = np.asarray(score)
        if method == "time_bin":

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
        if method == "position_bin":
            pass

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
        neighbours = int(self.decode_margin / self.ratemap.xbin_size)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(radon_transform)(
                epoch,
                nlines=self.nlines,
                dt=self.bin_size,
                dx=self.pos_bin_size,
                neighbours=neighbours,
            )
            for epoch in p
        )
        score, velocity, intercept = np.asarray(results).T

        return score, velocity, intercept

    def weighted_correlation(self, pmats):
        r = []
        for pmat in pmats:
            nt, nx = pmat.shape[1], pmat.shape[0]
            x_mat = np.tile(np.arange(nx)[:, np.newaxis], (1, nt))
            t_mat = np.tile(np.arange(nt), (nx, 1))
            pmat_sum = np.nansum(pmat)
            ex = np.nansum(pmat * x_mat) / pmat_sum
            et = np.nansum(pmat * t_mat) / pmat_sum
            cov_xt = np.nansum(pmat * (x_mat - ex) * (t_mat - et)) / pmat_sum
            cov_xx = np.nansum(pmat * (x_mat - ex) ** 2) / pmat_sum
            cov_tt = np.nansum(pmat * (t_mat - et) ** 2) / pmat_sum

            r.append(cov_xt / np.sqrt(cov_tt * cov_xx))
        return np.asarray(r)

    @property
    def p_value(self):
        """Monte Carlo p-value"""
        shuff_score = self.shuffle_score
        n_iter = shuff_score.shape[0]
        diff_score = shuff_score - self.score[np.newaxis, :]
        chance = np.where(diff_score > 0, 1, 0).sum(axis=0)
        return (chance + 1) / (n_iter + 1)

    def plot_in_bokeh(self):
        pass

    def plot_summary(self, prob_cmap="hot", count_cmap="binary", lc="#00E676"):
        n_posteriors = len(self.posterior)
        posterior_ind = np.random.default_rng().integers(0, n_posteriors, 5)
        arrs = [self.posterior[i] for i in posterior_ind]

        fig, axs = plt.subplots(4, 5, sharey="row", sharex="col", figsize=[11, 8])

        zsc_tuning = stats.zscore(self.ratemap.tuning_curves, axis=1)
        sort_ind = np.argsort(np.argmax(zsc_tuning, axis=1))
        n_neurons = self.neurons.n_neurons
        neighbours = int(self.decode_margin / self.ratemap.xbin_size)

        for i, arr in enumerate(arrs):

            t_start = self.epochs[posterior_ind[i]].flatten()[0]
            score = self.score[posterior_ind[i]]
            velocity, intercept = (
                self.velocity[posterior_ind[i]],
                self.intercept[posterior_ind[i]],
            )
            arr = np.apply_along_axis(
                np.convolve, axis=0, arr=arr, v=np.ones(2 * 2 + 1)
            )
            t = np.arange(arr.shape[1]) * self.bin_size + t_start
            pos = np.arange(arr.shape[0]) * 2

            axs[0, i].pcolormesh(t, pos, arr, cmap=prob_cmap)
            axs[0, i].plot(t, velocity * (t - t_start) + intercept, color=lc, lw=2)
            axs[0, i].set_ylim([pos.min(), pos.max()])

            arr_margin = np.apply_along_axis(
                np.convolve, axis=0, arr=arr, v=np.ones(2 * neighbours + 1), mode="same"
            )
            axs[0, i].set_title(
                f"#{posterior_ind[i]},\ns={np.round(score,2)}\nv={np.round(velocity,2)} cm/s"
            )

            axs[1, i].pcolormesh(t, pos, arr_margin, cmap=prob_cmap)
            axs[1, i].plot(t, velocity * (t - t_start) + intercept, color=lc, lw=2)
            axs[1, i].set_ylim([pos.min(), pos.max()])

            axs[2, i].pcolormesh(
                t,
                np.arange(n_neurons),
                self.spkcount[posterior_ind[i]],
                cmap=count_cmap,
            )
            plotting.plot_raster(
                self.neurons[sort_ind].time_slice(t_start=t[0], t_stop=t[-1]),
                ax=axs[3, i],
                color="k",
            )
            if i == 0:
                axs[0, i].set_ylabel("Position (cm)")
                axs[1, i].set_ylabel("Position (cm)")
                axs[2, i].set_ylabel("Neurons")
            if i > 0:
                axs[3, i].set_ylabel("")

        # fig.suptitle(
        #     f"Summary of decoding decode margin={self.decode_margin}\nBin size={self.bin_size}\nnlines={self.nlines}"
        # )


class Decode2d:
    pass
