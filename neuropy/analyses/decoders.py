import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm
import scipy.signal as sg
from typing import Union
from .. import core
from .. import plotting


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

    NOTE: when returning velocity the sign is flipped to match with position going from bottom to up

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


def wcorr(arr):
    """weighted correlation"""
    nx, ny = arr.shape[1], arr.shape[0]
    y_mat = np.tile(np.arange(ny)[:, np.newaxis], (1, nx))
    x_mat = np.tile(np.arange(nx), (ny, 1))
    arr_sum = np.nansum(arr)
    ey = np.nansum(arr * y_mat) / arr_sum
    ex = np.nansum(arr * x_mat) / arr_sum
    cov_xy = np.nansum(arr * (y_mat - ey) * (x_mat - ex)) / arr_sum
    cov_yy = np.nansum(arr * (y_mat - ey) ** 2) / arr_sum
    cov_xx = np.nansum(arr * (x_mat - ex) ** 2) / arr_sum

    return cov_xy / np.sqrt(cov_xx * cov_yy)


def jump_distance(posteriors, jump_stat="mean", norm=True):
    """Calculate jump distance for posterior matrices"""

    if jump_stat == "mean":
        f = np.mean
    elif jump_stat == "median":
        f = np.median
    elif jump_stat == "max":
        f = np.max
    else:
        raise ValueError("Invalid jump_stat. Valid values: mean, median, max")

    dx = 1 / posteriors[0].shape[0] if norm else 1
    jd = np.array([f(np.abs(np.diff(np.argmax(_, axis=0)))) for _ in posteriors])

    return jd * dx


def column_shift(arr, shifts=None):
    """Circular shift columns independently by a given amount"""

    assert arr.ndim == 2, "only 2d arrays accepted"

    if shifts is None:
        rng = np.random.default_rng()
        shifts = rng.integers(-arr.shape[0], arr.shape[0], arr.shape[1])

    assert arr.shape[1] == len(shifts)

    shifts = shifts % arr.shape[0]
    rows_indx, columns_indx = np.ogrid[: arr.shape[0], : arr.shape[1]]

    rows_indx = rows_indx - shifts[np.newaxis, :]

    return arr[rows_indx, columns_indx]


class Decode1d:
    def __init__(
        self,
        neurons: core.Neurons,
        ratemap: core.Ratemap,
        epochs: Union[core.Epoch, None] = None,
        bin_size=0.5,
        slideby=None,
        # score_method="wcorr",
        # radon_kw=dict(nlines=5000, decode_margin=15),
        # jump_distance=False,
        n_jobs=1,
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
            binning size to calculate spike counts, by default 0.5
        slideby : float, optional
            slide the binning window by this amount, by default None
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
        self.pos_bin_size = ratemap.x_binsize
        self.decoded_position = None
        self.epochs = epochs
        self.slideby = slideby
        # self.score = None
        # self.jump_distance
        # self.shuffle_score: Union[NDArray, None] = None
        # self.shuffle_jump_distance: Union[NDArray, None] = None
        # self.score_method = score_method
        # self.radon_kw = radon_kw
        self.n_jobs = n_jobs

        # Only available when using 'radon_transform'
        # self.velocity = None
        # self.intercept = None

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
            # ignore neurons/indx which have zero frate at this location to
            # avoid having frate product zero
            valid_indx = ratemaps[:, i] > 0
            if np.any(valid_indx):
                frate = (ratemaps[valid_indx, i, np.newaxis]) ** spkcount[valid_indx, :]
                exp_frate = np.exp(-tau * np.sum(ratemaps[valid_indx, i]))
                prob[i, :] = np.prod(frate, axis=0) * exp_frate

        old_settings = np.seterr(all="ignore")
        prob /= np.sum(prob, axis=0, keepdims=True)
        np.seterr(**old_settings)

        return prob

    def _estimate(self):
        """Estimates position within each bin"""

        tuning_curves = self.ratemap.tuning_curves
        bincntr = self.ratemap.x_coords()

        if self.epochs is not None:
            spkcount, nbins = self.neurons.get_spikes_in_epochs(
                self.epochs, self.bin_size, self.slideby
            )
            posterior = self._decoder(np.hstack(spkcount), tuning_curves)
            decodedPos = bincntr[np.argmax(posterior, axis=0)]
            cum_nbins = np.cumsum(nbins)[:-1]

            self.decoded_position = np.hsplit(decodedPos, cum_nbins)
            self.posterior = np.hsplit(posterior, cum_nbins)
            self.spkcount = spkcount
            self.nbins_epochs = nbins
            # score_results = self._score_posterior(self.posterior)
            # self.score = score_results[0]
            # if score_results.shape[0] == 3:
            #     self.velocity, self.intercept = score_results[1:, :]

        else:
            spkcount = self.neurons.get_binned_spiketrains(
                bin_size=self.bin_size
            ).spike_counts

            self.posterior = self._decoder(spkcount, tuning_curves)
            # self.decoded_position = bincntr[np.argmax(self.posterior, axis=0)]
            self.decoded_position = bincntr[np.argmax(self.posterior, axis=0)]
            self.score = None

    def _get_jd(self, posteriors, jump_stat="mean"):
        """Calculate jump distance for posterior matrices"""

        if jump_stat == "mean":
            f = np.mean
        elif jump_stat == "median":
            f = np.median
        elif jump_stat == "max":
            f = np.max
        else:
            raise ValueError("Invalid jump_stat. Valid values: mean, median, max")

        dx = 1 / posteriors[0].shape[0]
        jd = np.array([f(np.abs(np.diff(np.argmax(_, axis=0)))) for _ in posteriors])

        return jd * dx

    def get_trajectory_length(self, max_jump=40, min_distance=None, posteriors=None):
        if posteriors is None:
            assert self.posterior is not None, "No posteriors found"
            posteriors = self.posterior

        for p in posteriors:
            max_loc = np.argmax(p, axis=0)
            dist = np.abs(np.diff(max_loc))
            dist_logical = np.where(dist < 40, 1, 0)
            pad_dist = np.pad(dist_logical, (1, 1), "constant", constant_values=(0, 0))
            peaks_dict = sg.find_peaks(pad_dist, height=1, width=3, plateau_size=3)[1]
            lengths = peaks_dict["plateau_sizes"] + 1

            traj_length = 0
            traj_dist = 0

        return traj_length, traj_dist

    def get_wcorr(self, jump_stat=None, posteriors=None):
        if posteriors is None:
            assert self.posterior is not None, "No posteriors found"
            posteriors = self.posterior

        scores = Parallel(n_jobs=self.n_jobs)(delayed(wcorr)(_) for _ in posteriors)
        scores = np.array(scores)

        if jump_stat is not None:
            return scores, self._get_jd(posteriors, jump_stat)
        else:
            return scores

    def get_radon_transform(
        self, nlines=5000, margin=16, jump_stat=None, posteriors=None
    ):
        if posteriors is None:
            assert self.posterior is not None, "No posteriors found"
            posteriors = self.posterior

        neighbours = int(margin / self.ratemap.x_binsize)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(radon_transform)(
                epoch,
                nlines=nlines,
                dt=self.bin_size,
                dx=self.pos_bin_size,
                neighbours=neighbours,
            )
            for epoch in posteriors
        )
        score, velocity, intercept = np.asarray(results).T

        if jump_stat is not None:
            return score, velocity, intercept, self._get_jd(posteriors, jump_stat)
        else:
            return score, velocity, intercept

    def get_shuffled_wcorr(self, n_iter, method="neuron_id", **kwargs):
        return self._shuffler(self.get_wcorr, n_iter=n_iter, method=method, **kwargs)

    def get_shuffled_radon_transform(self, n_iter, method="neuron_id", **kwargs):
        return self._shuffler(
            self.get_radon_transform, n_iter=n_iter, method=method, **kwargs
        )

    def _shuffler(self, func, n_iter, method, **kwargs):
        assert callable(func), "scoring function is not callable"

        cum_nbins = np.cumsum(self.nbins_epochs)[:-1]
        stacked_posterior = np.hstack(self.posterior)
        spkcount = np.hstack(self.spkcount)

        score = []
        for i in tqdm(range(n_iter)):
            if method == "neuron_id":
                shuffled_tc = self.ratemap.tuning_curves.copy()
                np.random.default_rng().shuffle(shuffled_tc)
                shuffle_posteriors = np.hsplit(
                    self._decoder(spkcount, shuffled_tc), cum_nbins
                )

            if method == "column_cycle":
                shuffle_posteriors = np.hsplit(
                    column_shift(stacked_posterior), cum_nbins
                )

            score.append(func(posteriors=shuffle_posteriors, **kwargs))

        return np.array(score)

    @property
    def p_value(self):
        """Monte Carlo p-value"""
        if self.score_method == "radon_transorm":
            shuff_score = self.shuffle_score
            n_iter = shuff_score.shape[0]
            diff_score = shuff_score - self.score[np.newaxis, :]
            chance = np.where(diff_score > 0, 1, 0).sum(axis=0)
            return (chance + 1) / (n_iter + 1)
        if self.score_method == "wcorr":
            pass

    @property
    def percentile_score(self):
        return np.array(
            [
                stats.percentileofscore(
                    self.shuffle_score[:, i], self.score[i], kind="strict"
                )
                for i in range(self.epochs.n_epochs)
            ]
        )

    @property
    def sequence_score(self):
        if self.score_method == "wcorr":
            abs_score = np.abs(self.score)
            abs_shuffle_score = np.abs(self.shuffle_score)
            mean_shuffle_score = abs_shuffle_score.mean(axis=0)
            std_shuffle_score = abs_shuffle_score.std(axis=0)
            return (abs_score - mean_shuffle_score) / std_shuffle_score

    def plot_summary(self, **kwargs):
        if self.score_method == "radon_transform":
            self._plot_radon_transform(**kwargs)
        if self.score_method == "wcorr":
            self._plot_wcorr(**kwargs)

    def _plot_wcorr(self, prob_cmap="hot", count_cmap="binary"):
        n_posteriors = len(self.posterior)
        posterior_ind = np.random.default_rng().integers(0, n_posteriors, 5)
        arrs = [self.posterior[i] for i in posterior_ind]

        _, axs = plt.subplots(3, 5, sharey="row", sharex="col", figsize=[11, 8])

        zsc_tuning = stats.zscore(self.ratemap.tuning_curves, axis=1)
        sort_ind = np.argsort(np.argmax(zsc_tuning, axis=1))
        n_neurons = self.neurons.n_neurons

        for i, arr in enumerate(arrs):
            t_start = self.epochs[posterior_ind[i]].flatten()[0]
            score = self.score[posterior_ind[i]]

            arr = np.apply_along_axis(
                np.convolve, axis=0, arr=arr, v=np.ones(2 * 2 + 1)
            )
            t = np.arange(arr.shape[1]) * self.bin_size + t_start
            pos = np.arange(arr.shape[0]) * 2

            axs[0, i].pcolormesh(t, pos, arr, cmap=prob_cmap)
            axs[0, i].set_ylim([pos.min(), pos.max()])
            axs[0, i].set_title(f"#{posterior_ind[i]},\ns={np.round(score,2)}")

            axs[1, i].pcolormesh(
                t,
                np.arange(n_neurons),
                self.spkcount[posterior_ind[i]],
                cmap=count_cmap,
            )
            plotting.plot_raster(
                self.neurons[sort_ind].time_slice(t_start=t[0], t_stop=t[-1]),
                ax=axs[2, i],
                color="k",
            )
            if i == 0:
                axs[0, i].set_ylabel("Position (cm)")
                axs[1, i].set_ylabel("Neurons")
            if i > 0:
                axs[2, i].set_ylabel("")

    def _plot_radon_transform(self, prob_cmap="hot", count_cmap="binary", lc="#00E676"):
        n_posteriors = len(self.posterior)
        posterior_ind = np.random.default_rng().integers(0, n_posteriors, 5)
        arrs = [self.posterior[i] for i in posterior_ind]

        fig, axs = plt.subplots(4, 5, sharey="row", sharex="col", figsize=[11, 8])

        zsc_tuning = stats.zscore(self.ratemap.tuning_curves, axis=1)
        sort_ind = np.argsort(np.argmax(zsc_tuning, axis=1))
        n_neurons = self.neurons.n_neurons
        neighbours = int(self.decode_margin / self.ratemap.x_binsize)

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
