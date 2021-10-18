import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA, FastICA

from ..utils.mathutil import getICA_Assembly, parcorr_mult
from .. import core
from ..plotting import Fig
import pingouin as pg


class ExplainedVariance(core.DataWriter):
    colors = {"ev": "#4a4a4a", "rev": "#05d69e"}  # colors of each curve

    def __init__(
        self,
        neurons: core.Neurons,
        template,
        matching,
        control,
        bin_size=0.250,
        window: int = 900,
        slideby: int = None,
        cross_shanks=True,
        ignore_epochs: core.Epoch = None,
    ):
        super().__init__()
        self.neurons = neurons
        if cross_shanks:
            assert self.neurons.shank_ids is not None, "neurons should have shank_ids"

        self.template = template
        self.matching = matching
        self.control = control
        self.bin_size = bin_size
        self.window = window

        if slideby is None:
            self.slideby = window
        else:
            self.slideby = slideby
        self.cross_shanks = cross_shanks
        self.ignore_epochs = ignore_epochs
        self._calculate()

    def _calculate(self):

        template_corr = (
            self.neurons.time_slice(self.template[0], self.template[1])
            .get_binned_spiketrains(bin_size=self.bin_size)
            .get_pairwise_corr(cross_shanks=self.cross_shanks)
        )

        matching = np.arange(self.matching[0], self.matching[1])
        matching_windows = np.lib.stride_tricks.sliding_window_view(
            matching, self.window
        )[:: self.slideby, [0, -1]]

        n_matching_windows = matching_windows.shape[0]

        matching_paircorr = []
        for window in matching_windows:
            matching_paircorr.append(
                self.neurons.time_slice(window[0], window[1])
                .get_binned_spiketrains(self.bin_size)
                .get_pairwise_corr(cross_shanks=self.cross_shanks)
            )

        control = np.arange(self.control[0], self.control[1])
        control_windows = np.lib.stride_tricks.sliding_window_view(
            control, self.window
        )[:: self.slideby, [0, -1]]
        n_control_windows = control_windows.shape[0]
        control_paircorr = []
        for window in control_windows:
            control_paircorr.append(
                self.neurons.time_slice(window[0], window[1])
                .get_binned_spiketrains(self.bin_size)
                .get_pairwise_corr(cross_shanks=self.cross_shanks)
            )

        partial_corr = np.zeros((n_control_windows, n_matching_windows))
        rev_partial_corr = np.zeros((n_control_windows, n_matching_windows))
        for m_i, m_pairs in enumerate(matching_paircorr):
            for c_i, c_pairs in enumerate(control_paircorr):
                df = pd.DataFrame({"t": template_corr, "m": m_pairs, "c": c_pairs})
                partial_corr[c_i, m_i] = pg.partial_corr(df, x="t", y="m", covar="c").r
                rev_partial_corr[c_i, m_i] = pg.partial_corr(
                    df, x="t", y="c", covar="m"
                ).r

        self.ev = np.nanmean(partial_corr ** 2, axis=0)
        self.rev = np.nanmean(rev_partial_corr ** 2, axis=0)
        self.ev_std = np.nanstd(partial_corr ** 2, axis=0)
        self.rev_std = np.nanstd(rev_partial_corr ** 2, axis=0)
        self.partial_corr = partial_corr
        self.rev_partial_corr = rev_partial_corr
        self.n_pairs = len(template_corr)
        self.matching_time = np.mean(matching_windows, axis=1)
        self.control_time = np.mean(control_windows, axis=1)

    def compute_shuffle(
        self,
        template,
        match,
        binSize=0.250,
        window=900,
        slideby=None,
        cross_shanks=True,
        n_iter=10,
    ):
        """Calucate explained variance (EV) and reverse EV

        Parameters
        ----------
        template : list
            template period
        match : list
            match period whose similarity is calculated to template
        control : list
            control period, correlations in this period will be accounted for
        binSize : float,
            bin size within each window, defaults 0.250 seconds
        window : int,
            size of window in which ev is calculated, defaults 900 seconds
        slideby : int,
            calculate EV by sliding window, seconds

        References:
        1) Kudrimoti 1999
        2) Tastsuno et al. 2007
        """

        spikes = Spikes(self._obj)
        if slideby is None:
            slideby = window

        # ----- choosing cells ----------------
        spks = spikes.times
        stability = spikes.stability.info
        stable_cells = np.where(stability.stable == 1)[0]
        pyr_id = spikes.pyrid
        stable_pyr = np.intersect1d(pyr_id, stable_cells, assume_unique=True)
        print(f"Calculating EV for {len(stable_pyr)} stable cells")
        spks = [spks[_] for _ in stable_pyr]

        # ------- windowing the time periods ----------
        nbins_window = int(window / binSize)
        nbins_slide = int(slideby / binSize)

        # ---- function to calculate correlation in each window ---------
        def cal_corr(spikes, period, windowing=True):
            bin_period = np.arange(period[0], period[1], binSize)
            spkcnt = np.array([np.histogram(x, bins=bin_period)[0] for x in spikes])

            if windowing:
                t = np.arange(period[0], period[1] - window, slideby) + window / 2
                nwindow = len(t)

                window_spkcnt = [
                    spkcnt[:, i : i + nbins_window]
                    for i in range(0, int(nwindow) * nbins_slide, nbins_slide)
                ]

                # if nwindow % 1 > 0.3:
                #     window_spkcnt.append(spkcnt[:, int(nwindow) * nbins_window :])
                #     t = np.append(t, t[-1] + round(nwindow % 1, 3) / 2)

                corr = [
                    np.corrcoef(window_spkcnt[x]) for x in range(len(window_spkcnt))
                ]

            else:
                corr = np.corrcoef(spkcnt)
                t = None

            return corr, t

        # ---- correlation for each time period -----------
        template_corr, _ = cal_corr(spks, period=template, windowing=False)
        match_corr, self.t_match = cal_corr(spks, period=match)

        # ----- indices for cross shanks correlation -------
        shnkId = np.asarray(spikes.info.shank)
        shnkId = shnkId[stable_pyr]
        assert len(shnkId) == len(spks)

        selected_pairs = np.tril_indices(len(spks), k=-1)
        if cross_shanks:
            selected_pairs = np.nonzero(
                np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1))
            )

        template_corr = template_corr[selected_pairs]
        match_corr = [match_corr[x][selected_pairs] for x in range(len(match_corr))]

        ev_all, rev_all = [], []
        for i in range(n_iter):
            # pair_id = np.arange(len(template_corr))
            # np.random.shuffle(pair_id)
            # shuff_match = [window[pair_id] for window in match_corr]

            spks_shuff = random.sample(spks, len(spks))
            shuff_corr, _ = cal_corr(spks_shuff, period=match)
            shuff_match = [
                shuff_corr[x][selected_pairs] for x in range(len(shuff_corr))
            ]

            ev, rev = [], []
            for control, match_ in zip(shuff_match, match_corr):
                df = pd.DataFrame(
                    {"control": control, "template": template_corr, "match": match_}
                )
                ev_ = pg.partial_corr(data=df, x="template", y="match", covar="control")
                rev_ = pg.partial_corr(
                    data=df, x="template", y="control", covar="match"
                )
                ev.append(ev_.r2)
                rev.append(rev_.r2)

            ev_all.append(ev)
            rev_all.append(rev)

        ev_all = np.asarray(ev_all)
        rev_all = np.asarray(rev_all)

        self.ev = ev_all
        self.rev = rev_all
        self.npairs = template_corr.shape[0]

    def plot(self, ax=None, t_start=0, legend=True):

        if ax is None:
            fig, ax = plt.subplots()
        # ---- plot rev first ---------
        ax.fill_between(
            (self.matching_time - t_start) / 3600,
            self.rev - self.rev_std,
            self.rev + self.rev_std,
            color=self.colors["rev"],
            zorder=1,
            alpha=0.5,
            label="REV",
        )
        ax.plot(
            (self.matching_time - t_start) / 3600,
            self.rev,
            color=self.colors["rev"],
            zorder=2,
        )

        # ------- plot ev -------
        ax.fill_between(
            (self.matching_time - t_start) / 3600,
            self.ev - self.ev_std,
            self.ev + self.ev_std,
            color=self.colors["ev"],
            zorder=3,
            alpha=0.5,
            label="EV",
        )
        ax.plot(
            (self.matching_time - t_start) / 3600, self.ev, self.colors["ev"], zorder=4
        )

        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Explained variance")
        if legend:
            ax.legend()
        ax.set_xlim(
            [
                (self.matching_time[0] - t_start) / 3600,
                (self.matching_time[-1] - t_start) / 3600,
            ]
        )
        return ax


class NeuronEnsembles(core.DataWriter):
    """[summary]

    Parameters
    ----------
    neurons : [type]
        [description]
    t_start : float
        start of the ensemble detection period
    t_stop : float
        end of the ensemble detection period
    bin_size : float
        bining for calculating spike counts
    frate_thresh : float
        exclude neurons with firing rate below or equal to this number
    ignore_epochs: core.Epoch

    References
    ----------
    1) van de Ven, G. M., Trouche, S., McNamara, C. G., Allen, K., & Dupret, D. (2016). Hippocampal offline reactivation consolidates recently formed cell assembly patterns during sharp wave-ripples. Neuron, 92(5), 968-974.Gido M. van de Ven et al. 2016
    """

    def __init__(
        self,
        neurons: core.Neurons,
        t_start=None,
        t_stop=None,
        bin_size=0.250,
        frate_thresh=0,
        ignore_epochs: core.Epoch = None,
    ):
        super().__init__()

        # ---- selecting neurons which are above frate_thresh ------
        frate = neurons.time_slice(t_start, t_stop).firing_rate
        neuron_indx_thresh = frate > frate_thresh

        if len(np.argwhere(neuron_indx_thresh)) < neurons.n_neurons:
            print(
                f"Based on frate_thresh, excluded neuron_ids: {neurons.neuron_ids[~neuron_indx_thresh]}"
            )
        self.neurons = neurons[neuron_indx_thresh]

        self.t_start = t_start
        self.t_stop = t_stop
        self.bin_size = bin_size
        self.ignore_epochs = ignore_epochs
        self.ensembles = None
        self._estimate_ensembles()

    def _estimate_ensembles(self):
        """extracting statisticaly independent components from significant eigenvectors as detected using Marcenko-Pasteur distributionvinput = Matrix  (m x n) where 'm' are the number of cells and 'n' time bins ICA weights thus extracted have highiest weight positive V = ICA weights for each neuron in the coactivation (weight having the highiest value is kept positive) M1 =  originally extracted neuron weights"""

        template = (
            self.neurons.time_slice(self.t_start, self.t_stop)
            .get_binned_spiketrains(bin_size=self.bin_size)
            .spike_counts
        )
        n_spikes = np.sum(template, axis=1)
        assert np.all(
            n_spikes > 0
        ), f"You have neurons with no spikes between {self.t_start,self.t_stop} seconds."

        zsc_template = stats.zscore(template, axis=1)

        # corrmat = (zsc_x @ zsc_x.T) / x.shape[1]
        corrmat = np.corrcoef(zsc_template)
        lambda_max = (1 + np.sqrt(1 / (template.shape[1] / template.shape[0]))) ** 2
        eig_val, eig_mat = np.linalg.eigh(corrmat)
        get_sigeigval = np.where(eig_val > lambda_max)[0]
        n_sigComp = len(get_sigeigval)
        pca_fit = PCA(n_components=n_sigComp, whiten=False).fit_transform(zsc_template)

        ica_decomp = FastICA(n_components=None, whiten=False).fit(pca_fit)
        W = ica_decomp.components_
        V = eig_mat[:, get_sigeigval] @ W.T

        # --- making highest absolute weight positive and then normalizing ----------
        max_weight = V[np.argmax(np.abs(V), axis=0), range(V.shape[1])]
        V[:, np.where(max_weight < 0)[0]] = (-1) * V[:, np.where(max_weight < 0)[0]]
        V /= np.sqrt(np.sum(V ** 2, axis=0))  # making sum of squares=1

        self.weights = V

    @property
    def n_ensembles(self):
        return self.weights.shape[1]

    def calculate_activation(self, t_start=None, t_stop=None, bin_size=0.250):

        W = self.weights
        act_binspk = self.neurons.time_slice(t_start, t_stop).get_binned_spiketrains(
            bin_size=bin_size
        )
        spkcnts = act_binspk.spike_counts

        activation = []
        for i in range(W.shape[1]):
            projMat = np.outer(W[:, i], W[:, i])
            np.fill_diagonal(projMat, 0)
            activation.append(
                np.asarray(
                    [
                        spkcnts[:, t] @ projMat @ spkcnts[:, t]
                        for t in range(spkcnts.shape[1])
                    ]
                )
            )

        self.activation = np.asarray(activation)
        self.activation_time = act_binspk.time
        self.activation_bin_size = bin_size

    def plot_activation(self, nrows=None, ncols=None):
        activation = self.activation
        t = self.activation_time

        if nrows is None:
            nrows, ncols = self.n_ensembles // 2, 2

        _, ax = plt.subplots(nrows, ncols, sharex=True, squeeze=False, sharey=True)
        ax = ax.reshape(-1)
        for i, act in enumerate(activation):
            ax[i].plot(t / 3600, act, color="#fa895c", lw=1)
            Fig.remove_spines(ax[i])
            Fig.set_spines_width(ax[i], lw=2)

        ax[i].set_xlabel("Time (h)")
        ax[i].set_ylabel("Act.")

    def plot_ensembles(self, style="heatmap", sort=True):
        weights = self.weights

        if style == "heatmap":
            _, ax = plt.subplots()
            ax.pcolormesh(weights)
