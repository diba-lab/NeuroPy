import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA, FastICA
from typing import Union
from ..utils.mathutil import getICA_Assembly, parcorr_mult
from .. import core
from ..plotting import Fig


class ExplainedVariance(core.DataWriter):
    """Explained variance measure for assessing reactivation of neuronal activity using pairwise correlations.

    Attributes
    ----------
    neurons: core.Neurons
        neurons used for ev calculation
    ev: np.array of size n_matching_windows
        explained variance result for each window
    ev_std: np.array of size n_matching_windows
        standard deviation for each matching_window
    rev: array of size n_matching_windows
        reversed explained variance, an estimate of chance level
    matching_time: np.array
        midpoints of matching time windows in seconds
    control_time: np.array
        midpints of control time windows in seconds
    n_pairs: int
        maximum number of pairs


    References
    -------
    1) Kudrimoti, H. S., Barnes, C. A., & McNaughton, B. L. (1999). Reactivation of Hippocampal Cell Assemblies: Effects of Behavioral State, Experience, and EEG Dynamics. Journal of Neuroscience, 19(10), 4090–4101. https://doi.org/10/4090
    2) Tatsuno, M., Lipa, P., & McNaughton, B. L. (2006). Methodological Considerations on the Use of Template Matching to Study Long-Lasting Memory Trace Replay. Journal of Neuroscience, 26(42), 10727–10742. https://doi.org/10.1523/JNEUROSCI.3317-06.2006


    """

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
        pairs_bool=None,
        ignore_epochs: core.Epoch = None,
    ):
        """Explained variance measure for assessing reactivation of neuronal activity using pairwise correlations.

        Parameters
        ----------
        neurons : core.Neurons
            obj that holds spiketrains for multiple neurons
        template : list/array of length 2
            time in seconds, pairwise correlation calculated from this period will be compared to matching period
        matching : list/array of length 2
            time in seconds, template-correlations will be correlated with pariwise correlations of this period
        control : list/array of length 2
            time in seconds, control for pairwise correlations within this period
        bin_size : float, optional
            in seconds, binning size for spike counts, by default 0.250
        window : int or typle, optional
            window over which pairwise correlations will be calculated in matching and control time periods, if window is None entire time period is considered,in seconds, by default 900
        slideby : int, optional
            slide window by this much, in seconds, by default None
        pairs_bool : 2d array, optional
            a 2d symmetric boolean array of size n_neurons x n_neurons specifying which pairs to be kept for calcualting explained variance, by default None
        ignore_epochs : core.Epoch, optional
            ignore calculation for these epochs, helps with noisy epochs, by default None
        """
        super().__init__()
        self.neurons = neurons

        self.template = template
        self.matching = matching
        self.control = control
        self.bin_size = bin_size

        if (window is None) and (slideby is not None):
            print(
                "slideby can not be a number, if window is None, setting slideby to None"
            )
            slideby = None

        self.window = window
        self.slideby = slideby
        self.pairs_bool = pairs_bool
        self.ignore_epochs = ignore_epochs
        self._calculate()

    def _calculate(self):
        # TODO: Think about directly working on binned spiketrains, will be little faster but may require redundant additions like pariwise_corr as separate function

        matching = np.arange(self.matching[0], self.matching[1])
        control = np.arange(self.control[0], self.control[1])

        # truncate/delete windows if they fall within ignore_epochs
        if self.ignore_epochs is not None:
            ignore_bins = self.ignore_epochs.flatten()
            matching = matching[np.digitize(matching, ignore_bins) % 2 == 0]
            control = control[np.digitize(control, ignore_bins) % 2 == 0]

        if self.window is None:
            control_window_size = len(control)
            matching_window_size = len(matching)
            slideby = None
        elif self.window is not None and self.slideby is None:
            control_window_size = self.window
            matching_window_size = self.window
            slideby = None
        else:
            control_window_size = self.window
            matching_window_size = self.window
            slideby = self.slideby

        assert control_window_size <= len(control), "window is bigger than matching"
        assert matching_window_size <= len(matching), "window is bigger than matching"
        # assert slideby <= control_window_size, "slideby should be smaller than window"
        # assert slideby <= matching_window_size, "slideby should be smaller than window"

        matching_windows = np.lib.stride_tricks.sliding_window_view(
            matching, matching_window_size
        )[::slideby, [0, -1]]

        control_windows = np.lib.stride_tricks.sliding_window_view(
            control, control_window_size
        )[::slideby, [0, -1]]

        with np.errstate(all="ignore", invalid="ignore"):
            template_corr = (
                self.neurons.time_slice(self.template[0], self.template[1])
                .get_binned_spiketrains(
                    bin_size=self.bin_size, ignore_epochs=self.ignore_epochs
                )
                .get_pairwise_corr(pairs_bool=self.pairs_bool)
            )
            n_matching_windows = matching_windows.shape[0]
            matching_paircorr = []
            for w in matching_windows:
                matching_paircorr.append(
                    self.neurons.time_slice(w[0], w[1])
                    .get_binned_spiketrains(self.bin_size)
                    .get_pairwise_corr(pairs_bool=self.pairs_bool)
                )

            n_control_windows = control_windows.shape[0]
            control_paircorr = []
            for w in control_windows:
                control_paircorr.append(
                    self.neurons.time_slice(w[0], w[1])
                    .get_binned_spiketrains(self.bin_size)
                    .get_pairwise_corr(pairs_bool=self.pairs_bool)
                )

            partial_corr = np.zeros((n_control_windows, n_matching_windows))
            rev_partial_corr = np.zeros((n_control_windows, n_matching_windows))
            for m_i, m_pairs in enumerate(matching_paircorr):
                for c_i, c_pairs in enumerate(control_paircorr):
                    df = pd.DataFrame({"t": template_corr, "m": m_pairs, "c": c_pairs})
                    partial_corr[c_i, m_i] = pg.partial_corr(
                        df, x="t", y="m", covar="c"
                    ).r
                    rev_partial_corr[c_i, m_i] = pg.partial_corr(
                        df, x="t", y="c", covar="m"
                    ).r

        self.ev = np.nanmean(partial_corr**2, axis=0)
        self.rev = np.nanmean(rev_partial_corr**2, axis=0)
        self.ev_std = np.nanstd(partial_corr**2, axis=0)
        self.rev_std = np.nanstd(rev_partial_corr**2, axis=0)
        self.partial_corr = partial_corr
        self.rev_partial_corr = rev_partial_corr
        self.n_pairs = len(template_corr)
        self.matching_time = np.mean(matching_windows, axis=1)
        self.control_time = np.mean(control_windows, axis=1)

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

        if self.ignore_epochs is not None:
            for i, epoch in enumerate(self.ignore_epochs.itertuples()):
                ax.axvspan(
                    (epoch.start - t_start) / 3600,
                    (epoch.stop - t_start) / 3600,
                    color="k",
                    edgecolor=None,
                    # alpha=alpha,
                    zorder=5,
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
        V /= np.sqrt(np.sum(V**2, axis=0))  # making sum of squares=1

        self.weights = V

    @property
    def n_ensembles(self):
        return self.weights.shape[1]

    def get_activation(self, t_start=None, t_stop=None, bin_size=0.250):

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

        # self.activation = np.asarray(activation)
        # self.activation_time = act_binspk.time
        # self.activation_bin_size = bin_size

        return np.asarray(activation), act_binspk.time

    def plot_activation(self, time, activation, nrows=None, ncols=None):

        if nrows is None:
            nrows, ncols = self.n_ensembles // 2, 2

        _, ax = plt.subplots(nrows, ncols, sharex=True, squeeze=False, sharey=True)
        ax = ax.reshape(-1)
        for i, act in enumerate(activation):
            ax[i].plot(time / 3600, act, color="#fa895c", lw=1)
            Fig.remove_spines(ax[i])
            Fig.set_spines_width(ax[i], lw=2)

        ax[i].set_xlabel("Time (h)")
        ax[i].set_ylabel("Act.")

    def plot_ensembles(self, style="heatmap", sort=True):
        weights = self.weights

        if style == "heatmap":
            _, ax = plt.subplots()
            ax.pcolormesh(weights)
