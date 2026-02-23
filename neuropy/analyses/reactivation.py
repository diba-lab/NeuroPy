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
# from ..utils.mathutil import getICA_Assembly
from .. import core
try:
    from ..plotting import Fig
except ImportError:
    from neuropy.plotting.figure import Fig
from tqdm import tqdm


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

    def __init__(
        self,
        neurons: core.Neurons,
        template,
        matching,
        control,
        bin_size=0.250,
        window: int = 900,
        slideby: int = 300,
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
            time in seconds, template-correlations will be correlated with pairwise correlations of this period
        control : list/array of length 2
            time in seconds, control for pairwise correlations within this period
        bin_size : float, optional
            in seconds, binning size for spike counts, by default 0.250
        window : int or typle, optional
            window over which pairwise correlations will be calculated in matching and control time periods, if window is None entire time period is considered,in seconds, by default 900
        slideby : int, optional
            slide window by this much, in seconds, by default 300
        pairs_bool : 2d array, optional
            a 2d symmetric boolean array of size n_neurons x n_neurons specifying which pairs to be kept for calculating explained variance, by default None
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
            print(f"Calculating partial correlations for {len(matching_paircorr)} time windows")
            for m_i, m_pairs in enumerate(tqdm(matching_paircorr)):
                for c_i, c_pairs in enumerate(control_paircorr):
                    df = pd.DataFrame({"t": template_corr, "m": m_pairs, "c": c_pairs})
                    try:
                        # if ((~np.isnan(df["m"])).sum() > 2) and ((~np.isnan(df["c"])).sum() > 2):
                        # Make sure you have at least 3 neurons-pairs with valid pairwise correlations in the windows in question
                        if df[['t', 'm', 'c']].dropna().shape[0] > 2:
                            partial_corr[c_i, m_i] = pg.partial_corr(
                                df, x="t", y="m", covar="c"
                            ).r
                            rev_partial_corr[c_i, m_i] = pg.partial_corr(
                                df, x="t"
                                , y="c", covar="m"
                            ).r
                        else:  # Don't calculate ev and rev unless you have > 3 samples from both the control and matching epochs
                            partial_corr[c_i, m_i] = np.nan
                            rev_partial_corr[c_i, m_i] = np.nan
                    except AssertionError:

                        partial_corr[c_i, m_i] = pg.partial_corr(
                            df, x="t", y="m", covar="c"
                        ).r
                        pass


        self.ev = np.nanmean(partial_corr**2, axis=0)
        self.rev = np.nanmean(rev_partial_corr**2, axis=0)
        self.ev_std = np.nanstd(partial_corr**2, axis=0)
        self.rev_std = np.nanstd(rev_partial_corr**2, axis=0)
        self.partial_corr = partial_corr
        self.rev_partial_corr = rev_partial_corr
        self.n_pairs = len(template_corr)
        self.matching_time = np.mean(matching_windows, axis=1)
        self.control_time = np.mean(control_windows, axis=1)

    def plot(
        self,
        ax=None,
        t_start=0,
        legend=True,
        color_ev="#4a4a4a",
        color_rev="#05d69e",
        show_ignore_epochs=True,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        # ---- plot rev first ---------
        ax.fill_between(
            (self.matching_time - t_start) / 3600,
            self.rev - self.rev_std,
            self.rev + self.rev_std,
            color=color_rev,
            zorder=1,
            alpha=0.5,
            label="REV",
        )
        ax.plot(
            (self.matching_time - t_start) / 3600,
            self.rev,
            color=color_rev,
            zorder=2,
        )

        # ------- plot ev -------
        ax.fill_between(
            (self.matching_time - t_start) / 3600,
            self.ev - self.ev_std,
            self.ev + self.ev_std,
            color=color_ev,
            zorder=3,
            alpha=0.5,
            label="EV",
        )
        ax.plot((self.matching_time - t_start) / 3600, self.ev, color_ev, zorder=4)

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

        if show_ignore_epochs:
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

    References
    ----------
    1) van de Ven, G. M., Trouche, S., McNamara, C. G., Allen, K., & Dupret, D. (2016). Hippocampal offline reactivation consolidates recently formed cell assembly patterns during sharp wave-ripples. Neuron, 92(5), 968-974.Gido M. van de Ven et al. 2016
    """

    def __init__(
        self,
        neurons: core.Neurons,
        epochs: core.Epoch,
        bin_size=0.250,
        # ignore_epochs: core.Epoch = None,
        verbose=True,
    ):
        """

        Parameters
        ----------
        neurons : core.Neurons
            neurons to use for ensemble detection
        epochs : core.Epoch
            epochs with which ensembles are detected
        bin_size : float, optional
            binning size for spike counts, by default 0.250
        verbose : bool, optional
            _description_, by default True
        """
        super().__init__()

        self.bin_size = bin_size
        # self.ignore_epochs = ignore_epochs
        self.neurons, self.epochs = self._validate(neurons, epochs)
        self.weights = self._estimate_weights()
        self.verbose = verbose

    def _validate(self, neurons, epochs):
        assert isinstance(neurons, core.Neurons), "neurons is not of type core.Neurons"
        assert isinstance(epochs, core.Epoch), "epochs is not of type core.Epoch"

        # ---- removing neurons which do not fire during the epochs ------
        frate = np.zeros(len(neurons))
        for e in epochs.itertuples():
            frate += neurons.time_slice(e.start, e.stop).firing_rate

        neuron_indx_thresh = frate > 0
        if len(np.argwhere(neuron_indx_thresh)) < neurons.n_neurons:
            print(
                f"Removed neurons with no spikes within provided epochs, neuron_ids : {neurons.neuron_ids[~neuron_indx_thresh]}"
            )

            neurons = neurons[neuron_indx_thresh]

        return neurons, epochs

    def _estimate_weights(self):
        """extracting statisticaly independent components from significant eigenvectors as detected using Marcenko-Pasteur distribution vinput = Matrix  (m x n) where 'm' are the number of cells and 'n' time bins ICA weights thus extracted have highiest weight positive V = ICA weights for each neuron in the coactivation (weight having the highiest value is kept positive) M1 =  originally extracted neuron weights"""
        template = []
        for e in self.epochs.itertuples():
            template.append(
                self.neurons.time_slice(e.start, e.stop)
                .get_binned_spiketrains(bin_size=self.bin_size)
                .spike_counts
            )
        template = np.hstack(template)

        n_spikes = np.sum(template, axis=1)
        assert np.all(n_spikes > 0), f"Neurons with no spikes within epochs"

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
        return V

    @property
    def n_ensembles(self):
        return self.weights.shape[1]

    def get_activation(self, epochs: core.Epoch, bin_size=0.250):
        """Calculates activation strength of ensembles in given epochs. If number of epochs is more than one then activation strengths are calculated on combined binned spikecounts across epochs.

        Parameters
        ----------
        epochs : core.Epoch
            activation strength calculation is restricted to these epochs
        bin_size : float, optional
            bin size for spike counts, by default 0.250

        Returns
        -------
        array
            activation strength of the ensembles (n_neurons x n_bins)
        time
            time corresponding to each bin (n_bins x 0)
        """

        W = self.weights
        spkcnts, time = [], []
        for e in epochs.itertuples():
            e_binspk = self.neurons.time_slice(e.start, e.stop).get_binned_spiketrains(
                bin_size=bin_size
            )
            spkcnts.append(e_binspk.spike_counts)
            time.append(e_binspk.time)
        spkcnts = np.hstack(spkcnts)
        time = np.concatenate(time)

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

        return np.asarray(activation), time

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


if __name__ == "__main__":
    from pathlib import Path
    from neuropy.core.epoch import Epoch
    from neuropy.core.neurons import Neurons
    dir_use = Path('/Users/nkinsky/Documents/UM/Working/Octopamine_Rolipram/BG_2019-10-21_SDSAL')
    neurons_use = Neurons.from_file(sorted(dir_use.glob("*.neurons.npy"))[0])
    neurons_use = Neurons.from_dict(neurons_use)

    rec_epochs = Epoch(epochs=None, file=sorted(dir_use.glob("*.epoch.npy"))[0])
    t = ExplainedVariance(neurons_use, rec_epochs['maze'].as_array().squeeze(), rec_epochs['post'].as_array().squeeze(),
                          rec_epochs['pre'].as_array().squeeze(), slideby=300, window=900)
