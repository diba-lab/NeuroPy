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
from ..parsePath import Recinfo
from .. import core
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
        window=900,
        slideby=None,
        cross_shanks=True,
        ignore_epochs: core.Epoch = None,
    ):
        super().__init__()
        self.template = template
        self.matching = matching
        self.control = control
        self.neurons = neurons
        self.bin_size = bin_size
        self.window = window
        self.slideby = slideby
        self.cross_shanks = cross_shanks
        self.ignore_epochs = ignore_epochs

    def calculate(self):
        epochs = np.array((self.template, self.matching, self.control))
        epochs = core.Epoch(
            pd.DataFrame(
                {
                    "start": epochs[:, 0],
                    "stop": epochs[:, 1],
                    "label": ["template", "matching", "control"],
                }
            )
        )

        for del_epoch in self.ignore_epochs.to_dataframe().itertuples():
            epochs = epochs.delete_in_between(t1=del_epoch.start, t2=del_epoch.stop)

        for epoch in epochs.to_dataframe().itertuples():
            spktrn = self.neurons.get_binned_spiketrains(
                epoch.start, epoch.stop, self.bin_size
            )
            pair_corr = spktrn.get_pairwise_corr(cross_shanks=self.cross_shanks)

    def compute(
        self,
        neuron_ids,
        binSize=0.250,
        window=900,
        slideby=None,
        cross_shanks=True,
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

        spks = self._neurons.get_spiketrains(neuron_ids)
        shnkId = self._neurons.get_shankids(neuron_ids)

        if slideby is None:
            slideby = window

        # ----- choosing cells ----------------
        # spks = spikes.times
        # stability = spikes.stability.info
        # stable_cells = np.where(stability.stable == 1)[0]
        # pyr_id = spikes.pyrid
        # stable_pyr = np.intersect1d(pyr_id, stable_cells, assume_unique=True)
        # print(f"Calculating EV for {len(stable_pyr)} stable cells")
        # spks = [spks[_] for _ in stable_pyr]

        # ------- windowing the time periods ----------
        nbins_window = int(window / binSize)
        nbins_slide = int(slideby / binSize)

        # ---- function to calculate correlation in each window ---------
        def cal_corr(period, windowing=True):
            bin_period = np.arange(period[0], period[1], binSize)
            spkcnt = np.array([np.histogram(x, bins=bin_period)[0] for x in spks])

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
        control_corr, self.t_control = cal_corr(period=control)
        template_corr, _ = cal_corr(period=template, windowing=False)
        match_corr, self.t_match = cal_corr(period=match)

        # ----- indices for cross shanks correlation -------
        # shnkId = np.asarray(spikes.info.shank)
        # shnkId = shnkId[stable_pyr]
        assert len(shnkId) == len(spks)

        selected_pairs = np.tril_indices(len(spks), k=-1)
        if cross_shanks:
            selected_pairs = np.nonzero(
                np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1))
            )

        # --- selecting only pairwise correlations from different shanks -------
        control_corr = [
            control_corr[x][selected_pairs] for x in range(len(control_corr))
        ]
        template_corr = template_corr[selected_pairs]
        match_corr = [match_corr[x][selected_pairs] for x in range(len(match_corr))]

        parcorr_template_vs_match, rev_corr = parcorr_mult(
            [template_corr], match_corr, control_corr
        )

        ev_template_vs_match = parcorr_template_vs_match ** 2
        rev_corr = rev_corr ** 2

        self.ev = ev_template_vs_match
        self.rev = rev_corr
        self.npairs = template_corr.shape[0]

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

    def plot(self, ax=None, tstart=0, legend=True):

        ev_mean = np.nanmean(self.ev.squeeze(), axis=0)
        ev_std = np.nanstd(self.ev.squeeze(), axis=0)
        rev_mean = np.nanmean(self.rev.squeeze(), axis=0)
        rev_std = np.nanstd(self.rev.squeeze(), axis=0)

        if ax is None:
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        t = (self.t_match - tstart) / 3600  # converting to hour

        # ---- plot rev first ---------
        ax.fill_between(
            t,
            rev_mean - rev_std,
            rev_mean + rev_std,
            color=self.colors["rev"],
            zorder=1,
            alpha=0.5,
            label="REV",
        )
        ax.plot(t, rev_mean, color=self.colors["rev"], zorder=2)

        # ------- plot ev -------
        ax.fill_between(
            t,
            ev_mean - ev_std,
            ev_mean + ev_std,
            color=self.colors["ev"],
            zorder=3,
            alpha=0.5,
            label="EV",
        )
        ax.plot(t, ev_mean, self.colors["ev"], zorder=4)

        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Explained variance")
        if legend:
            ax.legend()

        return ax


class CellAssembly:
    def __init__(self, basepath, neurons: core.Neurons):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        self._neurons = neurons

    def getAssemblies(self, cell_ids, period, bnsz=0.25):
        """extracting statisticaly independent components from significant eigenvectors as detected using Marcenko-Pasteur distributionvinput = Matrix  (m x n) where 'm' are the number of cells and 'n' time bins ICA weights thus extracted have highiest weight positive (as done in Gido M. van de Ven et al. 2016) V = ICA weights for each neuron in the coactivation (weight having the highiest value is kept positive) M1 =  originally extracted neuron weights

        Arguments:
            x {[ndarray]} -- [an array of size n * m]

        Returns:
            [type] -- [Independent assemblies]
        """

        spikes = self._neurons.get_spiketrains(cell_ids)

        template_bin = np.arange(period[0], period[1], bnsz)
        template = np.asarray(
            [np.histogram(cell, bins=template_bin)[0] for cell in spikes]
        )

        # --- removing very low firing cells -----
        # nspikes = np.sum(template, axis=1)
        # good_cells = np.where(nspikes > 10)[0]
        # template = template[good_cells, :]
        # spikes = [spikes[_] for _ in good_cells]

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

        self.vectors = V
        self.spikes = spikes
        return self.vectors

    def getActivation(self, period, binsize=0.250):

        V = self.vectors
        spks = self.spikes

        match_bin = np.arange(period[0], period[1], binsize)
        match = np.asarray([np.histogram(cell, bins=match_bin)[0] for cell in spks])

        activation = []
        for i in range(V.shape[1]):
            projMat = np.outer(V[:, i], V[:, i])
            np.fill_diagonal(projMat, 0)
            activation.append(
                np.asarray(
                    [match[:, t] @ projMat @ match[:, t] for t in range(match.shape[1])]
                )
            )

        self.activation = np.asarray(activation)
        self.match_bin = match_bin

        return self.activation, self.match_bin

    def plotActivation(self):
        activation = self.activation
        vectors = self.vectors
        nvec = activation.shape[0]
        nCells = vectors.shape[0]
        t = self.match_bin[1:]

        fig = plt.figure(num=None, figsize=(10, 15))
        gs = gridspec.GridSpec(nvec, 6, figure=fig)
        fig.subplots_adjust(hspace=0.3)

        for vec in range(nvec):
            axact = plt.subplot(gs[vec, 3:])
            axact.plot(t / 3600, activation[vec, :])

            axvec = plt.subplot(gs[vec, :2])
            # axvec.stem(vectors[:, vec], markerfmt="C2o")
            axvec.vlines(np.arange(nCells), ymin=0, ymax=vectors[:, vec])
            if vec == nvec - 1:
                axact.set_xlabel("Time")
                axact.set_ylabel("Activation \n strength")

                axvec.set_xlabel("Neurons")
                axvec.set_ylabel("Weight")

            else:
                axact.set_xticks([])
                axact.set_xticklabels([])

                axvec.set_xticks([])
                axvec.set_xticklabels([])
                axvec.spines["bottom"].set_visible(False)
