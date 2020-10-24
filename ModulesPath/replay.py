import numpy as np

from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import pandas as pd
import os
from mathutil import parcorr_mult, getICA_Assembly
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from parsePath import Recinfo


class Replay:
    def __init__(self, basepath):
        self.expvar = ExplainedVariance(basepath)
        self.bayesian = Bayesian(basepath)
        self.assemblyICA = CellAssemblyICA(basepath)
        self.corr = Correlation(basepath)


class Bayesian:
    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def correlation(self, template_time=None, match_time=None, cells=None):
        """Pairwise correlation between template window and matching windows

        Args:
            template_time (array_like, optional): in seconds
            match_time (array_like, optional): in seconds
            cells ([type], optional): cells to calculate the correlation for.

        Returns:
            [array]: returns correlation
        """

        if template_time is None:
            template_time = self._obj.epochs.maze

        if match_time is None:
            match_time = self._obj.epochs.post

        if cells is None:
            unstable_units = self._obj.spikes.stability.unstable
            # stable_units = self._obj.spikes.stability.stable
            # stable_units = list(range(len(spks)))
            stable_units = self._obj.spikes.stability.stable
            quality = np.asarray(self._obj.spikes.info.q)
            pyr = np.where(quality < 5)[0]
            stable_pyr = stable_units[np.isin(stable_units, pyr)]

        spks = self._obj.spikes.times
        print(stable_pyr)

        # spks = [spks[x] for x in stable_units]

        nUnits = len(spks)
        windowSize = self.timeWindow
        spks = [spks[_] for _ in stable_pyr]

        maze_bin = np.arange(maze[0], maze[1], 0.250)
        post_bin = np.arange(post[0], post[1], 0.250)

        pre_spikecount = np.array([np.histogram(x, bins=pre_bin)[0] for x in spks])
        pre_spikecount = [
            pre_spikecount[:, i : i + windowSize]
            for i in range(0, 3 * windowSize, windowSize)
        ]
        maze_spikecount = np.array([np.histogram(x, bins=maze_bin)[0] for x in spks])
        post_spikecount = np.array([np.histogram(x, bins=post_bin)[0] for x in spks])
        post_spikecount = [
            post_spikecount[:, i : i + windowSize]
            for i in range(0, 40 * windowSize, windowSize)
        ]

        # --- pre_corr = np.corrcoef(pre_spikecount)
        pre_corr = [np.corrcoef(pre_spikecount[x]) for x in range(len(pre_spikecount))]
        maze_corr = np.corrcoef(maze_spikecount)
        post_corr = [
            np.corrcoef(post_spikecount[x]) for x in range(len(post_spikecount))
        ]

        # --- selecting only pairwise correlations from different shanks
        shnkId = np.asarray(self._obj.spikes.info.shank)
        shnkId = shnkId[stable_pyr]
        assert len(shnkId) == len(spks)
        cross_shnks = np.nonzero(np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1)))
        pre_corr = [pre_corr[x][cross_shnks] for x in range(len(pre_spikecount))]
        maze_corr = maze_corr[cross_shnks]
        post_corr = [post_corr[x][cross_shnks] for x in range(len(post_spikecount))]
        print(maze_corr)

        corr_all = []
        for window in range(len(post_corr)):
            nas = np.logical_or(np.isnan(maze_corr), np.isnan(post_corr[window]))
            corr_all.append(np.corrcoef(post_corr[window][~nas], maze_corr[~nas])[0, 1])

        return np.asarray(corr_all)

    def oneD(self):
        pass

    def twoD(self):
        pass


class ExplainedVariance:

    nChans = 16
    binSize = 0.250  # in seconds
    window = 900  # in seconds

    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    # TODO  smooth version of explained variance
    def compute(self, template=None, match=None, control=None):
        """ Calucate explained variance (EV) and reverse EV
        References:
        1) Kudrimoti 1999
        2) Tastsuno et al. 2007
        """

        if None in [template, match, control]:
            control = self._obj.epochs.pre
            template = self._obj.epochs.maze
            match = self._obj.epochs.post

        # ----- choosing cells ----------------
        spks = self._obj.spikes.times
        stability = self._obj.spikes.stability.info
        stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]
        print(f"Calculating EV for {len(stable_pyr)} stable cells")
        spks = [spks[_] for _ in stable_pyr]

        # ------- windowing the time periods ----------
        window = self.window
        nbins_window = int(window / self.binSize)

        # ---- function to calculate correlation in each window ---------
        def cal_corr(period, windowing=True):
            bin_period = np.arange(period[0], period[1], self.binSize)
            spkcnt = np.array([np.histogram(x, bins=bin_period)[0] for x in spks])

            if windowing:
                dur = np.diff(period)
                nwindow = (dur / window)[0]
                t = np.arange(period[0], period[1], window)[: int(nwindow)] + window / 2

                window_spkcnt = [
                    spkcnt[:, i : i + nbins_window]
                    for i in range(0, int(nwindow) * nbins_window, nbins_window)
                ]

                if nwindow % 1 > 0.3:
                    window_spkcnt.append(spkcnt[:, int(nwindow) * nbins_window :])
                    t = np.append(t, round(nwindow % 1, 3) / 2)

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
        shnkId = np.asarray(self._obj.spikes.info.shank)
        shnkId = shnkId[stable_pyr]
        assert len(shnkId) == len(spks)
        cross_shnks = np.nonzero(np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1)))

        # --- selecting only pairwise correlations from different shanks -------
        control_corr = [control_corr[x][cross_shnks] for x in range(len(control_corr))]
        template_corr = template_corr[cross_shnks]
        match_corr = [match_corr[x][cross_shnks] for x in range(len(match_corr))]

        parcorr_template_vs_match, rev_corr = parcorr_mult(
            [template_corr], match_corr, control_corr
        )

        ev_template_vs_match = parcorr_template_vs_match ** 2
        rev_corr = rev_corr ** 2

        self.ev = ev_template_vs_match
        self.rev = rev_corr

    def plot(self, ax=None, tstart=0):

        ev_mean = np.mean(self.ev.squeeze(), axis=0)
        ev_std = np.std(self.ev.squeeze(), axis=0)
        rev_mean = np.mean(self.rev.squeeze(), axis=0)
        rev_std = np.std(self.rev.squeeze(), axis=0)

        if ax is None:
            plt.clf()
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        t = (self.t_match - tstart) / 3600  # converting to hour

        ax.fill_between(
            t, rev_mean - rev_std, rev_mean + rev_std, color="#87d498", zorder=1
        )
        ax.plot(t, rev_mean, "#02c59b", zorder=2)
        ax.fill_between(
            t, ev_mean - ev_std, ev_mean + ev_std, color="#7c7979", zorder=3
        )

        ax.plot(t, ev_mean, "k", zorder=4)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Explained variance")
        ax.legend(["EV", "REV"])
        ax.text(0.2, 0.28, "POST SD", fontweight="bold")
        # ax.set_xlim([0, 4])


class CellAssemblyICA:
    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def getAssemblies(self, x):
        """extracting statisticaly independent components from significant eigenvectors as detected using Marcenko-Pasteur distributionvinput = Matrix  (m x n) where 'm' are the number of cells and 'n' time bins ICA weights thus extracted have highiest weight positive (as done in Gido M. van de Ven et al. 2016) V = ICA weights for each neuron in the coactivation (weight having the highiest value is kept positive) M1 =  originally extracted neuron weights

        Arguments:
            x {[ndarray]} -- [an array of size n * m]

        Returns:
            [type] -- [Independent assemblies]
        """

        zsc_x = stats.zscore(x, axis=1)

        # corrmat = (zsc_x @ zsc_x.T) / x.shape[1]
        corrmat = np.corrcoef(zsc_x)

        lambda_max = (1 + np.sqrt(1 / (x.shape[1] / x.shape[0]))) ** 2
        eig_val, eig_mat = np.linalg.eigh(corrmat)
        get_sigeigval = np.where(eig_val > lambda_max)[0]
        n_sigComp = len(get_sigeigval)
        pca_fit = PCA(n_components=n_sigComp, whiten=False).fit_transform(zsc_x)

        ica_decomp = FastICA(n_components=None, whiten=False).fit(pca_fit)
        W = ica_decomp.components_
        V = eig_mat[:, get_sigeigval] @ W.T

        # --- making highest absolute weight positive and then normalizing ----------
        max_weight = V[np.argmax(np.abs(V), axis=0), range(V.shape[1])]
        V[:, np.where(max_weight < 0)[0]] = (-1) * V[:, np.where(max_weight < 0)[0]]
        V /= np.sqrt(np.sum(V ** 2, axis=0))  # making sum of squares=1

        self.vectors = V
        return self.vectors

    def getActivation(self, template, match, spks=None, binsize=0.250):

        if spks is None:
            spks = self._obj.spikes.pyr

        template_bin = np.arange(template[0], template[1], binsize)
        template = np.asarray(
            [np.histogram(cell, bins=template_bin)[0] for cell in spks]
        )

        V = self.getAssemblies(template)

        match_bin = np.arange(match[0], match[1], binsize)
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

    def plotActivation(self, ax=None):
        activation = self.activation
        vectors = self.vectors
        nvec = activation.shape[0]
        t = self.match_bin[1:]

        if ax is None:
            plt.clf()
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(nvec, 6, figure=fig)
            fig.subplots_adjust(hspace=0.3)

        else:
            gs = gridspec.GridSpecFromSubplotSpec(7, 6, ax, wspace=0.1)

        for vec in range(nvec):
            axact = plt.subplot(gs[vec, 3:])
            axact.plot(t / 3600, activation[vec, :])

            axvec = plt.subplot(gs[vec, :2])
            axvec.stem(vectors[:, vec], markerfmt="C2o")
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


class Correlation:
    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def comparePeriods(self, template, match, spks=None, window=900, bnsz=0.25):

        if spks is None:
            spks = self._obj.spikes.times

        template_corr = self.getcorr(period=template)
        match_corr = self.getcorr(period=match)

    def getcorr(self):
        bins = np.arange(period[0], period[1], binsize)
        spk_cnts = np.asarray([np.histogram(cell, bins=bins)[0] for cell in spikes])
        corr = np.corrcoef(spk_cnts)
        np.fill_diagonal(corr, 0)

        return corr

