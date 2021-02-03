from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.special import factorial

from behavior import behavior_epochs
from getSpikes import Spikes
from parsePath import Recinfo
from pfPlot import pf1d, pf2d
from plotUtil import Fig
from joblib import Parallel, delayed
from tqdm import tqdm


class DecodeBehav:
    def __init__(self, pf1d_obj: pf1d, pf2d_obj: pf2d):

        self.bayes1d = Bayes1d(pf1d_obj)
        self.bayes2d = bayes2d(pf2d_obj)


class Bayes1d:
    binsize = 0.02
    n_jobs = 8

    def __init__(self, pf1d_obj: pf1d):
        self._obj = pf1d_obj._obj
        self.ratemaps = pf1d_obj
        self._events = None
        self.posterior = None
        self.decodedPos = None

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, events: pd.DataFrame):

        if isinstance(events, pd.DataFrame):
            assert (
                pd.Series(["start", "end"]).isin(events.columns).all()
            ), "events should have start and end as column names"
        elif isinstance(events, np.ndarray):
            events = pd.DataFrame({"start": events[:, 0], "end": events[:, 1]})
        self._events = events

    def _decoder(self, spkcount, ratemaps):
        """
        ===========================
        Probability is calculated using this formula
        prob = (1 / nspike!)* ((tau * frate)^nspike) * exp(-tau * frate)
        where,
            tau = binsize
        ===========================
        """
        tau = self.binsize
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

    def estimate_behavior(self, speed_thresh=False, smooth=1, plot=True):

        """Estimates position on track using ratemaps and spike counts during behavior

        TODO: Needs furthher improvement/polish

        Parameters
        ----------
        binsize : float
            binsize in seconds
        """
        pf1d_obj = self.ratemaps
        spikes = Spikes(self._obj).pyr
        binsize = self.binsize

        mapinfo = pf1d_obj.no_thresh
        if speed_thresh:
            mapinfo = pf1d_obj.thresh

        ratemaps = np.asarray(mapinfo["ratemaps"])
        bincntr = pf1d_obj.bin + np.diff(pf1d_obj.bin).mean() / 2
        maze = pf1d_obj.period
        x = pf1d_obj.x
        time = pf1d_obj.t
        speed = pf1d_obj.speed

        tmz = np.arange(maze[0], maze[1], binsize)
        actualposx = stats.binned_statistic(time, values=x, bins=tmz)[0]
        meanspeed = stats.binned_statistic(time, speed, bins=tmz)[0]

        spkcount = np.asarray([np.histogram(cell, bins=tmz)[0] for cell in spikes])

        self.posterior = self._decoder(spkcount, ratemaps)
        self.decodedPos = bincntr[np.argmax(self.posterior, axis=0)]
        self.decodingtime = tmz
        self.actualpos = actualposx

        if plot:
            _, gs = Fig().draw(grid=(3, 4), size=(15, 6))
            axpos = plt.subplot(gs[0, :3])
            axpos.plot(self.actualpos, "k")
            axpos.set_ylabel("Actual position")

            axdec = plt.subplot(gs[1, :3], sharex=axpos)
            axdec.plot(
                np.abs(
                    gaussian_filter1d(self.decodedPos, sigma=smooth) - self.actualpos
                ),
                "r",
            )
            axdec.set_ylabel("Error")

            axpost = plt.subplot(gs[2, :3], sharex=axpos)
            axpost.pcolormesh(
                self.posterior / np.max(self.posterior, axis=0, keepdims=True),
                cmap="binary",
            )
            axpost.set_ylabel("Posterior")

            axconf = plt.subplot(gs[:, 3])
            actual_ = self.actualpos[np.where(meanspeed > 20)[0]]
            decoded_ = self.decodedPos[np.where(meanspeed > 20)[0]]
            bin_ = np.histogram2d(decoded_, actual_, bins=[pf1d_obj.bin, pf1d_obj.bin])[
                0
            ]
            bin_ = bin_ / np.max(bin_, axis=0, keepdims=True)
            axconf.pcolormesh(pf1d_obj.bin, pf1d_obj.bin, bin_, cmap="binary")
            axconf.set_xlabel("Actual position (cm)")
            axconf.set_ylabel("Estimated position (cm)")
            axconf.set_title("Confusion matrix")

    def decode_events(self, speed_thresh=False):
        """Decoding events like population bursts or ripples

        Parameters
        ----------
        events : pd.Dataframe
            dataframe with column names start and end
        binsize : float
            bin size within each events
        slideby : float
            sliding window by this much, in seconds
        """

        events = self.events
        spks = Spikes(self._obj).pyr
        pf1d_obj = self.ratemaps

        mapinfo = pf1d_obj.no_thresh
        if speed_thresh:
            mapinfo = pf1d_obj.thresh

        ratemaps = np.asarray(mapinfo["ratemaps"])
        bincntr = pf1d_obj.bin + np.diff(pf1d_obj.bin).mean() / 2

        # ----- removing cells that fire < 1 HZ --------
        good_cells = np.where(np.max(ratemaps, axis=1) > 1)[0]
        spks = [spks[_] for _ in good_cells]
        ratemaps = ratemaps[good_cells, :]

        # --- sorting the cells according to pf location -------
        sort_ind = np.argsort(np.argmax(ratemaps, axis=1))
        spks = [spks[_] for _ in sort_ind]
        ratemaps = ratemaps[sort_ind, :]

        # ----- calculating binned spike counts -------------
        # Ncells = len(spks)
        nbins_events = np.zeros(len(events))  # number of bins in each event
        bins_events = []
        for i, epoch in enumerate(events.itertuples()):
            bins = np.arange(epoch.start, epoch.end, self.binsize)
            nbins_events[i] = len(bins) - 1
            bins_events.extend(bins)
        spkcount = np.asarray([np.histogram(_, bins=bins_events)[0] for _ in spks])

        # ---- deleting unwanted columns that represent time between events ------
        cumsum_nbins = np.cumsum(nbins_events)
        del_columns = cumsum_nbins[:-1] + np.arange(len(cumsum_nbins) - 1)
        spkcount = np.delete(spkcount, del_columns.astype(int), axis=1)

        posterior = self._decoder(spkcount, ratemaps)
        decodedPos = bincntr[np.argmax(posterior, axis=0)]
        cum_nbins = np.append(0, np.cumsum(nbins_events)).astype(int)

        posterior = [
            posterior[:, cum_nbins[i] : cum_nbins[i + 1]]
            for i in range(len(cum_nbins) - 1)
        ]

        decodedPos = [
            decodedPos[cum_nbins[i] : cum_nbins[i + 1]]
            for i in range(len(cum_nbins) - 1)
        ]
        spkcount = [
            spkcount[:, cum_nbins[i] : cum_nbins[i + 1]]
            for i in range(len(cum_nbins) - 1)
        ]
        self.decodedPos = decodedPos
        self.posterior = posterior
        self.spkcount = spkcount
        self.nbins_events = nbins_events
        self.score, self.slope = self._score_events(posterior)

    def decode_shuffle(self, speed_thresh=False, n_iter=100, kind="column"):
        """Decoding events like population bursts or ripples

        Parameters
        ----------
        events : pd.Dataframe
            dataframe with column names start and end
        binsize : float
            bin size within each events
        slideby : float
            sliding window by this much, in seconds
        """

        # print(f"Using {kind} shuffle")
        score = []

        if kind == "cellid":
            spks = Spikes(self._obj).pyr
            pf1d_obj = self.ratemaps

            mapinfo = pf1d_obj.no_thresh
            if speed_thresh:
                mapinfo = pf1d_obj.thresh

            ratemaps = np.asarray(mapinfo["ratemaps"])
            bincntr = pf1d_obj.bin + np.diff(pf1d_obj.bin).mean() / 2

            # ----- removing cells that fire < 1 HZ --------
            good_cells = np.where(np.max(ratemaps, axis=1) > 1)[0]
            spks = [spks[_] for _ in good_cells]
            ratemaps = ratemaps[good_cells, :]

            # --- sorting the cells according to pf location -------
            sort_ind = np.argsort(np.argmax(ratemaps, axis=1))
            spks = [spks[_] for _ in sort_ind]
            ratemaps = ratemaps[sort_ind, :]

            posterior, decodedPos = [], []
            for i in range(n_iter):
                np.random.shuffle(ratemaps)

                posterior_ = self._decoder(np.hstack(self.spkcount), ratemaps)
                decodedPos_ = bincntr[np.argmax(posterior_, axis=0)]
                cum_nbins = np.append(0, np.cumsum(self.nbins_events)).astype(int)

                posterior.extend(
                    [
                        posterior_[:, cum_nbins[i] : cum_nbins[i + 1]]
                        for i in range(len(cum_nbins) - 1)
                    ]
                )

                decodedPos.extend(
                    [
                        decodedPos_[cum_nbins[i] : cum_nbins[i + 1]]
                        for i in range(len(cum_nbins) - 1)
                    ]
                )

        if kind == "column":

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

        self.shuffle_score = np.array(score)

    def _score_events(self, posterior):
        """Scoring of events

        Returns
        -------
        [type]
            [description]

        References
        ----------
        1) Kloosterman et al. 2012
        """
        # ------ similar to radon transform ------------

        def score_event(evt):
            t = np.arange(evt.shape[1])
            nt = len(t)
            tmid = (nt + 1) / 2
            pos = np.arange(evt.shape[0])
            npos = len(pos)
            pmid = (npos + 1) / 2
            evt = np.apply_along_axis(np.convolve, axis=0, arr=evt, v=np.ones(3))

            nlines = 5000
            theta = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=nlines)
            diag_len = np.sqrt((nt - 1) ** 2 + (npos - 1) ** 2)
            intercept = np.random.uniform(
                low=-diag_len / 2, high=diag_len / 2, size=nlines
            )

            cmat = np.tile(intercept, (nt, 1)).T
            mmat = np.tile(theta, (nt, 1)).T
            tmat = np.tile(t, (nlines, 1))
            posterior = np.zeros((nlines, nt))

            y_line = (
                ((cmat - (tmat - tmid) * np.cos(mmat)) / np.sin(mmat)) + pmid
            ).astype(int)
            t_out = np.where((y_line < 0) | (y_line > npos - 1))
            t_in = np.where((y_line >= 0) & (y_line <= npos - 1))
            posterior[t_out] = np.median(evt[:, t_out[1]], axis=0)
            posterior[t_in] = evt[y_line[t_in], t_in[1]]

            posterior_sum = np.nanmean(posterior, axis=1)
            max_line = np.argmax(posterior_sum)
            slope = -(1 / np.tan(theta[max_line]))
            return posterior_sum[max_line], slope

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(score_event)(evt) for evt in posterior
        )
        score = [res[0] for res in results]
        slope = [res[1] for res in results]

        return np.asarray(score), np.asarray(slope)

    @property
    def p_val_events(self):
        shuff_score = self.shuffle_score
        n_iter = shuff_score.shape[0]
        diff_score = shuff_score - np.tile(self.score, (n_iter, 1))
        chance = np.where(diff_score > 0, 1, 0).sum(axis=0)
        return (chance + 1) / (n_iter + 1)

    def plot_replay_events(self, pval=0.05, speed_thresh=True, cmap="hot"):
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

        mapinfo = pf1d_obj.no_thresh
        if speed_thresh:
            mapinfo = pf1d_obj.thresh

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


class bayes2d:
    def __init__(self, pf2d_obj: pf2d):

        assert isinstance(pf2d_obj, pf2d)
        self._obj = pf2d_obj._obj
        self.ratemaps = pf2d_obj

    def fit(self):
        trackingSrate = self._obj.position.tracking_sRate
        spkAll = self._obj.spikes.pyr
        x = self._obj.position.x
        y = self._obj.position.y
        t = self._obj.position.t
        maze = self._obj.epochs.maze  # in seconds

        # --- we require only maze portion -----
        ind_maze = np.where((t > maze[0]) & (t < maze[1]))
        x = x[ind_maze]
        y = y[ind_maze]
        t = t[ind_maze]

        x_grid, xstep = np.linspace(min(x), max(x), 50, retstep=True)
        y_grid, ystep = np.linspace(min(y), max(y), 50, retstep=True)
        mesh = np.meshgrid(x_grid[:-1] + xstep / 2, y_grid[:-1] + ystep / 2)
        ngrid_centers = mesh[0].size

        x_center = np.reshape(mesh[0], [ngrid_centers, 1])
        y_center = np.reshape(mesh[1], [ngrid_centers, 1])
        xy_center = np.hstack((x_center, y_center))

        # ----- Speed calculation -------
        diff_posx = np.diff(x)
        diff_posy = np.diff(y)
        dt = 1 / trackingSrate
        speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2) / dt
        speed_thresh = np.where(speed / dt > 0)[0]

        occupancy = np.histogram2d(x, y, bins=(x_grid, y_grid))[0]
        occupancy = (occupancy + np.spacing(1)) / trackingSrate
        # occupancy = gaussian_filter(occupancy, sigma=1)

        ratemap, spk_pos = [], []
        for cell in spkAll:

            spk_maze = cell[np.where((cell > maze[0]) & (cell < maze[1]))]
            spk_speed = np.interp(spk_maze, t[1:], speed)
            spk_y = np.interp(spk_maze, t, y)
            spk_x = np.interp(spk_maze, t, x)

            # speed threshold
            spd_ind = np.where(spk_speed > 5)
            spk_spd = spk_speed[spd_ind]
            spk_x = spk_x[spd_ind]
            spk_y = spk_y[spd_ind]
            spk_t = spk_maze[spd_ind]

            spk_map = np.histogram2d(spk_x, spk_y, bins=(x_grid, y_grid))[0]
            spk_map = (gaussian_filter(spk_map / occupancy, sigma=1)).flatten("F")
            ratemap.append(spk_map)
            spk_pos.append([spk_x, spk_y])

        self.ratemap = np.asarray(ratemap)
        self._spks = spkAll
        self.gridcenter = xy_center.T
        self.grid = [x_grid, y_grid]

    def estimateBehav(self, binsize=0.25):
        ratemap = self.ratemap
        gridcntr = self.gridcenter
        spks = self._spks
        speed = self._obj.position.speed
        t = self._obj.position.t
        x = self._obj.position.x
        y = self._obj.position.y

        maze = self._obj.epochs.maze
        tmz = np.arange(maze[0], maze[1], binsize)
        actualposx = binned_statistic(t, values=x, bins=tmz)[0]
        actualposy = binned_statistic(t, values=y, bins=tmz)[0]
        meanspeed = binned_statistic(t[1:], speed, bins=tmz)[0]
        actualpos = np.vstack((actualposx, actualposy))

        spkcount = np.asarray([np.histogram(cell, bins=tmz)[0] for cell in spks])

        """ 
        ===========================
        Probability is calculated using this formula
        prob = (1 / nspike!)* ((0.1 * frate)^nspike) * exp(-0.1 * frate)
        =========================== 
        """

        Ncells = len(spks)
        cell_prob = np.zeros((ratemap.shape[1], spkcount.shape[1], Ncells))
        for cell in range(Ncells):
            cell_spkcnt = spkcount[cell, :][np.newaxis, :]
            cell_ratemap = ratemap[cell, :][:, np.newaxis]

            coeff = 1 / (factorial(cell_spkcnt))
            # broadcasting
            cell_prob[:, :, cell] = (((0.1 * cell_ratemap) ** cell_spkcnt) * coeff) * (
                np.exp(-0.1 * cell_ratemap)
            )

        posterior = np.prod(cell_prob, axis=2)
        posterior /= np.sum(posterior, axis=0)
        self.posterior = posterior
        self.decodedPos = gridcntr[:, np.argmax(self.posterior, axis=0)]
        self.decodingtime = tmz
        self.actualPos = actualpos
        self.speed = meanspeed

    def decode(self, epochs, binsize=0.02, slideby=0.005):

        assert isinstance(epochs, pd.DataFrame)

        spks = self._spks
        Ncells = len(spks)
        # self.fit()
        ratemap = self.ratemap
        gridcntr = self.gridcenter

        nbins = np.zeros(len(epochs))
        spkcount = []
        for i, epoch in enumerate(epochs.itertuples()):
            bins = np.arange(epoch.start, epoch.end - binsize, slideby)
            nbins[i] = len(bins)
            for j in bins:
                spkcount.append(
                    np.asarray(
                        [np.histogram(_, bins=[j, j + binsize])[0] for _ in spks]
                    )
                )

        spkcount = np.hstack(spkcount)
        print(spkcount.shape)

        """ 
        ===========================
        Probability is calculated using this formula
        prob = (1 / nspike!)* ((0.1 * frate)^nspike) * exp(-0.1 * frate)
        =========================== 
        """

        cell_prob = np.zeros((ratemap.shape[1], spkcount.shape[1], Ncells))
        for cell in range(Ncells):
            cell_spkcnt = spkcount[cell, :][np.newaxis, :]
            cell_ratemap = ratemap[cell, :][:, np.newaxis]

            coeff = 1 / (factorial(cell_spkcnt))
            # broadcasting
            cell_prob[:, :, cell] = (((0.1 * cell_ratemap) ** cell_spkcnt) * coeff) * (
                np.exp(-0.1 * cell_ratemap)
            )
        posterior = np.prod(cell_prob, axis=2)
        posterior /= np.sum(posterior, axis=0)

        decodedPos = gridcntr[:, np.argmax(posterior, axis=0)]
        cum_nbins = np.append(0, np.cumsum(nbins)).astype(int)

        posterior = [
            posterior[:, cum_nbins[i] : cum_nbins[i + 1]]
            for i in range(len(cum_nbins) - 1)
        ]

        decodedPos = [
            decodedPos[:, cum_nbins[i] : cum_nbins[i + 1]]
            for i in range(len(cum_nbins) - 1)
        ]

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
