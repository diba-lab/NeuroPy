import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, binned_statistic
from sklearn.naive_bayes import GaussianNB
import math
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.special import factorial
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from parsePath import Recinfo


class DecodeBehav:
    def __init__(self, basepath):

        # self._obj = basepath
        self.bayes1d = bayes1d(basepath)
        self.bayes2d = bayes2d(basepath)


class bayes1d:
    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def fit(self):
        spkAll = self._obj.spikes.times
        x = self._obj.position.x
        y = self._obj.position.y
        t = self._obj.position.t
        maze = self._obj.epochs.maze  # in seconds
        maze[0] = maze[0] + 60
        maze[1] = maze[1] - 90

        # we require only maze portion
        ind_maze = np.where((t > maze[0]) & (t < maze[1]))[0]
        x = y[ind_maze]
        y = y[ind_maze]
        t = t[ind_maze]

        x = x + abs(min(x))
        x_grid = np.arange(min(x), max(x), 10)

        diff_posx = np.diff(x)
        diff_posy = np.diff(y)

        speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2)
        dt = t[1] - t[0]
        speed_thresh = np.where(speed / dt > 0)[0]

        x_thresh = x[speed_thresh]
        y_thresh = y[speed_thresh]
        t_thresh = t[speed_thresh]

        occupancy = np.histogram(x, bins=x_grid)[0]
        shape_occ = occupancy.shape
        occupancy = occupancy + np.spacing(1)
        occupancy = occupancy / 120  # converting to seconds

        bin_t = np.arange(t[0], t[-1], 0.1)
        x_bin = np.interp(bin_t, t, x)
        y_bin = np.interp(bin_t, t, y)

        bin_number_t = np.digitize(x_bin, bins=x_grid)

        spkcount = np.asarray([np.histogram(x, bins=bin_t)[0] for x in spkAll])
        ratemap, spk_pos = [], []
        for cell in spkAll:

            spk_maze = cell[np.where((cell > maze[0]) & (cell < maze[1]))]
            spk_speed = np.interp(spk_maze, t[1:], speed)
            spk_y = np.interp(spk_maze, t, y)
            spk_x = np.interp(spk_maze, t, x)

            spk_map = np.histogram(spk_y, bins=x_grid)[0]
            spk_map = spk_map / occupancy
            ratemap.append(spk_map)
            spk_pos.append([spk_x, spk_y])

        ratemap = np.asarray(ratemap)
        print(ratemap.shape)

        ntbin = len(bin_t)
        nposbin = len(x_grid) - 1
        prob = (
            lambda nspike, rate: (1 / math.factorial(nspike))
            * ((0.1 * rate) ** nspike)
            * (np.exp(-0.1 * rate))
        )

        pos_decode = []
        for timebin in range(len(bin_t) - 1):
            spk_bin = spkcount[:, timebin]

            prob_allbin = []
            for posbin in range(nposbin):
                rate_bin = ratemap[:, posbin]
                spk_prob_bin = [prob(spk, rate) for spk, rate in zip(spk_bin, rate_bin)]
                prob_thisbin = np.prod(spk_prob_bin)
                prob_allbin.append(prob_thisbin)

            prob_allbin = np.asarray(prob_allbin)

            posterior = prob_allbin / np.sum(prob_allbin)
            predict_bin = np.argmax(posterior)

            pos_decode.append(predict_bin)

        plt.plot(bin_number_t, "k")
        plt.plot(pos_decode, "r")


class bayes2d:
    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

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

