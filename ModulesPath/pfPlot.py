import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.decomposition import PCA
from plotUtil import Colormap
from parsePath import Recinfo
from getPosition import ExtractPosition
from getSpikes import Spikes
from behavior import behavior_epochs
from plotUtil import pretty_plot


class pf:
    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        self.pf1d = pf1d(basepath)
        self.pf2d = pf2d(basepath)


class pf1d:
    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def compute(self, period):
        assert len(period) == 2, "period should have length 2"
        spikes = Spikes(self._obj)
        position = ExtractPosition(self._obj)

        trackingSRate = position.tracking_sRate
        spks = spikes.pyr
        xcoord = position.x
        ycoord = position.y

        time = position.t

        ind_maze = np.where((time > period[0]) & (time < period[1]))
        self.x = xcoord[ind_maze]
        self.y = ycoord[ind_maze]
        self.t = time[ind_maze]

        # --- Making sure x-axis is along the length of track -------
        # xrange = np.ptp(self.x)
        # yrange = np.ptp(self.y)
        # if yrange > xrange:
        #     self.x, self.y = self.y, self.x
        position = np.vstack((self.x, self.y)).T
        pca = PCA(n_components=1)
        self.xlinear = pca.fit_transform(position).squeeze()

        diff_posx = np.diff(self.y)

        dt = self.t[1] - self.t[0]

        # location = np.sqrt((xcoord) ** 2 + (ycoord) ** 2)
        self.speed = np.abs(diff_posx) / dt

        spk_pfx, spk_pft = [], []
        for cell in spks:

            spk_maze = cell[np.where((cell > period[0]) & (cell < period[1]))]
            spk_spd = np.interp(spk_maze, self.t[:-1], self.speed)
            spk_x = np.interp(spk_maze, self.t, self.xlinear)

            # speed threshold
            spd_ind = np.where(spk_spd > 0)
            spk_spd = spk_spd[spd_ind]
            spk_x = spk_x[spd_ind]
            spk_t = spk_maze[spd_ind]
            spk_pfx.append(spk_x)
            spk_pft.append(spk_t)

        self.spkx = spk_pfx
        self.spkt = spk_pft

        self.xbin = np.arange(min(self.x), max(self.x), 5)
        occupancy = np.histogram(self.x, bins=self.xbin)[0] / trackingSRate
        # occupancy = gaussian_filter1d(occupancy, sigma=1)

        self.ratemap = []
        for cell in range(10):
            spkpos = self.spkx[cell]
            spkmap = np.histogram(spkpos, bins=self.xbin)[0]
            # spkmap = gaussian_filter1d(spkmap, sigma=2)
            self.ratemap.append(spkmap / occupancy)

    def plot(self, ax=None, pad=2, normalize=False):

        ratemap = self.ratemap
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if normalize:
            ratemap = [cellmap / np.max(cellmap) for cellmap in ratemap]

        for cellid, cell in enumerate(ratemap):

            ax.fill_between(
                self.xbin[:-1],
                cellid * pad,
                cellid * pad + cell,
                color="gray",
                alpha=0.5,
                zorder=cellid + 1,
            )


class pf2d:
    def __init__(self, basepath, **kwargs):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def compute(self, period, gridbin=10, speed_thresh=10, smooth=2):
        """Calculates 2D placefields

        Parameters
        ----------
        period : list/array
            in seconds, time period between which placefields are calculated
        gridbin : int, optional
            bin size of grid in centimeters, by default 10
        speed_thresh : int, optional
            speed threshold in cm/s, by default 10 cm/s

        Returns
        -------
        [type]
            [description]
        """
        assert len(period) == 2, "period should have length 2"
        spikes = Spikes(self._obj)
        position = ExtractPosition(self._obj)
        # ------ Cell selection ---------
        spkAll = spikes.pyr
        # spkinfo = self._obj.spikes.info
        # pyrid = np.where(spkinfo.q < 4)[0]
        # spkAll = [spkAll[_] for _ in pyrid]

        # ----- Position---------
        xcoord = position.x
        ycoord = position.y
        time = position.t
        trackingRate = position.tracking_sRate

        ind_maze = np.where((time > period[0]) & (time < period[1]))
        x = xcoord[ind_maze]
        y = ycoord[ind_maze]
        t = time[ind_maze]

        x_grid = np.arange(min(x), max(x), gridbin)
        y_grid = np.arange(min(y), max(y), gridbin)
        # x_, y_ = np.meshgrid(x_grid, y_grid)

        diff_posx = np.diff(x)
        diff_posy = np.diff(y)

        speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2) / (1 / trackingRate)
        speed = gaussian_filter1d(speed, sigma=smooth)
        print(np.ptp(speed))
        dt = t[1] - t[0]
        running = np.where(speed / dt > speed_thresh)[0]

        x_thresh = x[running]
        y_thresh = y[running]
        t_thresh = t[running]

        def make_pfs(
            t_, x_, y_, spkAll_, occupancy_, speed_thresh_, maze_, x_grid_, y_grid_
        ):
            maps, spk_pos, spk_t = [], [], []
            for cell in spkAll_:
                # assemble spikes and position data
                spk_maze = cell[np.where((cell > maze_[0]) & (cell < maze_[1]))]
                spk_speed = np.interp(spk_maze, t_[1:], speed)
                spk_y = np.interp(spk_maze, t_, y_)
                spk_x = np.interp(spk_maze, t_, x_)

                # speed threshold
                spd_ind = np.where(spk_speed > speed_thresh_)
                # spk_spd = spk_speed[spd_ind]
                spk_x = spk_x[spd_ind]
                spk_y = spk_y[spd_ind]

                # Calculate maps
                spk_map = np.histogram2d(spk_x, spk_y, bins=(x_grid_, y_grid_))[0]
                spk_map = gaussian_filter(spk_map, sigma=smooth)
                maps.append(spk_map / occupancy_)

                spk_t.append(spk_maze[spd_ind])
                spk_pos.append([spk_x, spk_y])

            return maps, spk_pos, spk_t

        occupancy = np.histogram2d(x, y, bins=(x_grid, y_grid))[0]
        occupancy = occupancy / trackingRate + 10e-16  # converting to seconds
        occupancy = gaussian_filter(occupancy, sigma=smooth)

        maps, spk_pos, spk_t = make_pfs(
            t, x, y, spkAll, occupancy, 0, period, x_grid, y_grid
        )

        run_occupancy = np.histogram2d(x_thresh, y_thresh, bins=(x_grid, y_grid))[0]
        run_occupancy = run_occupancy / trackingRate + 10e-16  # converting to seconds
        run_occupancy = gaussian_filter(
            run_occupancy, sigma=2
        )  # NRK todo: might need to normalize this so that total occupancy adds up to 1 here...

        run_maps, run_spk_pos, run_spk_t = make_pfs(
            t, x, y, spkAll, run_occupancy, speed_thresh, period, x_grid, y_grid
        )

        # NRK todo: might be nicer to make spk_pos, spk_t, maps, and occupancy into two separate dicts: no thresh, speed_thresh
        self.spk_pos = spk_pos
        self.spk_t = spk_t
        self.maps = maps
        self.run_spk_pos = run_spk_pos
        self.run_spk_t = run_spk_t
        self.run_maps = run_maps
        self.speed = speed
        self.x = x
        self.y = y
        self.t = t
        self.occupancy = occupancy
        self.run_occupancy = run_occupancy
        self.xgrid = x_grid
        self.ygrid = y_grid
        self.speed_thresh = speed_thresh

    def plotMap(self, speed_thresh=False, subplots=(7, 4), fignum=None):
        """Plots heatmaps of placefields with peak firing rate

        Parameters
        ----------
        speed_thresh : bool, optional
            [description], by default False
        subplots : tuple, optional
            number of cells within each figure window. If cells exceed the number of subplots, then cells are plotted in successive figure windows of same size, by default (10, 8)
        fignum : int, optional
            figure number to start from, by default None
        """

        map_use = thresh = None
        if speed_thresh:
            map_use, thresh = self.run_maps, self.speed_thresh
        elif not speed_thresh:
            map_use, thresh = self.maps, 0

        nCells = len(map_use)
        nfigures = nCells // np.prod(subplots) + 1

        if fignum is None:
            if (f := plt.get_fignums()) :
                fignum = f[-1] + 1
            else:
                fignum = 1

        figures, gs = [], []
        for fig_ind in range(nfigures):
            fig = plt.figure(fignum + fig_ind, figsize=(6, 10), clear=True)
            gs.append(GridSpec(subplots[0], subplots[1], figure=fig))
            fig.subplots_adjust(hspace=0.4)
            fig.suptitle(
                "Place maps with peak firing rate (speed_threshold = "
                + str(thresh)
                + ")"
            )
            figures.append(fig)

        for cell, pfmap in enumerate(map_use):
            ind = cell // np.prod(subplots)
            subplot_ind = cell % np.prod(subplots)
            ax1 = figures[ind].add_subplot(gs[ind][subplot_ind])
            im = ax1.pcolorfast(
                self.xgrid,
                self.ygrid,
                np.rot90(np.fliplr(pfmap)) / np.max(pfmap),
                cmap="Spectral_r",
                vmin=0,
            )  # rot90(flipud... is necessary to match plotRaw configuration.
            # max_frate =
            ax1.axis("off")
            ax1.set_title(f"{round(np.nanmax(pfmap),2)} Hz")

            # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
            # cbar = fig.colorbar(im, cax=cbar_ax)
            # cbar.set_label("firing rate (Hz)")

    def plotRaw(
        self,
        speed_thresh=False,
        subplots=(10, 8),
        fignum=None,
        alpha=0.5,
        label_cells=False,
    ):
        fig = plt.figure(fignum, figsize=(6, 10))
        gs = GridSpec(subplots[0], subplots[1], figure=fig)
        # fig.subplots_adjust(hspace=0.4)

        if not speed_thresh:
            spk_pos_use = self.spk_pos
        elif speed_thresh:
            spk_pos_use = self.run_spk_pos

        for cell, (spk_x, spk_y) in enumerate(spk_pos_use):
            ax1 = fig.add_subplot(gs[cell])
            ax1.plot(self.x, self.y, color="#d3c5c5")
            ax1.plot(spk_x, spk_y, ".r", markersize=0.8, color=[1, 0, 0, alpha])
            ax1.axis("off")
            if label_cells:
                # Put info on title
                info = self._obj.spikes.info.iloc[cell]
                ax1.set_title("Cell " + str(info["id"]))

        if speed_thresh:
            fig.suptitle(
                "Place maps for cells with their peak firing rate (with speed threshold)"
            )
        elif not speed_thresh:
            fig.suptitle(
                "Place maps for cells with their peak firing rate (no speed threshold)"
            )

    def plotRaw_v_time(self, cellind, speed_thresh=False, alpha=0.5):
        fig, ax = plt.subplots(2, 1, sharex=True)
        fig.set_size_inches([23, 9.7])

        # plot trajectories
        for a, pos, ylabel in zip(
            ax, [self.x, self.y], ["X position (cm)", "Y position (cm)"]
        ):
            a.plot(self.t, pos)
            a.set_xlabel("Time (seconds)")
            a.set_ylabel(ylabel)
            pretty_plot(a)

        # Grab correct spike times/positions
        if speed_thresh:
            spk_pos_, spk_t_ = self.run_spk_pos, self.run_spk_t
        else:
            spk_pos_, spk_t_ = self.spk_pos, self.spk_t

        # plot spikes on trajectory
        for a, pos in zip(ax, spk_pos_[cellind]):
            a.plot(spk_t_[cellind], pos, "r.", color=[1, 0, 0, alpha])

        # Put info on title
        info = self._obj.spikes.info.iloc[cellind]
        ax[0].set_title(
            "Cell "
            + str(info["id"])
            + ": q = "
            + str(info["q"])
            + ", speed_thresh="
            + str(self.speed_thresh)
        )
