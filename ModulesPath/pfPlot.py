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
    def __init__(self, basepath, **kwargs):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        self.pf1d = pf1d(basepath)
        self.pf2d = pf2d(basepath)


class pf1d:
    def __init__(self, basepath, **kwargs):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)
        self._obj.position = ExtractPosition(basepath)
        self._obj.spikes = Spikes(basepath)
        self._obj.epochs = behavior_epochs(basepath)

    def compute(self):
        trackingSRate = self._obj.position.tracking_sRate
        spks = self._obj.spikes.pyr

        xcoord = self._obj.position.x
        ycoord = self._obj.position.y

        time = self._obj.position.t

        maze = self._obj.epochs.maze

        ind_maze = np.where((time > maze[0]) & (time < maze[1]))
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

            spk_maze = cell[np.where((cell > maze[0]) & (cell < maze[1]))]
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

    def compute(self, gridbin=10, speed_thresh=10):
        """Calculate 2d placefields.  Gridbin in centimeters."""
        # ------ Cell selection ---------
        spkAll = self._obj.spikes.pyr
        # spkinfo = self._obj.spikes.info
        # pyrid = np.where(spkinfo.q < 4)[0]
        # spkAll = [spkAll[_] for _ in pyrid]

        # ----- Position---------
        xcoord = self._obj.position.x
        ycoord = self._obj.position.y
        time = self._obj.position.t
        maze = [self._obj.epochs.maze[0], self._obj.epochs.maze[1]]  # in seconds
        trackingRate = self._obj.position.tracking_sRate

        ind_maze = np.where((time > maze[0]) & (time < maze[1]))
        x = xcoord[ind_maze]
        y = ycoord[ind_maze]
        t = time[ind_maze]

        x_grid = np.arange(min(x), max(x), gridbin)
        y_grid = np.arange(min(y), max(y), gridbin)
        # x_, y_ = np.meshgrid(x_grid, y_grid)

        diff_posx = np.diff(x)
        diff_posy = np.diff(y)

        speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2) / (1 / trackingRate)
        speed = gaussian_filter1d(speed, sigma=2)
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
                spk_map = gaussian_filter(spk_map, sigma=2)
                maps.append(spk_map / occupancy_)

                spk_t.append(spk_maze[spd_ind])
                spk_pos.append([spk_x, spk_y])

            return maps, spk_pos, spk_t

        occupancy = np.histogram2d(x, y, bins=(x_grid, y_grid))[0]
        occupancy = occupancy / trackingRate + 10e-16  # converting to seconds
        occupancy = gaussian_filter(occupancy, sigma=2)

        maps, spk_pos, spk_t = make_pfs(
            t, x, y, spkAll, occupancy, 0, maze, x_grid, y_grid
        )

        run_occupancy = np.histogram2d(x_thresh, y_thresh, bins=(x_grid, y_grid))[0]
        run_occupancy = run_occupancy / trackingRate + 10e-16  # converting to seconds
        run_occupancy = gaussian_filter(
            run_occupancy, sigma=2
        )  # NRK todo: might need to normalize this so that total occupancy adds up to 1 here...

        run_maps, run_spk_pos, run_spk_t = make_pfs(
            t, x, y, spkAll, run_occupancy, speed_thresh, maze, x_grid, y_grid
        )

        # spk_pfx, spk_pfy, spk_pft = [], [], []
        # pf, spk_pos, spk_t = [], [], []
        # for cell in spkAll:
        #
        #     spk_maze = cell[np.where((cell > maze[0]) & (cell < maze[1]))]
        #     spk_speed = np.interp(spk_maze, t[1:], speed)
        #     spk_y = np.interp(spk_maze, t, y)
        #     spk_x = np.interp(spk_maze, t, x)
        #
        #     spk_map = np.histogram2d(spk_x, spk_y, bins=(x_grid, y_grid))[0]
        #     spk_map = gaussian_filter(spk_map, sigma=2)
        #     pf.append(spk_map / occupancy)
        #
        #     # speed threshold
        #     spd_ind = np.where(spk_speed > 0)
        #     spk_spd = spk_speed[spd_ind]
        #     spk_x = spk_x[spd_ind]
        #     spk_y = spk_y[spd_ind]
        #     spk_t.append(spk_maze[spd_ind])
        #     spk_pos.append([spk_x, spk_y])
        #
        #     # spk_pfx.append(spk_x)
        #     # spk_pfy.append(spk_y)
        #     # spk_pft.append(spk_t)

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
        # self.spkx = spk_pfx
        # self.spky = spk_pfy
        # self.spkt = spk_pft

    def plotMap(self, speed_thresh=False, subplots=(10, 8), fignum=None):
        plt.clf()
        fig = plt.figure(fignum, figsize=(6, 10))
        gs = GridSpec(subplots[0], subplots[1], figure=fig)
        fig.subplots_adjust(hspace=0.4)

        if speed_thresh:
            map_use, thresh = self.run_maps, self.speed_thresh
        elif not speed_thresh:
            map_use, thresh = self.maps, 0

        for cell, pfmap in enumerate(map_use):
            ax1 = fig.add_subplot(gs[cell])
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

        fig.suptitle(
            "Place maps with peak firing rate (speed_threshold = " + str(thresh) + ")"
        )

    def plotRaw(
        self,
        speed_thresh=False,
        subplots=(10, 8),
        fignum=None,
        alpha=0.5,
        label_cells=False,
        ax=None,
        clus_use=None,
    ):
        if ax is None:
            fig = plt.figure(fignum, figsize=(6, 10))
            gs = GridSpec(subplots[0], subplots[1], figure=fig)
            # fig.subplots_adjust(hspace=0.4)
        else:
            assert len(ax) == len(clus_use), "Number of axes must match number of clusters to plot"
            fig = ax[0].get_figure()



        if not speed_thresh:
            spk_pos_use = self.spk_pos
        elif speed_thresh:
            spk_pos_use = self.run_spk_pos

        if clus_use is not None:
            spk_pos_tmp = spk_pos_use
            spk_pos_use = []
            [spk_pos_use.append(spk_pos_tmp[a]) for a in clus_use]

        for cell, (spk_x, spk_y) in enumerate(spk_pos_use):
            if ax is None:
                ax1 = fig.add_subplot(gs[cell])
            else:
                ax1 = ax[cell]
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

    def plotRaw_v_time(self, cellind, speed_thresh=False, alpha=0.5, ax=None):
        if ax is None:
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
        ipbool = self._obj.spikes.pyrid[cellind] == self._obj.spikes.info.index
        info = self._obj.spikes.info.iloc[ipbool]
        ax[0].set_title(
            "Cell "
            + str(info["id"])
            + ": q = "
            + str(info["q"])
            + ", speed_thresh="
            + str(self.speed_thresh)
        )

    def plot_all(self, cellind, speed_thresh=True, alpha=0.4, fig=None):
        if fig is None:
            fig_use = plt.figure(figsize=[28.25, 11.75])
        else:
            fig_use = fig
        gs = GridSpec(2, 4, figure=fig_use)
        ax2d = fig_use.add_subplot(gs[0, 0])
        axccg = np.asarray(fig_use.add_subplot(gs[1, 0]))
        axx = fig_use.add_subplot(gs[0, 1:])
        axy = fig_use.add_subplot(gs[1, 1:], sharex=axx)

        self.plotRaw(speed_thresh=speed_thresh, clus_use=[cellind], ax=[ax2d])
        self.plotRaw_v_time(cellind, speed_thresh=speed_thresh, ax=[axx, axy], alpha=alpha)
        self._obj.spikes.plot_ccg(clus_use=[cellind], type='acg', ax=axccg)

        return fig_use

