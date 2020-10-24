import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.decomposition import PCA
from plotUtil import Colormap
from parsePath import Recinfo


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

    def compute(self, gridbin=10):

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
        x_, y_ = np.meshgrid(x_grid, y_grid)

        diff_posx = np.diff(x)
        diff_posy = np.diff(y)

        speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2) / (1 / trackingRate)
        speed = gaussian_filter1d(speed, sigma=2)
        print(np.ptp(speed))
        dt = t[1] - t[0]
        speed_thresh = np.where(speed / dt > 0)[0]

        x_thresh = x[speed_thresh]
        y_thresh = y[speed_thresh]
        t_thresh = t[speed_thresh]

        occupancy = np.histogram2d(x, y, bins=(x_grid, y_grid))[0]
        occupancy = occupancy / trackingRate + 10e-16  # converting to seconds
        occupancy = gaussian_filter(occupancy, sigma=2)

        # spk_pfx, spk_pfy, spk_pft = [], [], []
        pf, spk_pos = [], []
        for cell in spkAll:

            spk_maze = cell[np.where((cell > maze[0]) & (cell < maze[1]))]
            spk_speed = np.interp(spk_maze, t[1:], speed)
            spk_y = np.interp(spk_maze, t, y)
            spk_x = np.interp(spk_maze, t, x)

            spk_map = np.histogram2d(spk_x, spk_y, bins=(x_grid, y_grid))[0]
            spk_map = gaussian_filter(spk_map, sigma=2)
            pf.append(spk_map / occupancy)

            # speed threshold
            spd_ind = np.where(spk_speed > 0)
            spk_spd = spk_speed[spd_ind]
            spk_x = spk_x[spd_ind]
            spk_y = spk_y[spd_ind]
            spk_t = spk_maze[spd_ind]
            spk_pos.append([spk_x, spk_y])

            # spk_pfx.append(spk_x)
            # spk_pfy.append(spk_y)
            # spk_pft.append(spk_t)

        self.spk_pos = spk_pos
        self.maps = pf
        self.speed = speed
        self.x = x
        self.y = y
        self.occupancy = occupancy
        self.xgrid = x_grid
        self.ygrid = y_grid
        # self.spkx = spk_pfx
        # self.spky = spk_pfy
        # self.spkt = spk_pft

    def plotMap(self):
        plt.clf()
        fig = plt.figure(1, figsize=(6, 10))
        gs = GridSpec(10, 8, figure=fig)
        fig.subplots_adjust(hspace=0.4)

        for cell, pfmap in enumerate(self.maps):
            ax1 = fig.add_subplot(gs[cell])
            im = ax1.pcolorfast(
                self.xgrid, self.ygrid, pfmap / np.max(pfmap), cmap="Spectral_r", vmin=0
            )
            # max_frate =
            ax1.axis("off")
            ax1.set_title(f"{round(np.nanmax(pfmap),2)} Hz")

        # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
        # cbar = fig.colorbar(im, cax=cbar_ax)
        # cbar.set_label("firing rate (Hz)")

        fig.suptitle(
            "Place maps for cells with their peak firing rate (no speed threshold)"
        )

    def plotRaw(self):
        fig = plt.figure(1, figsize=(6, 10))
        gs = GridSpec(10, 8, figure=fig)
        # fig.subplots_adjust(hspace=0.4)

        for cell, (spk_x, spk_y) in enumerate(self.spk_pos):
            ax1 = fig.add_subplot(gs[cell])
            ax1.plot(self.x, self.y, color="#d3c5c5")
            ax1.plot(spk_x, spk_y, ".r", markersize=0.8)
            ax1.axis("off")
