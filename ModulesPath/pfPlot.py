from dataclasses import dataclass

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from getPosition import ExtractPosition
from getSpikes import Spikes
from parsePath import Recinfo
from plotUtil import Colormap, Fig, pretty_plot
from track import Track
from signal_process import ThetaParams


class pf:
    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        self.pf1d = pf1d(basepath)
        self.pf2d = pf2d(basepath)


class pf1d:
    """1D place field computation

    Attributes
    ----------
    x : array_like
        position used for place field calculation
    thresh : dict
        place maps with speed threshold
    no_thresh : dict
        place maps without speed threshold

    """

    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ------- defining file names ---------
        filePrefix = self._obj.files.filePrefix

        @dataclass
        class files:
            placemap: str = filePrefix.with_suffix(".pf1d.npy")

        self.files = files()
        self.thresh = []
        self.no_thresh = []

    # def _load(self):
    #     if (f := self.files.placemap) :
    #         self.ratemap = np.load(f, allow_pickle=True)

    def compute(self, track_name, run_dir=None, grid_bin=5, speed_thresh=5, smooth=1):
        """computes 1d place field using linearized coordinates. It always computes two place maps with and
        without speed thresholds.

        Parameters
        ----------
        track_name : str
            name of track
        direction : forward, backward or None
            direction of running, by default None which means direction is ignored
        grid_bin : int
            bin size of position bining, by default 5
        speed_thresh : int
            speed threshold for calculating place field
        """

        tracks = Track(self._obj)
        assert track_name in tracks.names, f"{track_name} doesn't exist"
        spikes = Spikes(self._obj)
        trackingSRate = ExtractPosition(self._obj).tracking_sRate

        maze = tracks.data[track_name]
        assert (
            "linear" in maze
        ), f"Track {track_name} doesn't have linearized coordinates. First run tracks.linearize_position(track_name='{track_name}')"

        x = maze.linear
        speed = maze.speed
        t = maze.time
        period = [np.min(t), np.max(t)]
        spks = spikes.pyr
        xbin = np.arange(min(x), max(x), grid_bin)  # binning of x position

        # ------ if direction then restrict to those epochs --------
        if run_dir in ["forward", "backward"]:
            print(f" using {run_dir} running only")
            run_epochs = tracks.get_laps(track_name)
            run_epochs = run_epochs[run_epochs["direction"] == run_dir]
            spks = [
                np.concatenate(
                    [
                        cell[(cell > epc.start) & (cell < epc.end)]
                        for epc in run_epochs.itertuples()
                    ]
                )
                for cell in spks
            ]
            # changing x, speed, time to only run epochs so occupancy map is consistent with that
            indx = np.concatenate(
                [
                    np.where((t > epc.start) & (t < epc.end))[0]
                    for epc in run_epochs.itertuples()
                ]
            )
            x = x[indx]
            speed = speed[indx]
            t = t[indx]
        else:
            spks = [cell[(cell > period[0]) & (cell < period[1])] for cell in spks]

        # --- occupancy with no speed threshold ---------
        # NOTE: x is changed if direction is taken into account
        occupancy = np.histogram(x, bins=xbin)[0] / trackingSRate + 1e-16
        occupancy = gaussian_filter1d(occupancy, sigma=smooth)

        # --- speed thresh occupancy----
        x_speed = x[speed > speed_thresh]
        run_occupancy = np.histogram(x_speed, bins=xbin)[0] / trackingSRate + 1e-16
        run_occupancy = gaussian_filter1d(run_occupancy, sigma=smooth)

        run_spk_pos, run_spk_t, spk_pos, spk_t = [], [], [], []
        ratemap, run_ratemap = [], []
        for cell in spks:
            spk_spd = np.interp(cell, t, speed)
            spk_x = np.interp(cell, t, x)

            spk_pos.append(spk_x)
            spk_t.append(cell)

            # speed threshold
            spd_ind = np.where(spk_spd > speed_thresh)[0]
            spk_x_spd = spk_x[spd_ind]
            run_spk_pos.append(spk_x_spd)
            run_spk_t.append(cell[spd_ind])

            # ratemap calculation
            ratemap_ = (
                gaussian_filter1d(np.histogram(spk_x, bins=xbin)[0], sigma=smooth)
                / occupancy
            )

            run_ratemap_ = (
                gaussian_filter1d(np.histogram(spk_x_spd, bins=xbin)[0], sigma=smooth)
                / run_occupancy
            )
            ratemap.append(ratemap_)
            run_ratemap.append(run_ratemap_)

        self.no_thresh = {
            "pos": spk_pos,
            "spikes": spk_t,
            "ratemaps": ratemap,
            "occupancy": occupancy,
        }

        self.thresh = {
            "pos": run_spk_pos,
            "spikes": run_spk_t,
            "ratemaps": run_ratemap,
            "occupancy": run_occupancy,
        }

        self.speed = maze.speed
        self.x = maze.linear
        self.t = maze.time
        self.bin = xbin
        self.track_name = track_name
        self.period = period
        self.run_dir = run_dir

    def phase_precession(self, theta_chan):
        """Calculates phase of spikes computed for placefields

        Parameters
        ----------
        theta_chan : int
            lfp channel to use for calculating theta phases
        """
        lfpmaze = self._obj.geteeg(chans=theta_chan, timeRange=self.period)
        lfpt = np.linspace(self.period[0], self.period[1], len(lfpmaze))
        thetaparam = ThetaParams(lfpmaze, fs=self._obj.lfpSrate)
        # phase_bin = np.linspace(0, 360, 37)

        phase, run_phase = [], []

        for spk, run_spk in zip(self.no_thresh["spikes"], self.thresh["spikes"]):
            spk_phase = np.interp(spk, lfpt, thetaparam.angle)
            run_spk_phase = np.interp(run_spk, lfpt, thetaparam.angle)

            phase.append(spk_phase)
            run_phase.append(run_spk_phase)
            # precess = np.histogram2d(spk_pos, spk_phase, bins=[pos_bin, phase_bin])[0]
            # precess = gaussian_filter(precess, sigma=1)

        self.no_thresh["phases"] = phase
        self.thresh["phases"] = run_phase

    def plot_with_phase(
        self,
        ax=None,
        speed_thresh=False,
        normalize=True,
        stack=True,
        cmap="tab20b",
        subplots=(5, 8),
    ):
        cmap = mpl.cm.get_cmap(cmap)

        mapinfo = self.no_thresh
        if speed_thresh:
            mapinfo = self.thresh

        ratemaps = mapinfo["ratemaps"]
        if normalize:
            ratemaps = [map_ / np.max(map_) for map_ in ratemaps]
        phases = mapinfo["phases"]
        position = mapinfo["pos"]
        nCells = len(ratemaps)
        bin_cntr = self.bin[:-1] + np.diff(self.bin).mean() / 2

        def plot_(cell, ax, axphase):
            color = cmap(cell / nCells)
            if subplots is None:
                ax.clear()
                axphase.clear()
            ax.fill_between(bin_cntr, 0, ratemaps[cell], color=color, alpha=0.3)
            ax.plot(bin_cntr, ratemaps[cell], color=color, alpha=0.2)
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Normalized frate")
            ax.set_title(
                " ".join(filter(None, ("Cell", str(cell), self.run_dir.capitalize())))
            )
            if normalize:
                ax.set_ylim([0, 1])
            axphase.scatter(position[cell], phases[cell], c="k", s=0.6)
            if stack:  # double up y-axis as is convention for phase precession plots
                axphase.scatter(position[cell], phases[cell] + 360, c="k", s=0.6)
            axphase.set_ylabel(r"$\theta$ Phase")

        if ax is None:

            if subplots is None:
                _, gs = Fig().draw(grid=(1, 1), size=(10, 5))
                ax = plt.subplot(gs[0])
                ax.spines["right"].set_visible(True)
                axphase = ax.twinx()
                widgets.interact(
                    plot_,
                    cell=widgets.IntSlider(
                        min=0,
                        max=nCells - 1,
                        step=1,
                        description="Cell ID:",
                    ),
                    ax=widgets.fixed(ax),
                    axphase=widgets.fixed(axphase),
                )
            else:
                _, gs = Fig().draw(grid=subplots, size=(15, 10))
                for cell in range(nCells):
                    ax = plt.subplot(gs[cell])
                    axphase = ax.twinx()
                    plot_(cell, ax, axphase)

        return ax

    def plot(
        self,
        ax=None,
        speed_thresh=False,
        pad=2,
        normalize=False,
        sortby=None,
        cmap="tab20b",
    ):
        """Plot 1D place fields stacked

        Parameters
        ----------
        ax : [type], optional
            [description], by default None
        speed_thresh : bool, optional
            [description], by default False
        pad : int, optional
            [description], by default 2
        normalize : bool, optional
            [description], by default False
        sortby : bool, optional
            [description], by default True
        cmap : str, optional
            [description], by default "tab20b"

        Returns
        -------
        [type]
            [description]
        """
        cmap = mpl.cm.get_cmap(cmap)

        mapinfo = self.no_thresh
        if speed_thresh:
            mapinfo = self.thresh

        ratemaps = mapinfo["ratemaps"]
        nCells = len(ratemaps)
        bin_cntr = self.bin[:-1] + np.diff(self.bin).mean() / 2

        if ax is None:
            _, gs = Fig().draw(grid=(1, 1), size=(4.5, 11))
            ax = plt.subplot(gs[0])

        if normalize:
            ratemaps = [_ / np.max(_) for _ in ratemaps]
            pad = 1

        if sortby is None:
            sort_ind = np.argsort(np.argmax(np.asarray(ratemaps), axis=1))
        elif isinstance(sortby, (list, np.ndarray)):
            sort_ind = sortby
        else:
            sort_ind = np.arange(len(ratemaps))
        for cellid, cell in enumerate(sort_ind):
            color = cmap(cellid / nCells)

            ax.fill_between(
                bin_cntr,
                cellid * pad,
                cellid * pad + ratemaps[cell],
                color=color,
                ec=None,
                alpha=0.5,
                zorder=cellid + 1,
            )
            ax.plot(
                bin_cntr,
                cellid * pad + ratemaps[cell],
                color=color,
                alpha=0.7,
            )

        ax.set_yticks(list(range(nCells)))
        ax.set_yticklabels(list(sort_ind))
        ax.set_xlabel("Position")
        ax.spines["left"].set_visible(False)
        ax.set_xlim([np.min(bin_cntr), np.max(bin_cntr)])
        ax.tick_params("y", length=0)
        ax.set_ylim([0, nCells])
        if self.run_dir is not None:
            ax.set_title(self.run_dir.capitalize() + " Runs only")

        return ax

    def plot_raw(self, speed_thresh=False, ax=None, subplots=(8, 9)):
        """Plot spike location on animal's path

        Parameters
        ----------
        speed_thresh : bool, optional
            [description], by default False
        ax : [type], optional
            [description], by default None
        subplots : tuple, optional
            [description], by default (8, 9)
        """

        mapinfo = self.no_thresh
        if speed_thresh:
            mapinfo = self.thresh
        nCells = len(mapinfo["pos"])

        def plot_(cell, ax):
            if subplots is None:
                ax.clear()
            ax.plot(self.x, self.t, color="gray", alpha=0.6)
            ax.plot(mapinfo["pos"][cell], mapinfo["spikes"][cell], ".", color="#ff5f5c")
            ax.set_title(
                " ".join(filter(None, ("Cell", str(cell), self.run_dir.capitalize())))
            )
            ax.invert_yaxis()
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Time (s)")

        if ax is None:

            if subplots is None:
                _, gs = Fig().draw(grid=(1, 1), size=(6, 8))
                ax = plt.subplot(gs[0])
                widgets.interact(
                    plot_,
                    cell=widgets.IntSlider(
                        min=0,
                        max=nCells - 1,
                        step=1,
                        description="Cell ID:",
                    ),
                    ax=widgets.fixed(ax),
                )
            else:
                _, gs = Fig().draw(grid=subplots, size=(10, 11))
                for cell in range(nCells):
                    ax = plt.subplot(gs[cell])
                    ax.set_yticks([])
                    plot_(cell, ax)


class pf2d:
    def __init__(self, basepath, **kwargs):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def compute(self, period, gridbin=10, speed_thresh=5, smooth=2):
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
        ax=None,
        clus_use=None,
    ):
        if ax is None:
            fig = plt.figure(fignum, figsize=(6, 10))
            gs = GridSpec(subplots[0], subplots[1], figure=fig)
            # fig.subplots_adjust(hspace=0.4)
        else:
            assert len(ax) == len(
                clus_use
            ), "Number of axes must match number of clusters to plot"
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
        self.plotRaw_v_time(
            cellind, speed_thresh=speed_thresh, ax=[axx, axy], alpha=alpha
        )
        self._obj.spikes.plot_ccg(clus_use=[cellind], type="acg", ax=axccg)

        return fig_use
