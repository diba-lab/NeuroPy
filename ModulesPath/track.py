from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from mathutil import threshPeriods
from behavior import behavior_epochs
from getPosition import ExtractPosition
from parsePath import Recinfo
from scipy.ndimage import gaussian_filter1d


class Track:
    def __init__(self, basepath: Recinfo) -> None:
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ------- defining file names ---------
        filePrefix = self._obj.files.filePrefix

        @dataclass
        class files:
            trackinfo: str = filePrefix.with_suffix(".tracks.info.npy")
            laps: str = filePrefix.with_suffix(".tracks.laps.npy")

        self.files = files()
        self.names = None

        self._load()

    def _load(self):
        if (f := self.files.trackinfo).is_file():
            tracks = np.load(f, allow_pickle=True).item()
            self.names = list(tracks.keys())
            self.data = tracks

        if (f := self.files.laps).is_file():
            lapdata = np.load(f, allow_pickle=True).item()
            self.laps = lapdata

    def create(self, epoch_names):

        position = ExtractPosition(self._obj)
        assert hasattr(position, "x"), "First extract position"
        epochs = behavior_epochs(self._obj)
        periods = None
        if isinstance(epoch_names, str):
            periods = [epochs.times[epoch_names].to_list()]
            epoch_names = [epoch_names]
        elif all(isinstance(name, str) for name in epoch_names):
            periods = [epochs.times[_].to_list() for _ in epoch_names]

        posdata = position.data

        maze_data = {}
        for name, epch in zip(epoch_names, periods):
            maze_data[name] = posdata[
                (posdata.time > epch[0]) & (posdata.time < epch[1])
            ].reset_index(drop=True)

        np.save(self.files.trackinfo, maze_data)
        self._load()

    def __getitem__(self, track_name):
        return self.data[track_name]

    def __len__(self):
        return len(self.data)

    def linearize_position(
        self, track_names=None, sample_sec=3, method="isomap", plot=True
    ):
        """linearize trajectory. Use method='PCA' for off-angle linear track, method='ISOMAP' for any non-linear track.
        ISOMAP is more versatile but also more computationally expensive.

        Parameters
        ----------
        track_names: list of track names, each must match an epoch in epochs class.
        sample_sec : int, optional
            sample a point every sample_sec seconds for training ISOMAP, by default 3. Lower it if inaccurate results
        method : str, optional
            by default 'ISOMAP' (for any continuous track, untested on t-maze as of 12/22/2020) or
            'PCA' (for straight tracks)

        """
        posinfo = ExtractPosition(self._obj)
        tracking_sRate = posinfo.tracking_sRate

        if track_names is None:
            track_names = self.names

        # ---- loading the data ----------
        alldata = np.load(self.files.trackinfo, allow_pickle=True).item()

        for name in track_names:
            xpos = alldata[name].x
            ypos = alldata[name].y
            position = np.vstack((xpos, ypos)).T
            xlinear = None
            if method == "pca":
                pca = PCA(n_components=1)
                xlinear = pca.fit_transform(position).squeeze()
            elif method == "isomap":
                imap = Isomap(n_neighbors=5, n_components=2)
                # downsample points to reduce memory load and time
                pos_ds = position[0 : -1 : np.round(int(tracking_sRate) * sample_sec)]
                imap.fit(pos_ds)
                iso_pos = imap.transform(position)
                # Keep iso_pos here in case we want to use 2nd dimension (transverse to track) in future...
                if iso_pos.std(axis=0)[0] < iso_pos.std(axis=0)[1]:
                    iso_pos[:, [0, 1]] = iso_pos[:, [1, 0]]
                xlinear = iso_pos[:, 0]
            if plot:
                fig, ax = plt.subplots()
                fig.set_size_inches([28, 8.6])
                ax.plot(xlinear)
                ax.set_xlabel("Frame #")
                ax.set_ylabel("Linear Position")
                ax.set_title(method.upper() + " Sanity Check Plot")

            alldata[name]["linear"] = xlinear

        # ---- saving the updated data -----------
        np.save(self.files.trackinfo, alldata)

        self._load()

    def plot(self, track_names=None, linear=False):
        """track_names: list of tracks
        linear: boolean to plot 2d (False, default) or linear (True)"""

        if track_names is None:
            track_name = self.names

        _, ax = plt.subplots(1, len(track_names), squeeze=False)
        ax = ax.reshape(-1)

        for ind, name in enumerate(track_names):
            posdata = self[name]
            if not linear:
                ax[ind].plot(posdata.x, posdata.y)
            elif linear:
                ax[ind].plot(posdata.time, posdata.linear)
                ax[ind].set_xlabel("Time (s)")
                ax[ind].set_ylabel("Linear Position (cm)")
            ax[ind].set_title(name)

    def estimate_run_laps(
        self,
        track_name,
        speedthresh=(10, 20),
        merge_dur=2,
        min_dur=2,
        smooth_speed=50,
        min_dist=50,
        plot=True,
    ):
        """Divide running epochs into forward and backward

        Parameters
        ----------
        track_name : str
            name of track
        speedthresh : tuple, optional
            low and high speed threshold for speed, by default (10, 20)
        merge_dur : int, optional
            two epochs if less than merge_dur (seconds) apart they will be merged , by default 2 seconds
        min_dur : int, optional
            minimum duration of a run epoch, by default 2 seconds
        smooth_speed : int, optional
            speed is smoothed, increase if epochs are fragmented, by default 50
        min_dist : int, optional
            the animal should cover this much distance in one direction within the lap to be included, by default 50
        plot : bool, optional
            plots the epochs with position and speed data, by default True
        """

        trackingSrate = ExtractPosition(self._obj).tracking_sRate
        track_period = behavior_epochs(self._obj)[track_name]
        posdata = self[track_name]
        x = posdata.linear
        time = posdata.time
        speed = gaussian_filter1d(posdata.speed, sigma=smooth_speed)

        high_speed = threshPeriods(
            speed,
            lowthresh=speedthresh[0],
            highthresh=speedthresh[1],
            minDistance=merge_dur * trackingSrate,
            minDuration=min_dur * trackingSrate,
        )
        val = []
        for epoch in high_speed:
            displacement = x[epoch[1]] - x[epoch[0]]
            # distance = np.abs(np.diff(x[epoch[0] : epoch[1]])).sum()

            if np.abs(displacement) > min_dist:
                if displacement < 0:
                    val.append(-1)
                elif displacement > 0:
                    val.append(1)
            else:
                val.append(0)
        val = np.asarray(val)

        # ---- deleting epochs where animal ran a little distance------
        high_speed = np.delete(high_speed, np.where(val == 0)[0], axis=0)
        val = np.delete(val, np.where(val == 0)[0])

        high_speed = np.around(high_speed / trackingSrate + track_period[0], 2)
        data = pd.DataFrame(high_speed, columns=["start", "end"])
        data["duration"] = np.diff(high_speed, axis=1)
        data["direction"] = np.where(val > 0, "forward", "backward")

        if plot:
            _, axall = plt.subplots(2, 1, sharex=True)

            # ---- position and epoch plot ----------
            ax = axall[0]
            ax.plot(time, x, color="gray")
            for epoch in data.itertuples():
                if epoch.direction == "forward":
                    color = "#3eccc7"
                else:
                    color = "#ff928a"
                ax.axvspan(
                    epoch.start,
                    epoch.end,
                    ymax=np.ptp(x),
                    facecolor=color,
                    alpha=0.7,
                )
            ax.set_ylabel("Linear cordinates")

            # ----- velocity plot ----------
            ax = axall[1]
            ax.plot(time, speed, color="gray")
            ax.axhline(y=speedthresh[0], ls="--", color="r", label="lower speed limit")
            ax.axhline(y=speedthresh[1], ls="--", color="g", label="upper speed limit")
            ax.set_ylabel("speed (cm/s)")
            ax.set_xlabel("Time (s)")
            ax.legend()

        alldata = {}
        if (f := self.files.laps).is_file():
            alldata = np.load(f, allow_pickle=True).item()

        alldata[track_name] = data
        np.save(self.files.laps, alldata)
        self._load()

    def get_laps(self, track_name):
        return self.laps[track_name]

    def plot_laps(self, track_name: str, ax=None):
        """Plots run epochs for the track

        Parameters
        ----------
        track_name : str
            name of the track
        ax : axis object, optional
            axis to plot onto, by default None

        Returns
        -------
        axis
            [description]
        """

        track = self[track_name]
        x = track.linear
        time = track.time
        data = self.laps[track_name]

        if ax is None:
            _, ax = plt.subplots(1, 1)

        # ---- position and epoch plot ----------
        ax.plot(time, x, color="gray")
        for epoch in data.itertuples():
            if epoch.direction == "forward":
                color = "#3eccc7"
            else:
                color = "#ff928a"
            ax.axvspan(
                epoch.start,
                epoch.end,
                ymax=np.ptp(x),
                facecolor=color,
                alpha=0.7,
            )
        ax.set_ylabel("Linear cordinates")
        ax.set_title(f"{track_name} linear position with run epochs")

        return ax
