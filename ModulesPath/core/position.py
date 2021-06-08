import numpy as np
from ..mathutil import threshPeriods
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from .epoch import Epoch
from .datawriter import DataWriter
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


class Position(DataWriter):
    def __init__(
        self, time=None, x=None, y=None, z=None, sampling_rate=120, filename=None
    ) -> None:

        self._time = time
        self._x = x
        self._y = y
        self._z = z
        self._sampling_rate = sampling_rate
        super().__init__(filename=filename)

        if self._x is not None:
            self.linear = np.nan * np.zeros(len(self._x))
        else:
            self.linear = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, z):
        self._z = z

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        self._time = time

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate

    def load(self):
        data = super().load()

        if data is not None:
            data = np.load(self.filename, allow_pickle=True).item()
            self.time, self.x, self.y, self.z, self.sampling_rate, self.linear = (
                data["time"],
                data["x"],
                data["y"],
                data["z"],
                data["sampling_rate"],
                data["linear"],
            )

    def save(self):
        data = {
            "time": self.time,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "sampling_rate": self.sampling_rate,
            "linear": self.linear,
        }
        super().save(data)

    @property
    def speed(self):
        self.speed = np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2) / (
            1 / self.sampling_rate
        )

    def linearize(self, period=None, sample_sec=3, method="isomap"):
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

        indices = (self.time > period[0]) & (self.time < period[1])
        xpos = self.x[indices]
        ypos = self.y[indices]

        position = np.vstack((xpos, ypos)).T
        xlinear = None
        if method == "pca":
            pca = PCA(n_components=1)
            xlinear = pca.fit_transform(position).squeeze()
        elif method == "isomap":
            imap = Isomap(n_neighbors=5, n_components=2)
            # downsample points to reduce memory load and time
            pos_ds = position[0 : -1 : np.round(int(self.sampling_rate) * sample_sec)]
            imap.fit(pos_ds)
            iso_pos = imap.transform(position)
            # Keep iso_pos here in case we want to use 2nd dimension (transverse to track) in future...
            if iso_pos.std(axis=0)[0] < iso_pos.std(axis=0)[1]:
                iso_pos[:, [0, 1]] = iso_pos[:, [1, 0]]
            xlinear = iso_pos[:, 0]
        self.linear[indices] = xlinear - np.min(xlinear)

    def to_dataframe(self):
        self.data = pd.DataFrame(
            {
                "time": self.time[1:],
                "x": self.x[1:],
                "y": self.y[1:],
                "z": self.z[1:],
                "speed": self.speed,
                "linear": self.linear[1:],
            }
        )


class Track:
    def __init__(self, position: Position, **kwargs) -> None:
        self._position = position
        super().__init__(**kwargs)

    def calculate_run_epochs(
        self,
        period,
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

        trackingsampling_rate = self._position.time
        posdata = self._position.to_dataframe()

        posdata = posdata[(posdata.time > period[0]) & (posdata.time < period[1])]
        x = posdata.linear
        time = posdata.time
        speed = posdata.speed
        speed = gaussian_filter1d(posdata.speed, sigma=smooth_speed)

        high_speed = threshPeriods(
            speed,
            lowthresh=speedthresh[0],
            highthresh=speedthresh[1],
            minDistance=merge_dur * trackingsampling_rate,
            minDuration=min_dur * trackingsampling_rate,
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

        high_speed = np.around(high_speed / trackingsampling_rate + period[0], 2)
        data = pd.DataFrame(high_speed, columns=["start", "stop"])
        # data["duration"] = np.diff(high_speed, axis=1)
        data["direction"] = np.where(val > 0, "forward", "backward")

        self.epochs = run_epochs

        return run_epochs