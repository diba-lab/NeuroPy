import numpy as np
from ..mathutil import threshPeriods
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from .epoch import Epoch
from .datawriter import DataWriter


class Position(DataWriter):
    def __init__(
        self, time=None, x=None, y=None, z=None, tracking_srate=120, filename=None
    ) -> None:

        self._time = time
        self._x = x
        self._y = y
        self._z = z
        self._tracking_srate = tracking_srate
        super().__init__(filename=filename)
        self.linear = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._x

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

    def load(self):
        data = super().load()

        if data is not None:
            data = np.load(self.filename, allow_pickle=True).item()
            self.time, self.x, self.y, self.z, self.tracking_srate, self.linear = (
                data["time"],
                data["x"],
                data["y"],
                data["z"],
                data["tracking_srate"],
                data["linear"],
            )

    def save(self):
        data = {
            "time": self.time,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "tracking_srate": self.tracking_srate,
            "linear": self.linear,
        }
        super().save(data)

    @property
    def speed(self):
        pass

    def linearize(self, period, method="isomap"):
        pass

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
    def __init__(self, position: Position) -> None:
        self._position = position

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

        trackingSrate = self._position.time
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

        high_speed = np.around(high_speed / trackingSrate + period[0], 2)
        data = pd.DataFrame(high_speed, columns=["start", "stop"])
        # data["duration"] = np.diff(high_speed, axis=1)
        data["direction"] = np.where(val > 0, "forward", "backward")

        return data