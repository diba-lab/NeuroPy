import numpy as np
from ..utils import mathutil
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from .epoch import Epoch
from .datawriter import DataWriter


class Position(DataWriter):
    def __init__(
        self, time, x, y=None, z=None, sampling_rate=120, filename=None
    ) -> None:

        self._time = time
        self._x = x
        self._y = y
        self._z = z
        self._sampling_rate = sampling_rate
        super().__init__(filename=filename)

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
    def t_start(self):
        return np.min(self.time)

    @property
    def t_stop(self):
        return np.max(self.time)

    @property
    def ndim(self):
        ndim = 1

        if self._y is not None:
            ndim += 1
        if self._z is not None:
            ndim += 1

        return ndim

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate

    def to_dict(self):
        data = {
            "time": self.time[1:],
            "x": self.x[1:],
            "y": self.y[1:],
            "z": self.z[1:],
            "sampling_rate": self._sampling_rate,
            "filename": self.filename,
        }
        return data

    @staticmethod
    def from_dict(d):
        time = d["time"]
        x = d["x"]
        y = d["y"]
        z = d["z"]
        sampling_rate = d["sampling_rate"]
        filename = d["filename"]

        return Position(time, x, y, z, sampling_rate, filename)

    @property
    def speed(self):
        self.speed = np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2) / (
            1 / self.sampling_rate
        )

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict)

    def speed_in_epochs(self, epochs: Epoch):
        assert isinstance(epochs, Epoch), "epochs must be neuropy.Epoch object"
        pass

    def time_slice(self, t_start, t_stop):
        pass
