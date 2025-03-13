import numpy as np
from ..utils import mathutil
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from .epoch import Epoch
from .datawriter import DataWriter


class Position(DataWriter):
    def __init__(
        self,
        traces: np.ndarray,
        t_start=0,
        sampling_rate=120,
        time=None,
        metadata=None,
    ) -> None:
        if traces.ndim == 1:
            traces = traces.reshape(1, -1)

        assert traces.shape[0] <= 3, "Maximum possible dimension of position is 3"
        self.traces = traces
        self._t_start = t_start
        self._sampling_rate = sampling_rate
        super().__init__(metadata=metadata)

    @property
    def x(self):
        return self.traces[0]

    @x.setter
    def x(self, x):
        self.traces[0] = x

    @property
    def y(self):
        assert self.ndim > 1, "No y for one-dimensional position"
        return self.traces[1]

    @y.setter
    def y(self, y):
        assert self.ndim > 1, "Position data has only one dimension"
        self.traces[1] = y

    @property
    def z(self):
        assert self.ndim == 3, "Position data is not three-dimensional"
        return self.traces[2]

    @z.setter
    def z(self, z):
        self.traces[2] = z

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, t):
        self._t_start = t

    @property
    def n_frames(self):
        return self.traces.shape[1]

    @property
    def duration(self):
        return self.n_frames / self.sampling_rate

    @property
    def t_stop(self):
        return self.time[-1]

    @property
    def time(self):
        # return np.linspace(self.t_start, self.t_stop, self.n_frames)
        return np.arange(self.n_frames) * (1 / self.sampling_rate) + self.t_start

    @property
    def ndim(self):
        return self.traces.shape[0]

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate

    @property
    def speed(self):
        dt = 1 / self.sampling_rate
        speed = np.sqrt(((np.abs(np.diff(self.traces, axis=1))) ** 2).sum(axis=0)) / dt
        return np.hstack(([0], speed))

    def get_smoothed(self, sigma):
        dt = 1 / self.sampling_rate
        smooth = lambda x: gaussian_filter1d(x, sigma=sigma / dt, axis=-1)
        return Position(
            traces=smooth(self.traces),
            sampling_rate=self.sampling_rate,
            t_start=self.t_start,
        )

    def to_dataframe(self):
        pos_dict = {"time": self.time}

        # Only add x, y, and z if they exist
        for axis in ["x", "y", "z"]:
            if hasattr(self, axis):
                pos_dict[axis] = getattr(self, axis)

        # Include speed if it exists
        if hasattr(self, "speed"):
            pos_dict["speed"] = self.speed

        # Create the DataFrame with the available data
        position_df = pd.DataFrame(pos_dict)

        # Prepare metadata using available attributes
        metadata = {}
        for key in ["t_start", "t_stop", "sampling_rate"]:
            if hasattr(self, key):
                metadata[key] = getattr(self, key)
        position_df.attrs["metadata"] = metadata

        return position_df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, sampling_rate: float = 120, t_start: float = 0):
        # Build traces from DataFrame columns.
        # Check which coordinate columns exist in case only 1 or 2 dimensions were input.
        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            # Create 2D traces array with shape (number of dimensions, number of samples)
            traces = np.vstack((df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()))
        elif 'x' in df.columns and 'y' in df.columns:
            traces = np.vstack((df['x'].to_numpy(), df['y'].to_numpy()))
        elif 'x' in df.columns and 'z' in df.columns:
            traces = np.vstack((df['x'].to_numpy(), df['z'].to_numpy()))
        elif 'x' in df.columns:
            # Only x column is available; the constructor will reshape it to (1, n_samples)
            traces = df['x'].to_numpy()
        else:
            raise ValueError("DataFrame must contain at least an 'x' column")

        # Pass along metadata if available.
        metadata = df.attrs.get('metadata', {})
        # Use metadata values if they exist, otherwise fall back to the provided defaults.
        sampling_rate = metadata.get('sampling_rate', sampling_rate)
        t_start = metadata.get('t_start', t_start)

        print("Traces samples:", traces.shape[1])
        print("Using sampling_rate:", sampling_rate)
        print("Using t_start:", t_start)

        return cls(traces, t_start=t_start, sampling_rate=sampling_rate, metadata=metadata)

    def speed_in_epochs(self, epochs: Epoch):
        assert isinstance(epochs, Epoch), "epochs must be neuropy.Epoch object"
        pass

    def time_slice(self, t_start, t_stop, zero_times=False):
        indices = super()._time_slice_params(t_start, t_stop)
        if zero_times:
            t_stop = t_stop - t_start
            t_start = 0

        return Position(
            traces=self.traces[:, indices],
            t_start=t_start,
            sampling_rate=self.sampling_rate,
        )
