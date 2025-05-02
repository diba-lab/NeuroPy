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
        traces_rot: np.ndarray = None, #rotation
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

        if isinstance(traces_rot,np.ndarray):
            if traces_rot.ndim == 1:
                traces_rot = traces_rot.reshape(1,-1)
            assert traces_rot.shape[0] <= 3, "Maximum possible dimension of rotation is 3"
            self.traces_rot =  traces_rot
        else:
            self.traces_rot = None

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
    def x_rot(self):
        return self.traces_rot[0]

    @x_rot.setter
    def x_rot(self, x_rot):
        self.traces_rot[0] = x_rot

    @property
    def y_rot(self):
        assert self.traces_rot.shape[0] > 1, "No y for one-dimensional rotation"
        return self.traces_rot[1]

    @y.setter
    def y_rot(self, y_rot):
        assert self.traces_rot.shape[0] > 1, "Rotation data has only one dimension"
        self.traces_rot[1] = y_rot

    @property
    def z_rot(self):
        assert self.traces_rot.shape[0] == 3, "Rotation data is not three-dimensional"
        return self.traces_rot[2]

    @z.setter
    def z_rot(self, z_rot):
        self.traces_rot[2] = z_rot

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

    @property
    def _df(self):
        return self.to_dataframe()

    def get_smoothed(self, sigma):
        dt = 1 / self.sampling_rate
        smooth = lambda x: gaussian_filter1d(x, sigma=sigma / dt, axis=-1)

        if self.traces_rot is not None:
            return Position(
                traces=smooth(self.traces),
                traces_rot=smooth(self.traces_rot),
                sampling_rate=self.sampling_rate,
                t_start=self.t_start,
        )
        else:
            return Position(
                traces=smooth(self.traces),
                sampling_rate=self.sampling_rate,
                t_start=self.t_start,
        )


    def to_dataframe(self):
        pos_dict = {"time": self.time}

        for axis in ["x", "y", "z"]:
            try:
                pos_dict[axis] = getattr(self, axis)
            except AssertionError as e:
                print(f"Skipping axis '{axis}': {e}")
                continue  # Axis doesn't exist, skip it

        try:
            pos_dict["speed"] = self.speed
        except Exception:
            pass  # Only include if computable

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
        df_time_col = 't' if 't' in df.columns else 'time' if 'time' in df.columns else None

        # Collect valid position dimensions
        trace_components = []
        axis_names = []  # To keep track of which axis corresponds to each row

        for axis in ['x', 'y', 'z']:
            if axis in df.columns:
                trace_components.append(df[axis].to_numpy())
                axis_names.append(axis)

        # If none of x/y/z, try lin
        if not trace_components and 'lin' in df.columns:
            trace_components.append(df['lin'].to_numpy())
            axis_names.append('lin')

        if not trace_components:
            raise ValueError("DataFrame must contain at least one of 'x', 'y', 'z', or 'lin' columns")

        traces = np.vstack(trace_components)

        # Warn if lin is included with x/y/z
        if 'lin' in df.columns and 'lin' not in axis_names:
            print("Note: 'lin' column is present but will be ignored since x/y/z are being used.")

        # Load metadata if present
        metadata = df.attrs.get('metadata', {})
        print(metadata)
        # Optional t_start consistency check
        if df_time_col and not np.isclose(metadata.get('t_start', t_start), df.iloc[0][df_time_col]):
            print("Warning: t_start metadata does not match time column start value.")

        sampling_rate = metadata.get('sampling_rate', sampling_rate)
        t_start = metadata.get('t_start', t_start)

        print(f"Loaded dimensions: {axis_names}")
        print("Trace shape:", traces.shape)
        print("Using sampling_rate:", sampling_rate)
        print("Using t_start:", t_start)

        return cls(traces=traces, t_start=t_start, sampling_rate=sampling_rate, metadata=metadata)

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

    def epoch_slice(self, epochs: Epoch):
        assert isinstance(epochs, Epoch), "epochs must be neuropy.Epoch object"
        t_start = epochs.starts[0]
        t_stop = epochs.stops[0]

        print(f"Slicing from {t_start} to {t_stop}")

        return self.time_slice(t_start=t_start, t_stop=t_stop)
