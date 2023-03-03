from typing import Sequence, Union
import numpy as np
from neuropy.utils import mathutil
import pandas as pd
from scipy.ndimage import gaussian_filter1d


from .epoch import Epoch
from .signal import Signal
from .datawriter import DataWriter
from neuropy.utils.load_exported import import_mat_file 
from neuropy.utils.mixins.time_slicing import TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from neuropy.utils.mixins.concatenatable import ConcatenationInitializable


class PositionOld(ConcatenationInitializable, TimeSlicableIndiciesMixin, TimeSlicableObjectProtocol, DataWriter):
    def __init__(
        self,
        traces: np.ndarray,
        computed_traces: np.ndarray=None,
        t_start=0,
        sampling_rate=120,
        metadata=None,
    ) -> None:

        if traces.ndim == 1:
            traces = traces.reshape(1, -1)

        assert traces.shape[0] <= 3, "Maximum possible dimension of position is 3"
        self.traces = traces
        self.computed_traces = computed_traces
        self._t_start = t_start
        self._sampling_rate = sampling_rate
        self.metadata = metadata
        super().__init__()

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
        assert self.ndim > 1, "PositionOld data has only one dimension"
        self.traces[1] = y

    @property
    def z(self):
        assert self.ndim == 3, "PositionOld data is not three-dimensional"
        return self.traces[2]

    @z.setter
    def z(self, z):
        self.traces[2] = z

    @property
    def linear_pos_obj(self):
        # returns a PositionOld object containing only the linear_pos as its trace. This is used for compatibility with Bapun's Pf1D function 
        return PositionOld(
            traces=self.linear_pos,
            computed_traces=self.linear_pos,
            t_start=self.t_start,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata,
        )


    @property
    def linear_pos(self):
        assert self.computed_traces.shape[0] >= 1, "Linear PositionOld data has not yet been computed."
        return self.computed_traces[0]

    @linear_pos.setter
    def linear_pos(self, linear_pos):
        self.computed_traces[0] = linear_pos

    @property
    def has_linear_pos(self):
        if (self.computed_traces.shape[0] >= 1):
            return not np.isnan(self.computed_traces[0]).all() # check if all are nan
        else:
            # Linear PositionOld data has not yet been computed.
            return False

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
        return float(self.n_frames) / float(self.sampling_rate)

    @property
    def t_stop(self):
        return self.t_start + self.duration

    @property
    def time(self):
        return np.linspace(self.t_start, self.t_stop, self.n_frames)

    @property
    def ndim(self):
        return self.traces.shape[0]

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate

    def to_dict(self, recurrsively=False):
        data = {
            "traces": self.traces,
            "computed_traces": self.computed_traces,
            "t_start": self.t_start,
            "sampling_rate": self._sampling_rate,
            "metadata": self.metadata,
        }
        return data

    @staticmethod
    def from_dict(d):
        return PositionOld(
            traces=d["traces"],
            computed_traces=d.get('computed_traces', np.full([1, d["traces"].shape[1]], np.nan)),
            t_start=d["t_start"],
            sampling_rate=d["sampling_rate"],
            metadata=d["metadata"],
        )
            
    @property
    def speed(self):
        dt = 1 / self.sampling_rate
        return np.insert((np.sqrt(((np.abs(np.diff(self.traces, axis=1))) ** 2).sum(axis=0)) / dt), 0, 0.0) # prepends a 0.0 value to the front of the result array so it's the same length as the other position vectors (x, y, etc)
    

    def to_dataframe(self):
        df = pd.DataFrame({"t": self.time.flatten().copy(), "x": self.x.flatten().copy()})
        if self.has_linear_pos:
            df["lin_pos"] = self.linear_pos.flatten().copy()
        if self.ndim >= 2:
            df["y"] = self.y.flatten().copy()
        if self.ndim >= 3:
            df["z"] = self.z.flatten().copy()
        df["speed"] = self.speed.flatten()
        return df


    def speed_in_epochs(self, epochs: Epoch):
        assert isinstance(epochs, Epoch), "epochs must be neuropy.Epoch object"
        pass

    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop):
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        indices = self.time_slice_indicies(t_start, t_stop) # from TimeSlicableIndiciesMixin
        return PositionOld(
            traces=self.traces[:, indices],
            computed_traces=self.computed_traces[:, indices],
            t_start=t_start,
            sampling_rate=self.sampling_rate,
        )

    @classmethod
    def from_separate_arrays(cls, t, x, y):
        # TODO: t is unused, and sampling rate isn't set correctly. Also, this class assumes uniform sampling!!
        return cls(traces=np.vstack((x, y)))
    
    
    @classmethod
    def from_vt_mat_file(cls, position_mat_file_path):
        # example: PositionOld.from_vt_mat_file(position_mat_file_path = Path(basedir).joinpath('{}vt.mat'.format(session_name)))
        position_mat_file = import_mat_file(mat_import_file=position_mat_file_path)
        tt = position_mat_file['tt'] # 1, 63192
        xx = position_mat_file['xx'] # 10 x 63192
        yy = position_mat_file['yy'] # 10 x 63192
        tt = tt.flatten()
        tt_rel = tt - tt[0] # relative timestamps
        # timestamps_conversion_factor = 1e6
        # timestamps_conversion_factor = 1e4
        timestamps_conversion_factor = 1.0
        t = tt / timestamps_conversion_factor  # (63192,)
        t_rel = tt_rel / timestamps_conversion_factor  # (63192,)
        position_sampling_rate_Hz = 1.0 / np.mean(np.diff(tt / 1e6)) # In Hz, returns 29.969777
        num_samples = len(t);
        x = xx[0,:].flatten() # (63192,)
        y = yy[0,:].flatten() # (63192,)
        # active_t_start = t[0] # absolute t_start
        active_t_start = 0.0 # relative t_start
        return cls(traces=np.vstack((x, y)), computed_traces=np.full([1, num_samples], np.nan), t_start=active_t_start, sampling_rate=position_sampling_rate_Hz)
    
    
    # ConcatenationInitializable protocol:
    @classmethod
    def concat(cls, objList: Union[Sequence, np.array]):
        """ Concatenates the object list """
        objList = np.array(objList)
        t_start_times = np.array([obj.t_start for obj in objList])
        sort_idx = list(np.argsort(t_start_times))
        # print(sort_idx)
        # sort the objList by t_start
        objList = objList[sort_idx]
        
        new_t_start = objList[0].t_start # new t_start is the earliest t_start in the array
        new_sampling_rate = objList[0].sampling_rate
        
        # Concatenate the elements:
        traces_list = np.concatenate([obj.traces for obj in objList], axis=1)
        computed_traces_list = np.concatenate([obj.computed_traces for obj in objList], axis=1)
        
        return cls(
            traces=traces_list,
            computed_traces=computed_traces_list,
            t_start=new_t_start,
            sampling_rate=new_sampling_rate,
        )
        
    def print_debug_str(self):
        print('<core.PositionOld :: np.shape(traces): {}\t time: {}\n duration: {}\n time[-1]: {}\n time[0]: {}\n sampling_rate: {}\n t_start: {}\n t_stop: {}\n>\n'.format(np.shape(self.traces), self.time,
            self.duration,
            self.time[-1],
            self.time[0],
            self.sampling_rate,
            self.t_start,
            self.t_stop)
        )
        # print('self.time: {}\n self.duration: {}\n self.time[-1]: {}\n self.time[0]: {}\n self.sampling_rate: {}\n self.t_start: {}\n self.t_stop: {}\n>\n'.format(
        #     self.time,
        #     self.duration,
        #     self.time[-1],
        #     self.time[0],
        #     self.sampling_rate,
        #     self.t_start,
        #     self.t_stop))
         