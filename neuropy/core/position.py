from copy import deepcopy
from typing import Sequence, Union
import itertools # for flattening lists with itertools.chain.from_iterable()
import numpy as np
import h5py # for to_hdf and read_hdf definitions
from pandas.core.indexing import IndexingError

import pandas as pd
from .epoch import Epoch
from .datawriter import DataWriter
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin, TimeSlicedMixin
from neuropy.utils.mixins.concatenatable import ConcatenationInitializable
from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from neuropy.utils.mixins.time_slicing import TimePointEventAccessor

""" --- Helper FUNCTIONS """
def build_position_df_time_window_idx(active_pos_df: pd.DataFrame, curr_active_time_windows, debug_print=False):
    """ adds the time_window_idx column to the active_pos_df
    Usage:
        curr_active_time_windows = np.array(pho_custom_decoder.active_time_windows)
        active_pos_df = build_position_df_time_window_idx(sess.position.to_dataframe(), curr_active_time_windows)
    """
    active_pos_df['time_window_idx'] = np.full_like(active_pos_df['t'], -1, dtype='int')
    starts = curr_active_time_windows[:,0]
    stops = curr_active_time_windows[:,1]
    num_slices = len(starts)
    if debug_print:
        print(f'starts: {np.shape(starts)}, stops: {np.shape(stops)}, num_slices: {num_slices}')
    for i in np.arange(num_slices):
        active_pos_df.loc[active_pos_df[active_pos_df.position.time_variable_name].between(starts[i], stops[i], inclusive='both'), ['time_window_idx']] = int(i) # set the 'time_window_idx' identifier on the object
    active_pos_df['time_window_idx'] = active_pos_df['time_window_idx'].astype(int) # ensure output is the correct datatype
    return active_pos_df


def build_position_df_resampled_to_time_windows(active_pos_df: pd.DataFrame, time_bin_size=0.02):
    """ Note that this returns a TimedeltaIndexResampler, not a dataframe proper. To get the real dataframe call .nearest() on output.

    Usage:
        time_binned_position_resampler = build_position_df_resampled_to_time_windows(computation_result.sess.position.to_dataframe(), time_bin_size=computation_result.computation_config.pf_params.time_bin_size) # TimedeltaIndexResampler
        time_binned_position_df = time_binned_position_resampler.nearest() # an actual dataframe
    """
    position_time_delta = pd.to_timedelta(active_pos_df[active_pos_df.position.time_variable_name], unit="sec")
    active_pos_df['time_delta_sec'] = position_time_delta
    active_pos_df = active_pos_df.set_index('time_delta_sec')
    window_resampled_pos_df = active_pos_df.resample(f'{time_bin_size}S', base=0) #.nearest() # '0.02S' 0.02 second bins
    # window_resampled_pos_df = active_pos_df.resample(f'{time_bin_size}S') # , origin='start'
    # What happened to the `base` parameter in Pandas v2? I used to `window_resampled_pos_df = active_pos_df.resample(f'{time_bin_size}S', base=0)` and now I get `TypeError: resample() got an unexpected keyword argument 'base'`

    return window_resampled_pos_df



""" --- Helper MIXINS """
class PositionDimDataMixin:
    """ Implementors gain convenience properties to access .x, .y, and .z variables as properties. 
    Requirements:
        Implement  
            @property
            def df(self):
                return <a dataframe>
    """
    __time_variable_name = 't' # currently hardcoded
    
    @property
    def df(self):
        """The df property."""
        raise NotImplementedError # must be overriden by implementor 
        # return self._obj # for PositionAccessor
        # return self._data # for Position
    @df.setter
    def df(self, value):
        raise NotImplementedError # must be overriden by implementor 
        # self._obj = value # for PositionAccessor
        # self._data = value # for Position
    
    
    # Position
    @property
    def x(self):
        return self.df['x'].to_numpy()

    @x.setter
    def x(self, x):
        self.df.loc[:, 'x'] = x

    @property
    def y(self):
        assert self.ndim > 1, "No y for one-dimensional position"
        return self.df['y'].to_numpy()

    @y.setter
    def y(self, y):
        assert self.ndim > 1, "Position data has only one dimension"
        self.df.loc[:, 'y'] = y

    @property
    def z(self):
        assert self.ndim == 3, "Position data is not three-dimensional"
        return self.df['z'].to_numpy()

    @z.setter
    def z(self, z):
        self.df.loc[:, 'z'] = z
    
    @property
    def traces(self):
        """ Compatibility method for the old-style implementation. """
        # print('traces accessed with self.ndim of {}'.format(self.ndim))
        if self.ndim == 1:
            return self.df[['x']].to_numpy().T
        elif self.ndim >= 2:
            return self.df[['x','y']].to_numpy().T
        elif self.ndim >= 3:
            return self.df[['x','y','z']].to_numpy().T
        else:
            raise IndexingError
        
    # Time Properties:
    @property
    def time_variable_name(self):
        return PositionDimDataMixin.__time_variable_name
    @property
    def time(self):
        return self.df[self.time_variable_name].to_numpy()
    @property
    def t_start(self):
        return self.df[self.time_variable_name].iloc[0]
    @t_start.setter
    def t_start(self, t):
        raise NotImplementedError
        # self._t_start = t
    @property
    def t_stop(self):
        return self.df[self.time_variable_name].iloc[-1]
    @property
    def duration(self):
        return self.t_stop - self.t_start
        # return float(self.n_frames) / float(self.sampling_rate)

    # Dimension Properties:
    @property
    def ndim(self):
        """ returns the count of the spatial columns that the dataframe has """
        return np.sum(np.isin(['x','y','z'], self.df.columns))
    @property
    def dim_columns(self):
        """ returns the labels of the columns that correspond to spatial columns 
            If ndim == 1, returns ['x'], 
            if ndim == 2, returns ['x','y'], etc.
        """
        spatial_column_labels = np.array(['x','y','z'])
        return list(spatial_column_labels[np.isin(spatial_column_labels, self.df.columns)])
    
    @property
    def n_frames(self):
        return len(self.df.index)

class PositionComputedDataMixin:
    """ Requires conformance to PositionDimDataMixin as well. Adds computed properties like velocity and acceleration (higher-order derivatives of position) and smoothed values to the dataframe."""
    
    ## Computed Variable Labels Properties:
    @staticmethod
    def _computed_column_component_labels(component_label):
        return [f'velocity_{component_label}', f'acceleration_{component_label}']
    @property
    def dim_computed_columns(self, include_dt=False):
        """ returns the labels for the computed columns
            output: ['dt', 'velocity_x', 'acceleration_x', 'velocity_y', 'acceleration_y']
        """
        computed_column_labels = [PositionComputedDataMixin._computed_column_component_labels(a_dim_label) for a_dim_label in self.dim_columns]
        if include_dt:
            computed_column_labels.insert(0, ['dt']) # insert 'dt' at the start of the list
        # itertools.chain.from_iterable converts ['dt', ['velocity_x', 'acceleration_x'], ['velocity_y', 'acceleration_y']] to ['dt', 'velocity_x', 'acceleration_x', 'velocity_y', 'acceleration_y']
        computed_column_labels = list(itertools.chain.from_iterable(computed_column_labels)) # ['dt', 'velocity_x', 'acceleration_x', 'velocity_y', 'acceleration_y']
        return computed_column_labels
    
    @staticmethod
    def _perform_compute_higher_order_derivatives(pos_df: pd.DataFrame, component_label: str, time_variable_name: str = 't'):
        """Computes the higher-order positional derivatives for a single component (given by component_label) of the pos_df
        Args:
            pos_df (pd.DataFrame): [description]
            component_label (str): [description]
        Returns:
            pd.DataFrame: The updated dataframe with the dt, velocity, and acceleration columns added.
        """
        # compute each component separately:
        velocity_column_key = f'velocity_{component_label}'
        acceleration_column_key = f'acceleration_{component_label}'
                
        dt = np.insert(np.diff(pos_df[time_variable_name]), 0, np.nan)
        velocity_comp = np.insert(np.diff(pos_df[component_label]), 0, 0.0) / dt
        velocity_comp[np.isnan(velocity_comp)] = 0.0 # replace NaN components with zero
        acceleration_comp = np.insert(np.diff(velocity_comp), 0, 0.0) / dt
        acceleration_comp[np.isnan(acceleration_comp)] = 0.0 # replace NaN components with zero
        dt[np.isnan(dt)] = 0.0 # replace NaN components with zero
        
        # add the columns to the dataframe:
        # pos_df.loc[:, 'dt'] = dt
        pos_df['dt'] = dt
        pos_df[velocity_column_key] = velocity_comp
        pos_df[acceleration_column_key] = acceleration_comp
        
        return pos_df
    
    def compute_higher_order_derivatives(self):
        """Computes the higher-order positional derivatives for all spatial dimensional components of self.df. Adds the dt, velocity, and acceleration columns
        """
        for dim_i in np.arange(self.ndim):
            curr_column_label = self.dim_columns[dim_i]
            self.df = PositionComputedDataMixin._perform_compute_higher_order_derivatives(self.df, curr_column_label, time_variable_name=self.time_variable_name)
        return self.df
    
    
    ## Smoothed Computed Variables:
    @staticmethod
    def _smoothed_column_labels(non_smoothed_column_labels):
        """ returns the smoothed_column_labels given the specified non_smoothed_column_labels (which is a list of str) """
        return [f'{a_label}_smooth' for a_label in non_smoothed_column_labels]
    @property
    def dim_smoothed_columns(self, non_smoothed_column_labels=None):
        """ returns the labels for the smoothed columns
            non_smoothed_column_labels: list can be specified to only get the smoothed labels for the variables in the non_smoothed_column_labels list
            output: ['x_smooth', 'y_smooth', 'velocity_x_smooth', 'acceleration_x_smooth', 'velocity_y_smooth', 'acceleration_y_smooth']
        """
        if non_smoothed_column_labels is None:
            non_smoothed_column_labels = (self.dim_columns + self.dim_computed_columns)
        smoothed_column_labels = PositionComputedDataMixin._smoothed_column_labels(non_smoothed_column_labels)
        # smoothed_column_labels = list(itertools.chain.from_iterable(smoothed_column_labels)) # flattens list, but I don't think we need this
        return smoothed_column_labels
    
    @staticmethod
    def _perform_compute_smoothed_position_info(pos_df: pd.DataFrame, non_smoothed_column_labels,  N: int = 20):
        """Computes the smoothed quantities for a single component (given by component_label) of the pos_df
        Args:
            pos_df (pd.DataFrame): [description]
            non_smoothed_column_labels (list(str)): a list of the columns to be smoothed
            N (int): 20 # roll over the last N samples
            
        Returns:
            pd.DataFrame: The updated dataframe with the dt, velocity, and acceleration columns added.
            
        Usage:
            smoothed_pos_df = Position._perform_compute_smoothed_position_info(pos_df, (position_obj.dim_columns + position_obj.dim_computed_columns), N=20)
        
        """
        smoothed_column_names = PositionComputedDataMixin._smoothed_column_labels(non_smoothed_column_labels)
        pos_df[smoothed_column_names] = pos_df[non_smoothed_column_labels].rolling(window=N).mean()
        return pos_df
    
    def compute_smoothed_position_info(self, N: int = 20, non_smoothed_column_labels=None):
        """Computes smoothed position variables and adds them as columns to the internal dataframe
        Args:
            N (int, optional): Number of previous samples to smooth over. Defaults to 20.
        Returns:
            [type]: [description]
        """
        if non_smoothed_column_labels is None:
            non_smoothed_column_labels = (self.dim_columns + self.dim_computed_columns)
        # smoothed_column_names = self.dim_smoothed_columns(non_smoothed_column_labels)
        self.df = PositionComputedDataMixin._perform_compute_smoothed_position_info(self.df, non_smoothed_column_labels, N=N)
        return self.df        
    
    ## Linear Position Properties:
    @property
    def linear_pos_obj(self) -> "Position":
        """ returns a Position object containing only the linear_pos as its trace. This is used for compatibility with Bapun's Pf1D function """ 
        if not self.has_linear_pos:
            self.compute_linearized_position()
            assert self.has_linear_pos, "Doesn't have linear position even after `self.compute_linearized_position()` was called!"
            
        lin_pos_df = deepcopy(self.df[[self.time_variable_name, 'lin_pos']])
        # lin_pos_df.rename({'lin_pos':'x'}, axis='columns', errors='raise', inplace=True)
        lin_pos_df['x'] = lin_pos_df['lin_pos'].copy() # duplicate the lin_pos column to the 'x' column
        out_obj = Position(lin_pos_df, metadata=None) ## build position object out of the dataframe
        out_obj.compute_higher_order_derivatives()
        out_obj.compute_smoothed_position_info()
        out_obj.speed; # ensure speed is calculated for the new object
        return out_obj
    @property
    def linear_pos(self):
        assert 'lin_pos' in self.df.columns, "Linear Position data has not yet been computed."
        return self.df['lin_pos'].to_numpy()
    @linear_pos.setter
    def linear_pos(self, linear_pos):
        self.df.loc[:, 'lin_pos'] = linear_pos
    @property
    def has_linear_pos(self):
        if 'lin_pos' in self.df.columns:
            return not np.isnan(self.df['lin_pos'].to_numpy()).all() # check if all are nan
        else:
            # Linear Position data has not yet been computed.
            return False


    def compute_linearized_position(self, method='isomap', **kwargs) -> "Position":
        """ computes and adds the linear position to this Position object """
        from neuropy.utils import position_util
        # out_linear_position_obj = position_util.linearize_position(self, method=method, **kwargs)
        # self._data['lin_pos'] = out_linear_position_obj.to_dataframe()['lin_pos'] # add the `lin_pos` column to the pos_df
        self.df = position_util.linearize_position_df(self.df, method=method, **kwargs) # adds 'lin_pos' column to `self.df`
        return self
    
    ## Computed Variable Properties:
    @property
    def speed(self):
        if 'speed' in self.df.columns:
            return self.df['speed'].to_numpy()
        else:
            # Compute the speed if not already done upon first access
            dt = np.mean(np.diff(self.time))
            self.df['speed'] = np.insert((np.sqrt(((np.abs(np.diff(self.traces, axis=1))) ** 2).sum(axis=0)) / dt), 0, 0.0) # prepends a 0.0 value to the front of the result array so it's the same length as the other position vectors (x, y, etc)        
        return self.df['speed'].to_numpy()
    
    @property
    def dt(self):
        if 'dt' in self.df.columns:
            return self.df['dt'].to_numpy()
        else:
            # compute the higher_order_derivatives if not already done upon first access
            self.df = self.compute_higher_order_derivatives()   
        return self.df['dt'].to_numpy()
    
    @property
    def velocity_x(self):
        if 'velocity_x' in self.df.columns:
            return self.df['velocity_x'].to_numpy()
        else:
            # compute the higher_order_derivatives if not already done upon first access
            self.df = self.compute_higher_order_derivatives()   
        return self.df['velocity_x'].to_numpy()
    
    @property
    def acceleration_x(self):
        if 'acceleration_x' in self.df.columns:
            return self.df['acceleration_x'].to_numpy()
        else:
            # compute the higher_order_derivatives if not already done upon first access
            self.df = self.compute_higher_order_derivatives()   
        return self.df['acceleration_x'].to_numpy()
    
    @property
    def velocity_y(self):
        if 'velocity_y' in self.df.columns:
            return self.df['velocity_y'].to_numpy()
        else:
            # compute the higher_order_derivatives if not already done upon first access
            self.df = self.compute_higher_order_derivatives()   
        return self.df['velocity_y'].to_numpy()
    
    @property
    def acceleration_y(self):
        if 'acceleration_y' in self.df.columns:
            return self.df['acceleration_y'].to_numpy()
        else:
            # compute the higher_order_derivatives if not already done upon first access
            self.df = self.compute_higher_order_derivatives()   
        return self.df['acceleration_y'].to_numpy()
    
    


def adding_lap_info_to_position_df(position_df: pd.DataFrame, laps_df: pd.DataFrame, debug_print:bool=False):
    """ Adds a 'lap' column to the position dataframe:
        Also adds a 'lap_dir' column, containing 0 if it's an outbound trial, 1 if it's an inbound trial, and -1 if it's neither.
    Usage:
    
        from neuropy.core.position import adding_lap_info_to_position_df
        
        curr_position_df = self.position.to_dataframe() # get the position dataframe from the session
        curr_laps_df = self.laps.to_dataframe()
        curr_position_df = adding_lap_info_to_position_df(position_df=curr_position_df, laps_df=curr_laps_df)
        
        # update:
        self.position._data['lap'] = curr_position_df['lap']
        self.position._data['lap_dir'] = curr_position_df['lap_dir']
        
    """
    assert 'lap_id' in laps_df
    
    possible_lap_columns_to_add = ['maze_id', 'lap_dir', 'is_LR_dir', 'truth_decoder_name']
    possible_lap_columns_to_add_not_found_values = {'maze_id':-1, 'lap_dir':-1, 'is_LR_dir':False, 'truth_decoder_name':''}
    lap_columns_to_add = [v for v in possible_lap_columns_to_add if v in laps_df.columns] ## can only add the columns if they're present in laps_df

    
    position_df['lap'] = np.NaN # set all 'lap' column to NaN
    position_df['lap_dir'] = np.full_like(position_df['lap'], possible_lap_columns_to_add_not_found_values['lap_dir']) # set all 'lap_dir' to -1

    unique_lap_ids = np.unique(laps_df['lap_id'])
    lap_id_to_dir_dict = {a_row.lap_id:a_row.lap_dir for a_row in laps_df[['lap_id', 'lap_dir']].itertuples(index=False)}
    if debug_print:
        print(f'lap_id_to_dir_dict: {lap_id_to_dir_dict}')
    assert len(unique_lap_ids) == len(lap_id_to_dir_dict)
    
    n_laps: int = len(unique_lap_ids)
    for i in np.arange(n_laps):
        curr_lap_id = laps_df.loc[laps_df.index[i], 'lap_id'] # The second epoch in a session doesn't start with indicies of the first lap, so instead we need to get laps_df.index[i] to get the correct index
        curr_lap_t_start, curr_lap_t_stop = laps_df.loc[laps_df.index[i], 'start'], laps_df.loc[laps_df.index[i], 'stop']
        # curr_lap_t_start, curr_lap_t_stop = self.laps.get_lap_times(i)
        if debug_print:
            print('lap[{}]: ({}, {}): '.format(curr_lap_id, curr_lap_t_start, curr_lap_t_stop))
        curr_lap_position_df_is_included = position_df['t'].between(curr_lap_t_start, curr_lap_t_stop, inclusive='both') # returns a boolean array indicating inclusion in teh current lap
        position_df.loc[curr_lap_position_df_is_included, ['lap']] = curr_lap_id # set the 'lap' identifier on the object
        curr_lap_dir = lap_id_to_dir_dict[curr_lap_id]
        
        position_df.loc[curr_lap_position_df_is_included, ['lap_dir']] = curr_lap_dir # set the 'lap' identifier on the object

        # curr_position_df.query('-0.5 <= t < 0.5')
    
    # laps_df.epoch

    # update the lap_dir variable:
    # position_df.loc[np.logical_not(np.isnan(position_df.lap.to_numpy())), 'lap_dir'] = np.mod(position_df.loc[np.logical_not(np.isnan(position_df.lap.to_numpy())), 'lap'], 2.0)
    # position_df['lap_dir'] = position_df['lap'].map(lambda v: lap_id_to_dir_dict.get(v, -1))
    
    # return the extracted traces and the updated curr_position_df
    return position_df


""" --- """
@pd.api.extensions.register_dataframe_accessor("position")
class PositionAccessor(PositionDimDataMixin, PositionComputedDataMixin, TimeSlicedMixin, TimePointEventAccessor):
    """ A Pandas DataFrame-based Position helper. """
    
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column for timestamps ('t') and a column for at least 1D positions ('x')
        if "t" not in obj.columns:
            raise AttributeError("Must have at least one time variable: specifically 't' for PositionAccessor.")
        if "x" not in obj.columns:
            raise AttributeError("Must have at least one position dimension column 'x'.")
        # if "lin_pos" not in obj.columns or "speed" not in obj.columns:
        #     raise AttributeError("Must have 'lin_pos' column and 'x'.")

    # for PositionDimDataMixin & PositionComputedDataMixin
    @property
    def df(self):
        return self._obj # for PositionAccessor
    @df.setter
    def df(self, value):
        self._obj = value # for PositionAccessor
    
    def to_Position_obj(self, metadata=None):
        """ builds a Position object from the PositionAccessor's dataframe 
        Usage:
            pos_df.position.to_Position_obj()
        """
        return Position(self._obj, metadata=metadata)

    def drop_dimensions_above(self, desired_ndim:int, inplace:bool=False):
        """ drops all columns related to dimensions above `desired_ndim`.

        e.g. desired_ndim = 1:
            would drop 'y' related columns

        if inplace is True, None is returned and the dataframe is modified in place

        """
        z_related_column_names = [str(c) for c in self._obj.columns if str(c).endswith('z')] # Find z (3D) related columns
        y_related_column_names = [str(c) for c in self._obj.columns if str(c).endswith('y')] # Find y (2D) related columns
        if inplace:
            out_df = None
        else:
            out_df = self._obj.copy()

        if desired_ndim < 3:
            if inplace:
                self._obj.drop(columns=z_related_column_names, inplace=inplace)
            else:
                out_df = out_df.drop(columns=z_related_column_names, inplace=inplace)
        if desired_ndim < 2:
            if inplace:
                self._obj.drop(columns=y_related_column_names, inplace=inplace)
            else:
                out_df = out_df.drop(columns=y_related_column_names, inplace=inplace)

        return out_df
    


    def adding_lap_info(self, laps_df: pd.DataFrame, inplace:bool=False, debug_print:bool=False):
        """ Adds a 'lap' column to the position dataframe:
            Also adds a 'lap_dir' column, containing 0 if it's an outbound trial, 1 if it's an inbound trial, and -1 if it's neither.
        Usage:
        
            from neuropy.core.position import adding_lap_info_to_position_df
            
            curr_position_df = self.position.to_dataframe() # get the position dataframe from the session
            curr_laps_df = self.laps.to_dataframe()
            curr_position_df = curr_position_df.position.adding_lap_info(laps_df=curr_laps_df, inplace=False)
            
            # update:
            self.position._data['lap'] = curr_position_df['lap']
            self.position._data['lap_dir'] = curr_position_df['lap_dir']
            
        """
        if inplace:
            self._obj = adding_lap_info_to_position_df(position_df=self._obj, laps_df=laps_df, debug_print=debug_print)
            return self._obj
        else:
            out_pos_df = self._obj.copy()
            out_pos_df = adding_lap_info_to_position_df(position_df=out_pos_df, laps_df=laps_df, debug_print=debug_print)
            return out_pos_df


    
""" --- """
class Position(HDFMixin, PositionDimDataMixin, PositionComputedDataMixin, ConcatenationInitializable, StartStopTimesMixin, TimeSlicableObjectProtocol, DataFrameRepresentable, DataWriter):
    def __init__(self, pos_df: pd.DataFrame, metadata=None) -> None:
        """[summary]
        Args:
            pos_df (pd.DataFrame): Each column is a pd.Series(["t", "x", "y"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        super().__init__(metadata=metadata)
        self._data = pos_df # set to the laps dataframe
        self._data = self._data.sort_values(by=[self.time_variable_name]) # sorts all values in ascending order
        
    def time_slice_indicies(self, t_start, t_stop):
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        included_indicies = self._data[self.time_variable_name].between(t_start, t_stop, inclusive='both') # returns a boolean array indicating inclusion in teh current lap
        return self._data.index[included_indicies]# note that this currently returns a Pandas.Series object. I could get the normal indicis by using included_indicies.to_numpy()
        
    @classmethod
    def init(cls, traces: np.ndarray, computed_traces: np.ndarray=None, t_start=0, sampling_rate=120, metadata=None):
        """ Comatibility initializer """
        if traces.ndim == 1:
            traces = traces.reshape(1, -1) # required before setting ndim
            
        ndim = traces.shape[0]
        assert ndim <= 3, "Maximum possible dimension of position is 3"
        
        # generate time vector:
        n_frames = traces.shape[1]
        duration = float(n_frames) / float(sampling_rate)
        t_stop = t_start + duration
        time = np.linspace(t_start, t_stop, n_frames)

        x = traces[0].flatten().copy()       
        df = pd.DataFrame({'t': time, 'x': x})
        if computed_traces is not None:
            if computed_traces.ndim >= 1:
                df["lin_pos"] = computed_traces[0].flatten().copy()
        
        if ndim >= 2:
            y = traces[1]
            df["y"] = y.flatten().copy()
        if ndim >= 3:
            z = traces[2]
            df["z"] = z.flatten().copy()
        return Position(df, metadata=metadata)


    ## Compatibility:
    @classmethod
    def legacy_from_dict(cls, dict_rep: dict):
        """ Tries to load the dict using previous versions of this code. """
        # Legacy fallback:
        print(f'Position falling back to legacy loading protocol...: dict_rep: {dict_rep}')
        return Position.init(**({'computed_traces': None, 't_start': 0, 'sampling_rate': 120, 'metadata': None} | dict_rep))
        
    # for PositionDimDataMixin
    @property
    def df(self):
        return self._data # for Position
    @df.setter
    def df(self, value):
        self._data = value # for Position

    @property
    def sampling_rate(self):
        # raise NotImplementedError
        return 1.0/np.nanmean(np.diff(self.time))

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        raise NotImplementedError
    

    def to_dict(self):
        data = {
            "df": self._data,
            "metadata": self.metadata,
        }
        return data

    @staticmethod
    def from_dict(d):
        return Position(
            d["df"],
            metadata=d["metadata"],
        )

    # @staticmethod
    # def is_fixed_sampling_rate(time):
    #     dt = np.diff(time)
        
    
    def to_dataframe(self):
        return self._data.copy()

    def speed_in_epochs(self, epochs: Epoch):
        assert isinstance(epochs, Epoch), "epochs must be neuropy.Epoch object"
        pass

    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop):
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        included_df = deepcopy(self._data)
        included_df = included_df[((included_df[self.time_variable_name] >= t_start) & (included_df[self.time_variable_name] <= t_stop))]
        return Position(included_df, metadata=deepcopy(self.metadata))
        

    @classmethod
    def from_separate_arrays(cls, t, x, y=None, z=None, lin_pos=None, metadata=None):
        temp_dict = {'t':t,'x':x}
        if y is not None:
            temp_dict['y'] = y
        if z is not None:
            temp_dict['z'] = z
        if lin_pos is not None:
            temp_dict['lin_pos'] = lin_pos
        return cls(pd.DataFrame(temp_dict), metadata=metadata)                            
        # return cls(traces=np.vstack((x, y)))
    
    
    # ConcatenationInitializable protocol:
    @classmethod
    def concat(cls, objList: Union[Sequence, np.array]):
        """ Concatenates the object list """
        objList = np.array(objList)
        concat_df = pd.concat([obj._data for obj in objList])
        return cls(concat_df)

        
    def drop_dimensions_above(self, desired_ndim:int):
        """ modifies the internal dataframe to drop dimensions above a certain number. Always done in place, and returns None. """
        return self.df.position.drop_dimensions_above(desired_ndim, inplace=True)


    def print_debug_str(self):
        print('<core.Position :: np.shape(traces): {}\t time: {}\n duration: {}\n time[-1]: {}\n time[0]: {}\n sampling_rate: {}\n t_start: {}\n t_stop: {}\n>\n'.format(np.shape(self.traces), self.time,
            self.duration,
            self.time[-1],
            self.time[0],
            self.sampling_rate,
            self.t_start,
            self.t_stop)
        )
         
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pos_obj: Position = long_one_step_decoder_1D.pf.position
            _pos_obj.to_hdf(hdf5_output_path, key='pos')
        """
        _df = self.to_dataframe()
        
        # Save the DataFrame using pandas
        # Unable to open/create file '/media/MAX/Data/KDIBA/gor01/one/2006-6-12_15-55-31/output/pipeline_results.h5'
        with pd.HDFStore(file_path) as store:
            _df.to_hdf(path_or_buf=store, key=key, format=kwargs.pop('format', 'table'), data_columns=kwargs.pop('data_columns',True), **kwargs)

        # Open the file with h5py to add attributes to the dataset
        with h5py.File(file_path, 'r+') as f:
            dataset = f[key]
            metadata = {
                'time_variable_name': self.time_variable_name,
                'sampling_rate': self.sampling_rate,
                't_start': self.t_start,
                't_stop': self.t_stop,
            }
            for k, v in metadata.items():
                dataset.attrs[k] = v

    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> "Position":
        """ Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pos_obj = Position.read_hdf(hdf5_output_path, key='pos')
            _reread_pos_obj
        """
        # Read the DataFrame using pandas
        pos_df = pd.read_hdf(file_path, key=key)

        # Open the file with h5py to read attributes
        with h5py.File(file_path, 'r') as f:
            dataset = f[key]
            metadata = {
                'time_variable_name': dataset.attrs['time_variable_name'],
                'sampling_rate': dataset.attrs['sampling_rate'],
                't_start': dataset.attrs['t_start'],
                't_stop': dataset.attrs['t_stop'],
            }

        # Reconstruct the object using the class constructor
        _out = cls(pos_df=pos_df, metadata=metadata)
        _out.filename = file_path # set the filename it was loaded from
        return _out




