from copy import deepcopy
from typing import Sequence, Union
import itertools # for flattening lists with itertools.chain.from_iterable()
import numpy as np
from pandas.core.indexing import IndexingError
# from neuropy.analyses.placefields import PfND # for _bin_pos_nD
# from neuropy.analyses.placefields.PfND import _bin_pos_nD # for _bin_pos_nD

import pandas as pd
from .epoch import Epoch
from .datawriter import DataWriter
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin, TimeSlicedMixin
from neuropy.utils.mixins.concatenatable import ConcatenationInitializable
from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable

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
    """ Requires conformance to PositionDimDataMixin as well."""
    
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
    

    ## TODO: _perform_compute_higher_order_derivatives has been depricated in favor of factoring this functionality into the pho_custom_placefields "position" pandas Dataframe extension. (See PositionAccessor)
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
            # self.df = self.df.position._perform_compute_higher_order_derivatives(curr_column_label) # uses the position accessor
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
        """Computes the higher-order positional derivatives for a single component (given by component_label) of the pos_df
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
            N (int, optional): [description]. Defaults to 20.
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
    def linear_pos_obj(self):
        # returns a Position object containing only the linear_pos as its trace. This is used for compatibility with Bapun's Pf1D function 
        lin_pos_df = self.df[[self.time_variable_name,'lin_pos']].copy()
        # lin_pos_df.rename({'lin_pos':'x'}, axis='columns', errors='raise', inplace=True)
        lin_pos_df['x'] = lin_pos_df['lin_pos'].copy() # duplicate the lin_pos column to the 'x' column
        out_obj = Position(lin_pos_df, metadata=None)
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



    
    ## Computed Variable Properties:
    @property
    def speed(self):
        if 'speed' in self.df.columns:
            return self.df['speed'].to_numpy()
        else:
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
    
    
""" --- """
@pd.api.extensions.register_dataframe_accessor("position")
class PositionAccessor(PositionDimDataMixin, PositionComputedDataMixin, TimeSlicedMixin):
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
    
    
    # # TODO: The documentation says this is the preferred method, but it doesn't actually seem to be used anywhere!
    # def compute_higher_order_derivatives(self, component_label: str):
    #     """Computes the higher-order positional derivatives for a single component (given by component_label) of the pos_df
    #     Args:
    #         self (pd.DataFrame): [a pos_df]
    #         component_label (str): [description]
    #     Returns:
    #         pd.DataFrame: The updated dataframe with the dt, velocity, and acceleration columns added.
    #     """
    #     # compute each component separately:
    #     velocity_column_key = f'velocity_{component_label}'
    #     acceleration_column_key = f'acceleration_{component_label}'
                
    #     dt = np.insert(np.diff(self._obj['t']), 0, np.nan)
    #     velocity_comp = np.insert(np.diff(self._obj[component_label]), 0, 0.0) / dt
    #     velocity_comp[np.isnan(velocity_comp)] = 0.0 # replace NaN components with zero
    #     acceleration_comp = np.insert(np.diff(velocity_comp), 0, 0.0) / dt
    #     acceleration_comp[np.isnan(acceleration_comp)] = 0.0 # replace NaN components with zero
    #     dt[np.isnan(dt)] = 0.0 # replace NaN components with zero
        
    #     # add the columns to the dataframe:
    #     self._obj['dt'] = dt
    #     self._obj[velocity_column_key] = velocity_comp
    #     self._obj[acceleration_column_key] = acceleration_comp
        
    #     return self._obj  
    
    
""" --- """
class Position(PositionDimDataMixin, PositionComputedDataMixin, ConcatenationInitializable, StartStopTimesMixin, TimeSlicableObjectProtocol, DataFrameRepresentable, DataWriter):
    def __init__(
        self,
        pos_df: pd.DataFrame,
        metadata=None,
    ) -> None:
        """[summary]
        Args:
            pos_df (pd.DataFrame): Each column is a pd.Series(["t", "x", "y"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        super().__init__(metadata=metadata)
        self._data = pos_df # set to the laps dataframe
        # self._data = Position._update_dataframe_computed_vars(self._data) # maybe initialize equivalent for laps
        self._data = self._data.sort_values(by=[self.time_variable_name]) # sorts all values in ascending order
        
    def time_slice_indicies(self, t_start, t_stop):
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        # indicies = np.where((self._data['t'].to_numpy() >= t_start) & (self._data['t'].to_numpy() <= t_stop))[0]
        # included_indicies = ((self._data['t'] >= t_start) & (self._data['t'] <= t_stop))
        # print('time_slice_indicies(...): t_start: {}, t_stop: {}, included_indicies: {}'.format(t_start, t_stop, included_indicies))
        included_indicies = self._data[self.time_variable_name].between(t_start, t_stop, inclusive='both') # returns a boolean array indicating inclusion in teh current lap
        # position_df.loc[curr_lap_position_df_is_included, ['lap']] = curr_lap_id # set the 'lap' identifier on the object
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
        return 1.0/np.mean(np.diff(self.time))

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
        # indices = self.time_slice_indicies(t_start, t_stop) # from TimeSlicableIndiciesMixin
        included_df = deepcopy(self._data)
        included_df = included_df[((included_df[self.time_variable_name] >= t_start) & (included_df[self.time_variable_name] <= t_stop))]
        
        # df.query('Salary_in_1000 >= 100 & Age < 60 & FT_Team.str.startswith("S").values')
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

        
    def print_debug_str(self):
        print('<core.Position :: np.shape(traces): {}\t time: {}\n duration: {}\n time[-1]: {}\n time[0]: {}\n sampling_rate: {}\n t_start: {}\n t_stop: {}\n>\n'.format(np.shape(self.traces), self.time,
            self.duration,
            self.time[-1],
            self.time[0],
            self.sampling_rate,
            self.t_start,
            self.t_stop)
        )
         
         



