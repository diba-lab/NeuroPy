from copy import deepcopy
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from neuropy.core.epoch import Epoch
from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable

from neuropy.utils.mixins.print_helpers import SimplePrintable
from .datawriter import DataWriter

from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol

## Import:
# from neuropy.core.laps import Laps


# TODO: implement: NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, TimeSlicableObjectProtocol
class Laps(DataFrameRepresentable, DataWriter):
    
    df_all_fieldnames = ['lap_id','maze_id','start_spike_index', 'end_spike_index', 'start_t', 'end_t', 'start_t_seconds', 'end_t_seconds', 'duration_seconds']
    
    def __init__(self, laps: pd.DataFrame, metadata=None) -> None:
        """[summary]
        Args:
            laps (pd.DataFrame): Each column is a pd.Series(["start", "stop", "label"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        super().__init__(metadata=metadata)
        self._data = laps # set to the laps dataframe
        self._data = Laps._update_dataframe_computed_vars(self._data)
        self._data = self._data.sort_values(by=['start']) # sorts all values in ascending order

    @property
    def lap_id(self):
        return self._data['lap_id'].to_numpy()
    
    @property
    def maze_id(self):
        return self._data['maze_id'].to_numpy()
            
    @property
    def n_laps(self):
        return len(self.lap_id)
        
    @property
    def starts(self):
        return self._data['start'].to_numpy()

    @property
    def stops(self):
        return self._data['stop'].to_numpy()
    
    def filtered_by_lap_flat_index(self, lap_indicies):
        return self.filtered_by_lap_id(self.lap_id[lap_indicies])
       
    def filtered_by_lap_id(self, lap_ids):
        sliced_copy = deepcopy(self) # get copy of the dataframe
        sliced_copy._data = sliced_copy._data[np.isin(sliced_copy.lap_id, lap_ids)]
        return sliced_copy
        
    @classmethod
    def _update_dataframe_computed_vars(cls, laps_df: pd.DataFrame):
        # laps_df[['lap_id','maze_id','start_spike_index', 'end_spike_index']] = laps_df[['lap_id','maze_id','start_spike_index', 'end_spike_index']].astype('int')
        laps_df[['lap_id']] = laps_df[['lap_id']].astype('int')
        if 'maze_id' in laps_df.columns:
            laps_df[['maze_id']] = laps_df[['maze_id']].astype('int')
        if set(['start_spike_index','end_spike_index']).issubset(laps_df.columns):
            laps_df[['start_spike_index', 'end_spike_index']] = laps_df[['start_spike_index', 'end_spike_index']].astype('int')
            laps_df['num_spikes'] = laps_df['end_spike_index'] - laps_df['start_spike_index']
    
        if 'lap_dir' in laps_df.columns:
            laps_df['lap_dir'] = laps_df['lap_dir'].astype('int')        
            
        else:
            # compute the lap_dir if that field doesn't exist:
            laps_df['lap_dir'] = np.full_like(laps_df['lap_id'].to_numpy(), -1)
            laps_df.loc[np.logical_not(np.isnan(laps_df.lap_id.to_numpy())), 'lap_dir'] = np.mod(laps_df.loc[np.logical_not(np.isnan(laps_df.lap_id.to_numpy())), 'lap_id'], 2)
            laps_df['lap_dir'] = laps_df['lap_dir'].astype('int')
        
        laps_df['label'] = laps_df['lap_id'].astype('str') # add the string "label" column
        return laps_df
     
    @classmethod
    def build_dataframe(cls, mat_file_loaded_dict: dict, time_variable_name='t_rel_seconds', absolute_start_timestamp=None):
        laps_df = pd.DataFrame(mat_file_loaded_dict) # 1014937 rows × 11 columns
        laps_df = Laps._update_dataframe_computed_vars(laps_df)
        # Build output Laps object to add to session
        print('setting laps object.')
        if time_variable_name == 't_seconds':
            t_variable_column_names = ['start_t_seconds', 'end_t_seconds']
            t_variable = laps_df[t_variable_column_names].to_numpy()
        elif time_variable_name == 't_rel_seconds':
            t_variable_column_names = ['start_t_seconds', 'end_t_seconds']
            t_variable = laps_df[t_variable_column_names].to_numpy()
            # need to subtract off the absolute start timestamp
            t_variable = t_variable - absolute_start_timestamp
            laps_df[['start_t_rel_seconds', 'end_t_rel_seconds']] = t_variable # assign the new variable
        else:
            t_variable_column_names = ['start_t', 'end_t']
            t_variable = laps_df[t_variable_column_names].to_numpy()
            
        # finally assign the 'start' and 'stop' time columns to the appropriate variable
        laps_df[['start','stop']] = t_variable # assign the active t_variable to the start & end columns of the DataFrame
        return laps_df

    @classmethod
    def from_estimated_laps(cls, pos_t_rel_seconds, desc_crossing_begining_idxs, desc_crossing_ending_idxs, asc_crossing_begining_idxs, asc_crossing_ending_idxs):
        """Builds a Laps object from the output of the neuropy.analyses.laps.estimate_laps function.
        Args:
            pos_t_rel_seconds ([type]): [description]
            desc_crossing_beginings ([type]): [description]
            desc_crossing_endings ([type]): [description]
            asc_crossing_beginings ([type]): [description]
            asc_crossing_endings ([type]): [description]

        Returns:
            [type]: [description]
        """
        ## Build a custom Laps dataframe from the found points:
        ### Note that these crossing_* indicies are for the position dataframe, not the spikes_df (which is what the previous Laps object was computed from).
            # This means we don't have 'start_spike_index' or 'end_spike_index', and we'd have to compute them if we want them.
        custom_test_laps_df = pd.DataFrame({
            'start_position_index': np.concatenate([desc_crossing_begining_idxs, asc_crossing_begining_idxs]),
            'end_position_index': np.concatenate([desc_crossing_ending_idxs, asc_crossing_ending_idxs]),
            'lap_dir': np.concatenate([np.zeros_like(desc_crossing_begining_idxs), np.ones_like(asc_crossing_begining_idxs)])
        })
        # Get start/end times from the indicies
        custom_test_laps_df['start_t_rel_seconds'] = np.concatenate([pos_t_rel_seconds[desc_crossing_begining_idxs], pos_t_rel_seconds[asc_crossing_begining_idxs]])
        custom_test_laps_df['end_t_rel_seconds'] = np.concatenate([pos_t_rel_seconds[desc_crossing_ending_idxs], pos_t_rel_seconds[asc_crossing_ending_idxs]])
        custom_test_laps_df['start'] = custom_test_laps_df['start_t_rel_seconds']
        custom_test_laps_df['stop'] = custom_test_laps_df['end_t_rel_seconds']
        # Sort the laps based on the start time, reset the index, and finally assign lap_id's from the sorted laps
        custom_test_laps_df = custom_test_laps_df.sort_values(by=['start']).reset_index(drop=True) # sorts all values in ascending order
        custom_test_laps_df['lap_id'] = (custom_test_laps_df.index + 1) # set the lap_id column to the index starting at 1
        return Laps(custom_test_laps_df)

    @classmethod
    def from_dataframe(cls, df):
        return cls(df)

    def to_dataframe(self):
        return self._data

    def get_lap_flat_indicies(self, lap_id):
        return self._data.loc[lap_id, ['start_spike_index', 'end_spike_index']].to_numpy()

    def get_lap_times(self, lap_id):
        return self._data.loc[lap_id, ['start', 'stop']].to_numpy()
    
    def as_epoch_obj(self):
        """ Converts into a core.Epoch object containing the time periods """
        return Epoch(self.to_dataframe())

    @staticmethod
    def from_dict(d: dict):
        return Laps(d['_data'], metadata = d.get('metadata', None))
        
    def to_dict(self):
        return self.__dict__
    
    #TODO: #WM: Test this, it's not done! It should filter out the laps that occur outside of the start/end times that 
    def time_slice(self, t_start=None, t_stop=None):
        # raise NotImplementedError
        laps_obj = deepcopy(self)
        included_indicies = (((laps_obj._data.start >= t_start) & (laps_obj._data.start <= t_stop)) & ((laps_obj._data.stop >= t_start) & (laps_obj._data.stop <= t_stop)))
        included_df = laps_obj._data[included_indicies]
        # included_df = laps_obj._data[((laps_obj._data.start >= t_start) & (laps_obj._data.start <= t_stop))]
        return Laps(included_df, metadata=laps_obj.metadata)
        
        
