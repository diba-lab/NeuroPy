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
        # df_fieldnames = ['lap_dir']
        # position_df.loc[np.logical_not(np.isnan(position_df.lap.to_numpy())), 'lap_dir'] = np.mod(position_df.loc[np.logical_not(np.isnan(position_df.lap.to_numpy())), 'lap'], 2.0)
        # laps["label"] = laps["label"].astype("str")
        # self._data = laps.sort_values(by=['start']) # sorts all values in ascending order
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
        # included_indicies = np.isin(self.lap_id, lap_ids)
        # sliced_copy.lap_id = sliced_copy.lap_id[included_indicies]
        # sliced_copy.laps_spike_counts = sliced_copy.laps_spike_counts[included_indicies]
        # sliced_copy.lap_start_stop_flat_idx = sliced_copy.lap_start_stop_flat_idx[included_indicies, :]
        # sliced_copy.lap_start_stop_time = sliced_copy.lap_start_stop_time[included_indicies, :]
        # return sliced_copy
        
    @classmethod
    def _update_dataframe_computed_vars(cls, laps_df: pd.DataFrame):
        laps_df[['lap_id','maze_id','start_spike_index', 'end_spike_index']] = laps_df[['lap_id','maze_id','start_spike_index', 'end_spike_index']].astype('int')
        laps_df['num_spikes'] = laps_df['end_spike_index'] - laps_df['start_spike_index']
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
    def from_dataframe(cls, df):
        return cls(df)

    def to_dataframe(self):
        return self._data
        # return pd.DataFrame({'id': self.lap_id, 'start':self.lap_start_stop_time[:,0],'stop':self.lap_start_stop_time[:,1],'label':self.lap_id})
                
    def get_lap_flat_indicies(self, lap_id):
        return self._data.loc[lap_id, ['start_spike_index', 'end_spike_index']].to_numpy()
        # start_stop = self.lap_start_stop_flat_idx[lap_id,:] # array([ 15841., 900605.]) the start_stop time for the first lap
        # return start_stop[0], start_stop[1]

    def get_lap_times(self, lap_id):
        return self._data.loc[lap_id, ['start', 'stop']].to_numpy()
        # start_stop = self.lap_start_stop_time[lap_id,:] # array([ 886.4489000000001, 931.6386]) the start_stop time for the first lap
        # return start_stop[0], start_stop[1]
    
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
        
        

# ## OLD:
# class LapsOLD(DataWriter):
#     """Class to hold computed info about laps and how they relate to other information like times, flat linear indicies, etc.
    
#     ## TODO: Look at Epoch class for implementation guidance 
#     """

#     def __init__(
#         self,
#         lap_id: np.ndarray,
#         laps_spike_counts=None,
#         lap_start_stop_flat_idx=None,
#         lap_start_stop_time=None,
#         metadata=None,
#     ) -> None:
#         super().__init__(metadata=metadata)

#         self.lap_id = np.array(lap_id)
#         assert (len(laps_spike_counts) == len(lap_id)), "laps_spike_counts first dimension must match number of laps"
#         self.laps_spike_counts = laps_spike_counts
#         assert (lap_start_stop_flat_idx.shape[0] == len(lap_id)), "lap_start_stop_flat_idx first dimension must match number of laps"
#         self.lap_start_stop_flat_idx = lap_start_stop_flat_idx
#         assert (lap_start_stop_time.shape[0] == len(lap_id)), "lap_start_stop_time first dimension must match number of laps"
#         self.lap_start_stop_time = lap_start_stop_time

#     # @property
#     # def n_laps(self):
#     #     return len(self.lap_id)
        
#     # @property
#     # def starts(self):
#     #     return self.lap_start_stop_time[:,0]

#     # @property
#     # def stops(self):
#     #     return self.lap_start_stop_time[:,1]
    
    
#     @staticmethod
#     def from_dict(d: dict):
#         return Laps(d['lap_id'], laps_spike_counts = d['laps_spike_counts'], lap_start_stop_flat_idx = d['lap_start_stop_flat_idx'],
#                     lap_start_stop_time = d['lap_start_stop_time'], metadata = d.get('metadata', None))
        
        
#     def to_dict(self, recurrsively=False):
#         simple_dict = self.__dict__
#         # if recurrsively:
#         #     simple_dict['paradigm'] = simple_dict['paradigm'].to_dict()
#         #     simple_dict['position'] = simple_dict['position'].to_dict()
#         #     simple_dict['neurons'] = simple_dict['neurons'].to_dict()        
#         return simple_dict
    
#     #TODO: #WM: Fix this, it's not done! It should filter out the laps that occur outside of the start/end times that 
#     def time_slice(self, t_start=None, t_stop=None):
#         # t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        
#         sliced_copy = deepcopy(self) # get copy of the dataframe
        
#         # slice forward: 
#         new_t_start = t_start
#         include_indicies = np.argwhere(new_t_start < sliced_copy.stops)
#         if (np.size(include_indicies) == 0):
#             # this proposed t_start is after any contained epochs, so the returned object would be empty
#             print('Error: this proposed new_t_start ({}) is after any contained epochs, so the returned object would be empty'.format(new_t_start))
#             raise ValueError
#         first_include_index = include_indicies[0]
#         # print('\t first_include_index: {}'.format(first_include_index))
#         # print('\t changing ._data.loc[first_include_index, (\'start\')] from {} to {}'.format(self._data.loc[first_include_index, ('start')].item(), t))
#         if (first_include_index > 0):
#             # drop the epochs preceeding the first_include_index:
#             drop_indicies = np.arange(first_include_index)
#             print('drop_indicies: {}'.format(drop_indicies))
#             raise NotImplementedError # doesn't yet drop the indicies before the first_include_index
        
#         # np.unique(curr_position_df.lap.to_numpy())
        
        
#         raise NotImplementedError
#         # laps_obj = deepcopy(self)
#         # included_df = flattened_spiketrains.spikes_df[((flattened_spiketrains.spikes_df.t_seconds > t_start) & (flattened_spiketrains.spikes_df.t_seconds < t_stop))]
#         # return FlattenedSpiketrains(included_df, t_start=flattened_spiketrains.t_start, metadata=flattened_spiketrains.metadata)
    
#     # def filtered_by_lap_flat_index(self, lap_indicies):
#     #     return self.filtered_by_lap_id(self.lap_id[lap_indicies])
       
#     # def filtered_by_lap_id(self, lap_ids):
#     #     sliced_copy = deepcopy(self) # get copy of the dataframe
#     #     included_indicies = np.isin(self.lap_id, lap_ids)
#     #     sliced_copy.lap_id = sliced_copy.lap_id[included_indicies]
#     #     sliced_copy.laps_spike_counts = sliced_copy.laps_spike_counts[included_indicies]
#     #     sliced_copy.lap_start_stop_flat_idx = sliced_copy.lap_start_stop_flat_idx[included_indicies, :]
#     #     sliced_copy.lap_start_stop_time = sliced_copy.lap_start_stop_time[included_indicies, :]
#     #     return sliced_copy
    
#     # def get_lap_flat_indicies(self, lap_id):
#     #     start_stop = self.lap_start_stop_flat_idx[lap_id,:] # array([ 15841., 900605.]) the start_stop time for the first lap
#     #     return start_stop[0], start_stop[1]

#     # def get_lap_times(self, lap_id):
#     #     start_stop = self.lap_start_stop_time[lap_id,:] # array([ 886.4489000000001, 931.6386]) the start_stop time for the first lap
#     #     return start_stop[0], start_stop[1]

#     @staticmethod
#     def build_lap_specific_lists(active_epoch_session, include_empty_lists=True, time_variable_name='t_rel_seconds'):
#         ## Usage:
#         """Usage: lap_specific_subsessions, lap_specific_dataframes, lap_spike_indicies, lap_spike_t_seconds = build_lap_specific_lists(active_epoch_session)
#         """
#         # Group by the lap column:
#         lap_grouped_spikes_df = active_epoch_session.flattened_spiketrains.spikes_df.groupby(['lap']) #  as_index=False keeps the original index
#         lap_specific_subsessions = list()
#         for i in np.arange(active_epoch_session.laps.n_laps):
#             curr_lap_id = active_epoch_session.laps.lap_id[i]
#             if curr_lap_id in lap_grouped_spikes_df.groups.keys():
#                 curr_lap_dataframe = lap_grouped_spikes_df.get_group(curr_lap_id)
#                 lap_specific_subsessions.append(active_epoch_session.time_slice(curr_lap_dataframe[time_variable_name].values[0], curr_lap_dataframe[time_variable_name].values[-1]))
#             else:
#                 if include_empty_lists:
#                     lap_specific_subsessions.append(None)  
#         return lap_specific_subsessions
            
            
#     # def to_dataframe(self):
#     #     return pd.DataFrame({'id': self.lap_id, 'start':self.lap_start_stop_time[:,0],'stop':self.lap_start_stop_time[:,1],'label':self.lap_id})
        
#     # def as_epoch_obj(self):
#     #     """ Converts into a core.Epoch object containing the time periods """
#     #     return Epoch(self.to_dataframe())
        
#     # @staticmethod
#     # def build_lap_filtered_objects(active_epoch_session, include_empty_lists=True):


