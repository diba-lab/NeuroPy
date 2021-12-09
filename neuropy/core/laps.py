from copy import deepcopy
import numpy as np
import pandas as pd
from neuropy.core.epoch import Epoch

from neuropy.utils.mixins.print_helpers import SimplePrintable
from .datawriter import DataWriter

from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol

## Import:
# from neuropy.core.laps import Laps

# TODO: implement: NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, TimeSlicableObjectProtocol
class Laps(DataWriter):
    """Class to hold computed info about laps and how they relate to other information like times, flat linear indicies, etc.
    
    ## TODO: Look at Epoch class for implementation guidance 
    """

    def __init__(
        self,
        lap_id: np.ndarray,
        laps_spike_counts=None,
        lap_start_stop_flat_idx=None,
        lap_start_stop_time=None,
        metadata=None,
    ) -> None:
        super().__init__(metadata=metadata)

        self.lap_id = np.array(lap_id)
        assert (len(laps_spike_counts) == len(lap_id)), "laps_spike_counts first dimension must match number of laps"
        self.laps_spike_counts = laps_spike_counts
        assert (lap_start_stop_flat_idx.shape[0] == len(lap_id)), "lap_start_stop_flat_idx first dimension must match number of laps"
        self.lap_start_stop_flat_idx = lap_start_stop_flat_idx
        assert (lap_start_stop_time.shape[0] == len(lap_id)), "lap_start_stop_time first dimension must match number of laps"
        self.lap_start_stop_time = lap_start_stop_time

    @property
    def n_laps(self):
        return len(self.lap_id)
        
    @property
    def starts(self):
        return self.lap_start_stop_time[:,0]

    @property
    def stops(self):
        return self.lap_start_stop_time[:,1]
    
    
    @staticmethod
    def from_dict(d: dict):
        return Laps(d['lap_id'], laps_spike_counts = d['laps_spike_counts'], lap_start_stop_flat_idx = d['lap_start_stop_flat_idx'],
                    lap_start_stop_time = d['lap_start_stop_time'], metadata = d.get('metadata', None))
        
        
    def to_dict(self, recurrsively=False):
        simple_dict = self.__dict__
        # if recurrsively:
        #     simple_dict['paradigm'] = simple_dict['paradigm'].to_dict()
        #     simple_dict['position'] = simple_dict['position'].to_dict()
        #     simple_dict['neurons'] = simple_dict['neurons'].to_dict()        
        return simple_dict
    
    #TODO: #WM: Fix this, it's not done! It should filter out the laps that occur outside of the start/end times that 
    def time_slice(self, t_start=None, t_stop=None):
        # t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        
        sliced_copy = deepcopy(self) # get copy of the dataframe
        
        # slice forward: 
        new_t_start = t_start
        include_indicies = np.argwhere(new_t_start < sliced_copy.stops)
        if (np.size(include_indicies) == 0):
            # this proposed t_start is after any contained epochs, so the returned object would be empty
            print('Error: this proposed new_t_start ({}) is after any contained epochs, so the returned object would be empty'.format(new_t_start))
            raise ValueError
        first_include_index = include_indicies[0]
        # print('\t first_include_index: {}'.format(first_include_index))
        # print('\t changing ._data.loc[first_include_index, (\'start\')] from {} to {}'.format(self._data.loc[first_include_index, ('start')].item(), t))
        if (first_include_index > 0):
            # drop the epochs preceeding the first_include_index:
            drop_indicies = np.arange(first_include_index)
            print('drop_indicies: {}'.format(drop_indicies))
            raise NotImplementedError # doesn't yet drop the indicies before the first_include_index
        
        # np.unique(curr_position_df.lap.to_numpy())
        
        
        raise NotImplementedError
        # laps_obj = deepcopy(self)
        # included_df = flattened_spiketrains.spikes_df[((flattened_spiketrains.spikes_df.t_seconds > t_start) & (flattened_spiketrains.spikes_df.t_seconds < t_stop))]
        # return FlattenedSpiketrains(included_df, t_start=flattened_spiketrains.t_start, metadata=flattened_spiketrains.metadata)
    
    def filtered_by_lap_flat_index(self, lap_indicies):
        return self.filtered_by_lap_id(self.lap_id[lap_indicies])
       
    def filtered_by_lap_id(self, lap_ids):
        sliced_copy = deepcopy(self) # get copy of the dataframe
        included_indicies = np.isin(self.lap_id, lap_ids)
        sliced_copy.lap_id = sliced_copy.lap_id[included_indicies]
        sliced_copy.laps_spike_counts = sliced_copy.laps_spike_counts[included_indicies]
        sliced_copy.lap_start_stop_flat_idx = sliced_copy.lap_start_stop_flat_idx[included_indicies, :]
        sliced_copy.lap_start_stop_time = sliced_copy.lap_start_stop_time[included_indicies, :]
        return sliced_copy
    
    def get_lap_flat_indicies(self, lap_id):
        start_stop = self.lap_start_stop_flat_idx[lap_id,:] # array([ 15841., 900605.]) the start_stop time for the first lap
        return start_stop[0], start_stop[1]

    def get_lap_times(self, lap_id):
        start_stop = self.lap_start_stop_time[lap_id,:] # array([ 886.4489000000001, 931.6386]) the start_stop time for the first lap
        return start_stop[0], start_stop[1]

    @staticmethod
    def build_lap_specific_lists(active_epoch_session, include_empty_lists=True, time_variable_name='t_rel_seconds'):
        ## Usage:
        """Usage: lap_specific_subsessions, lap_specific_dataframes, lap_spike_indicies, lap_spike_t_seconds = build_lap_specific_lists(active_epoch_session)
        """
        # Group by the lap column:
        lap_grouped_spikes_df = active_epoch_session.flattened_spiketrains.spikes_df.groupby(['lap']) #  as_index=False keeps the original index
        lap_specific_subsessions = list()
        for i in np.arange(active_epoch_session.laps.n_laps):
            curr_lap_id = active_epoch_session.laps.lap_id[i]
            if curr_lap_id in lap_grouped_spikes_df.groups.keys():
                curr_lap_dataframe = lap_grouped_spikes_df.get_group(curr_lap_id)
                lap_specific_subsessions.append(active_epoch_session.time_slice(curr_lap_dataframe[time_variable_name].values[0], curr_lap_dataframe[time_variable_name].values[-1]))
            else:
                if include_empty_lists:
                    lap_specific_subsessions.append(None)  
        return lap_specific_subsessions
            
            
    def to_dataframe(self):
        return pd.DataFrame({'id': self.lap_id, 'start':self.lap_start_stop_time[:,0],'stop':self.lap_start_stop_time[:,1],'label':self.lap_id})
        
    def as_epoch_obj(self):
        """ Converts into a core.Epoch object containing the time periods """
        return Epoch(self.to_dataframe())
        
    # @staticmethod
    # def build_lap_filtered_objects(active_epoch_session, include_empty_lists=True):


