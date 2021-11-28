import numpy as np
import pandas as pd
from copy import deepcopy


from .datawriter import DataWriter
from ..utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from ..utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol

from .neurons import NeuronType


# class FlattenedSpiketrains(StartStopTimesMixin, TimeSlicableObjectProtocol, DataWriter):
class FlattenedSpiketrains(NeuronUnitSlicableObjectProtocol, TimeSlicableObjectProtocol, DataWriter):
    """Class to hold flattened spikes for all cells"""
    # flattened_sort_indicies: allow you to sort any naively flattened array (such as position info) using naively_flattened_variable[self.flattened_sort_indicies]
    def __init__(self, spikes_df: pd.DataFrame, t_start=0.0, metadata=None):
        super().__init__(metadata=metadata)
        self._spikes_df = spikes_df
        # self.flattened_sort_indicies = flattened_sort_indicies
        # self.flattened_spike_identities = flattened_spike_identities
        # self.flattened_spike_times = flattened_spike_times
        self.t_start = t_start
        self.metadata = metadata
        
    # @staticmethod
    # def from_separate_flattened_variables(flattened_sort_indicies: np.ndarray, flattened_spike_identities: np.ndarray,
    #     flattened_spike_times: np.ndarray, t_start=0.0, metadata=None):
    #     self.flattened_sort_indicies = flattened_sort_indicies
    #     self.flattened_spike_identities = flattened_spike_identities
    #     self.flattened_spike_times = flattened_spike_times
        
    #     'qclu'

    @property
    def spikes_df(self):
        """The spikes_df property."""
        return self._spikes_df

    # @spikes_df.setter
    # def spikes_df(self, value):
    #     self._spikes_df = value
        
    @property
    def flattened_sort_indicies(self):
        if self._spikes_df is None:
            return self._flattened_sort_indicies
        else:
            return self._spikes_df['flat_spike_idx'].values ## TODO: this might be wrong
    # @flattened_sort_indicies.setter
    # def flattened_sort_indicies(self, arr):
    #     self._flattened_sort_indicies = arr

    @property
    def flattened_spike_identities(self):
        if self._spikes_df is None:
            return self._flattened_spike_identities
        else:
            return self._spikes_df['aclu'].values

    # @flattened_spike_identities.setter
    # def flattened_spike_identities(self, arr):
    #     self._flattened_spike_identities = arr

    @property
    def flattened_spike_times(self):
        if self._spikes_df is None:
            return self._flattened_spike_times
        else:
            return self._spikes_df['t_seconds'].values

    # @flattened_spike_times.setter
    # def flattened_spike_times(self, arr):
    #     self._flattened_spike_times = arr

    # def add_metadata(self):
    #     pass
    
    def to_dict(self, recurrsively=False):
        d = {'spikes_df': self._spikes_df, 't_start': self.t_start, 'metadata': self.metadata}
        return d

    @staticmethod
    def from_dict(d: dict):
        return FlattenedSpiketrains(d["spikes_df"], t_start=d.get('t_start',0.0), metadata=d.get('metadata',None))
    
    
    def to_dataframe(self):
        df = self._spikes_df.copy()
        # df['t_start'] = self.t_start
        return df


    def time_slice(self, t_start=None, t_stop=None):
        # t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        flattened_spiketrains = deepcopy(self)
        included_df = flattened_spiketrains.spikes_df[((flattened_spiketrains.spikes_df.t_seconds > t_start) & (flattened_spiketrains.spikes_df.t_seconds < t_stop))]
        return FlattenedSpiketrains(included_df, t_start=flattened_spiketrains.t_start, metadata=flattened_spiketrains.metadata)
        
    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids):
        """Returns neurons object with neuron_ids equal to ids"""
        flattened_spiketrains = deepcopy(self)
        included_df = flattened_spiketrains.spikes_df[np.isin(flattened_spiketrains.spikes_df.aclu, ids)]
        return FlattenedSpiketrains(included_df, t_start=flattened_spiketrains.t_start, metadata=flattened_spiketrains.metadata)
    

    def get_neuron_type(self, query_neuron_type):
        """ filters self by the specified query_neuron_type, only returning neurons that match. """
        if isinstance(query_neuron_type, NeuronType):
            query_neuron_type = query_neuron_type
        elif isinstance(query_neuron_type, str):
            query_neuron_type_str = query_neuron_type
            query_neuron_type = NeuronType.from_string(query_neuron_type_str) ## Works
        else:
            print('error!')
            return []
        flattened_spiketrains = deepcopy(self)
        included_df = flattened_spiketrains.spikes_df[(flattened_spiketrains.spikes_df.cell_type == query_neuron_type)]
        return FlattenedSpiketrains(included_df, t_start=flattened_spiketrains.t_start, metadata=flattened_spiketrains.metadata)
        
    
    
    # def to_dict(self, recurrsively=False):
    #     return {
    #         "flattened_sort_indicies": self.flattened_sort_indicies,
    #         "flattened_spike_identities": self.flattened_spike_identities,
    #         "flattened_spike_times": self.flattened_spike_times,
    #         "t_start": self.t_start,
    #         "metadata": self.metadata,
    #     }

    # @staticmethod
    # def from_dict(d):
    #     return FlattenedSpiketrains(
    #         flattened_sort_indicies=d["flattened_sort_indicies"],
    #         flattened_spike_times=d["flattened_spike_times"],
    #         flattened_spike_identities=d["flattened_spike_identities"],
    #         t_start=d["t_start"],
    #         metadata=d["metadata"],
    #     )








# class FlattenedSpiketrains(StartStopTimesMixin, TimeSlicableObjectProtocol, DataWriter):
#     """Class to hold flattened spikes for all cells"""
#     # flattened_sort_indicies: allow you to sort any naively flattened array (such as position info) using naively_flattened_variable[self.flattened_sort_indicies]
#     def __init__(
#         self,
#         flattened_sort_indicies: np.ndarray,
#         flattened_spike_identities: np.ndarray,
#         flattened_spike_times: np.ndarray,
#         t_start=0.0,
#         metadata=None,
#     ) -> None:
#         super().__init__(metadata=metadata)
#         self.flattened_sort_indicies = flattened_sort_indicies
#         self.flattened_spike_identities = flattened_spike_identities
#         self.flattened_spike_times = flattened_spike_times
#         self.t_start = t_start
#         self.metadata = metadata

#     @property
#     def flattened_sort_indicies(self):
#         return self._flattened_sort_indicies
#     @flattened_sort_indicies.setter
#     def flattened_sort_indicies(self, arr):
#         self._flattened_sort_indicies = arr

#     @property
#     def flattened_spike_identities(self):
#         return self._flattened_spike_identities
#     @flattened_spike_identities.setter
#     def flattened_spike_identities(self, arr):
#         self._flattened_spike_identities = arr

#     @property
#     def flattened_spike_times(self):
#         return self._flattened_spike_times

#     @flattened_spike_times.setter
#     def flattened_spike_times(self, arr):
#         self._flattened_spike_times = arr

#     def add_metadata(self):
#         pass

#     def time_slice(self, t_start=None, t_stop=None):
#         t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
#         flattened_spiketrains = deepcopy(self)
#         included_indicies = ((flattened_spiketrains.flattened_spike_times > t_start) & (flattened_spiketrains.flattened_spike_times < t_stop))
#         flattened_spike_times = flattened_spiketrains.flattened_spike_times[included_indicies]
#         flattened_spike_identities = flattened_spiketrains.flattened_spike_identities[included_indicies]
#         return FlattenedSpiketrains(
#             flattened_spike_times=flattened_spiketrains.flattened_spike_times[included_indicies],
#             flattened_spike_identities=flattened_spiketrains.flattened_spike_identities[included_indicies],
#             t_start=flattened_spiketrains.t_start,
#             metadata=flattened_spiketrains.metadata,
#         )


#     def to_dict(self, recurrsively=False):
#         return {
#             "flattened_sort_indicies": self.flattened_sort_indicies,
#             "flattened_spike_identities": self.flattened_spike_identities,
#             "flattened_spike_times": self.flattened_spike_times,
#             "t_start": self.t_start,
#             "metadata": self.metadata,
#         }

#     @staticmethod
#     def from_dict(d):
#         return FlattenedSpiketrains(
#             flattened_sort_indicies=d["flattened_sort_indicies"],
#             flattened_spike_times=d["flattened_spike_times"],
#             flattened_spike_identities=d["flattened_spike_identities"],
#             t_start=d["t_start"],
#             metadata=d["metadata"],
#         )

