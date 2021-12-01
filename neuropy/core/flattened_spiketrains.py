import numpy as np
import pandas as pd
from copy import deepcopy


from .datawriter import DataWriter
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol

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
        
    @staticmethod
    def build_spike_dataframe(active_session, timestamp_scale_factor=(1/1E4)):
        flattened_spike_identities = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_ids[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_id for each spike that belongs to that neuron
        flattened_spike_types = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_type[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_type for each spike that belongs to that neuron
        flattened_spike_linear_unit_spike_idx = np.concatenate([np.arange(active_session.neurons.n_spikes[i]) for i in np.arange(active_session.neurons.n_neurons)]) # gives the index that would be needed to index into a given spike's position within its unit's spiketrain.
        flattened_spike_times = np.concatenate(active_session.neurons.spiketrains)
        
        # All these flattened arrays start off just concatenated with all the results for the first unit, and then the next, etc. They aren't sorted. flattened_sort_indicies are used to sort them.
        # Get the indicies required to sort the flattened_spike_times
        flattened_sort_indicies = np.argsort(flattened_spike_times)

        num_flattened_spikes = np.size(flattened_sort_indicies)
        spikes_df = pd.DataFrame({'flat_spike_idx': np.arange(num_flattened_spikes),
            't_seconds':flattened_spike_times[flattened_sort_indicies],
            'aclu':flattened_spike_identities[flattened_sort_indicies],
            'unit_id': np.array([int(active_session.neurons.reverse_cellID_index_map[original_cellID]) for original_cellID in flattened_spike_identities[flattened_sort_indicies]]),
            'flattened_spike_linear_unit_spike_idx': flattened_spike_linear_unit_spike_idx[flattened_sort_indicies],
            'cell_type': flattened_spike_types[flattened_sort_indicies]
            }
        )
        
        # # Determine the x and y positions each spike occured for each cell
        print('build_spike_dataframe(session): interpolating {} position values over {} spike timepoints. This may take a minute...'.format(len(active_session.position.time), num_flattened_spikes))
        ## TODO: spike_positions_list is in terms of cell_ids for some reason, maybe it's temporary?
        # num_cells = len(spike_list)
        # spike_positions_list = list()
        # for cell_id in np.arange(num_cells):
        #     spike_positions_list.append(np.vstack((np.interp(spike_list[cell_id], t, x), np.interp(spike_list[cell_id], t, y), np.interp(spike_list[cell_id], t, linear_pos), np.interp(spike_list[cell_id], t, speeds))))

        # # Gets the flattened spikes, sorted in ascending timestamp for all cells.
        # # Build the Active UnitIDs        
        # # reverse_cellID_idx_lookup_map: get the current filtered index for this cell given using reverse_cellID_idx_lookup_map
        # ## Build the flattened spike positions list
        # flattened_spike_positions_list = np.concatenate(tuple(spike_positions_list), axis=1) # needs tuple(...) to conver the list into a tuple, which is the format it expects
        # flattened_spike_positions_list = flattened_spike_positions_list[:, flattened_sort_indicies] # ensure the positions are ordered the same as the other flattened items so they line up
        # ## flattened_spike_positions_list = np.vstack((np.interp(spike_list[cell_id], t, x), np.interp(spike_list[cell_id], t, y), np.interp(spike_list[cell_id], t, linear_pos), np.interp(spike_list[cell_id], t, speeds))
        
        spikes_df['x'] = np.interp(spikes_df['t_seconds'], active_session.position.time, active_session.position.x)
        spikes_df['y'] = np.interp(spikes_df['t_seconds'], active_session.position.time, active_session.position.y)
        spikes_df['linear_pos'] = np.interp(spikes_df['t_seconds'], active_session.position.time, active_session.position.linear_pos)
        spikes_df['speed'] = np.interp(spikes_df['t_seconds'], active_session.position.time, active_session.position.speed)
        
        ## TODO: you could reconstruct flattened_spike_positions_list if you wanted.         
        # print('flattened_spike_positions_list: {}'.format(np.shape(flattened_spike_positions_list))) # (2, 19647)
        # spikes_df['x'] = flattened_spike_positions_list[0, :]
        # spikes_df['y'] = flattened_spike_positions_list[1, :]
        # spikes_df['linear_pos'] = flattened_spike_positions_list[2, :]
        # spikes_df['speed'] = flattened_spike_positions_list[3, :]
        
        spikes_df['t'] = spikes_df['t_seconds'] / timestamp_scale_factor

        print('\t done.')
        # spikes_df = pd.DataFrame({'flat_spike_idx': np.arange(num_flattened_spikes),
        #     't_seconds':flattened_spike_times[flattened_sort_indicies],
        #     'aclu':flattened_spike_identities[flattened_sort_indicies],
        #     'unit_id': np.array([int(reverse_cellID_idx_lookup_map[original_cellID]) for original_cellID in flattened_spike_identities[flattened_sort_indicies]]),
        #     }
        # )
        return spikes_df
        