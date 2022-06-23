from collections import OrderedDict
from typing import Sequence, Union
import numpy as np
import pandas as pd
from copy import deepcopy

from neuropy.core.neuron_identities import NeuronExtendedIdentityTuple
from .datawriter import DataWriter
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin, TimeSlicedMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
from neuropy.utils.mixins.concatenatable import ConcatenationInitializable
from .neurons import NeuronType





@pd.api.extensions.register_dataframe_accessor("spikes")
class SpikesAccessor(TimeSlicedMixin):
    """ Part of the December 2021 Rewrite of the neuropy.core classes to be Pandas DataFrame based and easily manipulatable """
    __time_variable_name = 't_rel_seconds' # currently hardcoded
    
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('cell_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """       
        if "aclu" not in obj.columns or "cell_type" not in obj.columns:
            raise AttributeError("Must have unit id column 'aclu' and 'cell_type' column.")
        if "flat_spike_idx" not in obj.columns:
            raise AttributeError("Must have 'flat_spike_idx' column.")
        if "t" not in obj.columns and "t_seconds" not in obj.columns and "t_rel_seconds" not in obj.columns:
            raise AttributeError("Must have at least one time column: either 't' and 't_seconds', or 't_rel_seconds'.")
        
    @property
    def time_variable_name(self):
        return SpikesAccessor.__time_variable_name
    
    def set_time_variable_name(self, new_time_variable_name):
        print(f'WARNING: SpikesAccessor.set_time_variable_name(new_time_variable_name: {new_time_variable_name}) has been called. Be careful!')
        SpikesAccessor.__time_variable_name = new_time_variable_name
        print('\t time variable changed!')
        
    @property
    def neuron_ids(self):
        """ return the unique cell identifiers (given by the unique values of the 'aclu' column) for this DataFrame """
        unique_aclus = np.unique(self._obj['aclu'].values)
        return unique_aclus
    
    
    @property
    def neuron_probe_tuple_ids(self):
        """ returns a list of NeuronExtendedIdentityTuple tuples where the first element is the shank_id and the second is the cluster_id. Returned in the same order as self.neuron_ids """
        # groupby the multi-index [shank, cluster]:
        # shank_cluster_grouped_spikes_df = self._obj.groupby(['shank','cluster'])
        aclu_grouped_spikes_df = self._obj.groupby(['aclu'])
        shank_cluster_reference_df = aclu_grouped_spikes_df[['aclu','shank','cluster']].first() # returns a df indexed by 'aclu' with only the 'shank' and 'cluster' columns
        output_tuples_list = [NeuronExtendedIdentityTuple(an_id.shank, an_id.cluster, an_id.aclu) for an_id in shank_cluster_reference_df.itertuples()] # returns a list of tuples where the first element is the shank_id and the second is the cluster_id. Returned in the same order as self.neuron_ids
        return output_tuples_list
        
    @property
    def n_total_spikes(self):
        return np.shape(self._obj)[0]

    @property
    def n_neurons(self):
        return len(self.neuron_ids)
    
    
    def get_split_by_unit(self):
        """ returns a list containing the spikes dataframe split by the 'aclu' column. """
        # self.neuron_ids is the list of 'aclu' values found in the spikes_df table.
        return [self._obj.groupby('aclu').get_group(neuron_id) for neuron_id in self.neuron_ids] # dataframes split for each ID
        
    
    # sets the 'x','y' positions by interpolating over a position data frame
    def interpolate_spike_positions(self, position_sampled_times, position_x, position_y, position_linear_pos=None, position_speeds=None):
        spike_timestamp_column_name=self.time_variable_name
        self._obj['x'] = np.interp(self._obj[spike_timestamp_column_name], position_sampled_times, position_x)
        self._obj['y'] = np.interp(self._obj[spike_timestamp_column_name], position_sampled_times, position_y)
        if position_linear_pos is not None:
            self._obj['lin_pos'] = np.interp(self._obj[spike_timestamp_column_name], position_sampled_times, position_linear_pos)
        if position_speeds is not None:
            self._obj['speed'] = np.interp(self._obj[spike_timestamp_column_name], position_sampled_times, position_speeds)
        return self._obj
    
    
    def add_same_cell_ISI_column(self):
        """ Compute the inter-spike-intervals (ISIs) for each cell/unit separately. Meaning the list should be the difference from the current spike to the last spike of the previous unit.
            spikes: curr_active_pipeline.sess.spikes_df
            adds column 'scISI' to spikes df.
            
            # Created Columns:
                'scISI'
        """
        spike_timestamp_column_name=self.time_variable_name # 't_rel_seconds'
        self._obj['scISI'] = -1 # initialize the 'scISI' column (same-cell Intra-spike-interval) to -1

        for (i, a_cell_id) in enumerate(self._obj.spikes.neuron_ids):
            # loop through the cell_ids
            curr_df = self._obj.groupby('aclu').get_group(a_cell_id)
            curr_series_differences = curr_df[spike_timestamp_column_name].diff() # These are the ISIs
            #set the properties for the points in question:
            self._obj.loc[curr_df.index,'scISI'] = curr_series_differences


    def rebuild_fragile_linear_neuron_IDXs(self, debug_print=False):
        """ Rebuilds the 'fragile_linear_neuron_IDX' and 'neuron_IDX' columns from the complete list of 'aclu' values in the current spike dataframe so that they're monotonic and without gaps. Ensures that all the fragile_linear_neuron_IDXs are valid after removing neurons or filtering cells.
        
        History:
            Refactored from a static function in SpikeRenderingBaseMixin.
    
                
        Called by helper_setup_neuron_colors_and_order(...)
        
        # Created/Updated Columns:
            'old_fragile_linear_neuron_IDX'
            'fragile_linear_neuron_IDX'
            'neuron_IDX'
            
        """
        new_neuron_IDXs = np.arange(self.n_neurons)
        neuron_id_to_new_IDX_map = OrderedDict(zip(self.neuron_ids, new_neuron_IDXs)) # provides the new_IDX corresponding to any neuron_id (aclu value)
        return self._overwrite_invalid_fragile_linear_neuron_IDXs(neuron_id_to_new_IDX_map, debug_print=debug_print), neuron_id_to_new_IDX_map

    # This is a stupid way of preserving this functionality, but it was brought in from another class:
    def _overwrite_invalid_fragile_linear_neuron_IDXs(self, neuron_id_to_new_IDX_map, debug_print=False):
        """ A helper function that allows passing in a custom neuron_id_to_new_IDX_map OrderedDict to provide the mapping.
        
        Inputs:
            neuron_id_to_new_IDX_map: an OrderedDict from neuron_ids (aclu values) to a monotonically ascending sequence with no gaps.
        History:
            Refactored from a static function in SpikeRenderingBaseMixin.
        
        Called only by rebuild_fragile_linear_neuron_IDXs()
        
        # Created/Updated Columns:
            'old_fragile_linear_neuron_IDX'
            'fragile_linear_neuron_IDX'
            'neuron_IDX'
        
        """
        if debug_print:
            print("WARNING: Overwriting spikes_df's 'fragile_linear_neuron_IDX' and 'neuron_IDX' columns!")
        self._obj['old_fragile_linear_neuron_IDX'] = self._obj['fragile_linear_neuron_IDX'].copy()
        included_cell_INDEXES = np.array([neuron_id_to_new_IDX_map[an_included_cell_ID] for an_included_cell_ID in self._obj['aclu'].to_numpy()], dtype=int) # get the indexes from the cellIDs
        if debug_print:
            print('\t computed included_cell_INDEXES.')
        self._obj['fragile_linear_neuron_IDX'] = included_cell_INDEXES.copy()
        if debug_print:
            print("\t set self._obj['fragile_linear_neuron_IDX']")
        self._obj['neuron_IDX'] = self._obj['fragile_linear_neuron_IDX'].copy()
        
        if debug_print:
            print("\t set self._obj['neuron_IDX']")
            print("\t done updating 'fragile_linear_neuron_IDX' and 'neuron_IDX'.")
        return self._obj
        


# class FlattenedSpiketrains(StartStopTimesMixin, TimeSlicableObjectProtocol, DataWriter):
class FlattenedSpiketrains(ConcatenationInitializable, NeuronUnitSlicableObjectProtocol, TimeSlicableObjectProtocol, DataWriter):
    """Class to hold flattened spikes for all cells"""
    # flattened_sort_indicies: allow you to sort any naively flattened array (such as position info) using naively_flattened_variable[self.flattened_sort_indicies]
    def __init__(self, spikes_df: pd.DataFrame, time_variable_name = 't_rel_seconds', t_start=0.0, metadata=None):
        super().__init__(metadata=metadata)
        self._time_variable_name = time_variable_name
        self._spikes_df = spikes_df
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
            return self._spikes_df[self._time_variable_name].values

    # @flattened_spike_times.setter
    # def flattened_spike_times(self, arr):
    #     self._flattened_spike_times = arr

    # def add_metadata(self):
    #     pass
    
    def to_dict(self, recurrsively=False):
        d = {'spikes_df': self._spikes_df, 't_start': self.t_start, 'time_variable_name': self._time_variable_name, 'metadata': self.metadata}
        return d

    @staticmethod
    def from_dict(d: dict):
        return FlattenedSpiketrains(d["spikes_df"], t_start=d.get('t_start',0.0), time_variable_name=d.get('time_variable_name','t_rel_seconds'), metadata=d.get('metadata',None))
    
    
    
    def to_dataframe(self):
        df = self._spikes_df.copy()
        # df['t_start'] = self.t_start
        return df


    # @classmethod
    # def from_dataframe(cls, spikes_df, 

    
    def time_slice(self, t_start=None, t_stop=None):
        # t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        flattened_spiketrains = deepcopy(self)
        included_df = flattened_spiketrains.spikes_df[((flattened_spiketrains.spikes_df[self._time_variable_name] > t_start) & (flattened_spiketrains.spikes_df[self._time_variable_name] < t_stop))]
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
            
    # ConcatenationInitializable protocol:
    @classmethod
    def concat(cls, objList: Union[Sequence, np.array]):
        """ Concatenates the object list """
        objList = np.array(objList)
        t_start_times = np.array([obj.t_start for obj in objList])
        sort_idx = list(np.argsort(t_start_times))
        # sort the objList by t_start
        objList = objList[sort_idx]
        new_t_start = objList[0].t_start # new t_start is the earliest t_start in the array
        # Concatenate the elements:
        new_df = pd.concat([obj.to_dataframe() for obj in objList])
        return FlattenedSpiketrains(new_df, t_start=new_t_start, metadata=objList[0].metadata)
        
        
        
    @staticmethod
    def interpolate_spike_positions(spikes_df, position_sampled_times, position_x, position_y, position_linear_pos=None, position_speeds=None, spike_timestamp_column_name='t_rel_seconds'):
        spikes_df['x'] = np.interp(spikes_df[spike_timestamp_column_name], position_sampled_times, position_x)
        spikes_df['y'] = np.interp(spikes_df[spike_timestamp_column_name], position_sampled_times, position_y)
        if position_linear_pos is not None:
            spikes_df['lin_pos'] = np.interp(spikes_df[spike_timestamp_column_name], position_sampled_times, position_linear_pos)
        if position_speeds is not None:
            spikes_df['speed'] = np.interp(spikes_df[spike_timestamp_column_name], position_sampled_times, position_speeds)
        return spikes_df
    
    
        
    # @staticmethod
    # def build_spike_dataframe(active_session, timestamp_scale_factor=(1/1E4), spike_timestamp_column_name='t_rel_seconds'):
    #     """ Builds the FlattenedSpiketrains from the active_session's .neurons object.

    #     Args:
    #         active_session (_type_): _description_
    #         timestamp_scale_factor (tuple, optional): _description_. Defaults to (1/1E4).

    #     Returns:
    #         _type_: _description_
    #     """
    #     return 
    #     flattened_spike_identities = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_ids[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_id for each spike that belongs to that neuron
    #     flattened_spike_types = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_type[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_type for each spike that belongs to that neuron
    #     flattened_spike_linear_unit_spike_idx = np.concatenate([np.arange(active_session.neurons.n_spikes[i]) for i in np.arange(active_session.neurons.n_neurons)]) # gives the index that would be needed to index into a given spike's position within its unit's spiketrain.
    #     flattened_spike_times = np.concatenate(active_session.neurons.spiketrains)
        
    #     # All these flattened arrays start off just concatenated with all the results for the first unit, and then the next, etc. They aren't sorted. flattened_sort_indicies are used to sort them.
    #     # Get the indicies required to sort the flattened_spike_times
    #     flattened_sort_indicies = np.argsort(flattened_spike_times)

    #     num_flattened_spikes = np.size(flattened_sort_indicies)
    #     spikes_df = pd.DataFrame({'flat_spike_idx': np.arange(num_flattened_spikes),
    #         't_seconds':flattened_spike_times[flattened_sort_indicies],
    #         'aclu':flattened_spike_identities[flattened_sort_indicies],
    #         'unit_id': np.array([int(active_session.neurons.reverse_cellID_index_map[original_cellID]) for original_cellID in flattened_spike_identities[flattened_sort_indicies]]),
    #         'flattened_spike_linear_unit_spike_idx': flattened_spike_linear_unit_spike_idx[flattened_sort_indicies],
    #         'cell_type': flattened_spike_types[flattened_sort_indicies]
    #         }
    #     )
        
        
    #     # need 'cell_type':
        
        
        
    #     active_session.neurons.
    #     active_session.c
        
    #     return pd.DataFrame({'flat_spike_idx':sorted_indicies, 'aclu': flattened_spike_identities[sorted_indicies], spike_timestamp_column_name: flattened_spike_times[sorted_indicies]})
        
    #     # return FlattenedSpiketrains(
    #     #     sorted_indicies,
    #     #     flattened_spike_identities[sorted_indicies],
    #     #     flattened_spike_times[sorted_indicies],
    #     #     t_start=active_session.neurons.t_start
    #     # )
        
    #     # raise NotImplementedError


    @staticmethod
    def build_spike_dataframe(active_session, timestamp_scale_factor=(1/1E4), spike_timestamp_column_name='t_rel_seconds'):
        """ Builds the spike_df from the active_session's .neurons object.

        Args:
            active_session (_type_): _description_
            timestamp_scale_factor (tuple, optional): _description_. Defaults to (1/1E4).

        Returns:
            _type_: _description_
        """
        flattened_spike_identities = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_ids[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_id for each spike that belongs to that neuron
        flattened_spike_types = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_type[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_type for each spike that belongs to that neuron
        flattened_spike_shank_identities = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.shank_ids[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_id for each spike that belongs to that neuron
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
            'shank_id': flattened_spike_shank_identities[flattened_sort_indicies],
            'flattened_spike_linear_unit_spike_idx': flattened_spike_linear_unit_spike_idx[flattened_sort_indicies],
            'cell_type': flattened_spike_types[flattened_sort_indicies]
            }
        )
        
        # # # Determine the x and y positions each spike occured for each cell
        # print('build_spike_dataframe(session): interpolating {} position values over {} spike timepoints. This may take a minute...'.format(len(active_session.position.time), num_flattened_spikes))
        # ## TODO: spike_positions_list is in terms of cell_ids for some reason, maybe it's temporary?
        # # num_cells = len(spike_list)
        # # spike_positions_list = list()
        # # for cell_id in np.arange(num_cells):
        # #     spike_positions_list.append(np.vstack((np.interp(spike_list[cell_id], t, x), np.interp(spike_list[cell_id], t, y), np.interp(spike_list[cell_id], t, linear_pos), np.interp(spike_list[cell_id], t, speeds))))

        # # # Gets the flattened spikes, sorted in ascending timestamp for all cells.
        # # # Build the Active UnitIDs        
        # # # reverse_cellID_idx_lookup_map: get the current filtered index for this cell given using reverse_cellID_idx_lookup_map
        # # ## Build the flattened spike positions list
        # # flattened_spike_positions_list = np.concatenate(tuple(spike_positions_list), axis=1) # needs tuple(...) to conver the list into a tuple, which is the format it expects
        # # flattened_spike_positions_list = flattened_spike_positions_list[:, flattened_sort_indicies] # ensure the positions are ordered the same as the other flattened items so they line up
        # # ## flattened_spike_positions_list = np.vstack((np.interp(spike_list[cell_id], t, x), np.interp(spike_list[cell_id], t, y), np.interp(spike_list[cell_id], t, linear_pos), np.interp(spike_list[cell_id], t, speeds))
        
        # spikes_df['x'] = np.interp(spikes_df[spike_timestamp_column_name], active_session.position.time, active_session.position.x)
        # spikes_df['y'] = np.interp(spikes_df[spike_timestamp_column_name], active_session.position.time, active_session.position.y)
        # spikes_df['linear_pos'] = np.interp(spikes_df[spike_timestamp_column_name], active_session.position.time, active_session.position.linear_pos)
        # spikes_df['speed'] = np.interp(spikes_df[spike_timestamp_column_name], active_session.position.time, active_session.position.speed)
        
        # ## TODO: you could reconstruct flattened_spike_positions_list if you wanted.         
        # # print('flattened_spike_positions_list: {}'.format(np.shape(flattened_spike_positions_list))) # (2, 19647)
        # # spikes_df['x'] = flattened_spike_positions_list[0, :]
        # # spikes_df['y'] = flattened_spike_positions_list[1, :]
        # # spikes_df['linear_pos'] = flattened_spike_positions_list[2, :]
        # # spikes_df['speed'] = flattened_spike_positions_list[3, :]
        
        # spikes_df['t'] = spikes_df[spike_timestamp_column_name] / timestamp_scale_factor

        # print('\t done.')
        # spikes_df = pd.DataFrame({'flat_spike_idx': np.arange(num_flattened_spikes),
        #     spike_timestamp_column_name:flattened_spike_times[flattened_sort_indicies],
        #     'aclu':flattened_spike_identities[flattened_sort_indicies],
        #     'unit_id': np.array([int(reverse_cellID_idx_lookup_map[original_cellID]) for original_cellID in flattened_spike_identities[flattened_sort_indicies]]),
        #     }
        # )
        return spikes_df
        