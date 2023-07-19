from collections import OrderedDict
from typing import Sequence, Union
# from warnings import warn
import logging

from neuropy.utils.misc import safe_pandas_get_group
module_logger = logging.getLogger('com.PhoHale.neuropy') # create logger
import numpy as np
import pandas as pd
from copy import deepcopy

from neuropy.core.neuron_identities import NeuronExtendedIdentityTuple
from neuropy.utils.mixins.binning_helpers import BinningInfo # for add_binned_time_column
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter
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
        return self.__time_variable_name
    
    def set_time_variable_name(self, new_time_variable_name):
        module_logger.warning(f'WARNING: SpikesAccessor.set_time_variable_name(new_time_variable_name: {new_time_variable_name}) has been called. Be careful!')
        if self._obj.spikes.time_variable_name == new_time_variable_name:
            # no change in the time_variable_name:
            module_logger.warning(f'\t no change in time_variable_name. It will remain {new_time_variable_name}.')
        else:
            assert new_time_variable_name in self._obj.columns, f"spikes_df.spikes.set_time_variable_name(new_time_variable_name='{new_time_variable_name}') was called but '{new_time_variable_name}' is not a column of the dataframe! Original spk_df.spikes.time_variable_name: '{self._obj.spikes.time_variable_name}'.\n\t valid_columns: {list(self._obj.columns)}"
            # otherwise it's okay and we can continue
            original_time_variable_name = self._obj.spikes.time_variable_name
            SpikesAccessor.__time_variable_name = new_time_variable_name # set for the class
            self.__time_variable_name = new_time_variable_name # also set for the instance, as the class properties won't be retained when doing deepcopy and hopefully the instance properties will.
            module_logger.warning(f"\t time variable changed from '{original_time_variable_name}' to '{new_time_variable_name}'.")
            print('\t time variable changed!')
        

    @property
    def times(self):
        """ convenience property to access the times of the spikes in the dataframe 
            ## TODO: why doesn't this have a `times` property to access `self._obj[self.time_variable_name].values`?
        """
        return self._obj[self.time_variable_name].values

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
    
    
    def get_split_by_unit(self, included_neuron_ids=None):
        """ returns a list containing the spikes dataframe split by the 'aclu' column. """
        # self.neuron_ids is the list of 'aclu' values found in the spikes_df table.
        if included_neuron_ids is None:
            included_neuron_ids = self.neuron_ids
        return [safe_pandas_get_group(self._obj.groupby('aclu'), neuron_id) for neuron_id in included_neuron_ids] # dataframes split for each ID
    

    def sliced_by_neuron_id(self, included_neuron_ids):
        """ gets the slice of spikes with the specified `included_neuron_ids` """
        if included_neuron_ids is None:
            included_neuron_ids = self.neuron_ids
        return self._obj[self._obj['aclu'].isin(included_neuron_ids)] ## restrict to only the shared aclus for both short and long
        

    def get_unit_spiketrains(self, included_neuron_ids=None):
        """ returns an array of the spiketrains (an array of the times that each spike occured) for each unit """
        return np.asarray([a_unit_spikes_df[self.time_variable_name].to_numpy() for a_unit_spikes_df in self.get_split_by_unit(included_neuron_ids=included_neuron_ids)])
        

    def sliced_by_neuron_type(self, query_neuron_type):
        """ returns a copy of self._obj filtered by the specified query_neuron_type, only returning neurons that match.
            e.g. query_neuron_type = NeuronType.PYRAMIDAL 
                or query_neuron_type = 'PYRAMIDAL' 
                or query_neuron_type = 'Pyr'
         """
        try:
            # Try with the assumption that it's a string first:
            query_neuron_type = NeuronType.from_string(query_neuron_type) ## Works
        except ValueError:
            # Try to interpret as a NeuronType object:
            query_neuron_type = query_neuron_type
        except Exception as e:
            raise e
        
        # Compare via .shortClassName for both query_neuron_type and self._obj.cell_type
        inclusion_mask = np.isin(np.array([a_type.shortClassName for a_type in self._obj.cell_type]), [query_neuron_type.shortClassName])
        return self._obj.loc[inclusion_mask, :].copy()
        # return self._obj[np.isin(np.array([a_type.shortClassName for a_type in self._obj.cell_type]), [query_neuron_type.shortClassName])]
        

    # ==================================================================================================================== #
    # Additive Mutating Functions: Adds or Update Columns in the Dataframe                                                 #
    # ==================================================================================================================== #
    
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
            
            TODO: PERFORMANCE: This takes over a minute to compute for Bapun's data.
            
            # Created Columns:
                'scISI'
                
            # Called only from _default_add_spike_scISIs_if_needed(...)
        """
        if 'scISI' in self._obj.columns:
            print(f'column "scISI" already exists in df! Skipping recomputation.')
            return
        else:
            spike_timestamp_column_name=self.time_variable_name # 't_rel_seconds'
            self._obj['scISI'] = -1 # initialize the 'scISI' column (same-cell Intra-spike-interval) to -1

            for (i, a_cell_id) in enumerate(self._obj.spikes.neuron_ids):
                # loop through the cell_ids
                curr_df = safe_pandas_get_group(self._obj.groupby('aclu'), a_cell_id)
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
        assert 'aclu' in self._obj.columns, f"spikes_df is missing the required 'aclu' column!"
        if 'fragile_linear_neuron_IDX' in self._obj.columns:
            # Backup the old value if it exists:
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

    def add_binned_time_column(self, time_window_edges, time_window_edges_binning_info:BinningInfo, debug_print:bool=False):
        """ adds a 'binned_time' column to spikes_df given the time_window_edges and time_window_edges_binning_info provided 
        
        """
        spike_timestamp_column_name = self.time_variable_name # 't_rel_seconds'
        if not (np.shape(time_window_edges)[0] < np.shape(self._obj)[0]):
            print('WARNING: PREVIOUSLY ASSERT: ')
            print(f'\t spikes_df[time_variable_name]: {np.shape(self._obj[spike_timestamp_column_name])} should be less than time_window_edges: {np.shape(time_window_edges)}!') # 2023-03-06 - I no longer know why this should be the case.... more spikes than time windows?
            # assert np.shape(time_window_edges)[0] < np.shape(self._obj)[0], f'self._obj[time_variable_name]: {np.shape(self._obj[time_variable_name])} should be less than time_window_edges: {np.shape(time_window_edges)}!'

        if debug_print:
            print(f'self._obj[time_variable_name]: {np.shape(self._obj[spike_timestamp_column_name])}\ntime_window_edges: {np.shape(time_window_edges)}')
            # assert (np.shape(out_digitized_variable_bins)[0] == np.shape(self._obj)[0]), f'np.shape(out_digitized_variable_bins)[0]: {np.shape(out_digitized_variable_bins)[0]} should equal np.shape(self._obj)[0]: {np.shape(self._obj)[0]}'
            print(time_window_edges_binning_info)

        bin_labels = time_window_edges_binning_info.bin_indicies[1:] # edge bin indicies: [0,     1,     2, ..., 11878, 11879, 11880][1:] -> [ 1,     2, ..., 11878, 11879, 11880]
        self._obj['binned_time'] = pd.cut(self._obj[spike_timestamp_column_name].to_numpy(), bins=time_window_edges, include_lowest=True, labels=bin_labels) # same shape as the input data (time_binned_self._obj: (69142,))
        return self._obj

    



class FlattenedSpiketrains(ConcatenationInitializable, NeuronUnitSlicableObjectProtocol, TimeSlicableObjectProtocol, DataWriter):
    """Class to hold flattened spikes for all cells"""
    # flattened_sort_indicies: allow you to sort any naively flattened array (such as position info) using naively_flattened_variable[self.flattened_sort_indicies]
    def __init__(self, spikes_df: pd.DataFrame, time_variable_name = 't_rel_seconds', t_start=0.0, metadata=None):
        super().__init__(metadata=metadata)
        self._time_variable_name = time_variable_name
        self._spikes_df = spikes_df
        self.t_start = t_start
        self.metadata = metadata
        
    @property
    def spikes_df(self):
        """The spikes_df property."""
        return self._spikes_df

    @property
    def flattened_sort_indicies(self):
        if self._spikes_df is None:
            return self._flattened_sort_indicies
        else:
            return self._spikes_df['flat_spike_idx'].values ## TODO: this might be wrong

    @property
    def flattened_spike_identities(self):
        if self._spikes_df is None:
            return self._flattened_spike_identities
        else:
            return self._spikes_df['aclu'].values
        
    @property
    def flattened_spike_times(self):
        if self._spikes_df is None:
            return self._flattened_spike_times
        else:
            return self._spikes_df[self._time_variable_name].values
    
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
    
    

    @staticmethod
    def build_spike_dataframe(active_session, timestamp_scale_factor=(1/1E4), spike_timestamp_column_name='t_rel_seconds', progress_tracing=True):
        """ Builds the spike_df from the active_session's .neurons object.

        Args:
            active_session (_type_): _description_
            timestamp_scale_factor (tuple, optional): _description_. Defaults to (1/1E4).

        Returns:
            _type_: _description_
        
        # TODO: only use ProgressMessagePrinter if progress_tracing is True
        
        """
        
        with ProgressMessagePrinter('build_spike_dataframe(...)', 'Computing', 'flattened_spike_identities'):
            flattened_spike_identities = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_ids[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_id for each spike that belongs to that neuron
    
        with ProgressMessagePrinter('build_spike_dataframe(...)', 'Computing', 'flattened_spike_types'):
            flattened_spike_types = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_type[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_type for each spike that belongs to that neuron
        with ProgressMessagePrinter('build_spike_dataframe(...)', 'Computing', 'flattened_spike_shank_identities'):
            flattened_spike_shank_identities = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.shank_ids[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_id for each spike that belongs to that neuron
        with ProgressMessagePrinter('build_spike_dataframe(...)', 'Computing', 'flattened_spike_linear_unit_spike_idx'):
            flattened_spike_linear_unit_spike_idx = np.concatenate([np.arange(active_session.neurons.n_spikes[i]) for i in np.arange(active_session.neurons.n_neurons)]) # gives the index that would be needed to index into a given spike's position within its unit's spiketrain.
        with ProgressMessagePrinter('build_spike_dataframe(...)', 'Computing', 'flattened_spike_times'):
            flattened_spike_times = np.concatenate(active_session.neurons.spiketrains)
        
        # All these flattened arrays start off just concatenated with all the results for the first unit, and then the next, etc. They aren't sorted. flattened_sort_indicies are used to sort them.
        # Get the indicies required to sort the flattened_spike_times
        with ProgressMessagePrinter('build_spike_dataframe(...)', 'Sorting', 'flattened_sort_indicies'):
            flattened_sort_indicies = np.argsort(flattened_spike_times)

        num_flattened_spikes = np.size(flattened_sort_indicies)
    
        with ProgressMessagePrinter('build_spike_dataframe(...)', f'Building final dataframe (containing {num_flattened_spikes} spikes)', 'spikes_df'):
            spikes_df = pd.DataFrame({'flat_spike_idx': np.arange(num_flattened_spikes),
                't_seconds':flattened_spike_times[flattened_sort_indicies],
                'aclu':flattened_spike_identities[flattened_sort_indicies],
                'unit_id': np.array([int(active_session.neurons.reverse_cellID_index_map[original_cellID]) for original_cellID in flattened_spike_identities[flattened_sort_indicies]]),
                'shank': flattened_spike_shank_identities[flattened_sort_indicies],
                'intra_unit_spike_idx': flattened_spike_linear_unit_spike_idx[flattened_sort_indicies],
                'cell_type': flattened_spike_types[flattened_sort_indicies]
                }
            )
        
            spikes_df[['shank', 'aclu']] = spikes_df[['shank', 'aclu']].astype('int') # convert integer calumns to correct datatype

        # Renaming {'shank_id':'shank', 'flattened_spike_linear_unit_spike_idx':'intra_unit_spike_idx'}
        return spikes_df

        