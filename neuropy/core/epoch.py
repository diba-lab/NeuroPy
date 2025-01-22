from typing import Optional, List, Dict, Tuple, Union
from importlib import metadata
import warnings
from warnings import warn
import numpy as np
from nptyping import NDArray
import pandas as pd
import portion as P # Required for interval search: portion~=2.3.0

from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable, DataFrameInitializable
from .datawriter import DataWriter
from neuropy.utils.mixins.print_helpers import SimplePrintable, OrderedMeta
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicedMixin, TimeColumnAliasesProtocol
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs, deduplicate_epochs # for EpochsAccessor's .get_non_overlapping_df()
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin


def find_data_indicies_from_epoch_times(a_df: pd.DataFrame, epoch_times: NDArray, t_column_names=None, atol:float=1e-3, not_found_action='skip_index', debug_print=False) -> NDArray:
    """ returns the matching data indicies corresponding to the epoch [start, stop] times 
    epoch_times: S x 2 array of epoch start/end times


    skip_index: drops indicies that can't be found, meaning that the number of returned indicies might be less than len(epoch_times)


    Returns: (S, ) array of data indicies corresponding to the times.

    Uses:
        from neuropy.core.epoch import find_data_indicies_from_epoch_times

        selection_start_stop_times = deepcopy(active_epochs_df[['start', 'stop']].to_numpy())
        print(f'np.shape(selection_start_stop_times): {np.shape(selection_start_stop_times)}')

        test_epochs_data_df: pd.DataFrame = deepcopy(ripple_simple_pf_pearson_merged_df)
        print(f'np.shape(test_epochs_data_df): {np.shape(test_epochs_data_df)}')

        # 2D_search (for both start, end times):
        found_data_indicies = find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=selection_start_stop_times)
        print(f'np.shape(found_data_indicies): {np.shape(found_data_indicies)}')

        # 1D_search (only for start times):
        found_data_indicies_1D_search = find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=np.squeeze(selection_start_stop_times[:, 0]))
        print(f'np.shape(found_data_indicies_1D_search): {np.shape(found_data_indicies_1D_search)}')
        found_data_indicies_1D_search

        assert np.array_equal(found_data_indicies, found_data_indicies_1D_search)
    

    - [X] FIXED 2024-03-04 19:55 - This function was peviously incorrect and could return multiple matches for each passed time due to the tolerance.
        
    """
    def _subfn_find_epoch_times(epoch_slices_df: pd.DataFrame, epoch_times: NDArray, active_t_column_names=['start','stop'], ndim:int=2) -> NDArray:
        """Loop through each pair of epoch_times and find the closest start and end time
        
        Captures: atol, debug_print, not_found_action

        """
        assert len(active_t_column_names) == ndim, f"ndim: {ndim}, active_t_column_names: {active_t_column_names}"
        assert not_found_action in ['skip_index', 'full_nan']

        if (ndim == 0):
            epoch_times = np.atleast_1d(epoch_times)
            ndim = 1

        indices = []
        if (ndim == 1):
            for start_time in epoch_times:
                # Find the index with the closest start time
                start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin() # idxmin returns a .loc index apparently?

                ## Numpy-only version:
                # start_index: NDArray = np.argmin(epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().to_numpy())

                # start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin() 
                selected_index = start_index
                
                ## End if
                ## Check the tolerance
                assert selected_index is not None
                was_index_found: bool = True # true by default

                if atol is not None:
                    
                    # Can convert to an actual integer index like this:
                    # selected_integer_position_index = epoch_slices_df.index.get_loc(selected_index) # to match with .iloc do this
                    # selected_index_diff = epoch_slices_df.iloc[selected_integer_position_index].sub(start_time)

                    ## See how the selecteded index's values diff from the search values
                    selected_index_diff = epoch_slices_df.loc[selected_index].sub(start_time) # IndexError: single positional indexer is out-of-bounds

                    ## Check against tolerance:
                    exceeds_tolerance: bool = np.any((selected_index_diff.abs() > atol))
                    if exceeds_tolerance:
                        if debug_print:
                            print(f'WARN: CLOSEST FOUND INDEX EXCEEDS TOLERANCE (atol={atol}):\n\tsearch_time: {start_time}, closest: {epoch_slices_df.loc[selected_index].to_numpy()}, {selected_index_diff}. No matching index was found.')
                        selected_index = np.nan
                        was_index_found = False

                if was_index_found:
                    # index was found
                    indices.append(selected_index)
                else:
                    ## index not found:
                    if not_found_action == 'skip_index':
                        # skip without adding this index. This means the the output array will be smaller than the epoch_times
                        pass
                    elif (not_found_action == 'full_nan'):
                        ## append the nan anyway
                        indices.append(selected_index)
                    else:
                        raise NotImplementedError(f"not_found_action: {not_found_action}")


            # end for
                    
        elif (ndim == 2):
            for start_time, end_time in epoch_times:
                # Find the index with the closest start time
                start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin()
                # Find the index with the closest end time
                end_index = epoch_slices_df[active_t_column_names[1]].sub(end_time).abs().idxmin()
                
                was_index_found: bool = True # true by default
                
                selected_index = None
                # If the start and end indices are the same, we have a match
                if (start_index == end_index):
                    ## Good, this is how it should be, they correspond to the same (single) row:
                    selected_index = start_index
                else:
                    ## MODE: CLOSEST START
                    if debug_print:
                        print(f'WARNING: CLOSEST START INDEX: {start_index} is not equal to the closest END index: {end_index}. Using start index.')
                    selected_index = start_index

                    # ## MODE: CLOSEST START OR STOP
                    # start_diff = epoch_slices_df.iloc[start_index].sub([start_time, end_time]).abs().sum()
                    # end_diff = epoch_slices_df.iloc[end_index].sub([start_time, end_time]).abs().sum()
                    # # If not, find which one is closer overall (by comparing the sum of absolute differences to start_time and end_time)
                    # selected_index = (start_index if start_diff <= end_diff else end_index)

                ## End if
                ## Check the tolerance
                assert selected_index is not None
                if atol is not None:
                    ## See how the selecteded index's values diff from the search values
                    selected_index_diff = epoch_slices_df.loc[selected_index].sub([start_time, end_time]) # .loc[selected_index] method supposedly compatibile with .idxmin()
                    # selected_index_diff = epoch_slices_df.iloc[selected_index].sub([start_time, end_time]) #.abs() #.sum() # IndexError: single positional indexer is out-of-bounds -- selected_index: 319. SHIT. Confirmed it corresponds to df.Index == 319, which is at .iloc[134]
                    exceeds_tolerance: bool = np.any((selected_index_diff.abs() > atol))
                    if exceeds_tolerance:
                        if debug_print:
                            print(f'WARN: CLOSEST FOUND INDEX EXCEEDS TOLERANCE (atol={atol}):\n\tsearch_time: [{start_time}, {end_time}], closest: [{epoch_slices_df.loc[selected_index].to_numpy()}], diff: [{selected_index_diff.to_numpy()}]. No matching index was found.')
                        selected_index = np.nan
                        was_index_found = False

                if was_index_found:
                    # index was found
                    indices.append(selected_index)
                else:
                    ## index not found:
                    if not_found_action == 'skip_index':
                        # skip without adding this index. This means the the output array will be smaller than the epoch_times
                        pass
                    elif (not_found_action == 'full_nan'):
                        ## append the nan anyway
                        indices.append(selected_index)
                    else:
                        raise NotImplementedError(f"not_found_action: {not_found_action}")
                # end for
        else:
            raise NotImplementedError(f"ndim: {ndim}")
        
        # Return the indices as an ndarray
        return np.array(indices)


    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    assert not_found_action in ['skip_index', 'full_nan']

    if ((np.ndim(epoch_times) == 2) and (np.shape(epoch_times)[1] == 2)):
        if t_column_names is None:
            t_column_names = ['start', 'stop']
        assert (len(t_column_names) == 2), f"len(t_column_names): {len(t_column_names)} != 2)"
        num_query_times: int = np.shape(epoch_times)[0]
    elif (np.ndim(epoch_times) == 1):
        # start times only
        if t_column_names is None:
            t_column_names = ['start',]
        if len(t_column_names) > 1:
            t_column_names = [t_column_names[0],]
        num_query_times: int = len(epoch_times)


    elif (np.ndim(epoch_times) == 0):
        # single start time only
        if t_column_names is None:
            t_column_names = ['start',]
        if len(t_column_names) > 1:
            t_column_names = [t_column_names[0],]
        epoch_times = np.atleast_1d(epoch_times)
        num_query_times: int = 1

    else:
        raise NotImplementedError

    # start, stop epoch times:
    epoch_slices_df = a_df[t_column_names]

    found_data_indicies = _subfn_find_epoch_times(epoch_slices_df=epoch_slices_df, epoch_times=epoch_times, active_t_column_names=t_column_names, ndim=len(t_column_names))
    if not_found_action == 'skip_index':
        # skip without adding this index. This means the the output found_data_indicies might be smaller than the num_query_times
        assert (len(found_data_indicies) <= num_query_times), f"num_query_times: {num_query_times}, len(found_data_indicies): {len(found_data_indicies)}"
    elif (not_found_action == 'full_nan'):
        assert (len(found_data_indicies) == num_query_times), f"num_query_times: {num_query_times}, len(found_data_indicies): {len(found_data_indicies)}"
    else:
        raise NotImplementedError(f"not_found_action: {not_found_action}")

    return found_data_indicies


def find_epoch_times_to_data_indicies_map(a_df: pd.DataFrame, epoch_times: NDArray, t_column_names=None, atol:float=1e-3, not_found_action='skip_index', debug_print=False) -> Dict[Union[float, Tuple[float, float]], Union[int, NDArray]]:
    """ returns the matching data indicies corresponding to the epoch [start, stop] times 
    epoch_times: S x 2 array of epoch start/end times


    skip_index: drops indicies that can't be found, meaning that the number of returned indicies might be less than len(epoch_times)


    Returns: (S, ) array of data indicies corresponding to the times.

    Uses:
        from neuropy.core.epoch import find_epoch_times_to_data_indicies_map
        
        selection_start_stop_times = deepcopy(active_epochs_df[['start', 'stop']].to_numpy())
        print(f'np.shape(selection_start_stop_times): {np.shape(selection_start_stop_times)}')

        test_epochs_data_df: pd.DataFrame = deepcopy(ripple_simple_pf_pearson_merged_df)
        print(f'np.shape(test_epochs_data_df): {np.shape(test_epochs_data_df)}')

        # 2D_search (for both start, end times):
        found_data_indicies = find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=selection_start_stop_times)
        print(f'np.shape(found_data_indicies): {np.shape(found_data_indicies)}')

        # 1D_search (only for start times):
        found_data_indicies_1D_search = find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=np.squeeze(selection_start_stop_times[:, 0]))
        print(f'np.shape(found_data_indicies_1D_search): {np.shape(found_data_indicies_1D_search)}')
        found_data_indicies_1D_search

        assert np.array_equal(found_data_indicies, found_data_indicies_1D_search)
    

    - [X] FIXED 2024-03-04 19:55 - This function was peviously incorrect and could return multiple matches for each passed time due to the tolerance.
        
    """
    def _subfn_find_epoch_times_map(epoch_slices_df: pd.DataFrame, epoch_times: NDArray, active_t_column_names=['start','stop'], ndim:int=2):
        """Loop through each pair of epoch_times and find the closest start and end time
        
        Captures: atol, debug_print, not_found_action

        """
        assert len(active_t_column_names) == ndim, f"ndim: {ndim}, active_t_column_names: {active_t_column_names}"
        assert not_found_action in ['skip_index', 'full_nan']

        if (ndim == 0):
            epoch_times = np.atleast_1d(epoch_times)
            ndim = 1

        indices = []
        epoch_time_to_index_map = {}
        if (ndim == 1):
            for start_time in epoch_times:
                # Find the index with the closest start time
                start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin() # idxmin returns a .loc index apparently?

                ## Numpy-only version:
                # start_index: NDArray = np.argmin(epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().to_numpy())

                # start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin() 
                selected_index = start_index
                
                ## End if
                ## Check the tolerance
                assert selected_index is not None
                was_index_found: bool = True # true by default

                if atol is not None:
                    
                    # Can convert to an actual integer index like this:
                    # selected_integer_position_index = epoch_slices_df.index.get_loc(selected_index) # to match with .iloc do this
                    # selected_index_diff = epoch_slices_df.iloc[selected_integer_position_index].sub(start_time)

                    ## See how the selecteded index's values diff from the search values
                    selected_index_diff = epoch_slices_df.loc[selected_index].sub(start_time) # IndexError: single positional indexer is out-of-bounds

                    ## Check against tolerance:
                    exceeds_tolerance: bool = np.any((selected_index_diff.abs() > atol))
                    if exceeds_tolerance:
                        if debug_print:
                            print(f'WARN: CLOSEST FOUND INDEX EXCEEDS TOLERANCE (atol={atol}):\n\tsearch_time: {start_time}, closest: {epoch_slices_df.loc[selected_index].to_numpy()}, {selected_index_diff}. No matching index was found.')
                        selected_index = np.nan
                        was_index_found = False

                if was_index_found:
                    # index was found
                    indices.append(selected_index)
                    if isinstance(selected_index, (list, tuple, NDArray)):
                        ## if it's a list, tuple, NDArray, etc
                        if len(selected_index) > 0:
                            if (not isinstance(selected_index, NDArray)):
                                selected_index = NDArray(selected_index) ## convert to NDArray
                                
                    epoch_time_to_index_map[start_time] = selected_index
                    
                else:
                    ## index not found:
                    if not_found_action == 'skip_index':
                        # skip without adding this index. This means the the output array will be smaller than the epoch_times
                        pass
                    elif (not_found_action == 'full_nan'):
                        ## append the nan anyway
                        indices.append(selected_index)
                        if isinstance(selected_index, (list, tuple, NDArray)):
                            ## if it's a list, tuple, NDArray, etc
                            if len(selected_index) > 0:
                                if (not isinstance(selected_index, NDArray)):
                                    selected_index = NDArray(selected_index) ## convert to NDArray
                                
                        epoch_time_to_index_map[start_time] = selected_index
                    else:
                        raise NotImplementedError(f"not_found_action: {not_found_action}")


            # end for
                    
        elif (ndim == 2):
            for start_time, end_time in epoch_times:
                # Find the index with the closest start time
                start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin()
                # Find the index with the closest end time
                end_index = epoch_slices_df[active_t_column_names[1]].sub(end_time).abs().idxmin()
                
                was_index_found: bool = True # true by default
                
                selected_index = None
                # If the start and end indices are the same, we have a match
                if (start_index == end_index):
                    ## Good, this is how it should be, they correspond to the same (single) row:
                    selected_index = start_index
                else:
                    ## MODE: CLOSEST START
                    if debug_print:
                        print(f'WARNING: CLOSEST START INDEX: {start_index} is not equal to the closest END index: {end_index}. Using start index.')
                    selected_index = start_index

                    # ## MODE: CLOSEST START OR STOP
                    # start_diff = epoch_slices_df.iloc[start_index].sub([start_time, end_time]).abs().sum()
                    # end_diff = epoch_slices_df.iloc[end_index].sub([start_time, end_time]).abs().sum()
                    # # If not, find which one is closer overall (by comparing the sum of absolute differences to start_time and end_time)
                    # selected_index = (start_index if start_diff <= end_diff else end_index)

                ## End if
                ## Check the tolerance
                assert selected_index is not None
                if atol is not None:
                    ## See how the selecteded index's values diff from the search values
                    selected_index_diff = epoch_slices_df.loc[selected_index].sub([start_time, end_time]) # .loc[selected_index] method supposedly compatibile with .idxmin()
                    # selected_index_diff = epoch_slices_df.iloc[selected_index].sub([start_time, end_time]) #.abs() #.sum() # IndexError: single positional indexer is out-of-bounds -- selected_index: 319. SHIT. Confirmed it corresponds to df.Index == 319, which is at .iloc[134]
                    exceeds_tolerance: bool = np.any((selected_index_diff.abs() > atol))
                    if exceeds_tolerance:
                        if debug_print:
                            print(f'WARN: CLOSEST FOUND INDEX EXCEEDS TOLERANCE (atol={atol}):\n\tsearch_time: [{start_time}, {end_time}], closest: [{epoch_slices_df.loc[selected_index].to_numpy()}], diff: [{selected_index_diff.to_numpy()}]. No matching index was found.')
                        selected_index = np.nan
                        was_index_found = False
                        
                # if (not isinstance(selected_index, (float, int))):


                if was_index_found:
                    # index was found
                    indices.append(selected_index)
                    if isinstance(selected_index, (list, tuple, NDArray)):
                        ## if it's a list, tuple, NDArray, etc
                        if len(selected_index) > 0:
                            if (not isinstance(selected_index, NDArray)):
                                selected_index = NDArray(selected_index) ## convert to NDArray
                    epoch_time_to_index_map[(start_time, end_time,)] = selected_index
                else:
                    ## index not found:
                    if not_found_action == 'skip_index':
                        # skip without adding this index. This means the the output array will be smaller than the epoch_times
                        pass
                    elif (not_found_action == 'full_nan'):
                        ## append the nan anyway
                        indices.append(selected_index)
                        if isinstance(selected_index, (list, tuple, NDArray)):
                            ## if it's a list, tuple, NDArray, etc
                            if len(selected_index) > 0:
                                if (not isinstance(selected_index, NDArray)):
                                    selected_index = NDArray(selected_index) ## convert to NDArray
        
                        epoch_time_to_index_map[(start_time, end_time,)] = selected_index
                    else:
                        raise NotImplementedError(f"not_found_action: {not_found_action}")
                # end for
        else:
            raise NotImplementedError(f"ndim: {ndim}")
        
        # Return the indices as an ndarray
        return epoch_time_to_index_map, np.array(indices)


    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    assert not_found_action in ['skip_index', 'full_nan']

    if ((np.ndim(epoch_times) == 2) and (np.shape(epoch_times)[1] == 2)):
        if t_column_names is None:
            t_column_names = ['start', 'stop']
        assert (len(t_column_names) == 2), f"len(t_column_names): {len(t_column_names)} != 2)"
        num_query_times: int = np.shape(epoch_times)[0]
    elif (np.ndim(epoch_times) == 1):
        # start times only
        if t_column_names is None:
            t_column_names = ['start',]
        if len(t_column_names) > 1:
            t_column_names = [t_column_names[0],]
        num_query_times: int = len(epoch_times)


    elif (np.ndim(epoch_times) == 0):
        # single start time only
        if t_column_names is None:
            t_column_names = ['start',]
        if len(t_column_names) > 1:
            t_column_names = [t_column_names[0],]
        epoch_times = np.atleast_1d(epoch_times)
        num_query_times: int = 1

    else:
        raise NotImplementedError

    # start, stop epoch times:
    epoch_slices_df = a_df[t_column_names]

    epoch_time_to_index_map, found_data_indicies = _subfn_find_epoch_times_map(epoch_slices_df=epoch_slices_df, epoch_times=epoch_times, active_t_column_names=t_column_names, ndim=len(t_column_names))
    if not_found_action == 'skip_index':
        # skip without adding this index. This means the the output found_data_indicies might be smaller than the num_query_times
        assert (len(found_data_indicies) <= num_query_times), f"num_query_times: {num_query_times}, len(found_data_indicies): {len(found_data_indicies)}"
    elif (not_found_action == 'full_nan'):
        assert (len(found_data_indicies) == num_query_times), f"num_query_times: {num_query_times}, len(found_data_indicies): {len(found_data_indicies)}"
    else:
        raise NotImplementedError(f"not_found_action: {not_found_action}")

    # , found_data_indicies
    
    return epoch_time_to_index_map
            
def find_epochs_overlapping_other_epochs(epochs_df: pd.DataFrame, epochs_df_required_to_overlap: pd.DataFrame):
    """ 
    For example, you might wonder which epochs occur during laps:

        from neuropy.core.epoch import find_epoch_times_to_data_indicies_map
        
        ## INPUTS: time_bin_containers, global_laps
        left_edges = deepcopy(time_bin_containers.left_edges)
        right_edges = deepcopy(time_bin_containers.right_edges)
        continuous_time_binned_computation_epochs_df: pd.DataFrame = pd.DataFrame({'start': left_edges, 'stop': right_edges, 'label': np.arange(len(left_edges))})
        is_included: NDArray = find_epochs_overlapping_other_epochs(epochs_df=continuous_time_binned_computation_epochs_df, epochs_df_required_to_overlap=deepcopy(global_laps))
        continuous_time_binned_computation_epochs_df['is_in_laps'] = is_included
        continuous_time_binned_computation_epochs_df
        continuous_time_binned_computation_epochs_included_only_df = continuous_time_binned_computation_epochs_df[continuous_time_binned_computation_epochs_df['is_in_laps']].drop(columns=['is_in_laps'])
        continuous_time_binned_computation_epochs_included_only_df

    """
    ## INPUTS: continuous_time_binned_computation_epochs, laps
    epochs_df: pd.DataFrame = ensure_dataframe(epochs_df)
    epochs_df_required_to_overlap: pd.DataFrame = ensure_dataframe(epochs_df_required_to_overlap)
    epochs_df_required_to_overlap_portion: P.Interval = epochs_df_required_to_overlap.epochs.to_PortionInterval()
    continuous_time_binned_computation_epochs_portion_intervals: List[P.Interval] = [P.closedopen(a_row.start, a_row.stop) for a_row in epochs_df[['start', 'stop']].itertuples()]
    is_included: NDArray = np.array([an_interval.overlaps(epochs_df_required_to_overlap_portion) for an_interval in continuous_time_binned_computation_epochs_portion_intervals])
    return is_included

        
        
""" 
from neuropy.core.epoch import NamedTimerange, EpochsAccessor, Epoch

"""
class NamedTimerange(SimplePrintable, metaclass=OrderedMeta):
    """ A simple named period of time with a known start and end time """
    def __init__(self, name, start_end_times):
        self.name = name
        self.start_end_times = start_end_times
        
    @property
    def t_start(self):
        return self.start_end_times[0]
    
    @t_start.setter
    def t_start(self, t):
        self.start_end_times[0] = t

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def t_stop(self):
        return self.start_end_times[1]
    
    @t_stop.setter
    def t_stop(self, t):
        self.start_end_times[1] = t
    
    
    def to_Epoch(self):
        return Epoch(pd.DataFrame({'start': [self.t_start], 'stop': [self.t_stop], 'label':[self.name]}))
        

@pd.api.extensions.register_dataframe_accessor("epochs")
class EpochsAccessor(TimeColumnAliasesProtocol, TimeSlicedMixin, StartStopTimesMixin, TimeSlicableObjectProtocol):
    """ A Pandas pd.DataFrame representation of [start, stop, label] epoch intervals """
    
    _time_column_name_synonyms = {"start":{'begin','start_t'},
            "stop":['end','stop_t'],
            "label":['name', 'id', 'flat_replay_idx','lap_id']
        }
    

    _required_column_names = ['start', 'stop', 'label', 'duration']

    def __init__(self, pandas_obj):
        pandas_obj = self.renaming_synonym_columns_if_needed(pandas_obj, required_columns_synonym_dict=self._time_column_name_synonyms)       #@IgnoreException 
        pandas_obj = self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj = self._obj.sort_values(by=["start"]) # sorts all values in ascending order
        # Optional: If the 'label' column of the dataframe is empty, should populate it with the index (after sorting) as a string.
        # self._obj['label'] = self._obj.index
        self._obj["label"] = self._obj["label"].astype("str")
        # Optional: Add 'duration' column:
        self._obj["duration"] = self._obj["stop"] - self._obj["start"]
        # Optional: check for and remove overlaps

    @classmethod
    def _validate(cls, obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('neuron_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """       
        return obj # important! Must return the modified obj to be assigned (since its columns were altered by renaming

    @property
    def is_valid(self):
        """ The dataframe is valid (because it passed _validate(...) in __init__(...) so just return True."""
        return True

    @property
    def starts(self):
        return self._obj.start.values

    @property
    def midtimes(self): # -> NDArray
        """ since each epoch is given by a (start, stop) time, the midtimes are the center of this epoch. """
        return self._obj.start.values + ((self._obj.stop.values - self._obj.start.values)/2.0)

    @property
    def stops(self):
        return self._obj.stop.values
    
    @property
    def t_start(self):
        return self.starts[0]
    @t_start.setter
    def t_start(self, t):
        include_indicies = np.argwhere(t < self.stops)
        if (np.size(include_indicies) == 0):
            # this proposed t_start is after any contained epochs, so the returned object would be empty
            print('Error: this proposed t_start ({}) is after any contained epochs, so the returned object would be empty'.format(t))
            raise ValueError
        first_include_index = include_indicies[0]
        
        if (first_include_index > 0):
            # drop the epochs preceeding the first_include_index:
            drop_indicies = np.arange(first_include_index)
            print('drop_indicies: {}'.format(drop_indicies))
            raise NotImplementedError # doesn't yet drop the indicies before the first_include_index
        self._obj.loc[first_include_index, ('start')] = t # exclude the first short period where the animal isn't on the maze yet

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def t_stop(self):
        return self.stops[-1]

    @property
    def durations(self):
        return self.stops - self.starts

    @property
    def n_epochs(self):
        return len(self.starts)

    @property
    def labels(self):
        return self._obj.label.values


    @property
    def extra_data_column_names(self):
        """Any additional columns in the dataframe beyond those that exist by default. """
        return list(set(self._obj.columns) - set(self._required_column_names))
        
    @property
    def extra_data_dataframe(self) -> pd.DataFrame:
        """The subset of the dataframe containing additional information in its columns beyond that what is required. """
        return self._obj[self.extra_data_column_names]

    def as_array(self) -> NDArray:
        return self._obj[["start", "stop"]].to_numpy()

    def get_unique_labels(self):
        return np.unique(self.labels)

    def get_start_stop_tuples_list(self):
        """ returns a list of (start, stop) tuples. """
        return list(zip(self.starts, self.stops))

    def get_valid_df(self) -> pd.DataFrame:
        """ gets a validated copy of the dataframe. Looks better than doing `epochs_df.epochs._obj` """
        return self._obj.copy()

    # @classmethod
    # def _mergable(cls, a, b):
    #     """ 
    #     NOT YET IMPLEMENTED. Based off of Portion's mergable operation with intent to extend to Epochs.

    #     """
    #     # a - a single period in time
    #     # b - a single potentially overlapping period in time
    #     a_start, a_end = a # unwrap
    #     b_start, b_end = b # unwrap
    #     ## Check their lower bounds first

    #     ## Check their upper bounds for overlap

    
    
        
    ## Handling overlapping
    def get_non_overlapping_df(self, debug_print=False) -> pd.DataFrame:
        """ Returns a dataframe with overlapping epochs removed. """
        ## 2023-02-23 - PortionInterval approach to ensuring uniqueness:
        from neuropy.utils.efficient_interval_search import convert_PortionInterval_to_epochs_df, _convert_start_end_tuples_list_to_PortionInterval
        ## Capture dataframe properties beyond just the start/stop times:
        P_Interval_kwargs = {'merge_on_adjacent': False, 'enable_auto_simplification': True}

        ## Post-2024-04-01 13:11: - [ ] with auto-simplification disabled and such:
        # P_Interval_kwargs = {'merge_on_adjacent': False, 'enable_auto_simplification': False}

        # _intermedia_start_end_tuples_list = self.get_start_stop_tuples_list()
        _intermediate_portions_interval: P.Interval = _convert_start_end_tuples_list_to_PortionInterval(zip(self.starts, self.stops))
        filtered_epochs_df = convert_PortionInterval_to_epochs_df(_intermediate_portions_interval)
        # is_epoch_included = np.array([(a_tuple.start, a_tuple.stop) in _intermedia_start_end_tuples_list for a_tuple in list(filtered_epochs_df.itertuples(index=False))])

        if debug_print:
            before_num_rows = self.n_epochs
            filtered_epochs_df = convert_PortionInterval_to_epochs_df(_intermediate_portions_interval)
            after_num_rows = np.shape(filtered_epochs_df)[0]
            changed_num_rows = after_num_rows - before_num_rows
            print(f'Dataframe Changed from {before_num_rows} -> {after_num_rows} ({changed_num_rows = })')
            return filtered_epochs_df
        else:
            return filtered_epochs_df

    def get_epochs_longer_than(self, minimum_duration, debug_print=False) -> pd.DataFrame:
        """ returns a copy of the dataframe contining only epochs longer than the specified minimum_duration. """
        active_filter_epochs = self.get_valid_df()
        if debug_print:
            before_num_rows = np.shape(active_filter_epochs)[0]
        if 'duration' not in active_filter_epochs.columns:
            active_filter_epochs['duration'] = active_filter_epochs['stop'] - active_filter_epochs['start']
        if debug_print:
            filtered_epochs = active_filter_epochs[active_filter_epochs['duration'] >= minimum_duration]
            after_num_rows = np.shape(filtered_epochs)[0]
            changed_num_rows = after_num_rows - before_num_rows
            print(f'Dataframe Changed from {before_num_rows} -> {after_num_rows} ({changed_num_rows = })')
            return filtered_epochs
        else:
            return active_filter_epochs[active_filter_epochs['duration'] >= minimum_duration]

    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop) -> pd.DataFrame:
        """ trim the epochs down to the provided time range
        
        """
        # TODO time_slice should also include partial epochs falling in between the timepoints
        df = self._obj.copy() 
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        df = df[(df["start"] >= t_start) & (df["start"] < t_stop)].reset_index(drop=True) # 2023-11-13 - changed to `(df["start"] < t_stop)` from `(df["start"] <= t_stop)` because in the equals case the resulting included interval would be zero duration.
        return df
        
    def label_slice(self, label) -> pd.DataFrame:
        if isinstance(label, (list, NDArray)):
            df = self._obj[np.isin(self._obj["label"], label)].reset_index(drop=True)
        else:
            assert isinstance(label, str), "label must be string"
            df = self._obj[self._obj["label"] == label].reset_index(drop=True)
        return df

    def find_data_indicies_from_epoch_times(self, epoch_times: NDArray, atol:float=1e-3, t_column_names=None) -> NDArray:
        """ returns the matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times
        Returns: (S, ) array of data indicies corresponding to the times.

        Uses:
            self.plots_data.epoch_slices
        
        - [X] FIXED 2024-03-04 19:55 - This function was peviously incorrect and could return multiple matches for each passed time due to the tolerance.

        """
        # find_data_indicies_from_epoch_times(a_df, np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t',])
        return find_data_indicies_from_epoch_times(self._obj, epoch_times, t_column_names=t_column_names, atol=atol, debug_print=False)
    

    def find_epoch_times_to_data_indicies_map(self, epoch_times: NDArray, atol:float=1e-3, t_column_names=None) -> Dict[Union[float, Tuple[float, float]], Union[int, NDArray]]:
        """ returns the a Dict[Union[float, Tuple[float, float]], Union[int, NDArray]] matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times
        Returns: (S, ) array of data indicies corresponding to the times.

        Usage:
            epoch_time_to_index_map = deepcopy(dbgr.active_decoder_decoded_epochs_result).filter_epochs.epochs.find_epoch_times_to_data_indicies_map(epoch_times=[epoch_start_time, ])
        
        """
        return find_epoch_times_to_data_indicies_map(self._obj, epoch_times, t_column_names=t_column_names, atol=atol, debug_print=False)
    

            
    def matching_epoch_times_slice(self, epoch_times: NDArray, atol:float=1e-3, t_column_names=None) -> pd.DataFrame:
        """ slices the dataframe to return only the rows that match the epoch_times with some tolerance.
        
        Internally calls self.find_data_indicies_from_epoch_times(...)

        """
        # , not_found_action='skip_index'
        found_data_indicies = self._obj.epochs.find_data_indicies_from_epoch_times(epoch_times=epoch_times, atol=atol, t_column_names=t_column_names)
        # df = self._obj.iloc[found_data_indicies].copy().reset_index(drop=True)
        df = self._obj.loc[found_data_indicies].copy().reset_index(drop=True)
        return df

    def filtered_by_duration(self, min_duration=None, max_duration=None):
        return self._obj[(self.durations >= (min_duration or 0.0)) & (self.durations <= (max_duration or np.inf))].reset_index(drop=True)
        
    # Requires Optional `portion` library
    # import portion as P # Required for interval search: portion~=2.3.0
    @classmethod
    def from_PortionInterval(cls, portion_interval):
        from neuropy.utils.efficient_interval_search import convert_PortionInterval_to_epochs_df
        return convert_PortionInterval_to_epochs_df(portion_interval)

    def to_PortionInterval(self): # -> "P.Interval"
        from neuropy.utils.efficient_interval_search import _convert_start_end_tuples_list_to_PortionInterval
        return _convert_start_end_tuples_list_to_PortionInterval(zip(self.starts, self.stops))

    # Column Adding/Updating Methods _____________________________________________________________________________________ #
    def adding_active_aclus_information(self, spikes_df: pd.DataFrame, epoch_id_key_name: str = 'Probe_Epoch_id', add_unique_aclus_list_column: bool=False) -> pd.DataFrame:
        """ 
        adds the columns: ['unique_active_aclus', 'n_unique_aclus'] 

        Usage:

            active_epochs_df = active_epochs_df.epochs.adding_active_aclus_information(spikes_df=active_spikes_df, epoch_id_key_name='Probe_Epoch_id', add_unique_aclus_list_column=True)

        """
        active_epochs_df: pd.DataFrame = self._obj.epochs.get_valid_df()
        
        # Ensures the appropriate columns are added to spikes_df:
        # spikes_df = spikes_df.spikes.adding_epochs_identity_column(epochs_df=active_epochs_df, epoch_id_key_name=epoch_id_key_name, epoch_label_column_name='label', override_time_variable_name='t_rel_seconds',
        #     should_replace_existing_column=False, drop_non_epoch_spikes=True)
        assert epoch_id_key_name in spikes_df, f"""ERROR: epoch_id_key_name: '{epoch_id_key_name}' is not in spikes_df.columns: {spikes_df.columns}. Did you add the columns to spikes_df via:
## INPUTS: curr_active_pipeline, epochs_df
active_spikes_df: pd.DataFrame = get_proper_global_spikes_df(curr_active_pipeline).spikes.adding_epochs_identity_column(epochs_df=epochs_df, epoch_id_key_name='replay_id', epoch_label_column_name='label', override_time_variable_name='t_rel_seconds',
    should_replace_existing_column=False, drop_non_epoch_spikes=True) ## gets the proper spikes_df, and then adds the epoch_id columns (named 'replay_id') for the epochs provided in `epochs_df`
epochs_df = epochs_df.epochs.adding_active_aclus_information(spikes_df=active_spikes_df, epoch_id_key_name='replay_id', add_unique_aclus_list_column=True)
epochs_df
        """
        unique_values = np.unique(spikes_df[epoch_id_key_name]) # array([ 0,  1,  2,  3,  4,  7, 11, 12, 13, 14])
        grouped_df = spikes_df.groupby([epoch_id_key_name]) #  Groups on the specified column.
        epoch_unique_aclus_dict = {aValue:grouped_df.get_group(aValue).aclu.unique() for aValue in unique_values} # dataframes split for each unique value in the column

        # Convert label column in `active_epochs_df` to same dtype as the unique_values that were found
        active_epochs_df.label = active_epochs_df.label.astype(unique_values.dtype) ## WARNING: without this line it returns all np.nan results in the created columns!
        if add_unique_aclus_list_column:
            active_epochs_df['unique_active_aclus'] = active_epochs_df.label.map(epoch_unique_aclus_dict)
        epoch_num_unique_aclus_dict = {k:len(v) for k,v in epoch_unique_aclus_dict.items()}
        active_epochs_df['n_unique_aclus'] = active_epochs_df.label.map(epoch_num_unique_aclus_dict)
        active_epochs_df['n_unique_aclus'] = active_epochs_df['n_unique_aclus'].fillna(0).astype(int) # fill NaN values with 0s and convert to int
        return active_epochs_df


    
    @classmethod
    def add_maze_id_if_needed(cls, epochs_df: pd.DataFrame, t_start:float, t_delta:float, t_end:float, replace_existing:bool=True, labels_column_name:str='label', start_time_col_name: str='start', end_time_col_name: str='stop') -> pd.DataFrame:
        """ 2024-01-17 - adds the 'maze_id' column if it doesn't exist

        Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on
        
        WARNING: does NOT modify in place!

        Adds Columns: ['maze_id']
        Usage:
            from neuropy.core.session.dataSession import Laps

            t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
            laps_obj: Laps = curr_active_pipeline.sess.laps
            laps_df = laps_obj.to_dataframe()
            laps_df = laps_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
            laps_df

        """
        # epochs_df = epochs_df.epochs.to_dataframe()
        epochs_df[[labels_column_name]] = epochs_df[[labels_column_name]].astype('int')
        is_missing_column: bool = ('maze_id' not in epochs_df.columns)
        if (is_missing_column or replace_existing):
            # Create the maze_id column:
            epochs_df['maze_id'] = np.full_like(epochs_df[labels_column_name].to_numpy(), -1) # all -1 to start
            epochs_df.loc[(np.logical_and((epochs_df[start_time_col_name].to_numpy() >= t_start), (epochs_df[end_time_col_name].to_numpy() <= t_delta))), 'maze_id'] = 0 # first epoch
            epochs_df.loc[(np.logical_and((epochs_df[start_time_col_name].to_numpy() >= t_delta), (epochs_df[end_time_col_name].to_numpy() <= t_end))), 'maze_id'] = 1 # second epoch, post delta
            epochs_df['maze_id'] = epochs_df['maze_id'].astype('int') # note the single vs. double brakets in the two cases. Not sure if it makes a difference or not
        else:
            # already exists and we shouldn't overwrite it:
            epochs_df[['maze_id']] = epochs_df[['maze_id']].astype('int') # note the single vs. double brakets in the two cases. Not sure if it makes a difference or not
        return epochs_df
            

    def adding_maze_id_if_needed(self, t_start:float, t_delta:float, t_end:float, replace_existing:bool=True, labels_column_name:str='label') -> pd.DataFrame:
        """ 2024-01-17 - adds the 'maze_id' column if it doesn't exist

        Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on
        
        WARNING: does NOT modify in place!

        Adds Columns: ['maze_id']
        Usage:
            from neuropy.core.session.dataSession import Laps

            t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
            laps_obj: Laps = curr_active_pipeline.sess.laps
            laps_df = laps_obj.to_dataframe()
            laps_df = laps_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
            laps_df

        """
        epochs_df: pd.DataFrame = self._obj.epochs.get_valid_df()
        return self.add_maze_id_if_needed(epochs_df=epochs_df, t_start=t_start, t_delta=t_delta, t_end=t_end, replace_existing=replace_existing, labels_column_name=labels_column_name, start_time_col_name='start', end_time_col_name='stop')
    

    def adding_global_epoch_row(self, global_epoch_name='maze', first_included_epoch_name=None, last_included_epoch_name=None) -> pd.DataFrame:
        """ builds the 'global' epoch row for the entire session that includes by default the times from all other epochs in epochs_df. 
		e.g. builds the 'maze' epoch from ['maze1', 'maze2'] epochs
        
        Based off of `neuropy.core.session.Formats.BaseDataSessionFormats.DataSessionFormatBaseRegisteredClass.build_global_epoch_filter_config_dict` on 2025-01-15 15:56 
        
        Usage:
            from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch

            maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()
            maze_epochs_df = maze_epochs_df.epochs.adding_global_epoch_row()
            maze_epochs_df

        """
        all_epoch_names = list(self.get_unique_labels()) # all_epoch_names # ['maze1', 'maze2']
        if global_epoch_name in all_epoch_names:
            global_epoch_name = f"{global_epoch_name}_GLOBAL"
            print(f'WARNING: name collision "{global_epoch_name}" already exists in all_epoch_names: {all_epoch_names}! Using {global_epoch_name} instead.')
        
        if first_included_epoch_name is not None:
            # global_start_end_times[0] = sess.epochs[first_included_epoch_name][0] # 'maze1'
            pass
        else:
            first_included_epoch_name = self.get_unique_labels()[0]
            

        if last_included_epoch_name is not None:
            # global_start_end_times[1] = sess.epochs[last_included_epoch_name][1] # 'maze2'
            pass
        else:
            last_included_epoch_name = self.get_unique_labels()[-1]
        
        maze_epochs_df =  pd.DataFrame({'start': [*self.starts, self._obj.iloc[(self.labels == first_included_epoch_name)]['start'].tolist()[0]], 
                                        'stop': [*self.stops, self._obj.iloc[(self.labels == last_included_epoch_name)]['stop'].tolist()[0]],
        # 'stop': [*self.stops, self._obj[self.labels == last_included_epoch_name][1]],
        'label': [*self.labels, global_epoch_name],
        }) # .epochs.get_valid_df()    
        maze_epochs_df[['start', 'stop']] = maze_epochs_df[['start', 'stop']].astype(float) 
        maze_epochs_df['duration'] = maze_epochs_df['stop'] - maze_epochs_df['start']

        self._obj = maze_epochs_df
        return self._obj
        

    # ==================================================================================================================== #
    # `Epoch` object / pd.DataFrame exchangeability                                                                         #
    # ==================================================================================================================== #
    def to_dataframe(self) -> pd.DataFrame:
        """ Ensures code exchangeability of epochs in either `Epoch` object / pd.DataFrame """
        return self._obj.copy()

    def to_Epoch(self) -> "Epoch":
        """ Ensures code exchangeability of epochs in either `Epoch` object / pd.DataFrame """
        return Epoch(self._obj.copy())


class Epoch(HDFMixin, StartStopTimesMixin, TimeSlicableObjectProtocol, DataFrameRepresentable, DataFrameInitializable, DataWriter):
    """ An Epoch object holds one ore more periods of time (marked by start/end timestamps) along with their corresponding metadata.

    """
    def __init__(self, epochs: pd.DataFrame, metadata=None) -> None:
        """[summary]
        Args:
            epochs (pd.DataFrame): Each column is a pd.Series(["start", "stop", "label"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        if not isinstance(epochs, pd.DataFrame):
            _epochs_metadata = getattr(epochs, 'metadata', None)
            metadata = metadata or _epochs_metadata
            epochs = epochs.to_dataframe() # try to convert to dataframe if the object is an Epoch or other compatible object
            assert isinstance(epochs, pd.DataFrame)
        super().__init__(metadata=metadata)
        self._df = epochs.epochs.get_valid_df() # gets already sorted appropriately and everything. epochs.epochs uses the DataFrame accesor
        self._check_epochs(self._df) # check anyway
        # self._check_epochs(epochs) # check anyway

    @property
    def starts(self):
        return self._df.epochs.starts

    @property
    def stops(self):
        return self._df.epochs.stops
    
    @property
    def t_start(self):
        return self.starts[0]
    @t_start.setter
    def t_start(self, t):
        self._df.epochs.t_start = t

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def t_stop(self):
        return self.stops[-1]

    @property
    def durations(self):
        return self.stops - self.starts

    @property
    def midtimes(self): # NDArray
        """ since each epoch is given by a (start, stop) time, the midtimes are the center of this epoch. """
        return self._df.epochs.midtimes


    @property
    def n_epochs(self):
        return self._df.epochs.n_epochs
    @property
    def labels(self):
        return self._df.epochs.labels

    def get_unique_labels(self):
        return np.unique(self.labels)
    
    def get_named_timerange(self, epoch_name):
        return NamedTimerange(name=epoch_name, start_end_times=self[epoch_name])


    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """metadata compatibility"""
        self._metadata = metadata

        
    @property
    def epochs(self) -> "EpochsAccessor":
        """ a passthrough accessor to the Pandas dataframe `EpochsAccessor` to allow complete pass-thru compatibility with either Epoch or pd.DataFrame versions of epochs.
        Instead of testing whether it's an `Epoch` object or pd.DataFrame and then converting back and forth, should just be able to pretend it's a dataframe for the most part and use the `some_epochs.epochs.*` properties and methods.
        """
        return self._df.epochs


    def _check_epochs(self, epochs):
        assert isinstance(epochs, pd.DataFrame)
        # epochs.epochs.
        assert (
            pd.Series(["start", "stop", "label"]).isin(epochs.columns).all()
        ), f"Epoch dataframe should at least have columns with names: start, stop, label but it only has the columns: {list(epochs.columns)}"

    def __repr__(self) -> str:
        # return f"{len(self.starts)} epochs"
        return f"{len(self.starts)} epochs\n{self.as_array().__repr__()}\n"

    def _repr_pretty_(self, p, cycle=False):
        """ The cycle parameter will be true if the representation recurses - e.g. if you put a container inside itself. """
        # p.text(self.__repr__() if not cycle else '...')
        p.text(self.to_dataframe().__repr__() if not cycle else '...')

    def __str__(self) -> str:
        return f"{len(self.starts)} epochs\n{self.as_array().__repr__()}\n"
    
    def __len__(self):
        """ allows using `len(epochs_obj)` and getting the number of epochs. """
        return len(self.starts)

    def str_for_concise_display(self) -> str:
        """ returns a minimally descriptive string like: '60 epochs in (17.9, 524.1)' that doesn't print all the array elements only the number of epochs and the first and last. """
        return f"{len(self.starts)} epochs in ({self.starts[0]:.1f}, {self.stops[-1]:.1f})" # "60 epochs in (17.9, 524.1)"

    def str_for_filename(self) -> str:
        return f"Epoch[{len(self.starts)}]({self.starts[0]:.1f}-{self.stops[-1]:.1f})" #


    @property
    def __array_interface__(self):
        """ wraps the internal dataframe's `__array_interface__` which Pandas uses to provide numpy with information about dataframes such as np.shape(a_df) info.
        Allows np.shape(an_epoch_obj) to work.

        """
        # Get the numpy array's __array_interface__ from the DataFrame's values
        # The .to_numpy() method explicitly converts the DataFrame to a NumPy array
        return self._df.to_numpy().__array_interface__
        # return self._df.__array_interface__


    def __getitem__(self, slice_):
        """ Allows pass-thru indexing like it were a numpy array.

        2024-03-07 Potentially more dangerous than helpful.

        having issue whith this being called with pd.Dataframe columns (when assuming a pd.DataFrame epochs format but actually an Epoch object)

        IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
               Occurs because `_slice == ['lap_id']` which doesn't pass the first check because it's a list of strings not a string itself
        Example:
            Error line `laps_df[['lap_id']] = laps_df[['lap_id']].astype('int')`
        """
        if isinstance(slice_, str):
            indices = np.where(self.labels == slice_)[0]
            if len(indices) > 1:
                return np.vstack((self.starts[indices], self.stops[indices])).T
            else:
                return np.array([self.starts[indices], self.stops[indices]]).squeeze()
        elif ((slice_ is not None) and (len(slice_) > 0) and isinstance(slice_[0], str)): # TypeError: object of type 'int' has no len()
            # a list of strings, probably meant to use a dataframe indexing method
            # having issue whith this being called with pd.Dataframe columns (when assuming a pd.DataFrame epochs format but actually an Epoch object)
            # IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
            #     Occurs because `_slice == ['lap_id']` which doesn't pass the first check because it's a list of strings not a string itself
            # Example:
            #     Error line `laps_df[['lap_id']] = laps_df[['lap_id']].astype('int')`                
            raise IndexError(f"PHO: you're probably trying to treat the epochs as if they are in the pd.DataFrame format but they are an Epoch object! Use `actual_laps_df = incorrectly_assumed_laps_df.epochs.to_dataframe()` to convert.")

        else:
            return np.vstack((self.starts[slice_], self.stops[slice_])).T


    def adding_global_epoch_row(self, global_epoch_name='maze', first_included_epoch_name=None, last_included_epoch_name=None) -> "Epoch":
        """ builds the 'global' epoch row for the entire session that includes by default the times from all other epochs in epochs_df. 
		e.g. builds the 'maze' epoch from ['maze1', 'maze2'] epochs
        
        Based off of `neuropy.core.session.Formats.BaseDataSessionFormats.DataSessionFormatBaseRegisteredClass.build_global_epoch_filter_config_dict` on 2025-01-15 15:56 
        
        Usage:
            from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch

            maze_epochs_obj = ensure_Epoch(deepcopy(curr_active_pipeline.sess.epochs).to_dataframe())
            maze_epochs_obj.adding_global_epoch_row(global_epoch_name='maze', first_included_epoch_name=None, last_included_epoch_name=None)
            maze_epochs_obj
            
        """
        all_epoch_names = list(self.get_unique_labels()) # all_epoch_names # ['maze1', 'maze2']
        if global_epoch_name in all_epoch_names:
            global_epoch_name = f"{global_epoch_name}_GLOBAL"
            print(f'WARNING: name collision "{global_epoch_name}" already exists in all_epoch_names: {all_epoch_names}! Using {global_epoch_name} instead.')
        
        if first_included_epoch_name is not None:
            # global_start_end_times[0] = sess.epochs[first_included_epoch_name][0] # 'maze1'
            pass
        else:
            first_included_epoch_name = self.get_unique_labels()[0]
            
        if last_included_epoch_name is not None:
            # global_start_end_times[1] = sess.epochs[last_included_epoch_name][1] # 'maze2'
            pass
        else:
            last_included_epoch_name = self.get_unique_labels()[-1]
    
        # global_start_end_times = [epochs.t_start, epochs.t_stop]
        # global_start_end_times = [self[first_included_epoch_name][0], self[last_included_epoch_name][1]]
        # self.t_start, self.t_stop
        # maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()[['start', 'stop', 'label']] # ['start', 'stop', 'label', 'duration']
        # maze_epochs_numpy = maze_epochs_df.to_numpy()
        # maze_epochs_numpy.shape

        # new_df =  pd.DataFrame({'start': np.stack([self.starts, [self[first_included_epoch_name][0]]]), 
        # 'stop': np.stack([self.stops, [self[last_included_epoch_name][1]]]),
        # 'label': [*self.labels, global_epoch_name],
        # }).epochs.get_valid_df()
        
        maze_epochs_df =  pd.DataFrame({'start': [*self.starts, self[first_included_epoch_name][0]], 
        'stop': [*self.stops, self[last_included_epoch_name][1]],
        'label': [*self.labels, global_epoch_name],
        }) # .epochs.get_valid_df()    
        maze_epochs_df[['start', 'stop']] = maze_epochs_df[['start', 'stop']].astype(float) 
        maze_epochs_df['duration'] = maze_epochs_df['stop'] - maze_epochs_df['start']

        self._df = maze_epochs_df
        # self._df = new_df
        return self
        
        

        
    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop):
        return Epoch(epochs=self._df.epochs.time_slice(t_start, t_stop), metadata=self.metadata)
        
    def label_slice(self, label):
        return Epoch(epochs=self._df.epochs.label_slice(label), metadata=self.metadata)

    def boolean_indicies_slice(self, boolean_indicies):
        return Epoch(epochs=self._df[boolean_indicies], metadata=self.metadata)

    def filtered_by_duration(self, min_duration=None, max_duration=None):
        return Epoch(epochs=self._df.epochs.filtered_by_duration(min_duration, max_duration), metadata=self.metadata)

    @classmethod
    def filter_epochs(cls, curr_epochs: Union[pd.DataFrame, "Epoch"], pos_df:Optional[pd.DataFrame]=None, spikes_df:pd.DataFrame=None, require_intersecting_epoch:"Epoch"=None, min_epoch_included_duration=0.06, max_epoch_included_duration=0.6,
        maximum_speed_thresh=2.0, min_inclusion_fr_active_thresh=2.0, min_num_unique_aclu_inclusions=3, debug_print=False) -> "Epoch":
        """filters the provided replay epochs by specified constraints.

        Args:
            curr_epochs (Epoch): the epochs to filter on
            min_epoch_included_duration (float, optional): all epochs shorter than min_epoch_included_duration will be excluded from analysis. Defaults to 0.06.
            max_epoch_included_duration (float, optional): all epochs longer than max_epoch_included_duration will be excluded from analysis. Defaults to 0.6.
            maximum_speed_thresh (float, optional): epochs are only included if the animal's interpolated speed (as determined from the session's position dataframe) is below the speed. Defaults to 2.0 [cm/sec].
            min_inclusion_fr_active_thresh: minimum firing rate (in Hz) for a unit to be considered "active" for inclusion.
            min_num_unique_aclu_inclusions: minimum number of unique active cells that must be included in an epoch to have it included.

            save_on_compute (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            Epoch: the filtered epochs as an Epoch object

        NOTE: this really is a general method that works for any Epoch object or Epoch-type dataframe to filter it.

        TODO 2023-04-11 - This really belongs in the Epoch class or the epoch dataframe accessor. 

        """
        from neuropy.utils.efficient_interval_search import filter_epochs_by_speed
        from neuropy.utils.efficient_interval_search import filter_epochs_by_num_active_units

        if not isinstance(curr_epochs, pd.DataFrame):
            curr_epochs = curr_epochs.to_dataframe() # .get_valid_df() # convert to pd.DataFrame to start
    
        assert isinstance(curr_epochs, pd.DataFrame), f'curr_replays must be a pd.DataFrame or Epoch object, but is {type(curr_epochs)}'
        # Ensure the dataframe representation has the required columns. TODO: is this needed?
        if not 'stop' in curr_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            curr_epochs['stop'] = curr_epochs['end'].copy()
        if not 'label' in curr_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            curr_epochs['label'] = curr_epochs['flat_replay_idx'].copy()
        # must convert back from pd.DataFrame to Epoch object to use the Epoch methods
        curr_epochs = cls(curr_epochs)

        ## Use the existing replay epochs from the session but ensure they look valid:

        ## Filter based on required overlap with Ripples:
        if require_intersecting_epoch is not None:
            curr_epochs = cls.from_PortionInterval(require_intersecting_epoch.to_PortionInterval().intersection(curr_epochs.to_PortionInterval()))
        else:
            curr_epochs = cls.from_PortionInterval(curr_epochs.to_PortionInterval()) # just do this to ensure non-overlapping

        if curr_epochs.n_epochs == 0:
            warn(f'curr_epochs already empty prior to any filtering')

        # Filter by duration bounds:
        if (min_epoch_included_duration is not None) or (max_epoch_included_duration is not None):
            curr_epochs = curr_epochs.filtered_by_duration(min_duration=min_epoch_included_duration, max_duration=max_epoch_included_duration)

        # Filter *_replays_Interval by requiring them to be below the speed:
        if maximum_speed_thresh is not None:
            assert pos_df is not None, "must provide pos_df if filtering by speed"
            if curr_epochs.n_epochs > 0:
                curr_epochs, above_speed_threshold_intervals, below_speed_threshold_intervals = filter_epochs_by_speed(pos_df, curr_epochs, speed_thresh=maximum_speed_thresh, debug_print=debug_print)
            else:
                warn(f'curr_epochs already empty prior to filtering by speed')

        # 2023-02-10 - Trimming and Filtering Estimated Replay Epochs based on cell activity and pyramidal cell start/end times:
        if (min_inclusion_fr_active_thresh is not None) or (min_num_unique_aclu_inclusions is not None):
            assert spikes_df is not None, "must provide spikes_df if filtering by active units"
            active_spikes_df = spikes_df.spikes.sliced_by_neuron_type('pyr') # trim based on pyramidal cell activity only
            if curr_epochs.n_epochs > 0:
                curr_epochs, _extra_outputs = filter_epochs_by_num_active_units(active_spikes_df, curr_epochs, min_inclusion_fr_active_thresh=min_inclusion_fr_active_thresh, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions, include_intermediate_computations=False) # TODO: seems wasteful considering we compute all these spikes_df metrics and refinements and then don't return them.
            else:
                warn(f'curr_epochs already empty prior to filtering by firing rate or minimum active units')
                
        return curr_epochs

    def to_dict(self, recurrsively=False):
        d = {"epochs": self._df, "metadata": self._metadata}
        return d
    
    @staticmethod
    def from_dict(d: dict):
        return Epoch(d["epochs"], metadata=d["metadata"])

    ## TODO: refactor these methods into the 'epochs' pd.DataFrame accessor above and then wrap them:
    def fill_blank(self, method="from_left"):
        ep_starts = self.epochs["start"].values
        ep_stops = self.epochs["stop"].values
        ep_durations = self.epochs["duration"].values
        ep_labels = self.epochs["label"].values

        mask = (ep_starts[:-1] + ep_durations[:-1]) < ep_starts[1:]
        (inds,) = np.nonzero(mask)

        if method == "from_left":
            for ind in inds:
                ep_durations[ind] = ep_starts[ind + 1] - ep_starts[ind]

        elif method == "from_right":
            for ind in inds:
                gap = ep_starts[ind + 1] - (ep_starts[ind] + ep_durations[ind])
                ep_starts[ind + 1] -= gap
                ep_durations[ind + 1] += gap

        elif method == "from_nearest":
            for ind in inds:
                gap = ep_starts[ind + 1] - (ep_starts[ind] + ep_durations[ind])
                ep_durations[ind] += gap / 2.0
                ep_starts[ind + 1] -= gap / 2.0
                ep_durations[ind + 1] += gap / 2.0

        self.epochs["start"] = ep_starts
        self.epochs["stop"] = ep_starts + ep_durations
        self.epochs["duration"] = ep_durations

    def delete_in_between(self, t1, t2):
        epochs_df = self.to_dataframe()[["start", "stop", "label"]]
        # delete epochs if they are within t1, t2
        epochs_df = epochs_df[~((epochs_df["start"] >= t1) & (epochs_df["stop"] <= t2))]

        # truncate stop if start is less than t1 but stop is within t1,t2
        epochs_df.loc[
            (epochs_df["start"] < t1)
            & (t1 < epochs_df["stop"])
            & (epochs_df["stop"] <= t2),
            "stop",
        ] = t1

        # truncate start if stop is greater than t2 but start is within t1,t2
        epochs_df.loc[
            (epochs_df["start"] > t1)
            & (epochs_df["start"] <= t2)
            & (epochs_df["stop"] > t2),
            "start",
        ] = t2

        # if epoch starts before and ends after range,
        flank_start = epochs_df[
            (epochs_df["start"] < t1) & (epochs_df["stop"] > t2)
        ].copy()
        flank_start["stop"] = t1
        flank_stop = epochs_df[
            (epochs_df["start"] < t1) & (epochs_df["stop"] > t2)
        ].copy()
        flank_stop["start"] = t2
        epochs_df = epochs_df[~((epochs_df["start"] < t1) & (epochs_df["stop"] > t2))]
        epochs_df = epochs_df.append(flank_start)
        epochs_df = epochs_df.append(flank_stop)
        epochs_df = epochs_df.reset_index(drop=True)

        return Epoch(epochs_df)

    def get_proportion_by_label(self, t_start=None, t_stop=None):
        if t_start is None:
            t_start = self.starts[0]
        if t_stop is None:
            t_stop = self.stops[-1]

        duration = t_stop - t_start

        ep = self._df.copy()
        ep = ep[(ep.stop > t_start) & (ep.start < t_stop)].reset_index(drop=True)

        if ep["start"].iloc[0] < t_start:
            ep.at[0, "start"] = t_start

        if ep["stop"].iloc[-1] > t_stop:
            ep.at[ep.index[-1], "stop"] = t_stop

        ep["duration"] = ep.stop - ep.start

        ep_group = ep.groupby("label").sum().duration / duration

        label_proportion = {}
        for label in self.get_unique_labels():
            label_proportion[label] = 0.0

        for state in ep_group.index.values:
            label_proportion[state] = ep_group[state]

        return label_proportion

    def count(self, t_start=None, t_stop=None, binsize=300):
        if t_start is None:
            t_start = 0

        if t_stop is None:
            t_stop = np.max(self.stops)

        mid_times = self.starts + self.durations / 2
        bins = np.arange(t_start, t_stop + binsize, binsize)
        return np.histogram(mid_times, bins=bins)[0]

    def to_neuroscope(self, ext="PHO", override_filepath=None):
        """ exports to a Neuroscope compatable .evt file. """
        if not isinstance(ext, str):
            ext = '.'.join(ext) # assume it is an list, tuple or something. Join its elements by periods

        if override_filepath is not None:
            out_filepath = override_filepath.resolve()
            out_filepath = out_filepath.with_suffix(f".{ext}.evt")
        else:
            assert self.filename is not None
            out_filepath = self.filename.with_suffix(f".{ext}.evt")

        with out_filepath.open("w") as a:
            for event in self._df.itertuples():
                a.write(f"{event.start*1000} start\n{event.stop*1000} end\n")
        return out_filepath

    @classmethod
    def from_neuroscope(cls, in_filepath, metadata=None):
        """ imports from a Neuroscope compatible .evt file.
        Usage:
            from neuropy.core.epoch import Epoch

            evt_filepath = Path('/Users/pho/Downloads/2006-6-07_16-40-19.bst.evt').resolve()
            # evt_filepath = Path('/Users/pho/Downloads/2006-6-08_14-26-15.bst.evt').resolve()
            evt_epochs: Epoch = Epoch.from_neuroscope(in_filepath=evt_filepath)
            evt_epochs

        """
        if isinstance(in_filepath, str):
            in_filepath = Path(in_filepath).resolve()
        assert in_filepath.exists()

        # Read the .evt file and reconstruct the data
        data = []
        with in_filepath.open("r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    timestamp: float = float(parts[0]) / 1000.0
                    event_type = parts[1]
                    if event_type in ['start', 'st']:
                        start = timestamp
                    elif event_type in ['end', 'e']:
                        end = timestamp
                        data.append({'start': start, 'stop': end})

        # Convert the reconstructed data into a DataFrame
        df = pd.DataFrame(data)
        df['label'] = df.index.astype('str', copy=True)
        _obj = cls.from_dataframe(df=df)
        _obj.filename = in_filepath.stem
        _obj.metadata = metadata
        return _obj
        
    def as_array(self):
        return self.to_dataframe()[["start", "stop"]].to_numpy()

    # Requires Optional `portion` library
    @classmethod
    def from_PortionInterval(cls, portion_interval, metadata=None):
        return Epoch(epochs=EpochsAccessor.from_PortionInterval(portion_interval), metadata=metadata) 

    def to_PortionInterval(self):
        return self._df.epochs.to_PortionInterval()

    def get_non_overlapping(self, debug_print=False):
        """ Returns a copy with overlapping epochs removed. """
        return Epoch(epochs=self._df.epochs.get_non_overlapping_df(debug_print=debug_print), metadata=self.metadata)
    

    # HDF5 Serialization _________________________________________________________________________________________________ #
    # HDFMixin Conformances ______________________________________________________________________________________________ #

    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pos_obj: Position = long_one_step_decoder_1D.pf.position
            _pos_obj.to_hdf(hdf5_output_path, key='pos')
        """
        _df = self.to_dataframe()
        _df.to_hdf(path_or_buf=file_path, key=key, format=kwargs.pop('format', 'table'), data_columns=kwargs.pop('data_columns',True), **kwargs)
        return
    
        # # create_group
        # a_key = Path(key)
        # with tb.open_file(file_path, mode='r+') as f:
        #     # group = f.create_group(str(a_key.parent), a_key.name, title='epochs.', createparents=True)
        #     group = f.get_node(str(a_key.parent))
        #     # group = f[key]
        #     table = f.create_table(group, a_key.name, EpochTable, "Epochs")
        #     # Serialization
        #     for i, t_start, t_stop, a_label in zip(np.arange(self.n_epochs), self.starts, self.stops, self.labels):
        #         row = table.row
        #         row['t_start'] = t_start
        #         row['t_end'] = t_stop  # Provide an appropriate session identifier here
        #         row['label'] = str(a_label)
        #         row.append()
                
        #     table.flush()
        #     # Metadata:
        #     group.attrs['t_start'] = self.t_start
        #     group.attrs['t_stop'] = self.t_stop
        #     group.attrs['n_epochs'] = self.n_epochs

    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> "Epoch":
        """  Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pos_obj = Epoch.read_hdf(hdf5_output_path, key='pos')
            _reread_pos_obj
        """
        _df = pd.read_hdf(file_path, key=key, **kwargs)
        return cls(_df, metadata=None) # TODO: recover metadata


    # DataFrameInitializable Conformances ________________________________________________________________________________ #
    
    def to_dataframe(self) -> pd.DataFrame:
        df = self._df.copy()
        return df
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, metadata=None):
        return cls(df, metadata=metadata)


    # ==================================================================================================================== #
    # `Epoch` object / pd.DataFrame exchangeability                                                                         #
    # ==================================================================================================================== #
    # NOTE: `def to_dataframe(self) -> pd.DataFrame` is defined above

    def to_Epoch(self) -> "Epoch":
        """ Ensures code exchangeability of epochs in either `Epoch` object / pd.DataFrame """
        return Epoch(epochs=self._df.copy(), metadata=self.metadata)


def ensure_dataframe(epochs: Union[Epoch, pd.DataFrame]) -> pd.DataFrame:
    """ Ensures that the epochs are returned as an Pandas DataFrame, does nothing if they already are a DataFrame.
    
    Reciprocal to `ensure_Epoch(...)`

    Usage:

        from neuropy.core.epoch import ensure_dataframe

    """
    if isinstance(epochs, pd.DataFrame):
        return epochs
    else:
        return epochs.to_dataframe()


def ensure_Epoch(epochs: Union[Epoch, pd.DataFrame], metadata=None) -> Epoch:
    """ Ensures that the epochs are returned as an Epoch object, does nothing if they already are an Epoch object.

        Reciprocal to `ensure_dataframe(...)`

        Usage:

        from neuropy.core.epoch import ensure_Epoch

    """
    if isinstance(epochs, pd.DataFrame):
        ## convert to Epoch
        return Epoch.from_dataframe(epochs, metadata=metadata)
    else:
        ## assume already an Epoch
        if metadata is not None:
            if epochs.metadata is None:
                epochs.metadata = metadata
            else:
                epochs.metadata = epochs.metadata | metadata # merge metadata
        
        return epochs
    



def subdivide_epochs(df: pd.DataFrame, subdivide_bin_size: float, start_col='start', stop_col='stop') -> pd.DataFrame:
    """ splits each epoch into equally sized chunks determined by subidivide_bin_size.
    
    # Example usage
        from neuropy.core.epoch import subdivide_epochs, ensure_dataframe

        df: pd.DataFrame = ensure_dataframe(deepcopy(long_LR_epochs_obj)) 
        df['epoch_type'] = 'lap'
        df['interval_type_id'] = 666

        subdivide_bin_size = 0.100  # Specify the size of each sub-epoch in seconds
        subdivided_df: pd.DataFrame = subdivide_epochs(df, subdivide_bin_size)
        print(subdivided_df)

    """
    sub_epochs = []

    extra_column_names = list(set(df.columns) - set([start_col, stop_col, 'label', 'duration', 'lap_id', 'lap_dir', 'epoch_t_bin_idx', 'epoch_num_t_bins']))
    # print(f'extra_column_names: {extra_column_names}')

    for index, row in df.iterrows():
        start = row[start_col]
        stop = row[stop_col]
        
        duration = stop - start
        num_bins: int = int(duration / subdivide_bin_size) + 1
        
        for i in range(num_bins):
            sub_start = start + (i * subdivide_bin_size)
            sub_stop = min(start + (i + 1) * subdivide_bin_size, stop)
            sub_duration = sub_stop - sub_start
            
            sub_epochs.append({
                start_col: sub_start,
                stop_col: sub_stop,
                'label': row['label'],
                'duration': sub_duration,
                'lap_id': row['lap_id'],
                'lap_dir': row['lap_dir'],
                'epoch_t_bin_idx': i,
                'epoch_num_t_bins': num_bins,
                **{k:row[k] for k in extra_column_names},
            })
    
    sub_epochs_df = pd.DataFrame(sub_epochs)
    return sub_epochs_df