from typing import Dict, List, Optional, Tuple
from nptyping import NDArray
import numpy as np
import pandas as pd
from functools import reduce # intersection_of_arrays, union_of_arrays


def flatten(A):
    """ safely flattens lists of lists without flattening top-level strings also. 
    https://stackoverflow.com/questions/17864466/flatten-a-list-of-strings-and-lists-of-strings-and-lists-in-python

    Usage:

        from neuropy.utils.indexing_helpers import flatten

    Example:
        list(flatten(['format_name', ('animal','exper_name', 'session_name')] ))
        >>> ['format_name', 'animal', 'exper_name', 'session_name']

    """
    rt = []
    for i in A:
        if isinstance(i, (list, tuple)): rt.extend(flatten(i))
        else: rt.append(i)
    return rt




def find_desired_sort_indicies(extant_arr, desired_sort_arr):
    """ Finds the set of sort indicies that can be applied to extant_arr s.t.
        (extant_arr[out_sort_idxs] == desired_sort_arr)
    
    INEFFICIENT: O^n^2
    
    Usage:
        from neuropy.utils.indexing_helpers import find_desired_sort_indicies
        new_all_aclus_sort_indicies, desired_sort_arr = find_desired_sort_indicies(active_2d_plot.neuron_ids, all_sorted_aclus)
        assert len(new_all_aclus_sort_indicies) == len(active_2d_plot.neuron_ids), f"need to have one new_all_aclus_sort_indicies value for each neuron_id"
        assert np.all(active_2d_plot.neuron_ids[new_all_aclus_sort_indicies] == all_sorted_aclus), f"must sort "
        new_all_aclus_sort_indicies
    """
    missing_aclu_indicies = np.isin(extant_arr, desired_sort_arr, invert=True)
    missing_aclus = extant_arr[missing_aclu_indicies] # array([ 3,  4,  8, 13, 24, 34, 56, 87])
    if len(missing_aclus) > 0:
        desired_sort_arr = np.concatenate((desired_sort_arr, missing_aclus)) # the final desired output order of aclus. Want to compute the indicies that are required to sort an ordered array of indicies in this order
        ## TODO: what about entries in desired_sort_arr that might be missing in extant_arr?? Hopefully never happens.
    assert len(desired_sort_arr) == len(extant_arr), f"need to have one all_sorted_aclu value for each neuron_id but len(desired_sort_arr): {len(desired_sort_arr)} and len(extant_arr): {len(extant_arr)}"
    # sort_idxs = np.array([desired_sort_arr.tolist().index(v) for v in extant_arr])
    sort_idxs = np.array([extant_arr.tolist().index(v) for v in desired_sort_arr])
    assert len(sort_idxs) == len(extant_arr), f"need to have one new_all_aclus_sort_indicies value for each neuron_id"
    assert np.all(extant_arr[sort_idxs] == desired_sort_arr), f"must sort: extant_arr[sort_idxs]: {extant_arr[sort_idxs]}\n desired_sort_arr: {desired_sort_arr}"
    return sort_idxs, desired_sort_arr



# ==================================================================================================================== #
# Multi-list Operators                                                                                                 #
# ==================================================================================================================== #

def union_of_arrays(*arrays) -> np.array:
    """ 
    from neuropy.utils.indexing_helpers import union_of_arrays
    
    """
    return np.unique(np.concatenate(arrays)) # note that np.unique SORTS the items
    # return reduce(np.union1d, tuple(arrays)) # should be equivalent


def intersection_of_arrays(*arrays) -> np.array:
    """ 
    from neuropy.utils.indexing_helpers import union_of_arrays
    
    """
    return reduce(np.intersect1d, tuple(arrays))


# ==================================================================================================================== #
# Numpy NDArrays                                                                                                       #
# ==================================================================================================================== #

class NumpyHelpers:
    """ various extensions and generalizations for numpy arrays 
    
    from neuropy.utils.indexing_helpers import NumpyHelpers


    """
    @classmethod
    def all_array_generic(cls, pairwise_numpy_fn, list_of_arrays: List[NDArray], **kwargs) -> bool:
        """ A n-element generalization of a specified pairwise numpy function such as `np.array_equiv`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_array_generic(list_of_arrays=list_of_arrays)

        """
        # Input type checking
        if not np.all(isinstance(arr, np.ndarray) for arr in list_of_arrays):
            raise ValueError("All elements in 'list_of_arrays' must be NumPy arrays.")        
    
        if len(list_of_arrays) == 0:
            return True # empty arrays are all equal
        elif len(list_of_arrays) == 1:
            # if only a single array, make sure it's not accidentally passed in incorrect
            reference_array = list_of_arrays[0] # Use the first array as a reference for comparison
            assert isinstance(reference_array, np.ndarray)
            return True # as long as imput is intended, always True
        
        else:
            ## It has more than two elements:
            reference_array = list_of_arrays[0] # Use the first array as a reference for comparison
            # Check equivalence for each array in the list
            return np.all([pairwise_numpy_fn(reference_array, an_arr, **kwargs) for an_arr in list_of_arrays[1:]]) # can be used without the list comprehension just as a generator if you use all(...) instead.
            # return all(np.all(np.array_equiv(reference_array, an_arr) for an_arr in list_of_arrays[1:])) # the outer 'all(...)' is required, otherwise it returns a generator object like: `<generator object NumpyHelpers.all_array_equiv.<locals>.<genexpr> at 0x00000128E0482AC0>`

    @classmethod
    def assert_all_array_generic(cls, pairwise_numpy_assert_fn, list_of_arrays: List[NDArray], **kwargs):
        """ A n-element generalization of a specified pairwise np.testing.assert* function such as `np.testing.assert_array_equal` or `np.testing.assert_allclose`

        TODO: doesn't really work yet

        msg: a use-provided assert message
        

        
        Usage:

            list_of_arrays = list(xbins.values())
            NumpyHelpers.assert_all_array_generic(np.testing.assert_array_equal, list_of_arrays=list_of_arrays, msg=f'test message')
            NumpyHelpers.assert_all_array_generic(np.testing.assert_array_equal, list_of_arrays=list(neuron_ids.values()), msg=f'test message')

        """
        msg = kwargs.pop('msg', None)
        # Input type checking
        if not np.all(isinstance(arr, np.ndarray) for arr in list_of_arrays):
            raise ValueError("All elements in 'list_of_arrays' must be NumPy arrays.")        
    
        if len(list_of_arrays) == 0:
            return # empty arrays are all equal
        elif len(list_of_arrays) == 1:
            # if only a single array, make sure it's not accidentally passed in incorrect
            reference_array = list_of_arrays[0] # Use the first array as a reference for comparison
            assert isinstance(reference_array, np.ndarray)
            return # as long as imput is intended, always True
        
        else:
            ## It has more than two elements:
            reference_array = list_of_arrays[0] # Use the first array as a reference for comparison
            # Check equivalence for each array in the list
            for an_arr in list_of_arrays[1:]:
                try:
                    pairwise_numpy_assert_fn(reference_array, an_arr, **kwargs) 
                except AssertionError as e:
                    # print(f'e: {e}, e.args: {e.args}')
                    # msg = kwargs.get("msg", None)
                    if msg is not None:
                        e.args = (msg,) + e.args
                        # e.args = ':'.join(e.args) # join as a single string 
                        
                    # print(f'e: {e},\n e.args: {e.args},\n msg: {msg or ""}\n')
                    # print(f'e.args: {":".join(e.args)}')
                    raise e
                
            # return np.all([pairwise_numpy_assert_fn(reference_array, an_arr, *args, **kwargs) for an_arr in list_of_arrays[1:]]) # can be used without the list comprehension just as a generator if you use all(...) instead.
            # return all(np.all(np.array_equiv(reference_array, an_arr) for an_arr in list_of_arrays[1:])) # the outer 'all(...)' is required, otherwise it returns a generator object like: `<generator object NumpyHelpers.all_array_equiv.<locals>.<genexpr> at 0x00000128E0482AC0>`



    @classmethod
    def all_array_equal(cls, list_of_arrays: List[NDArray], equal_nan=True) -> bool:
        """ A n-element generalization of `np.array_equal`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_array_equal(list_of_arrays=list_of_arrays)

        """
        return cls.all_array_generic(np.array_equal, list_of_arrays=list_of_arrays, equal_nan=equal_nan)
    
    @classmethod
    def all_array_equiv(cls, list_of_arrays: List[NDArray]) -> bool:
        """ A n-element generalization of `np.array_equiv`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_array_equiv(list_of_arrays=list_of_arrays)

        """
        return cls.all_array_generic(np.array_equiv, list_of_arrays=list_of_arrays)


    @classmethod
    def all_allclose(cls, list_of_arrays: List[NDArray], rtol:float=1.e-5, atol:float=1.e-8, equal_nan:bool=True) -> bool:
        """ A n-element generalization of `np.allclose`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_allclose(list_of_arrays=list_of_arrays)

        """
        return cls.all_array_generic(np.allclose, list_of_arrays=list_of_arrays, rtol=rtol, atol=atol, equal_nan=equal_nan)
    


# ==================================================================================================================== #
# Sorting/Joint-sorting                                                                                                #
# ==================================================================================================================== #

# @function_attributes(short_name=None, tags=['sort', 'neuron_ID', 'neuron_IDX', 'pfs'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-28 00:59', related_items=[])
def paired_incremental_sorting(neuron_IDs_lists, sortable_values_lists):
    """ builds up a list of `sorted_lists` 

    Given a list of neuron_IDs (usually aclus) and equal-sized lists containing values which to sort the lists of neuron_IDs, returns the list of incrementally sorted neuron_IDs. e.g.:

    Inputs: neuron_IDs_lists, sortable_values_lists

    Usage:
        from neuropy.utils.indexing_helpers import paired_incremental_sorting

        neuron_IDs_lists = [a_decoder.neuron_IDs for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
        sortable_values_lists = [np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1) for a_decoder in decoders_dict.values()]
        sorted_neuron_IDs_lists = paired_incremental_sorting(neuron_IDs_lists, sortable_values_lists)
        sort_helper_original_neuron_id_to_IDX_dicts = [dict(zip(neuron_ids, np.arange(len(neuron_ids)))) for neuron_ids in neuron_IDs_lists] # just maps each neuron_id in the list to a fragile_linear_IDX 

        # `sort_helper_neuron_id_to_sort_IDX_dicts` dictionaries in the appropriate order (sorted order) with appropriate indexes. Its .values() can be used to index into things originally indexed with aclus.
        sort_helper_neuron_id_to_sort_IDX_dicts = [{aclu:a_sort_helper_neuron_id_to_IDX_map[aclu] for aclu in sorted_neuron_ids} for a_sort_helper_neuron_id_to_IDX_map, sorted_neuron_ids in zip(sort_helper_original_neuron_id_to_IDX_dicts, sorted_neuron_IDs_lists)]
        # sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[a_sort_list, :] for a_decoder, a_sort_list in zip(decoders_dict.values(), sorted_neuron_IDs_lists)]

        sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]


    """

    sorted_lists = []
    union_accumulator = []
    assert len(neuron_IDs_lists) == len(sortable_values_lists), f"value lists must be the same length"
    if (len(neuron_IDs_lists) == 0):
        return sorted_lists
    assert np.all([len(neuron_ids) == len(sortable_values) for neuron_ids, sortable_values in zip(neuron_IDs_lists, sortable_values_lists)]), f"all items must be the same length."
    
    sortable_neuron_id_dicts = [dict(zip(neuron_ids, sortable_values)) for neuron_ids, sortable_values in zip(neuron_IDs_lists, sortable_values_lists)]
    assert len(neuron_IDs_lists) == len(sortable_neuron_id_dicts)
    for a_sortable_neuron_id_dict in sortable_neuron_id_dicts:
        # neuron_ids, sortable_values
        prev_sorted_neuron_ids = [aclu for aclu in union_accumulator if aclu in a_sortable_neuron_id_dict.keys()] # loop through the accumulator
        # novel_sorted_neuron_ids = [aclu for aclu in a_sortable_neuron_id_dict.keys() if aclu not in union_accumulator] # doesn't sort them, passes them unsorted as-is
        novel_sorted_neuron_id_dicts = {aclu:sort_v for aclu, sort_v in a_sortable_neuron_id_dict.items() if aclu not in union_accumulator} # subset based on value not being in union_accumulator
        # Sort them now as needed:
        novel_sorted_neuron_id_dicts = dict(sorted(novel_sorted_neuron_id_dicts.items(), key=lambda item: item[1]))
        # Convert them into a list as expected now that they're sorted based on values:
        novel_sorted_neuron_ids	= list(novel_sorted_neuron_id_dicts.keys())
        curr_sorted_list = np.array([*prev_sorted_neuron_ids, *novel_sorted_neuron_ids])
        sorted_lists.append(curr_sorted_list)
        union_accumulator.extend(novel_sorted_neuron_ids) # by only adding the novel elements at each step, we should accumulate a list only consisting of the novel elements from each step.
        # prune the duplicates here, it should operate like a set.

    assert [len(neuron_ids) == len(sorted_neuron_ids) for neuron_ids, sorted_neuron_ids in zip(neuron_IDs_lists, sorted_lists)], f"all items must be the same length."
    
    return sorted_lists



def paired_individual_sorting(neuron_IDs_lists, sortable_values_lists):
    """ nothing "paired" about it, just individually sorts the items in `neuron_IDs_lists` by `sortable_values_lists`

    Given a list of neuron_IDs (usually aclus) and equal-sized lists containing values which to sort the lists of neuron_IDs, returns the list of incrementally sorted neuron_IDs. e.g.:

    Inputs: neuron_IDs_lists, sortable_values_lists

    Usage:
        from neuropy.utils.indexing_helpers import paired_individual_sorting

        neuron_IDs_lists = [a_decoder.neuron_IDs for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
        sortable_values_lists = [np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1) for a_decoder in decoders_dict.values()]
        sorted_neuron_IDs_lists = paired_individual_sorting(neuron_IDs_lists, sortable_values_lists)
        sort_helper_original_neuron_id_to_IDX_dicts = [dict(zip(neuron_ids, np.arange(len(neuron_ids)))) for neuron_ids in neuron_IDs_lists] # just maps each neuron_id in the list to a fragile_linear_IDX 

        # `sort_helper_neuron_id_to_sort_IDX_dicts` dictionaries in the appropriate order (sorted order) with appropriate indexes. Its .values() can be used to index into things originally indexed with aclus.
        sort_helper_neuron_id_to_sort_IDX_dicts = [{aclu:a_sort_helper_neuron_id_to_IDX_map[aclu] for aclu in sorted_neuron_ids} for a_sort_helper_neuron_id_to_IDX_map, sorted_neuron_ids in zip(sort_helper_original_neuron_id_to_IDX_dicts, sorted_neuron_IDs_lists)]
        # sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[a_sort_list, :] for a_decoder, a_sort_list in zip(decoders_dict.values(), sorted_neuron_IDs_lists)]

        sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]


    """

    sorted_lists = []
    union_accumulator = []
    assert len(neuron_IDs_lists) == len(sortable_values_lists), f"value lists must be the same length"
    if (len(neuron_IDs_lists) == 0):
        return sorted_lists
    assert np.all([len(neuron_ids) == len(sortable_values) for neuron_ids, sortable_values in zip(neuron_IDs_lists, sortable_values_lists)]), f"all items must be the same length."    
    sortable_neuron_id_dicts = [dict(zip(neuron_ids, sortable_values)) for neuron_ids, sortable_values in zip(neuron_IDs_lists, sortable_values_lists)]
    assert len(neuron_IDs_lists) == len(sortable_neuron_id_dicts)
    for a_sortable_neuron_id_dict in sortable_neuron_id_dicts:
        # Sort them now as needed:
        curr_sorted_list = np.array(list(dict(sorted(a_sortable_neuron_id_dict.items(), key=lambda item: item[1])).keys()))
        sorted_lists.append(curr_sorted_list)

    assert [len(neuron_ids) == len(sorted_neuron_ids) for neuron_ids, sorted_neuron_ids in zip(neuron_IDs_lists, sorted_lists)], f"all items must be the same length."
    
    return sorted_lists




def find_nearest_time(df: pd.DataFrame, target_time: float, time_column_name:str='start', max_allowed_deviation:float=0.01, debug_print=False):
    """ finds the nearest time in the time_column_name matching the provided target_time
    

    max_allowed_deviation: if provided, requires the difference between the found time in the dataframe and the target_time to be less than or equal to max_allowed_deviation
    Usage:

    from neuropy.utils.indexing_helpers import find_nearest_time
    df = deepcopy(_out_ripple_rasters.active_epochs_df)
    df, closest_index, closest_time, matched_time_difference = find_nearest_time(df=df, target_time=193.65)
    df.iloc[closest_index]

    """
    # Ensure the DataFrame is sorted (if you're not sure it's already sorted)
    assert time_column_name in df.columns
    if df[time_column_name].is_monotonic:
        if debug_print:
            print('The column is already sorted in ascending order.')
    else:
        print(f'WARNING: The column is not sorted in ascending order. Sorting now...')
        df = df.sort_values(by=time_column_name, inplace=False)

    # Use searchsorted to find the insertion point for the target time
    insertion_index = df[time_column_name].searchsorted(target_time)

    # Since searchsorted returns the index where the target should be inserted to
    # maintain order, the closest time could be at this index or the previous index.
    # We need to compare both to find the nearest time.
    if insertion_index == 0:
        # The target_time is smaller than all elements, so closest is the first item
        closest_index = 0
    elif insertion_index == len(df):
        # The target_time is bigger than all elements, so closest is the last item
        closest_index = len(df) - 1
    else:
        # The target_time is between two elements, find the nearest one
        prev_time = df[time_column_name].iloc[insertion_index - 1]
        next_time = df[time_column_name].iloc[insertion_index]
        # Compare the absolute difference to determine which is closer
        closest_index = insertion_index if (next_time - target_time) < (target_time - prev_time) else insertion_index - 1

    # Now extract the closest time using the index
    closest_time = df[time_column_name].iloc[closest_index]
    matched_time_difference = closest_time - target_time
    if (max_allowed_deviation is not None) and (abs(matched_time_difference) > max_allowed_deviation):
        # raise an error
        print(f'WARNING: The closest start time to {target_time} ({closest_time} at index {closest_index}) exceeds the max_allowed_deviation of {max_allowed_deviation} (obs. deviation {matched_time_difference})\n.\t\tReturning None')
        # None out the found values:
        closest_index = None
        closest_time = None
    else:
        if debug_print:
            print(f"The closest start time to {target_time} is {closest_time} at index {closest_index}. Deviating by {matched_time_difference}")

    return df, closest_index, closest_time, matched_time_difference


def find_nearest_times(df: pd.DataFrame, target_times: np.ndarray, time_column_name: str='start', max_allowed_deviation: float=0.01, debug_print=False):
    """
    Find the nearest time indices for each target time within the specified max_allowed_deviation.
    """
    # Ensure the DataFrame is sorted
    assert time_column_name in df.columns
    if not df[time_column_name].is_monotonic:
        if debug_print:
            print(f'WARNING: The column is not sorted in ascending order. Sorting now...')
        df = df.sort_values(by=time_column_name, inplace=False)

    # Prepare output Series for closest indices
    closest_indices = pd.Series(index=target_times, dtype='int')
    matched_time_differences = pd.Series(index=target_times, dtype='float')
    
    # Vectorized search for insertion points
    insertion_indices = df[time_column_name].searchsorted(target_times)
    
    # Process each target time
    for idx, target_time in np.ndenumerate(target_times):
        target_idx = idx[0]
        insertion_index = insertion_indices[target_idx]
        if insertion_index == 0:
            closest_index = 0
        elif insertion_index == len(df):
            closest_index = len(df) - 1
        else:
            prev_time = df[time_column_name].iloc[insertion_index - 1]
            next_time = df[time_column_name].iloc[insertion_index]
            closest_index = insertion_index if (next_time - target_time) < (target_time - prev_time) else insertion_index - 1
        
        # Now extract the closest time using the index
        closest_time = df[time_column_name].iloc[closest_index]
        matched_time_difference = closest_time - target_time
        
        # Check if the found time is within the max_allowed_deviation
        if (max_allowed_deviation is not None) and (abs(matched_time_difference) > max_allowed_deviation):
            if debug_print:
                print(f'WARNING: The closest start time to {target_time} ({closest_time} at index {closest_index}) exceeds the max_allowed_deviation of {max_allowed_deviation} (obs. deviation {matched_time_difference}).')
            matched_time_differences.iloc[target_idx] = np.nan
            closest_indices.iloc[target_idx] = np.nan  # or use some other sentinel value
        else:
            matched_time_differences.iloc[target_idx] = matched_time_difference
            closest_indices.iloc[target_idx] = closest_index
            if debug_print:
                print(f"The closest start time to {target_time} is {closest_time} at index {closest_index}. Deviating by {matched_time_difference}")

    return closest_indices, matched_time_differences