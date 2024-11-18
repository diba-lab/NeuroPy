from typing import Union, List, Dict, Set, Any, Optional, OrderedDict  # for OrderedMeta
from nptyping import NDArray
from contextlib import contextmanager, ContextDecorator

import numpy as np
import pandas as pd
from functools import reduce # intersection_of_arrays, union_of_arrays

from typing import Iterable, TypeVar, Dict, List, Tuple, Any, Optional

T = TypeVar('T')

def collapse_if_identical(iterable: Iterable[T], return_original_on_failure: bool = False) -> Optional[Iterable[T]]:
    """
    Collapse an iterable to its first item if all items in the iterable are identical.
    If not all items are identical and 'return_original_on_failure' is True, the original
    iterable is returned as much collapsed as possible; otherwise, None is returned.

    Parameters
    ----------
    iterable : Iterable[T]
        An iterable containing items of any type (denoted by T).
    return_original_on_failure : bool, default=False
        If True, return the original iterable when it's not collapsible. If False, return None.

    Returns
    -------
    Optional[Iterable[T]]
        The first item of the iterable if all items are identical, or if 'return_original_on_failure'
        is set to True, the original iterable is returned when items are not identical. Otherwise, None
        is returned.

    Raises
    ------
    StopIteration
        If the provided iterable is empty, a StopIteration exception is raised internally
        and caught by the function to return None or the original iterable.
        
    Examples
    --------
    from neuropy.utils.indexing_helpers import collapse_if_identical
    
    >>> identical_items = ["a", "a", "a"]
    >>> collapse_if_identical(identical_items)
    "a"
    
    >>> non_identical_items = ["a", "b", "a"]
    >>> collapse_if_identical(non_identical_items)
    None
    
    >>> collapse_if_identical(non_identical_items, return_original_on_failure=True)
    ["a", "b", "a"]
    
    >>> empty_iterable = []
    >>> collapse_if_identical(empty_iterable)
    None
    """
    # Get an iterator for the iterable
    iterator = iter(iterable)
    
    try:
        # Get the first item for comparison
        first_item = next(iterator)
    except StopIteration:
        # If the iterable is empty, nothing to compare, return None or the original iterable
        return None if not return_original_on_failure else iterable
    
    # Check if all subsequent items in the iterator are equal to the first
    collapsed = all(item == first_item for item in iterator)
    if collapsed:
        # All items are the same, return the item
        return first_item

    # Since the collapse failed, determine the return value
    return iterable if return_original_on_failure else None


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


def unwrap_single_item(lst):
    """ if the item contains at least one item, return it, otherwise return None.

    from neuropy.utils.indexing_helpers import unwrap_single_item


    """
    return lst[0] if len(lst) == 1 else None

        
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
    if (len(extant_arr) == 0) or (len(desired_sort_arr) == 0):
        raise ValueError(f"find_desired_sort_indicies(...): extant_arr or desired_sort_arr is empty! How are we going to find the indicies? : \n\textant_arr: {extant_arr}\n\t desired_sort_arr: {desired_sort_arr}")

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
    
    
    @classmethod
    def safe_concat(cls, np_concat_list: Union[List[NDArray], Dict[Any, NDArray]], **np_concat_kwargs) -> NDArray:
        """ returns an empty dataframe if the key isn't found in the group.
        Usage:
            from neuropy.utils.indexing_helpers import NumpyHelpers

            NumpyHelpers.safe_concat
            
        """
        if len(np_concat_list) > 0:
            if isinstance(np_concat_list, dict):
                np_concat_list = list(np_concat_list.values())
            out_concat_arr: NDArray = np.concatenate(np_concat_list, **np_concat_kwargs)
        else:
            # out_concat_arr = []
            out_concat_arr = np.array([]) # empty df would be better
        return out_concat_arr
    
    

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
        prev_time = df[time_column_name].iloc[insertion_index - 1] # @DA#TODO 2024-03-11 17:49: - [ ] Is .iloc okay here under all circumstances?
        next_time = df[time_column_name].iloc[insertion_index]
        # Compare the absolute difference to determine which is closer
        closest_index = insertion_index if (next_time - target_time) < (target_time - prev_time) else insertion_index - 1

    # Now extract the closest time using the index
    closest_time = df[time_column_name].iloc[closest_index] ## NOTE .iloc here!

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
    !! Untested !! a ChatGPT GPT4-turbo written find_nearest_times which extends find_nearest_time to multiple target times. 
    Find the nearest time indices for each target time within the specified max_allowed_deviation.

    Usage:

        from neuropy.utils.indexing_helpers import find_nearest_times

        closest_indices, matched_time_differences = find_nearest_times(df=a_df, target_times=arr, time_column_name='start')
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






# ==================================================================================================================== #
# Dictionary Helpers                                                                                                   #
# ==================================================================================================================== #

def convert_to_dictlike(other) -> Dict:
    """ try every known trick to get a plain `dict` out of the provided object. """
    # Verify that other is an instance of IdentifyingContext
    if isinstance(other, dict):
        return other
    elif hasattr(other, 'to_dict') and callable(getattr(other, 'to_dict')):
        return other.to_dict()
    # elif (hasattr(other, 'items') and callable(getattr(other, 'items'))):
    #     # Check if 'other' has an 'items' method, it's "close enough"?
    #     return other
    elif hasattr(other, '__dict__'):
        # Check if 'other' has a '__dict__' property
        return other.__dict__
    else:
        raise NotImplementedError(f"Object other of type: {type(other)} could not be converted to a python dict.\nother: {other}.")

def get_nested_value(d: Dict, keys: List[Any]) -> Any:
    """  how can I index into nested dictionaries using a list of keys? """
    for key in keys:
        d = d[key]
    return d

def flatten_dict(d: Dict, parent_key='', sep='/') -> Dict:
    """ flattens a dictionary to a single non-nested dict

    Example:

    Input:
        {'computation_params': {'merged_directional_placefields': {'laps_decoding_time_bin_size': 0.25, 'ripple_decoding_time_bin_size': 0.025, 'should_validate_lap_decoding_performance': False},
        'rank_order_shuffle_analysis': {'num_shuffles': 500, 'minimum_inclusion_fr_Hz': 5.0, 'included_qclu_values': [1, 2], 'skip_laps': False},
        'directional_decoders_decode_continuous': {'time_bin_size': None},
        'directional_decoders_evaluate_epochs': {'should_skip_radon_transform': False},
            ...
        }

    Output:

        {'merged_directional_placefields/laps_decoding_time_bin_size': 0.25,
            'merged_directional_placefields/ripple_decoding_time_bin_size': 0.025,
            'merged_directional_placefields/should_validate_lap_decoding_performance': False,
            'rank_order_shuffle_analysis/num_shuffles': 500,
            'rank_order_shuffle_analysis/minimum_inclusion_fr_Hz': 5.0,
            'rank_order_shuffle_analysis/included_qclu_values': [1, 2],
            'rank_order_shuffle_analysis/skip_laps': False,
            'directional_decoders_decode_continuous/time_bin_size': None,
            'directional_decoders_evaluate_epochs/should_skip_radon_transform': False,
            ...
            }

    Usage:
        from neuropy.utils.indexing_helpers import flatten_dict
    
    """
    if (not isinstance(d, dict)):
        if hasattr(d, 'to_dict'):
            d = d.to_dict() ## convert to dict and continue
        else:
            assert isinstance(parent_key, str), f"expected type(parent_key) == str but instead type(parent_key): {type(parent_key)}, parent_key: {parent_key}"
            return {parent_key:d}
    
    items = {}
    for k, v in d.items():
        # Construct the new key by concatenating the parent key and current key
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # If the value is a dictionary, recursively flatten it
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            # If the value is not a dictionary, add it to the items
            items[new_key] = v
    return items


# ==================================================================================================================== #
#region PandasDataFrameHelpers                                                                                               #
# ==================================================================================================================== #
    
class PandasHelpers:
    """ various extensions and generalizations for numpy arrays 
    
    from neuropy.utils.indexing_helpers import PandasHelpers


    """
    @classmethod
    def require_columns(cls, dfs: Union[pd.DataFrame, List[pd.DataFrame], Dict[Any, pd.DataFrame]], required_columns: List[str], print_missing_columns: bool = False) -> bool:
        """
        Check if all DataFrames in the given container have the required columns.
        
        Parameters:
            dfs: A container that may be a single DataFrame, a list/tuple of DataFrames, or a dictionary with DataFrames as values.
            required_columns: A list of column names that are required to be present in each DataFrame.
            print_changes: If True, prints the columns that are missing from each DataFrame.
        
        Returns:
            True if all DataFrames contain all the required columns, otherwise False.

        Usage:

            required_cols = ['missing_column', 'congruent_dir_bins_ratio', 'coverage', 'direction_change_bin_ratio', 'jump', 'laplacian_smoothness', 'longest_sequence_length', 'longest_sequence_length_ratio', 'monotonicity_score', 'sequential_correlation', 'total_congruent_direction_change', 'travel'] # Replace with actual column names you require
            has_required_columns = PandasHelpers.require_columns({a_name:a_result.filter_epochs for a_name, a_result in filtered_decoder_filter_epochs_decoder_result_dict.items()}, required_cols, print_missing_columns=True)
            has_required_columns
            


        """
        def _subfn_check_and_report(df: pd.DataFrame, columns: List[str], identifier: Any = None) -> bool:
            missing_columns = set(columns).difference(df.columns)
            has_all_columns = not missing_columns
            
            if print_missing_columns and not has_all_columns:
                df_name = f" (DataFrame Identifier: {identifier})" if identifier else ""
                print(f"Missing required columns in DataFrame{df_name}: {missing_columns}")

            return has_all_columns

        all_have_all_columns = True
        
        if isinstance(dfs, pd.DataFrame):
            all_have_all_columns = _subfn_check_and_report(dfs, required_columns)
        elif isinstance(dfs, (list, tuple)):
            all_have_all_columns = all(_subfn_check_and_report(df, required_columns, i) for i, df in enumerate(dfs))
        elif isinstance(dfs, dict):
            all_have_all_columns = all(_subfn_check_and_report(df, required_columns, key) for key, df in dfs.items())

        return all_have_all_columns

    @classmethod
    def reordering_columns(cls, df: pd.DataFrame, column_name_desired_index_dict: Dict[str, int]) -> pd.DataFrame:
        """Reorders specified columns in a DataFrame while preserving other columns.
        
        Pure: Does not modify the df

        Args:
            df (pd.DataFrame): The DataFrame to reorder.
            column_name_desired_index_dict (Dict[str, int]): A dictionary where keys are column names
                to reorder and values are their desired indices in the reordered DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with specified columns reordered while preserving remaining columns.

        Raises:
            ValueError: If any column in the dictionary is not present in the DataFrame.
            
            
        Usage:
        
            from neuropy.utils.indexing_helpers import PandasHelpers
            dict(zip(['Long_LR_evidence', 'Long_RL_evidence', 'Short_LR_evidence', 'Short_RL_evidence'], np.arange(4)+4))
            PandasHelpers.reorder_columns(merged_complete_epoch_stats_df, column_name_desired_index_dict=dict(zip(['Long_LR_evidence', 'Long_RL_evidence', 'Short_LR_evidence', 'Short_RL_evidence'], np.arange(4)+4)))
            
            ## Move the "height" columns to the end
            result_df = PandasHelpers.reorder_columns(result_df, column_name_desired_index_dict=dict(zip(list(filter(lambda column: column.endswith('_peak_heights'), result_df.columns)), np.arange(len(result_df.columns)-4, len(result_df.columns)))))
            result_df
                    
        """
        # Validate column names
        missing_columns = set(column_name_desired_index_dict.keys()) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

        # Ensure desired indices are unique and within range
        desired_indices = column_name_desired_index_dict.values()
        if len(set(desired_indices)) != len(desired_indices) or any(index < 0 or index >= len(df.columns) for index in desired_indices):
            raise ValueError("Desired indices must be unique and within the range of existing columns.")

        # Create a list of columns to reorder
        reordered_columns_desired_index_dict: Dict[str, int] = {column_name:desired_index for column_name, desired_index in sorted(column_name_desired_index_dict.items(), key=lambda item: item[1])}
        # print(reordered_columns_desired_index_dict)
        
        # # Reorder specified columns while preserving remaining columns
        remaining_columns = [col for col in df.columns if col not in column_name_desired_index_dict]
        
        reordered_columns_list: List[str] = remaining_columns.copy()
        for item_to_insert, desired_index in reordered_columns_desired_index_dict.items():    
            reordered_columns_list.insert(desired_index, item_to_insert)
            
        # print(reordered_columns_list)
        reordered_df = df[reordered_columns_list]
        return reordered_df

    @classmethod
    def reordering_columns_relative(cls, df: pd.DataFrame, column_names: list[str], relative_mode='end') -> pd.DataFrame:
        """Reorders specified columns in a DataFrame while preserving other columns.
        
        Pure: Does not modify the df

        Args:
            df (pd.DataFrame): The DataFrame to reorder.
            column_name_desired_index_dict (Dict[str, int]): A dictionary where keys are column names
                to reorder and values are their desired indices in the reordered DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with specified columns reordered while preserving remaining columns.

        Raises:
            ValueError: If any column in the dictionary is not present in the DataFrame.
            
            
        Usage:
        
            ffrom neuropy.utils.indexing_helpers import PandasHelpers
            
            ## Move the "height" columns to the end
            result_df = PandasHelpers.reordering_columns_relative(result_df, column_names=list(filter(lambda column: column.endswith('_peak_heights'), existing_columns)), relative_mode='end')
            result_df
                    
        """
        if relative_mode == 'end':
            existing_columns = list(df.columns)
            return cls.reordering_columns(df, column_name_desired_index_dict=dict(zip(column_names, np.arange(len(existing_columns)-4, len(existing_columns)))))
        else:
            raise NotImplementedError
        

    @classmethod
    def safe_pandas_get_group(cls, dataframe_group, key):
        """ returns an empty dataframe if the key isn't found in the group."""
        if key in dataframe_group.groups.keys():
            return dataframe_group.get_group(key)
        else:
            original_df = dataframe_group.obj
            return original_df.drop(original_df.index)


    @classmethod
    def safe_concat(cls, df_concat_list: Union[List[pd.DataFrame], Dict[Any, pd.DataFrame]], **pd_concat_kwargs) -> Optional[pd.DataFrame]:
        """ returns an empty dataframe if the list of dataframes is empty.
        
        NOTE: does not perform intellegent merging, just handles empty lists
            
            
        Usage:
            from neuropy.utils.indexing_helpers import PandasHelpers

            PandasHelpers.safe_concat
            
        """
        if len(df_concat_list) > 0:
            if isinstance(df_concat_list, dict):
                df_concat_list = list(df_concat_list.values())
            out_concat_df: pd.DataFrame = pd.concat(df_concat_list, **pd_concat_kwargs)
        else:
            out_concat_df = None # empty df would be better
        return out_concat_df

    @classmethod
    def convert_dataframe_columns_to_datatype_if_possible(cls, df: pd.DataFrame, datatype_str_column_names_list_dict, debug_print=False):
        """ If the columns specified in datatype_str_column_names_list_dict exist in the dataframe df, their type is changed to the key of the dict. See usage example below:
        
        Inputs:
            df: Pandas.DataFrame 
            datatype_str_column_names_list_dict: {'int':['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']}

        Usage:
            from neuropy.utils.indexing_helpers import PandasHelpers
            PandasHelpers.convert_dataframe_columns_to_datatype_if_possible(curr_active_pipeline.sess.spikes_df, {'int':['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']})
        """
        # spikes_df[['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']] = spikes_df[['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']].astype('int') # convert integer calumns to correct datatype
        for a_datatype_name, a_column_names_list in datatype_str_column_names_list_dict.items():
            is_included = np.isin(a_column_names_list, df.columns)
            if debug_print:
                print(f'datatype: {a_datatype_name}: {a_column_names_list}:{is_included}')
            a_column_names_list = np.array(a_column_names_list)
            subset_extant_columns = a_column_names_list[is_included] # Get only the column names thare are included in the dataframe
            if debug_print:
                print(f'\t subset_extant_columns: {subset_extant_columns}')
            # subset_extant_columns = a_column_names_list
            df[subset_extant_columns] = df[subset_extant_columns].astype(a_datatype_name) # convert integer calumns to correct datatype

    @classmethod
    def add_explicit_dataframe_columns_from_lookup_df(cls, df: pd.DataFrame, lookup_properties_map_df: pd.DataFrame, join_column_name='aclu') -> pd.DataFrame:
        """ Uses a value (specified by `join_column_name`) in each row of `df` to lookup the appropriate values in `lookup_properties_map_df` to be explicitly added as columns to `df`
        df: a dataframe. Each row has a join_column_name value (e.g. 'aclu')
        
        lookup_properties_map_df: a dataframe with one row for each `join_column_name` value (e.g. one row for each 'aclu', describing various properties of that neuron)
        
        
        By default lookup_properties_map_df can be obtained from curr_active_pipeline.sess.neurons._extended_neuron_properties_df and has the columns:
            ['aclu', 'qclu', 'neuron_type', 'shank', 'cluster']
        Which will be added to the spikes_df
        
        WARNING: the df will be unsorted after this operation, and you'll need to sort it again if you want it sorted
        
        
        Usage:
            from neuropy.utils.indexing_helpers import PandasHelpers
            curr_active_pipeline.sess.flattened_spiketrains._spikes_df = PandasHelpers.add_explicit_dataframe_columns_from_lookup_df(curr_active_pipeline.sess.spikes_df, curr_active_pipeline.sess.neurons._extended_neuron_properties_df)
            curr_active_pipeline.sess.spikes_df.sort_values(by=['t_seconds'], inplace=True) # Need to re-sort by timestamps once done
            curr_active_pipeline.sess.spikes_df

        """
        ## only find the columns in lookup_properties_map_df that are NOT in df. e.g. ['qclu', 'cluster']
        missing_spikes_df_columns = list(lookup_properties_map_df.columns[np.logical_not(np.isin(lookup_properties_map_df.columns, df.columns))]) 
        missing_spikes_df_columns.insert(0, join_column_name) # Insert 'aclu' into the list so we can join on it
        subset_neurons_properties_df = lookup_properties_map_df[missing_spikes_df_columns] # get the subset dataframe with only the requested columns
        ## Merge the result:
        # df = pd.merge(subset_neurons_properties_df, df, on=join_column_name, how='outer', suffixes=('_neurons_properties', '_spikes_df'))
        return pd.merge(df, subset_neurons_properties_df, on=join_column_name, how='left', suffixes=('_neurons_properties', '_spikes_df'), copy=False) # avoids copying if possible

    @classmethod
    def adding_additional_df_columns(cls, original_df: pd.DataFrame, additional_cols_df: pd.DataFrame) -> pd.DataFrame:
        """ Adds the columns in `additional_cols_df` to `original_df`, horizontally concatenating them without considering either index.

        Usage:
            
            from neuropy.utils.indexing_helpers import PandasHelpers

            a_result.filter_epochs = PandasHelpers.adding_additional_df_columns(original_df=a_result.filter_epochs, additional_cols_df=_out_new_scores[a_name]) # update the filter_epochs with the new columns

        """     
        assert np.shape(additional_cols_df)[0] == np.shape(original_df)[0], f"np.shape(additional_cols_df)[0]: {np.shape(additional_cols_df)[0]} != np.shape(original_df)[0]: {np.shape(original_df)[0]}"
        # For each column in additional_cols_df, add it to original_df
        for column in additional_cols_df.columns:
            if not isinstance(original_df, pd.DataFrame):
                original_df._df[column] = additional_cols_df[column].values # Assume an Epoch, set the internal df
            else:
                # just set the column
                original_df[column] = additional_cols_df[column].values # TypeError: 'Epoch' object does not support item assignment
            
            
        return original_df


    
class ColumnTracker(ContextDecorator):
    """A context manager to track changes in the columns of DataFrames.

    The ColumnTracker can handle a single DataFrame, a list or tuple of DataFrames,
    or a dictionary with DataFrames as values. It prints the new columns added to the
    DataFrames during the block where the context manager is active.

    Attributes:
        result_cont: Container (single DataFrame, list/tuple of DataFrames, dict of DataFrames) being monitored.
        pre_cols: Set or structure containing sets of column names before the block execution.
        post_cols: Set or structure containing sets of column names after the block execution.
        added_cols: Set or structure containing sets of added column names after the block execution.
        all_added_columns: List of all unique added column names across all DataFrames after the block execution.


    Usage:

        from neuropy.utils.indexing_helpers import ColumnTracker

        dfs_dict = {a_name:a_result.filter_epochs for a_name, a_result in filtered_decoder_filter_epochs_decoder_result_dict.items()}
        with ColumnTracker(dfs_dict) as tracker:
            # Here you perform the operations that might modify the DataFrame columns
            filtered_decoder_filter_epochs_decoder_result_dict, _out_new_scores = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=filtered_decoder_filter_epochs_decoder_result_dict)

        # >>> ['longest_sequence_length', 'travel', 'sequential_correlation', 'monotonicity_score', 'jump', 'congruent_dir_bins_ratio', 'total_congruent_direction_change', 'direction_change_bin_ratio', 'longest_sequence_length_ratio', 'laplacian_smoothness', 'coverage']

    """

    def __init__(self, result_cont: Union[pd.DataFrame, List[pd.DataFrame], Dict[Any, pd.DataFrame]]):
        self.result_cont = result_cont
        self.pre_cols: Union[Set[str], List[Set[str]], Dict[Any, Set[str]]] = None
        self.post_cols: Union[Set[str], List[Set[str]], Dict[Any, Set[str]]] = None
        self.added_cols: Union[Set[str], List[Set[str]], Dict[Any, Set[str]]] = None
        self.all_added_columns: List[str] = None

    def _record_cols(self, data_structure: Union[pd.DataFrame, List[pd.DataFrame], Dict[Any, pd.DataFrame]]
                     ) -> Union[Set[str], List[Set[str]], Dict[Any, Set[str]]]:
        """Record the column names of the provided data structure."""
        if isinstance(data_structure, pd.DataFrame):
            return set(data_structure.columns)
        elif isinstance(data_structure, (list, tuple)):
            return [set(df.columns) for df in data_structure]
        elif isinstance(data_structure, dict):
            return {key: set(value.columns) for key, value in data_structure.items()}
        else:
            raise ValueError("Unsupported data structure type.")

    def __enter__(self) -> 'ColumnTracker':
        """Enter the runtime context related to this object.

        The with statement will bind this method's return value to the target(s)
        specified in the as clause of the statement, if any.
        """
        self.pre_cols = self._record_cols(self.result_cont)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the runtime context and perform any finalization actions."""
        self.post_cols = self._record_cols(self.result_cont)

        # Calculate added columns
        if isinstance(self.result_cont, pd.DataFrame):
            self.added_cols = self.post_cols - self.pre_cols
            self.all_added_columns = list(self.added_cols)
        elif isinstance(self.result_cont, (list, tuple)):
            self.added_cols = [post - pre for post, pre in zip(self.post_cols, self.pre_cols)]
            self.all_added_columns = np.unique(np.concatenate([list(cols) for cols in self.added_cols])).tolist()
        elif isinstance(self.result_cont, dict):
            self.added_cols = {key: self.post_cols[key] - self.pre_cols[key] for key in self.result_cont}
            all_cols = set()
            for added in self.added_cols.values():
                all_cols.update(added)
            self.all_added_columns = list(all_cols)

        # Print added columns
        print(f'Added columns:', end='\t')
        if len(self.all_added_columns) > 0:
            print(self.all_added_columns)
        else:
            print('[] (no columns added)')    


#endregion PandasDataFrameHelpers ______________________________________________________________________________________________________ #
