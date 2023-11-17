import numpy as np


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
        new_all_aclus_sort_indicies = find_desired_sort_indicies(active_2d_plot.neuron_ids, all_sorted_aclus)
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


def union_of_arrays(*arrays) -> np.array:
    """ 
    from neuropy.utils.indexing_helpers import union_of_arrays
    
    """
    return np.unique(np.concatenate(arrays))