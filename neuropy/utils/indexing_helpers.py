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
    assert [len(neuron_ids) == len(sortable_values) for neuron_ids, sortable_values in zip(neuron_IDs_lists, sortable_values_lists)], f"all items must be the same length."
    
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

