import numpy as np

def interleave_elements(start_points, end_points, debug_print:bool=False):
    """ Given two equal sized arrays, produces an output array of double that size that contains elements of start_points interleaved with elements of end_points
    Example:
        from neuropy.utils.mixins.indexing_helpers import interleave_elements

        a_starts = ['A','B','C','D']
        a_ends = ['a','b','c','d']
        a_interleaved = interleave_elements(a_starts, a_ends)
        >> a_interleaved: ['A','a','B','b','C','c','D','d']
    """
    if not isinstance(start_points, np.ndarray):
        start_points = np.array(start_points)
    if not isinstance(end_points, np.ndarray):
        end_points = np.array(end_points)
    assert np.shape(start_points) == np.shape(end_points), f"start_points and end_points must be the same shape. np.shape(start_points): {np.shape(start_points)}, np.shape(end_points): {np.shape(end_points)}"
    # Capture initial shapes to determine if np.atleast_2d changed the shapes
    start_points_initial_shape = np.shape(start_points)
    end_points_initial_shape = np.shape(end_points)
    
    # Capture initial datatypes for building the appropriate empty np.ndarray later:
    start_points_dtype = start_points.dtype # e.g. 'str32'
    end_points_dtype = end_points.dtype # e.g. 'str32'
    assert start_points_dtype == end_points_dtype, f"start_points and end_points must be the same datatype. start_points.dtype: {start_points.dtype.name}, end_points.dtype: {end_points.dtype.name}"
    start_points = np.atleast_2d(start_points)
    end_points = np.atleast_2d(end_points)
    if debug_print:
        print(f'start_points: {start_points}\nend_points: {end_points}')
        print(f'np.shape(start_points): {np.shape(start_points)}\tnp.shape(end_points): {np.shape(end_points)}') # np.shape(start_points): (1, 4)	np.shape(end_points): (1, 4)
        print(f'start_points_dtype.name: {start_points_dtype.name}\tend_points_dtype.name: {end_points_dtype.name}')
      
    if (np.shape(start_points) != start_points_initial_shape) and (np.shape(start_points)[0] == 1):
        # Shape changed after np.atleast_2d(...) which erroniously adds the newaxis to the 0th dimension. Fix by transposing:
        start_points = start_points.T
    if (np.shape(end_points) != end_points_initial_shape) and (np.shape(end_points)[0] == 1):
        # Shape changed after np.atleast_2d(...) which erroniously adds the newaxis to the 0th dimension. Fix by transposing:
        end_points = end_points.T
    if debug_print:
        print(f'POST-TRANSFORM: np.shape(start_points): {np.shape(start_points)}\tnp.shape(end_points): {np.shape(end_points)}') # POST-TRANSFORM: np.shape(start_points): (4, 1)	np.shape(end_points): (4, 1)
    all_points_shape = (np.shape(start_points)[0] * 2, np.shape(start_points)[1]) # it's double the length of the start_points
    if debug_print:
        print(f'all_points_shape: {all_points_shape}') # all_points_shape: (2, 4)
    # all_points = np.zeros(all_points_shape)
    all_points = np.empty(all_points_shape, dtype=start_points_dtype) # Create an empty array with the appropriate dtype to hold the objects
    all_points[np.arange(0, all_points_shape[0], 2), :] = start_points # fill the even elements
    all_points[np.arange(1, all_points_shape[0], 2), :] = end_points # fill the odd elements
    assert np.shape(all_points)[0] == (np.shape(start_points)[0] * 2), f"newly created all_points is not of corrrect size! np.shape(all_points): {np.shape(all_points)}"
    return np.squeeze(all_points)



def get_dict_subset(a_dict: dict, subset_includelist=None, subset_excludelist=None) -> dict:
    """
    Returns a subset of the input dictionary based on the specified inclusion or exclusion lists.

    Inputs:
        a_dict: dict - The dictionary to subset.
        subset_includelist: list, optional - A list of keys to include in the subset. If None, all keys are included.
        subset_excludelist: list, optional - A list of keys to exclude from the subset. If None, no keys are excluded.

    Returns:
        dict: The subset of the input dictionary.
        
    Usage:
        from neuropy.utils.mixins.indexing_helpers import get_dict_subset
        
    """
    if subset_excludelist is not None:
        assert subset_includelist is None, "subset_includelist must be None when a subset_excludelist is provided!"
        subset_includelist = [key for key in a_dict.keys() if key not in subset_excludelist]

    if subset_includelist is None:
        return dict(a_dict)
    else:
        # benedict version:
        # return benedict(a_dict).subset(subset_includelist)
        # no benedict required version:
        subset_dict = {}
        for key in subset_includelist:
            if key in a_dict:
                subset_dict[key] = a_dict[key]

        return subset_dict


