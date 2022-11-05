import numpy as np
from numba import jit, njit, prange # numba acceleration


@jit(nopython=True, parallel = True) 
def _compiled_verify_non_overlapping(start_stop_times_arr): # Function is compiled by numba and runs in machine code
    # coming in: spk_times_arr, pbe_start_stop_arr, pbe_identity_label
    assert (np.shape(start_stop_times_arr)[1] == 2), "start_stop_times_arr should have two columns: start, stop"
    num_elements = np.shape(start_stop_times_arr)[0]
    if (num_elements < 2):
        return np.array([True]) # Trivially True
    else: 
        start_t = start_stop_times_arr[1:,0] # get the start times, starting from the second element.
        stop_t = start_stop_times_arr[:(num_elements-1),1] # get the stop times, neglecting the last element
        return (start_t > stop_t) # check if the (i+1)th start_t is later than the (i)th stop_t


def verify_non_overlapping(start_stop_times_arr):
    """Returns True if no members of the start_stop_times_arr overlap each other.

    Args:
        start_stop_times_arr (_type_): An N x 2 numpy array of start, stop times

    Returns:
        bool: Returns true if all members are non-overlapping
        
    Example:
        are_all_non_overlapping = verify_non_overlapping(pbe_epoch_df[['start','stop']].to_numpy())
        are_all_non_overlapping

    """
    is_non_overlapping = _compiled_verify_non_overlapping(start_stop_times_arr)
    are_all_non_overlapping = np.alltrue(is_non_overlapping)
    return are_all_non_overlapping

def get_non_overlapping_epochs(start_stop_times_arr):
    """Gets the indicies of any epochs that DON'T overlap one another.
    
    Args:
        start_stop_times_arr (_type_): An N x 2 numpy array of start, stop times
        
    Example:        
        from neuropy.utils.efficient_interval_search import drop_overlapping
        start_stop_times_arr = any_lap_specific_epochs.to_dataframe()[['start','stop']].to_numpy() # note that this returns one less than the number of epochs.
        non_overlapping_start_stop_times_arr = drop_overlapping(start_stop_times_arr)
        non_overlapping_start_stop_times_arr
    """
    # print(f'start_stop_times_arr: {start_stop_times_arr}')
    is_non_overlapping = _compiled_verify_non_overlapping(start_stop_times_arr)
    # print(f'is_non_overlapping: {is_non_overlapping}, np.shape(is_non_overlapping): {np.shape(is_non_overlapping)}')
    overlapping_lap_indicies = np.array(np.where(np.logical_not(is_non_overlapping))) # get the start indicies of all overlapping laps
    following_overlapping_lap = [i + 1 for i in overlapping_lap_indicies] # get the following index that it overlaps
    # Get the "good" (non-overlapping) laps only, dropping the rest:
    is_good_epoch = np.full((np.shape(start_stop_times_arr)[0],), True)
        
    if (len(overlapping_lap_indicies) == 0):
        # no epochs overlap, don't need to drop any
        return is_good_epoch
    else:
        # print(f'overlapping_lap_indicies: {overlapping_lap_indicies}, following_overlapping_lap: {following_overlapping_lap}, is_non_overlapping: {is_non_overlapping}')
        is_good_epoch[overlapping_lap_indicies] = False
        is_good_epoch[following_overlapping_lap] = False
        # print(f'is_good_lap: {is_good_epoch}, np.shape(is_good_lap): {np.shape(is_good_epoch)}')
        # return only the non-overlapping periods
        return is_good_epoch
    
    

def drop_overlapping(start_stop_times_arr):
    """Drops the overlapping epochs

    Args:
        start_stop_times_arr (_type_): An N x 2 numpy array of start, stop times
        
    Example:        
        from neuropy.utils.efficient_interval_search import drop_overlapping
        start_stop_times_arr = any_lap_specific_epochs.to_dataframe()[['start','stop']].to_numpy() # note that this returns one less than the number of epochs.
        non_overlapping_start_stop_times_arr = drop_overlapping(start_stop_times_arr)
        non_overlapping_start_stop_times_arr
    """
    is_good_epoch = get_non_overlapping_epochs(start_stop_times_arr)
    return start_stop_times_arr[is_good_epoch, :].copy()



@jit(nopython=True, parallel = True)
def _compiled_unsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=np.nan): # Function is compiled by numba and runs in machine code
    """MUCH slower than _compiled_searchsorted_event_interval_identity(...), but it works with non-sorted or overlapping start_stop intervals

    Args:
        times_arr (np.ndarray): An array of times of shape (N, ) in the same units as the start_stop_times_arr
        start_stop_times_arr (np.ndarray): An array of start and stop intervals of shape (L, 2), with start_stop_times_arr[:, 0] representing the start times and start_stop_times_arr[:, 1] representing the stop times.
        period_identity_labels (np.ndarray): An array of shape (L, ) specifying the appropriate id/identity of the interval in the corresponding row of start_stop_times_arr
        no_interval_fill_value: The value to be used when the event doesn't belong to any of the provided intervals. Defaults to np.nan
        
    Returns:
        np.ndarray: an array of length N that specifies the interval identity each event in times_arr belongs to, or np.nan if it occurs outside all specified intervals.
        
    Performance:
        # For: np.shape(spk_times_arr): (16318817,), p.shape(pbe_start_stop_arr): (10960, 2), p.shape(pbe_identity_label): (10960,)
            # Elapsed Time = 90.92654037475586, 93.46184754371643, 90.16610431671143, 89.04321789741516

    """ 
    event_interval_identity_arr = np.full((times_arr.shape[0],), no_interval_fill_value) # fill with NaN for all entries initially
    for i in range(start_stop_times_arr.shape[0]):
        # find the spikes that fall in the current PBE (PBE[i])
        curr_PBE_identity = period_identity_labels[i]
        curr_bool_mask = np.logical_and((start_stop_times_arr[i,0] <= times_arr), (times_arr < start_stop_times_arr[i,1]))
        # spike_pbe_identity_arr[((pbe_start_stop_arr[i,0] <= spk_times_arr) & (spk_times_arr < pbe_start_stop_arr[i,1]))] = curr_PBE_identity
        event_interval_identity_arr[curr_bool_mask] = curr_PBE_identity
        # print(f'')
    # returns the array containing the PBE identity for each spike
    return event_interval_identity_arr



def _searchsorted_find_event_interval_indicies(times_arr, start_stop_times_arr): # Function is compiled by numba and runs in machine code
    """Converts the L x 2 array of start and stop times (start_stop_times_arr) representing intervals in time to an array of indicies into times_arr of the same size

    Args:
        times_arr (np.ndarray): An array of times of shape (N, ) in the same units as the start_stop_times_arr
        start_stop_times_arr (np.ndarray): An array of start and stop intervals of shape (L, 2), with start_stop_times_arr[:, 0] representing the start times and start_stop_times_arr[:, 1] representing the stop times.

    Returns:
        np.ndarray: An array of start and stop indicies into times_arr of shape (L, 2)
        
    Example:
        # found_start_end_indicies = _searchsorted_find_event_interval_indicies(times_arr, start_stop_times_arr)
        # found_start_indicies = found_start_end_indicies[:,0]
        # found_end_indicies = found_start_end_indicies[:,1]
        
    """
    assert np.shape(start_stop_times_arr)[1] == 2
    # Vectorized np.searchsorted mode:
    found_start_indicies = np.searchsorted(times_arr, start_stop_times_arr[:,0], side='left')
    found_end_indicies = np.searchsorted(times_arr, start_stop_times_arr[:,1], side='right') # find the end of the range
    found_start_end_indicies = np.vstack((found_start_indicies, found_end_indicies)).T 
    assert np.shape(found_start_end_indicies)[1] == 2
    return found_start_end_indicies
    


@jit(nopython=True, parallel = True)
def _compiled_searchsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=np.nan): # Function is compiled by numba and runs in machine code
    """ Consider an L x 2 array of start and stop times (start_stop_times_arr) representing intervals in time with corresponding identities provided by the (L, ) array of period_identity_labels.
    The goal of this function is to efficienctly determine which of the intervals, if any, each event occuring at a time specified by times_arr occurs during.
    
    The output result will be an array of length N that specifies the interval identity each event in times_arr belongs to, or np.nan if none.
    
    Limitations:
        !! Works only with sorted and non-overlapping start_stop_times_arr !!

    Args:
        times_arr (np.ndarray): An array of times of shape (N, ) in the same units as the start_stop_times_arr
        start_stop_times_arr (np.ndarray): An array of start and stop intervals of shape (L, 2), with start_stop_times_arr[:, 0] representing the start times and start_stop_times_arr[:, 1] representing the stop times.
        period_identity_labels (np.ndarray): An array of shape (L, ) specifying the appropriate id/identity of the interval in the corresponding row of start_stop_times_arr
        no_interval_fill_value: The value to be used when the event doesn't belong to any of the provided intervals. Defaults to np.nan
        
    Returns:
        np.ndarray: an array of length N that specifies the interval identity each event in times_arr belongs to, or np.nan if it occurs outside all specified intervals.
        
    Performance:
        # For: np.shape(spk_times_arr): (16318817,), p.shape(pbe_start_stop_arr): (10960, 2), p.shape(pbe_identity_label): (10960,)
        # Elapsed Time = 1.1290626525878906 seconds
    """
    event_interval_identity_arr = np.full((times_arr.shape[0],), no_interval_fill_value) # fill with NaN for all entries initially
    # Vectorized np.searchsorted mode:
    found_start_indicies = np.searchsorted(times_arr, start_stop_times_arr[:,0], side='left')
    found_end_indicies = np.searchsorted(times_arr, start_stop_times_arr[:,1], side='right') # find the end of the range

    for i in range(start_stop_times_arr.shape[0]):
        # find the spikes that fall in the current PBE (PBE[i])
        curr_PBE_identity = period_identity_labels[i]        
        found_start_index = found_start_indicies[i]
        found_end_index = found_end_indicies[i] # find the end of the range
        event_interval_identity_arr[found_start_index:found_end_index] = curr_PBE_identity        
        
    # returns the array containing the PBE identity for each spike
    return event_interval_identity_arr


def determine_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels=None, no_interval_fill_value=np.nan):
    if period_identity_labels is None:
        period_identity_labels = np.arange(np.shape(start_stop_times_arr)[0]) # just label them ascending if they don't have labels
    assert verify_non_overlapping(start_stop_times_arr=start_stop_times_arr), 'Intervals in start_stop_times_arr must be non-overlapping'
    assert np.shape(start_stop_times_arr)[0] == np.shape(period_identity_labels)[0], f'np.shape(period_identity_labels)[0] and np.shape(start_stop_times_arr)[0] must be the same, but np.shape(period_identity_labels)[0]: {np.shape(period_identity_labels)[0]} and np.shape(start_stop_times_arr)[0]: {np.shape(start_stop_times_arr)[0]}'
    return _compiled_searchsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=no_interval_fill_value)


def determine_unsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=np.nan):
    assert verify_non_overlapping(start_stop_times_arr=start_stop_times_arr), 'Intervals in start_stop_times_arr must be non-overlapping'
    assert np.shape(start_stop_times_arr)[0] == np.shape(period_identity_labels)[0], f'np.shape(period_identity_labels)[0] and np.shape(start_stop_times_arr)[0] must be the same, but np.shape(period_identity_labels)[0]: {np.shape(period_identity_labels)[0]} and np.shape(start_stop_times_arr)[0]: {np.shape(start_stop_times_arr)[0]}'
    return _compiled_unsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=no_interval_fill_value)



@jit(nopython=True, parallel=True)
def _compiled_searchsorted_event_interval_is_included(times_arr, start_stop_times_arr): # Function is compiled by numba and runs in machine code
    """ Consider an L x 2 array of start and stop times (start_stop_times_arr) representing intervals in time.
    The goal of this function is to efficienctly determine whether each event occuring at a time specified by times_arr occurs any of the intervals.
    
    Note:
        If information about the identity of the interval the event belongs to is desired, look at the other functions (determine_event_interval_identity)
    
    The output result will be Boolean array of length N that specifies whether each event in times_arr is included in any interval.
    
    Limitations:
        !! Works only with sorted and non-overlapping start_stop_times_arr !!

    Args:
        times_arr (np.ndarray): An array of times of shape (N, ) in the same units as the start_stop_times_arr
        start_stop_times_arr (np.ndarray): An array of start and stop intervals of shape (L, 2), with start_stop_times_arr[:, 0] representing the start times and start_stop_times_arr[:, 1] representing the stop times.
        
    Returns:
        np.ndarray: Boolean array of length N that specifies whether each event in times_arr is included in any interval.
        
    Performance:
        # For: np.shape(spk_times_arr): (16318817,), p.shape(pbe_start_stop_arr): (10960, 2), p.shape(pbe_identity_label): (10960,)
        # Elapsed Time = 1.1290626525878906 seconds
    """
    event_interval_is_included_arr = np.full((times_arr.shape[0],), False) # fill with False for all entries initially
    # Vectorized np.searchsorted mode:
    found_start_indicies = np.searchsorted(times_arr, start_stop_times_arr[:,0], side='left')
    found_end_indicies = np.searchsorted(times_arr, start_stop_times_arr[:,1], side='right') # find the end of the range

    for i in range(start_stop_times_arr.shape[0]):
        # find the events that fall in the current range 
        found_start_index = found_start_indicies[i]
        found_end_index = found_end_indicies[i] # find the end of the range
        event_interval_is_included_arr[found_start_index:found_end_index] = True # set True for included range 
        
    # returns the array containing the whether each event is included in any interval
    return event_interval_is_included_arr


def determine_event_interval_is_included(times_arr, start_stop_times_arr):
    assert verify_non_overlapping(start_stop_times_arr=start_stop_times_arr), 'Intervals in start_stop_times_arr must be non-overlapping'
    return _compiled_searchsorted_event_interval_is_included(times_arr, start_stop_times_arr)
