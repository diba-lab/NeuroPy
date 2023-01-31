from copy import deepcopy
from enum import Enum
import numpy as np
import pandas as pd
from numba import jit, njit, prange # numba acceleration

import portion as P # Required for interval search: portion~=2.3.0



class OverlappingIntervalsFallbackBehavior(Enum):
    """Describes the behavior of the search when the provided epochs overlap each other.
        overlap_behavior: OverlappingIntervalsFallbackBehavior - If ASSERT_FAIL, an AssertionError will be thrown in the case that any of the intervals in provided_epochs_df overlap each other. Otherwise, if FALLBACK_TO_SLOW_SEARCH, a much slower search will be performed that will still work.
    """
    ASSERT_FAIL = "ASSERT_FAIL"
    FALLBACK_TO_SLOW_SEARCH = "FALLBACK_TO_SLOW_SEARCH"
    


# ==================================================================================================================== #
# Overlap Detection                                                                                                    #
# ==================================================================================================================== #
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
        non_overlapping_start_stop_times_arr = get_non_overlapping_epochs(start_stop_times_arr)
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



def get_overlapping_indicies(start_stop_times_arr):
    """Gets the indicies of any epochs that DO overlap one another.
    
    Args:
        start_stop_times_arr (_type_): An N x 2 numpy array of start, stop times
        
    Example:        
        from neuropy.utils.efficient_interval_search import get_overlapping_indicies
        curr_laps_obj = deepcopy(sess.laps)
        start_stop_times_arr = np.vstack([curr_laps_obj.starts, curr_laps_obj.stops]).T # (80, 2)
        all_overlapping_lap_indicies = get_overlapping_indicies(start_stop_times_arr)
        all_overlapping_lap_indicies
    """
    is_non_overlapping = _compiled_verify_non_overlapping(start_stop_times_arr)
    overlapping_lap_indicies = np.array(np.where(np.logical_not(is_non_overlapping))) # get the start indicies of all overlapping laps
    # print(f'overlapping_lap_indicies: {overlapping_lap_indicies}')
    following_overlapping_lap = [i + 1 for i in overlapping_lap_indicies] # get the following index that it overlaps
    # print(f'following_overlapping_lap: {following_overlapping_lap}')
    all_overlapping_lap_indicies = np.union1d(ar1=overlapping_lap_indicies, ar2=following_overlapping_lap)
    # print(f'all_overlapping_lap_indicies: {all_overlapping_lap_indicies}')
    return all_overlapping_lap_indicies


# ==================================================================================================================== #
# Event Interval Identity                                                                                              #
# ==================================================================================================================== #
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
    # interval_timestamp_start_stop_indicies_arr = np.full((start_stop_times_arr.shape[0], 2), no_interval_fill_value) # each period_identity_labels corresponds to a single two-element row, containing the start/stop spike index for that interval. fill with NaN for all entries initially. If an interval has no spikes, the indicies wil be left NaN
    # interval_timestamp_start_stop_indicies_list = []  
    interval_timestamp_indicies_lists = []  

    for i in range(start_stop_times_arr.shape[0]):
        # find the spikes that fall in the current PBE (PBE[i])
        curr_PBE_identity = period_identity_labels[i]
        curr_bool_mask = np.logical_and((start_stop_times_arr[i,0] <= times_arr), (times_arr < start_stop_times_arr[i,1]))
        # spike_pbe_identity_arr[((pbe_start_stop_arr[i,0] <= spk_times_arr) & (spk_times_arr < pbe_start_stop_arr[i,1]))] = curr_PBE_identity
        event_interval_identity_arr[curr_bool_mask] = curr_PBE_identity

        # np.transpose(np.nonzero(curr_bool_mask))
        curr_timestamp_indicies = np.flatnonzero(curr_bool_mask)
        interval_timestamp_indicies_lists.append(curr_timestamp_indicies)
        # curr_timestamp_start_stop_indicies = (curr_timestamp_indicies[0], curr_timestamp_indicies[-1])
        # interval_timestamp_start_stop_indicies_list.append(curr_timestamp_start_stop_indicies)

        # print(f'')
    # returns the array containing the PBE identity for each spike
    return event_interval_identity_arr, interval_timestamp_indicies_lists

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

def determine_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels=None, no_interval_fill_value=np.nan, overlap_behavior=OverlappingIntervalsFallbackBehavior.ASSERT_FAIL):
    if period_identity_labels is None:
        period_identity_labels = np.arange(np.shape(start_stop_times_arr)[0]) # just label them ascending if they don't have labels
    assert np.shape(start_stop_times_arr)[0] == np.shape(period_identity_labels)[0], f'np.shape(period_identity_labels)[0] and np.shape(start_stop_times_arr)[0] must be the same, but np.shape(period_identity_labels)[0]: {np.shape(period_identity_labels)[0]} and np.shape(start_stop_times_arr)[0]: {np.shape(start_stop_times_arr)[0]}'

    if overlap_behavior.name == OverlappingIntervalsFallbackBehavior.ASSERT_FAIL.name:
        assert verify_non_overlapping(start_stop_times_arr=start_stop_times_arr), 'Intervals in start_stop_times_arr must be non-overlapping'
        return _compiled_searchsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=no_interval_fill_value)
    elif overlap_behavior.name == OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH.name:
        are_intervals_overlapping = not verify_non_overlapping(start_stop_times_arr=start_stop_times_arr)
        if are_intervals_overlapping:
            print('WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.')
        return _compiled_unsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=no_interval_fill_value)
    else:
        raise NotImplementedError

def determine_unsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=np.nan, overlap_behavior=OverlappingIntervalsFallbackBehavior.ASSERT_FAIL):
    assert np.shape(start_stop_times_arr)[0] == np.shape(period_identity_labels)[0], f'np.shape(period_identity_labels)[0] and np.shape(start_stop_times_arr)[0] must be the same, but np.shape(period_identity_labels)[0]: {np.shape(period_identity_labels)[0]} and np.shape(start_stop_times_arr)[0]: {np.shape(start_stop_times_arr)[0]}'
    if overlap_behavior.name == OverlappingIntervalsFallbackBehavior.ASSERT_FAIL.name:
        assert verify_non_overlapping(start_stop_times_arr=start_stop_times_arr), 'Intervals in start_stop_times_arr must be non-overlapping'
        # return _compiled_unsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=no_interval_fill_value)
    elif overlap_behavior.name == OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH.name:
        are_intervals_overlapping = not verify_non_overlapping(start_stop_times_arr=start_stop_times_arr)
        if are_intervals_overlapping:
            print('WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.')
        # return _compiled_unsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=no_interval_fill_value)
    else:
        raise NotImplementedError
    # If we've made it this far, perform the unsorted version either way
    return _compiled_unsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=no_interval_fill_value)


# ==================================================================================================================== #
# Event Interval is_included                                                                                           #
# ==================================================================================================================== #
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



def debug_overlapping_epochs(epochs_df):
    """
    from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs, drop_overlapping, get_overlapping_indicies, OverlappingIntervalsFallbackBehavior
    curr_epochs_obj = deepcopy(sess.ripple)
    debug_overlapping_epochs(curr_epochs_obj.to_dataframe())
    
    """
    start_stop_times_arr = np.vstack([epochs_df.epochs.starts, epochs_df.epochs.stops]).T # (80, 2)
    # start_stop_times_arr.shape
    all_overlapping_lap_indicies = get_overlapping_indicies(start_stop_times_arr)
    n_total_epochs = start_stop_times_arr.shape[0]
    n_overlapping = len(all_overlapping_lap_indicies)
    print(f'n_overlapping: {n_overlapping} of n_total_epochs: {n_total_epochs}')
    return all_overlapping_lap_indicies


# ==================================================================================================================== #
# De-Duplication                                                                                                       #
# ==================================================================================================================== #
def deduplicate_epochs(epochs_df, agressive_deduplicate:bool=True):
    """ Attempts to remove literal duplicate ('start', 'stop') entries in the epochs_df. Does not do anything about overlap if the epochs don't match
    returns the non-duplicated epochs in epochs_df.

    Usage:
        from neuropy.utils.efficient_interval_search import deduplicate_epochs
        curr_epochs_df = deduplicate_epochs(epochs_df, agressive_deduplicate=True)

    """
    df = deepcopy(epochs_df)

    if agressive_deduplicate:
        ## A more aggressive method that re-sorts first and eliminates more duplicates (meaning the version above leaves in some duplicates):
        # original epochs_df: np.shape(curr_epochs_df) = (770, 9)
        # above version: np.shape(non_duplicated_epochs_df) = (412, 9)
        # this agressive version: np.shape(non_duplicated_epochs_df) = (358, 11)
        df['index_original'] = df.groupby(['start','stop']).start.transform('idxmin')
        df['integer_label'] = df['label'].astype('int')
        df.sort_values(by='integer_label')
        return df[df.duplicated(subset=['start','stop'], keep='first')]
    else:
        # less agressive (more conservative) de-duplication mode:
        is_duplicated_epoch = df.duplicated(subset=['start','stop'], keep='first')
        return df[np.logical_not(is_duplicated_epoch)]


# ==================================================================================================================== #
# `portion`-based interval search and operations                                                                       #
# ==================================================================================================================== #

def _convert_start_end_tuples_list_to_Intervals(start_end_tuples_list):
    return P.from_data([(P.CLOSED, start, stop, P.CLOSED) for start, stop in start_end_tuples_list])

def _convert_Intervals_to_4uples_list(intervals: P.Interval):
    return P.to_data(intervals)

def _convert_4uples_list_to_epochs_df(tuples_list):
    """ Convert tuples list to epochs dataframe """
    if len(tuples_list) > 0:
        assert len(tuples_list[0]) == 4
    combined_df = pd.DataFrame.from_records(tuples_list, columns=['is_interval_start_closed','start','stop','is_interval_end_closed'], exclude=['is_interval_start_closed','is_interval_end_closed'], coerce_float=True)
    combined_df['label'] = combined_df.index.astype("str") # add the required 'label' column so it can be convereted into an Epoch object
    combined_df[['start','stop']] = combined_df[['start','stop']].astype('float')
    combined_df = combined_df.epochs.get_valid_df() # calling the .epochs.get_valid_df() method ensures all the appropriate columns are added (such as 'duration') to be used as an epochs_df
    return combined_df

def convert_Intervals_to_epochs_df(intervals: P.Interval) -> pd.DataFrame:
    """ 
    Usage:
        epochs_df = convert_Intervals_to_epochs_df(long_replays_intervals)
    """
    return _convert_4uples_list_to_epochs_df(_convert_Intervals_to_4uples_list(intervals))

def convert_Intervals_to_Epoch_obj(intervals: P.Interval):
    """ build an Epoch object version
    Usage:
        combined_epoch_obj = convert_Intervals_to_Epoch_obj(long_replays_intervals)
    """
    from neuropy.core.epoch import Epoch # import here to avoid circular import
    return Epoch(epochs=convert_Intervals_to_epochs_df(intervals))

# # to_string/from_string:
# P.to_string(long_replays_intervals)
# P.from_string(P.to_string(long_replays_intervals), conv=float)

### Get global intervals above/below a given speed threshold:
def _find_intervals_above_speed(df: pd.DataFrame, speed_thresh: float, is_interpolated: bool) -> list:
    """ written by ChatGPT 2023-01-26
    Used by `_filter_epochs_by_speed`
    """
    # speed_threshold_comparison_operator_fn: defines the function to compare a given speed with the threshold (e.g. > or <)
    speed_threshold_comparison_operator_fn = lambda a_speed, a_speed_thresh: (a_speed > a_speed_thresh) # greater than
    # speed_threshold_comparison_operator_fn = lambda a_speed, a_speed_thresh: (a_speed < a_speed_thresh) # less than
    speed_threshold_condition_fn = lambda start_speed, end_speed, speed_thresh: speed_threshold_comparison_operator_fn(start_speed, speed_thresh) and speed_threshold_comparison_operator_fn(end_speed, speed_thresh)
    # curr_df_record_fn = lambda i, col_name='speed': df.loc[i, col_name]
    curr_df_record_fn = lambda i, col_name='speed': df.loc[df.index[i], col_name]
    df = df.copy()
    df.index = df.index.astype(int) #use astype to convert to int
    # df = df.reset_index(drop=True) # reset and drop the index so the `df.loc[i, *]` and  `df.loc[df.index[i], *]` align    
    intervals = []
    start_time = None
    for i in range(len(df)):
        # curr_t = curr_df_record_fn(i, col_name='t')
        curr_speed = curr_df_record_fn(i, col_name='speed')
        if speed_threshold_comparison_operator_fn(curr_speed, speed_thresh):
            if start_time is None:
                try:
                    start_time = df.loc[i, 't']
                except KeyError as e:
                    # start_time = df.loc[df.index[i], 't']
                    # start_time = curr_t
                    start_time = df.t.to_numpy()[i]
                except Exception as e:
                    raise e                
        else:
            if start_time is not None:
                try:
                    end_time = df.loc[i, 't']
                except KeyError as e:
                    # end_time = df.loc[df.index[i], 't']
                    # end_time = curr_t
                    end_time = df.t.to_numpy()[i]
                except Exception as e:
                    raise e

                if is_interpolated:
                    start_speed = np.interp(start_time, df.t, df.speed)
                    end_speed = np.interp(end_time, df.t, df.speed)
                    if speed_threshold_condition_fn(start_speed, end_speed, speed_thresh):
                        intervals.append((start_time, end_time))
                else:
                    intervals.append((start_time, end_time))
                start_time = None
                
    # Last (unclosed) interval:
    if start_time is not None:
        try:
            end_time = df.loc[len(df)-1, 't']
        except KeyError as e:
            # end_time = df.loc[df.index[len(df)-1], 't']
            # end_time = curr_df_record_fn((len(df)-1), col_name='t')
            end_time = df.t.to_numpy()[(len(df)-1)]
        except Exception as e:
            raise e

        if is_interpolated:
            start_speed = np.interp(start_time, df.t, df.speed)
            end_speed = np.interp(end_time, df.t, df.speed)
            if speed_threshold_condition_fn(start_speed, end_speed, speed_thresh):
                intervals.append((start_time, end_time))
        else:
            intervals.append((start_time, end_time))
    return intervals



def filter_epochs_by_speed(speed_df, *epoch_args, speed_thresh=2.0, debug_print=False):
    """ Filter *_replays_Interval by requiring them to be below the speed 
    *epoch_args = long_replays, short_replays, global_replays

    Usage:
        from neuropy.utils.efficient_interval_search import filter_epochs_by_speed
        speed_thresh = 2.0
        speed_df = global_session.position.to_dataframe()
        long_replays, short_replays, global_replays, above_speed_threshold_intervals, below_speed_threshold_intervals = filter_epochs_by_speed(speed_df, long_replays, short_replays, global_replays, speed_thresh=speed_thresh, debug_print=True)
    """
    start_end_tuples_interval_list = _find_intervals_above_speed(speed_df, speed_thresh, is_interpolated=True)
    above_speed_threshold_intervals = _convert_start_end_tuples_list_to_Intervals(start_end_tuples_interval_list)
    if debug_print:
        print(f'len(above_speed_threshold_intervals): {len(above_speed_threshold_intervals)}')
    # find the intervals below the threshold speed by taking the complement:
    below_speed_threshold_intervals = above_speed_threshold_intervals.complement()
    if debug_print:
        print(f'len(below_speed_threshold_intervals): {len(below_speed_threshold_intervals)}')
        print(f'Pre-speed-filtering: ' + ', '.join([f"n_epochs: {an_interval.n_epochs}" for an_interval in epoch_args]))
    
    # Only this part depends on *epoch_args:
    if debug_print:
        print(f'len(epoch_args): {len(epoch_args)}')
    epoch_args_Interval = [_convert_start_end_tuples_list_to_Intervals(zip(a_replays_epoch_obj.starts, a_replays_epoch_obj.stops)) for a_replays_epoch_obj in epoch_args] # returns P.Interval objects
    ## Filter *_replays_Interval by requiring them to be below the speed:
    epoch_args_Interval = [below_speed_threshold_intervals.intersection(a_replays_Interval_obj) for a_replays_Interval_obj in epoch_args_Interval] # returns P.Interval objects
    ## Convert back to Epoch objects:
    epoch_args = [convert_Intervals_to_Epoch_obj(a_replays_Interval_obj) for a_replays_Interval_obj in epoch_args_Interval] # returns P.Interval objects
    if debug_print:
        print(f'Post-speed-filtering: ' + ', '.join([f"n_epochs: {an_interval.n_epochs}" for an_interval in epoch_args]))
    return *epoch_args, above_speed_threshold_intervals, below_speed_threshold_intervals

