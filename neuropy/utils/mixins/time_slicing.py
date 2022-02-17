import numpy as np
from numba import jit, njit, prange # numba acceleration


class StartStopTimesMixin:
    def safe_start_stop_times(self, t_start, t_stop):
        """ Returns t_start and t_stop while ensuring the values passed in aren't None.
        Usage:
             t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        """
        if t_start is None:
            t_start = self.t_start
        if t_stop is None:
            t_stop = self.t_stop
        return t_start, t_stop

class TimeSlicableIndiciesMixin(StartStopTimesMixin):
    def time_slice_indicies(self, t_start, t_stop):
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        return (self.time > t_start) & (self.time < t_stop)
    
class TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop):
        """ Implementors return a copy of themselves with each of their members sliced at the specified indicies """
        raise NotImplementedError




class TimeSlicedMixin:
    """ Used in Pho's more recent Pandas DataFrame-based core classes """
    # time_variable_name = 't_rel_seconds' # currently hardcoded
    
    @property
    def time_variable_name(self):
        raise NotImplementedError


    def time_sliced(self, t_start=None, t_stop=None):
        """ returns a copy of the spikes dataframe filtered such that only elements within the time ranges specified by t_start[i]:t_stop[i] (inclusive) are included. """
        # included_df = self._obj[((self._obj[SpikesAccessor.time_variable_name] >= t_start) & (self._obj[self.time_variable_name] <= t_stop))] # single time slice for sclar t_start and t_stop
        inclusion_mask = np.full_like(self._obj[self.time_variable_name], False, dtype=bool) # initialize entire inclusion_mask to False        
        # wrap the inputs in lists if they are scalars
        if np.isscalar(t_start):
            t_start = np.array([t_start])
        if np.isscalar(t_stop):
            t_stop = np.array([t_stop])
        
        starts = t_start
        stops = t_stop
        assert np.shape(starts) == np.shape(stops), f"starts and stops must be the same shape, but np.shape(starts): {np.shape(starts)} and np.shape(stops): {np.shape(stops)}"
        num_slices = len(starts)
        
        for i in np.arange(num_slices):
            curr_slice_t_start, curr_slice_t_stop = starts[i], stops[i]
            # TODO: BUG: I think we'd be double-counting here?
            curr_lap_position_df_is_included = self._obj[self.time_variable_name].between(curr_slice_t_start, curr_slice_t_stop, inclusive='both') # returns a boolean array indicating inclusion
            inclusion_mask[curr_lap_position_df_is_included] = True
            # position_df.loc[curr_lap_position_df_is_included, ['lap']] = curr_lap_id # set the 'lap' identifier on the object
            
        # once all slices have been computed and the inclusion_mask is complete, use it to mask the output dataframe
        return self._obj.loc[inclusion_mask, :].copy()
    

@jit(nopython=True, parallel = True)
def _compiled_verify_non_overlapping(start_stop_arr): # Function is compiled by numba and runs in machine code
    # coming in: spk_times_arr, pbe_start_stop_arr, pbe_identity_label
    assert (np.shape(start_stop_arr)[1] == 2), "pbe_start_stop_arr should have two columns: start, stop"
    num_elements = np.shape(start_stop_arr)[0]
    if (num_elements < 2):
        return np.array([True]) # Trivially True
    else: 
        start_t = start_stop_arr[1:,0] # get the start times, starting from the second element.
        stop_t = start_stop_arr[:(num_elements-1),1] # get the stop times, neglecting the last element
        return (start_t > stop_t) # check if the (i+1)th start_t is later than the (i)th stop_t


def verify_non_overlapping(start_stop_arr):
    """Returns True if no members of the start_stop_arr overlap each other.

    Args:
        start_stop_arr (_type_): An N x 2 numpy array of start, stop times

    Returns:
        bool: Returns true if all members are non-overlapping
        
    Example:
        are_all_non_overlapping = verify_non_overlapping(pbe_epoch_df[['start','stop']].to_numpy())
        are_all_non_overlapping

    """
    is_non_overlapping = _compiled_verify_non_overlapping(start_stop_arr)
    are_all_non_overlapping = np.alltrue(is_non_overlapping)
    return are_all_non_overlapping



@jit(nopython=True, parallel = True)
def _parallel_compiled_PBE_identity(times_arr, start_stop_times_arr, period_identity_labels): # Function is compiled by numba and runs in machine code
    """Works with non-sorted or overlapping start_stop intervals

    Args:
        times_arr (_type_): _description_
        start_stop_times_arr (_type_): _description_
        period_identity_labels (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert verify_non_overlapping(start_stop_arr=start_stop_times_arr), 'Intervals in start_stop_arr must be non-overlapping'
    
    spike_pbe_identity_arr = np.full((times_arr.shape[0],), np.nan) # fill with NaN for all entries initially
    for i in range(start_stop_times_arr.shape[0]):
        # find the spikes that fall in the current PBE (PBE[i])
        curr_PBE_identity = period_identity_labels[i]
        curr_bool_mask = np.logical_and((start_stop_times_arr[i,0] <= times_arr), (times_arr < start_stop_times_arr[i,1]))
        # spike_pbe_identity_arr[((pbe_start_stop_arr[i,0] <= spk_times_arr) & (spk_times_arr < pbe_start_stop_arr[i,1]))] = curr_PBE_identity
        spike_pbe_identity_arr[curr_bool_mask] = curr_PBE_identity
        # print(f'')
    # returns the array containing the PBE identity for each spike
    return spike_pbe_identity_arr




@jit(nopython=True, parallel = True)
def _compiled_searchsorted_PBE_identity(times_arr, start_stop_times_arr, period_identity_labels): # Function is compiled by numba and runs in machine code
    """Works only with sorted and non-overlapping start_stop_times_arr

    Args:
        times_arr (_type_): _description_
        start_stop_times_arr (_type_): _description_
        period_identity_labels (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert verify_non_overlapping(start_stop_arr=start_stop_times_arr), 'Intervals in start_stop_times_arr must be non-overlapping'
    assert np.shape(start_stop_times_arr)[0] == np.shape(period_identity_labels)[0], f'np.shape(period_identity_labels)[0] and np.shape(start_stop_times_arr)[0] must be the same, but np.shape(period_identity_labels)[0]: {np.shape(period_identity_labels)[0]} and np.shape(start_stop_times_arr)[0]: {np.shape(start_stop_times_arr)[0]}'
    spike_pbe_identity_arr = np.full((times_arr.shape[0],), np.nan) # fill with NaN for all entries initially
    
    # Vectorized np.searchsorted mode:
    found_start_indicies = np.searchsorted(times_arr, start_stop_times_arr[:,0], side='left')
    found_end_indicies = np.searchsorted(times_arr, start_stop_times_arr[:,1], side='right') # find the end of the range
    
    for i in range(start_stop_times_arr.shape[0]):
        # find the spikes that fall in the current PBE (PBE[i])
        curr_PBE_identity = period_identity_labels[i]        
        found_start_index = found_start_indicies[i]
        found_end_index = found_end_indicies[i] # find the end of the range
        spike_pbe_identity_arr[found_start_index:found_end_index] = curr_PBE_identity        
        
    # returns the array containing the PBE identity for each spike
    return spike_pbe_identity_arr

def _compute_spike_PBE_ids(spk_df, pbe_epoch_df):
    """ Computes the PBE identities for the spikes_df
    
    Example:
        # np.shape(spk_times_arr): (16318817,), p.shape(pbe_start_stop_arr): (10960, 2), p.shape(pbe_identity_label): (10960,)
        spike_pbe_identity_arr # Elapsed Time (seconds) = 90.92654037475586, 93.46184754371643, 90.16610431671143 
    """
    # coming in: spk_df, pbe_epoch_df
    spk_times_arr = spk_df.t_seconds.to_numpy()
    pbe_start_stop_arr = pbe_epoch_df[['start','stop']].to_numpy()
    # pbe_identity_label = pbe_epoch_df['label'].to_numpy()
    pbe_identity_label = pbe_epoch_df.index.to_numpy() # currently using the index instead of the label.
    # print(f'np.shape(spk_times_arr): {np.shape(spk_times_arr)}, p.shape(pbe_start_stop_arr): {np.shape(pbe_start_stop_arr)}, p.shape(pbe_identity_label): {np.shape(pbe_identity_label)}')
    # spike_pbe_identity_arr = _parallel_compiled_PBE_identity(spk_times_arr, pbe_start_stop_arr, pbe_identity_label)
    spike_pbe_identity_arr = _compiled_searchsorted_PBE_identity(spk_times_arr, pbe_start_stop_arr, pbe_identity_label)
    
    return spike_pbe_identity_arr

def add_PBE_identity(spk_df, pbe_epoch_df):
    """ Adds the PBE identity to the spikes_df
    
    Example:
        # np.shape(spk_times_arr): (16318817,), p.shape(pbe_start_stop_arr): (10960, 2), p.shape(pbe_identity_label): (10960,)
        spike_pbe_identity_arr # Elapsed Time (seconds) = 90.92654037475586, 93.46184754371643, 90.16610431671143 , 89.04321789741516
    """
    spike_pbe_identity_arr = _compute_spike_PBE_ids(spk_df, pbe_epoch_df)
    # np.shape(spike_pbe_identity_arr) # (16318817,)
    # np.where(np.logical_not(np.isnan(spike_pbe_identity_arr))) # (1, 2537652)

    spk_df['PBE_id'] = spike_pbe_identity_arr
    return spk_df
