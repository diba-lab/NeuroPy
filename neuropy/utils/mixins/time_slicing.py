import numpy as np

from neuropy.utils.efficient_interval_search import determine_event_interval_identity, determine_event_interval_is_included # numba acceleration


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
        
        
        inclusion_mask = determine_event_interval_is_included(self._obj[self.time_variable_name].to_numpy(), np.hstack((starts, stops))
        
        # slow method:
        
        num_slices = len(starts)
        
        for i in np.arange(num_slices):
            curr_slice_t_start, curr_slice_t_stop = starts[i], stops[i]
            # TODO: BUG: I think we'd be double-counting here?
            curr_lap_position_df_is_included = self._obj[self.time_variable_name].between(curr_slice_t_start, curr_slice_t_stop, inclusive='both') # returns a boolean array indicating inclusion
            inclusion_mask[curr_lap_position_df_is_included] = True
            # position_df.loc[curr_lap_position_df_is_included, ['lap']] = curr_lap_id # set the 'lap' identifier on the object
            
        # once all slices have been computed and the inclusion_mask is complete, use it to mask the output dataframe
        return self._obj.loc[inclusion_mask, :].copy()
    



def _compute_spike_PBE_ids(spk_df, pbe_epoch_df, no_interval_fill_value=np.nan):
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
    spike_pbe_identity_arr = determine_event_interval_identity(spk_times_arr, pbe_start_stop_arr, pbe_identity_label, no_interval_fill_value=no_interval_fill_value)
    
    return spike_pbe_identity_arr

def add_PBE_identity(spk_df, pbe_epoch_df, no_interval_fill_value=np.nan):
    """ Adds the PBE identity to the spikes_df
    
    Example:
        # np.shape(spk_times_arr): (16318817,), p.shape(pbe_start_stop_arr): (10960, 2), p.shape(pbe_identity_label): (10960,)
        spike_pbe_identity_arr # Elapsed Time (seconds) = 90.92654037475586, 93.46184754371643, 90.16610431671143 , 89.04321789741516
    """
    spike_pbe_identity_arr = _compute_spike_PBE_ids(spk_df, pbe_epoch_df, no_interval_fill_value=no_interval_fill_value)
    # np.shape(spike_pbe_identity_arr) # (16318817,)
    # np.where(np.logical_not(np.isnan(spike_pbe_identity_arr))) # (1, 2537652)
    spk_df['PBE_id'] = spike_pbe_identity_arr
    return spk_df
