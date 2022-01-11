import numpy as np

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
    
    