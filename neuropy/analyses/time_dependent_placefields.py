from copy import deepcopy
from typing import OrderedDict
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, interpolation
from neuropy.analyses.placefields import PfND, PlacefieldComputationParameters
from neuropy.core.epoch import Epoch
from neuropy.core.position import Position
from neuropy.core.ratemap import Ratemap
from neuropy.analyses.placefields import _normalized_occupancy
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.misc import safe_pandas_get_group, copy_if_not_none
## Need to apply position binning to the spikes_df's position columns to find the bins they fall in:
from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns # for perform_time_range_computation only

class PfND_TimeDependent(PfND):
    """ Time Dependent N-dimensional Placefields
        A version PfND that can return the current state of placefields considering only up to a certain period of time.
    
        Represents a collection of placefields at a given time over binned, N-dimensional space. 
        
        
        from copy import deepcopy
        from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
        from neuropy.plotting.placemaps import plot_all_placefields

        included_epochs = None
        computation_config = active_session_computation_configs[0]
        print('Recomputing active_epoch_placefields2D...', end=' ')
        # PfND version:
        t_list = []
        ratemaps_list = []
        active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
                                        speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                        grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
        print('\t done.')
        # np.shape(active_time_dependent_placefields2D.curr_spikes_maps_matrix) # (64, 64, 29)
    """
    
    # is_additive_mode = True # Default, cannot backtrack to earlier times.
    is_additive_mode = False # allows selecting any times of time range, but recomputes on each update (is not additive). This means later times will take longer to calculate the earlier ones. 
    
    @property
    def smooth(self):
        """The smooth property."""
        return self.config.smooth

    ########## Overrides for temporal dependence:
    """ 
        Define all_time_* versions of the self.filtered_pos_df and self.filtered_spikes_df properties to allow access to the underlying dataframes for setup and other purposes.
    """
    @property
    def all_time_filtered_spikes_df(self):
        """The filtered_spikes_df property."""
        return self._filtered_spikes_df
        
    @property
    def all_time_filtered_pos_df(self):
        """The filtered_pos_df property."""
        return self._filtered_pos_df
        
    @property
    def earliest_spike_time(self):
        """The earliest spike time."""
        return self.all_time_filtered_spikes_df[self.all_time_filtered_spikes_df.spikes.time_variable_name].values[0]
    
    @property
    def earliest_position_time(self):
        """The earliest position sample time."""
        return self.all_time_filtered_pos_df['t'].values[0]
    
    @property
    def earliest_valid_time(self):
        """The earliest time that we have both position and spikes (the later of the two individual earliest times)"""
        return max(self.earliest_position_time, self.earliest_spike_time)
        
        
    """ 
        Override the filtered_spikes_df and filtered_pos_df properties such that they only return the dataframes up to the last time (self.last_t).
        This allows access via self.t, self.x, self.y, self.speed, etc as defined in the parent class to work as expected since they access the self.filtered_pos_df and self.filtered_spikes_df
        
        Note: these would be called curr_filtered_spikes_df and curr_filtered_pos_df in the nomenclature of this class, but they're defined without the curr_* prefix for compatibility and to override the parent implementation.
    """
    @property
    def filtered_spikes_df(self):
        """The filtered_spikes_df property."""
        return self._filtered_spikes_df.spikes.time_sliced(0, self.last_t)
        
    @property
    def filtered_pos_df(self):
        """The filtered_pos_df property."""
        return self._filtered_pos_df.position.time_sliced(0, self.last_t)
        
        
    @property
    def ratemap_spiketrains(self):
        """ a list of spike times for each cell. for compatibility with old plotting functions."""        
        ## Get only the relevant columns and the 'aclu' column before grouping on aclu for efficiency:
        return self.curr_ratemap_spiketrains(self.last_t)
        
    @property
    def ratemap_spiketrains_pos(self):
        """ a list of spike positions for each cell. for compatibility with old plotting functions."""
        return self.curr_ratemap_spiketrains_pos(self.last_t)
    
    @property
    def _position_variable_names(self):
        """The _position_variable_names property."""
        if (self.ndim > 1):
            return ['x', 'y']
        else:
            return ['x']
    
    def curr_ratemap_spiketrains_pos(self, t):
        """ gets the ratemap_spiketrains_pos variable at the time t """
        if (self.ndim > 1):
            return [safe_pandas_get_group(self.all_time_filtered_spikes_df.spikes.time_sliced(0, t)[['aclu', self.all_time_filtered_spikes_df.spikes.time_variable_name, 'x', 'y']].groupby('aclu')['x', 'y'], neuron_id).to_numpy().T for neuron_id in self.included_neuron_IDs] # dataframes split for each ID
        else:
            return [safe_pandas_get_group(self.all_time_filtered_spikes_df.spikes.time_sliced(0, t)[['aclu', self.all_time_filtered_spikes_df.spikes.time_variable_name, 'x']].groupby('aclu')['x'], neuron_id).to_numpy().T for neuron_id in self.included_neuron_IDs] # dataframes split for each ID
        
    
    def curr_ratemap_spiketrains(self, t):
        """ gets the ratemap_spiketrains variable at the time t """
        return [safe_pandas_get_group(self.all_time_filtered_spikes_df.spikes.time_sliced(0, t)[['aclu', self.all_time_filtered_spikes_df.spikes.time_variable_name]].groupby('aclu')[self.all_time_filtered_spikes_df.spikes.time_variable_name], neuron_id).to_numpy() for neuron_id in self.included_neuron_IDs] # dataframes split for each ID
    

    @property
    def ratemap(self):
        """The ratemap property is computed only as needed. Note, this might be the slowest way to get this data, it's like this just for compatibility with the other display functions."""
        # return Ratemap(self.curr_occupancy_weighted_tuning_maps_matrix, spikes_maps=self.curr_spikes_maps_matrix, xbin=self.xbin, ybin=self.ybin, neuron_ids=self.included_neuron_IDs, occupancy=self.curr_seconds_occupancy, neuron_extended_ids=self.frate_filter_fcn(self.all_time_filtered_spikes_df.spikes.neuron_probe_tuple_ids))
        # DO I need neuron_ids=self.frate_filter_fcn(self.included_neuron_IDs)?
        
        # curr_smoothed_spikes_maps_matrix
        
        
        return Ratemap(self.curr_occupancy_weighted_tuning_maps_matrix[self._included_thresh_neurons_indx,:,:], spikes_maps=self.curr_spikes_maps_matrix[self._included_thresh_neurons_indx,:,:],
                       xbin=self.xbin, ybin=self.ybin, neuron_ids=self.included_neuron_IDs, occupancy=self.curr_seconds_occupancy, neuron_extended_ids=self.frate_filter_fcn(self.all_time_filtered_spikes_df.spikes.neuron_probe_tuple_ids))
    
    
    


    def __init__(self, spikes_df: pd.DataFrame, position: Position, epochs: Epoch = None, frate_thresh=1, speed_thresh=5, grid_bin=(1,1), grid_bin_bounds=None, smooth=(1,1)):
        """computes 2d place field using (x,y) coordinates. It always computes two place maps with and
        without speed thresholds.

        Parameters
        ----------
        spikes_df: pd.DataFrame
        position : core.Position
        epochs : core.Epoch
            specifies the list of epochs to include.
        grid_bin : int
            bin size of position bining, by default 5
        speed_thresh : int
            speed threshold for calculating place field
        
        
        NOTE: doesn't call super().__init__(...)
        
        
        NOTE: _peak_frate_filter_function is only used in computing self.ratemap, meaning it keeps and does calculations for all neuron_IDs (not just those passing _peak_frate_filter_function) behind the scenes. This can be taken advantage of if you want a ratemap only for certain neurons by setting self._included_thresh_neurons_indx manually

        EXAMPLE of filtering by neuron_IDs:        
            # Find the neuron_IDs that are included in the active_pf_2D for filtering the active_pf_2D_dt's results:
            is_pf_2D_included_neuron = np.isin(active_pf_2D_dt.included_neuron_IDs, active_pf_2D.included_neuron_IDs)
            pf_2D_included_neuron_indx = active_pf_2D_dt._included_thresh_neurons_indx[is_pf_2D_included_neuron]

            # #NOTE: to reset and include all neurons:
            # active_pf_2D_dt._included_thresh_neurons_indx = np.arange(active_pf_2D_dt.n_fragile_linear_neuron_IDXs)

            active_pf_2D_dt._included_thresh_neurons_indx = pf_2D_included_neuron_indx
            active_pf_2D_dt._peak_frate_filter_function = lambda list_: [list_[_] for _ in active_pf_2D_dt._included_thresh_neurons_indx]

            assert (active_pf_2D_dt.ratemap.spikes_maps == active_pf_2D.ratemap.spikes_maps).all(), f"active_pf_2D_dt.ratemap.spikes_maps: {active_pf_2D_dt.ratemap.spikes_maps}\nactive_pf_2D.ratemap.spikes_maps: {active_pf_2D.ratemap.spikes_maps}"


        
        """
        # save the config that was used to perform the computations
        self.config = PlacefieldComputationParameters(speed_thresh=speed_thresh, grid_bin=grid_bin, grid_bin_bounds=grid_bin_bounds, smooth=smooth, frate_thresh=frate_thresh)
        self.position_srate = position.sampling_rate
        # Set the dimensionality of the PfND object from the position's dimensionality
        self.ndim = position.ndim
        
        self._included_thresh_neurons_indx = None
        self._peak_frate_filter_function = None
        self._filtered_pos_df = None
        self._filtered_spikes_df = None
        self.xbin = None
        self.ybin = None 
        self.bin_info = None

        self.last_t = np.finfo('float').max # set to maximum value (so all times are included) just for setup.
        
        # Perform the primary setup to build the placefield
        self.setup(position, spikes_df, epochs) # Sets up self.xbin, self.ybin, self.bin_info, self._filtered_pos_df, self.filtered_spikes_df
        
        if (self.ndim < 2):
            # Drop any 'y' related columns if it's a 1D version:
            print(f"dropping 'y'-related columns in self._filtered_spikes_df because self.ndim: {self.ndim} (< 2).")
            self._filtered_spikes_df.drop(columns=['y','y_loaded'], inplace=True)

        self._filtered_pos_df.dropna(axis=0, how='any', subset=[*self._position_variable_names], inplace=True) # dropped NaN values
        
        if 'binned_x' in self._filtered_pos_df:
            if (position.ndim > 1):
                self._filtered_pos_df.dropna(axis=0, how='any', subset=['binned_x', 'binned_y'], inplace=True) # dropped NaN values
            else:
                self._filtered_pos_df.dropna(axis=0, how='any', subset=['binned_x'], inplace=True) # dropped NaN values

        # Reset the rebuild_fragile_linear_neuron_IDXs:
        self._filtered_spikes_df, _reverse_cellID_index_map = self._filtered_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
        self.fragile_linear_neuron_IDXs = np.unique(self._filtered_spikes_df.fragile_linear_neuron_IDX) # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])
        self.n_fragile_linear_neuron_IDXs = len(self.fragile_linear_neuron_IDXs)
        self._included_thresh_neurons_indx = np.arange(self.n_fragile_linear_neuron_IDXs)
        
        self._peak_frate_filter_function = lambda list_: [list_[_] for _ in self._included_thresh_neurons_indx] # filter_function: takes any list of length n_neurons (original number of neurons) and returns only the elements that met the firing rate criteria
        
        ## Interpolate the spikes over positions
        self._filtered_spikes_df['x'] = np.interp(self._filtered_spikes_df[spikes_df.spikes.time_variable_name].to_numpy(), self.t, self.x)
        if 'binned_x' not in self._filtered_spikes_df:
            self._filtered_spikes_df['binned_x'] = pd.cut(self._filtered_spikes_df['x'].to_numpy(), bins=self.xbin, include_lowest=True, labels=self.xbin_labels) # same shape as the input data 
    
        if (self.ndim > 1):
            self._filtered_spikes_df['y'] = np.interp(self._filtered_spikes_df[spikes_df.spikes.time_variable_name].to_numpy(), self.t, self.y)
            if 'binned_y' not in self._filtered_spikes_df:
                self._filtered_spikes_df['binned_y'] = pd.cut(self._filtered_spikes_df['y'].to_numpy(), bins=self.ybin, include_lowest=True, labels=self.ybin_labels)
    
        self.setup_time_varying()
        

    @property
    def dims_coord_tuple(self):
        """Returns a tuple containing the number of bins in each dimension. For 1D it will be (n_xbins,) for 2D (n_xbins, n_ybins) """
        n_xbins = len(self.xbin) - 1 # the -1 is to get the counts for the centers only
        if (self.ndim > 1):
            n_ybins = len(self.ybin) - 1 # the -1 is to get the counts for the centers only
            dims_coord_ist = (n_xbins, n_ybins)
        else:
            # 1D Only
            n_ybins = None # singleton dimension along this axis. Decide how we want to shape it.
            dims_coord_ist = (n_xbins,)
        return dims_coord_ist



    def reset(self):
        """ used to reset the calculations to an initial value. """
        self.setup_time_varying()

    def setup_time_varying(self):
        # Initialize for the 0th timestamp:
        dims_coord_tuple = self.dims_coord_tuple

        self.curr_spikes_maps_matrix = np.zeros((self.n_fragile_linear_neuron_IDXs, *dims_coord_tuple), dtype=int) # create an initially zero occupancy map
        self.curr_smoothed_spikes_maps_matrix = None
        self.curr_num_pos_samples_occupancy_map = np.zeros(dims_coord_tuple, dtype=int) # create an initially zero occupancy map
        self.curr_num_pos_samples_smoothed_occupancy_map = None
        self.last_t = 0.0
        self.curr_seconds_occupancy = np.zeros(dims_coord_tuple, dtype=float)
        self.curr_normalized_occupancy = self.curr_seconds_occupancy.copy()
        self.curr_occupancy_weighted_tuning_maps_matrix = np.zeros((self.n_fragile_linear_neuron_IDXs, *dims_coord_tuple), dtype=float) # will have units of # spikes/sec
        self.historical_snapshots = OrderedDict({})


    def step(self, num_seconds_to_advance, should_snapshot=False):
        """ advance the computed time by a fixed number of seconds. """
        next_t = self.last_t + num_seconds_to_advance # add one second
        self.update(next_t, should_snapshot=should_snapshot)
        return next_t
    
    
    def update(self, t, start_relative_t:bool=False, should_snapshot=False):
        """ updates all variables to the latest versions """
        if start_relative_t:
            # if start_relative_t then t is assumed to be specified relative to the earliest_valid_time
            t = self.earliest_valid_time + t # add self.earliest_valid_time to the relative_t value to get the absolute t value
        
        if self.is_additive_mode and (self.last_t > t):
            print(f'WARNING: update(t: {t}) called with t < self.last_t ({self.last_t}! Skipping.')
        else:
            # Otherwise update to this t.
            # with np.errstate(divide='ignore', invalid='ignore'):
            with np.errstate(divide='warn', invalid='raise'):
                if self.is_additive_mode:
                    self._minimal_additive_update(t)
                    self._display_additive_update(t)
                else:
                    # non-additive mode, recompute:
                    self.complete_time_range_computation(0.0, t)
                    
                if should_snapshot:
                    self.snapshot()
                    
    # ==================================================================================================================== #
    # Snapshotting and state restoration                                                                                   #
    # ==================================================================================================================== #
    
    def snapshot(self):
        """ takes a snapshot of the current values at this time."""    
        # Add this entry to the historical snapshot dict:                
        self.historical_snapshots[self.last_t] = {
            'spikes_maps_matrix':self.curr_spikes_maps_matrix.copy(),
            'smoothed_spikes_maps_matrix': copy_if_not_none(self.curr_smoothed_spikes_maps_matrix),
            'raw_occupancy_map':self.curr_num_pos_samples_occupancy_map.copy(),
            'raw_smoothed_occupancy_map': copy_if_not_none(self.curr_num_pos_samples_smoothed_occupancy_map),
            'seconds_occupancy':self.curr_seconds_occupancy.copy(),
            'normalized_occupancy':self.curr_normalized_occupancy.copy(),
            'occupancy_weighted_tuning_maps_matrix':self.curr_occupancy_weighted_tuning_maps_matrix.copy()
        }
        return (self.last_t, self.historical_snapshots[self.last_t]) # return the (snapshot_time, snapshot_data) pair
        
    def apply_snapshot_data(self, snapshot_t, snapshot_data):
        """ applys the snapshot_data to replace the current state of this object (except for historical_snapshots) """
        self.curr_spikes_maps_matrix = snapshot_data['spikes_maps_matrix']
        self.curr_smoothed_spikes_maps_matrix = snapshot_data['smoothed_spikes_maps_matrix']
        self.curr_num_pos_samples_occupancy_map = snapshot_data['raw_occupancy_map']
        self.curr_num_pos_samples_smoothed_occupancy_map = snapshot_data['raw_smoothed_occupancy_map']
        self.curr_seconds_occupancy = snapshot_data['seconds_occupancy']
        self.curr_normalized_occupancy = snapshot_data['normalized_occupancy']
        self.curr_occupancy_weighted_tuning_maps_matrix = snapshot_data['occupancy_weighted_tuning_maps_matrix']
        self.last_t = snapshot_t
        
    def restore_from_snapshot(self, snapshot_t):
        """ restores the current state to that of a historic snapshot indexed by the time snapshot_t """
        snapshot_data = self.historical_snapshots[snapshot_t]
        self.apply_snapshot_data(snapshot_t, snapshot_data)
        
    def to_dict(self):
        # print(f'to_dict(...): {list(self.__dict__.keys())}')
        curr_snapshot_time, curr_snapshot_data = self.snapshot() # take a snapshot of the current state
        return {'config': self.config,
                'position_srate': self.position_srate,
                'ndim': self.ndim, 
                'xbin': self.xbin,
                'ybin': self.ybin,
                'bin_info': self.bin_info,
                '_filtered_spikes_df': self._filtered_spikes_df,
                '_filtered_pos_df': self._filtered_pos_df,
                'last_t': self.last_t,
                'historical_snapshots': self.historical_snapshots,
                # 'curr_spikes_maps_matrix': self.curr_spikes_maps_matrix,
                'fragile_linear_neuron_IDXs': self.fragile_linear_neuron_IDXs, # not strictly needed, could be recomputed easily
                'n_fragile_linear_neuron_IDXs': self.n_fragile_linear_neuron_IDXs, # not strictly needed, could be recomputed easily
                '_included_thresh_neurons_indx': self._included_thresh_neurons_indx, # not strictly needed, could be recomputed easily
                }

    ## For serialization/pickling:
    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        """ assumes state is a dict generated by calling self.__getstate__() previously"""        
        # print(f'__setstate__(self: {self}, state: {state})')
        # print(f'__setstate__(...): {list(self.__dict__.keys())}')
        self.__dict__ = state # set the dict
        self._save_intermediate_spikes_maps = True # False is not yet implemented
        self.restore_from_snapshot(self.last_t) # after restoring the object's __dict__ from state, self.historical_snapshots is populated and the last entry can be used to restore all the last-computed properties. Note this requires at least one snapshot.
        
        # Rebuild the filter function from self._included_thresh_neurons_indx
        # self._included_thresh_neurons_indx = np.arange(self.n_fragile_linear_neuron_IDXs)
        self._peak_frate_filter_function = lambda list_: [list_[_] for _ in self._included_thresh_neurons_indx] # filter_function: takes any list of length n_neurons (original number of neurons) and returns only the elements that met the firing rate criteria        
        
        
    # ==================================================================================================================== #
    # Common Methods                                                                                                       #
    # ==================================================================================================================== #
    @classmethod
    def compute_occupancy_weighted_tuning_map(cls, curr_seconds_occupancy_map, curr_spikes_maps_matrix, debug_print=False):
        """ Given the curr_occupancy_map and curr_spikes_maps_matrix for this timestamp, returns the occupancy weighted tuning map
        Inputs:
        # curr_seconds_occupancy_map: note that this is the occupancy map in seconds, not the raw counts
        """
        ## Simple occupancy shuffle:
        # occupancy_weighted_tuning_maps_matrix = curr_spikes_maps_matrix / curr_seconds_occupancy_map # dividing by positions with zero occupancy result in a warning and the result being set to NaN. Set to 0.0 instead.
        # occupancy_weighted_tuning_maps_matrix = np.nan_to_num(occupancy_weighted_tuning_maps_matrix, copy=True, nan=0.0) # set any NaN values to 0.0, as this is the correct weighted occupancy
        
        ## More advanced occumancy shuffle:
        curr_seconds_occupancy_map[curr_seconds_occupancy_map == 0.0] = np.nan # pre-set the zero occupancy locations to NaN to avoid a warning in the next step. They'll be replaced with zero afterwards anyway
        occupancy_weighted_tuning_maps_matrix = curr_spikes_maps_matrix / curr_seconds_occupancy_map # dividing by positions with zero occupancy result in a warning and the result being set to NaN. Set to 0.0 instead.
        occupancy_weighted_tuning_maps_matrix = np.nan_to_num(occupancy_weighted_tuning_maps_matrix, copy=True, nan=0.0) # set any NaN values to 0.0, as this is the correct weighted occupancy
        curr_seconds_occupancy_map[np.isnan(curr_seconds_occupancy_map)] = 0.0 # restore these entries back to zero
        return occupancy_weighted_tuning_maps_matrix
    

    # ==================================================================================================================== #
    # Additive update static methods:                                                                                      #
    # ==================================================================================================================== #
    def _minimal_additive_update(self, t):
        """ Updates the current_occupancy_map, curr_spikes_maps_matrix
        # t: the "current time" for which to build the best possible placefields
        
        Updates:
            self.curr_num_pos_samples_occupancy_map
            self.curr_spikes_maps_matrix
            self.last_t
        """
        # Post Initialization Update
        curr_t, self.curr_num_pos_samples_occupancy_map = PfND_TimeDependent.update_occupancy_map(self.last_t, self.curr_num_pos_samples_occupancy_map, t, self.all_time_filtered_pos_df)
        curr_t, self.curr_spikes_maps_matrix = PfND_TimeDependent.update_spikes_map(self.last_t, self.curr_spikes_maps_matrix, t, self.all_time_filtered_spikes_df)
        self.last_t = curr_t
    
    def _display_additive_update(self, t):
        """ updates the extended variables:
        
        Using:
            self.position_srate
            self.curr_num_pos_samples_occupancy_map
            self.curr_spikes_maps_matrix
            
        Updates:
            self.curr_raw_smoothed_occupancy_map
            self.curr_smoothed_spikes_maps_matrix
            self.curr_seconds_occupancy
            self.curr_normalized_occupancy
            self.curr_occupancy_weighted_tuning_maps_matrix
        
        """
        # Smooth if needed: OH NO! Don't smooth the occupancy map!!
        ## Occupancy:
        # NOTE: usually don't smooth occupancy. Unless self.should_smooth_spatial_occupancy_map is True, and in that case use the same smoothing values that are used to smooth the firing rates
        if (self.should_smooth_spatial_occupancy_map and (self.smooth is not None) and ((self.smooth[0] > 0.0) & (self.smooth[1] > 0.0))): 
            # Smooth the occupancy map:
            self.curr_num_pos_samples_smoothed_occupancy_map = gaussian_filter(self.curr_num_pos_samples_occupancy_map, sigma=(self.smooth[1], self.smooth[0])) # 2d gaussian filter
            self.curr_seconds_occupancy, self.curr_normalized_occupancy = _normalized_occupancy(self.curr_num_pos_samples_smoothed_occupancy_map, position_srate=self.position_srate)
        else:
            self.curr_seconds_occupancy, self.curr_normalized_occupancy = _normalized_occupancy(self.curr_num_pos_samples_occupancy_map, position_srate=self.position_srate)
            
        ## Spikes:
        if ((self.smooth is not None) and ((self.smooth[0] > 0.0) & (self.smooth[1] > 0.0))): 
            # Smooth the firing map:
            self.curr_smoothed_spikes_maps_matrix = gaussian_filter(self.curr_spikes_maps_matrix, sigma=(0, self.smooth[1], self.smooth[0])) # 2d gaussian filter
            self.curr_occupancy_weighted_tuning_maps_matrix = PfND_TimeDependent.compute_occupancy_weighted_tuning_map(self.curr_seconds_occupancy, self.curr_smoothed_spikes_maps_matrix)

        else:
            self.curr_occupancy_weighted_tuning_maps_matrix = PfND_TimeDependent.compute_occupancy_weighted_tuning_map(self.curr_seconds_occupancy, self.curr_spikes_maps_matrix)
    
    @classmethod
    def update_occupancy_map(cls, last_t, last_occupancy_matrix, t, active_pos_df, debug_print=False):
        """ Given the last_occupancy_matrix computed at time last_t, determines the additional positional occupancy from active_pos_df and adds them producing an updated version
        Inputs:
            t: the "current time" for which to build the best possible placefields

        TODO: MAKE_1D: remove 'binned_y' references
        """
        active_current_pos_df = active_pos_df.position.time_sliced(last_t, t) # [active_pos_df.position.time<t]
        # Compute the updated counts:
        current_bin_counts = active_current_pos_df.value_counts(subset=['binned_x', 'binned_y'], normalize=False, sort=False, ascending=True, dropna=True) # dropna=True
        # current_bin_counts: a series with a MultiIndex index for each bin that has nonzero counts
        # binned_x  binned_y
        # 2         12           2
        # 3         11           1
        #           12          30
        # ...
        # 57        9            5
        #           12           1
        #           13           4
        #           14           1
        # 58        9            3
        #           10           2
        #           12           4
        # Length: 247, dtype: int64
        if debug_print:
            print(f'np.shape(current_bin_counts): {np.shape(current_bin_counts)}') # (247,)
        for (xbin_label, ybin_label), count in current_bin_counts.iteritems():
            if debug_print:
                print(f'xbin_label: {xbin_label}, ybin_label: {ybin_label}, count: {count}')
            
            # last_occupancy_matrix[xbin_label, ybin_label] += count
            try:
                last_occupancy_matrix[xbin_label-1, ybin_label-1] += count
                # last_occupancy_matrix[xbin_label, ybin_label] += count
            except IndexError as e:
                print(f'e: {e}\n active_current_pos_df: {np.shape(active_current_pos_df)}, current_bin_counts: {np.shape(current_bin_counts)}\n last_occupancy_matrix: {np.shape(last_occupancy_matrix)}\n count: {count}')
                raise e
        return t, last_occupancy_matrix

    @classmethod
    def update_spikes_map(cls, last_t, last_spikes_maps_matrix, t, active_spike_df, debug_print=False):
        """ Given the last_spikes_maps_matrix computed at time last_t, determines the additional updates (spikes) from active_spike_df and adds them producing an updated version
        Inputs:
        # t: the "current time" for which to build the best possible placefields

        TODO: MAKE_1D: remove 'binned_y' references
        """
        active_current_spike_df = active_spike_df.spikes.time_sliced(last_t, t)
        
        # Compute the updated counts:
        current_spike_per_unit_per_bin_counts = active_current_spike_df.value_counts(subset=['fragile_linear_neuron_IDX', 'binned_x', 'binned_y'], normalize=False, sort=False, ascending=True, dropna=True) # dropna=True
        
        if debug_print:
            print(f'np.shape(current_spike_per_unit_per_bin_counts): {np.shape(current_spike_per_unit_per_bin_counts)}') # (247,)
        for (fragile_linear_neuron_IDX, xbin_label, ybin_label), count in current_spike_per_unit_per_bin_counts.iteritems():
            if debug_print:
                print(f'fragile_linear_neuron_IDX: {fragile_linear_neuron_IDX}, xbin_label: {xbin_label}, ybin_label: {ybin_label}, count: {count}')
            try:
                last_spikes_maps_matrix[fragile_linear_neuron_IDX, xbin_label-1, ybin_label-1] += count
            except IndexError as e:
                print(f'e: {e}\n active_current_spike_df: {np.shape(active_current_spike_df)}, current_spike_per_unit_per_bin_counts: {np.shape(current_spike_per_unit_per_bin_counts)}\n last_spikes_maps_matrix: {np.shape(last_spikes_maps_matrix)}\n count: {count}')
                print(f' last_spikes_maps_matrix[fragile_linear_neuron_IDX: {fragile_linear_neuron_IDX}, (xbin_label-1): {xbin_label-1}, (ybin_label-1): {ybin_label-1}] += count: {count}')
                raise e
        return t, last_spikes_maps_matrix

    
    # ==================================================================================================================== #
    # 2022-08-02 - New Simple Time-Dependent Placefield Overhaul                                                           #
    # ==================================================================================================================== #
    # Idea: use simple dataframes and operations on them to easily get the placefield results for a given time range.


        
    def complete_time_range_computation(self, start_time, end_time, assign_results_to_member_variables=True):
        """ recomputes the entire time period from start_time to end_time with few other assumptions """
        computed_out_results = PfND_TimeDependent.perform_time_range_computation(self.all_time_filtered_spikes_df, self.all_time_filtered_pos_df, position_srate=self.position_srate,
                                                             xbin=self.xbin, ybin=self.ybin,
                                                             start_time=start_time, end_time=end_time,
                                                             included_neuron_IDs=self.included_neuron_IDs, active_computation_config=self.config, override_smooth=self.smooth) # previously active_computation_config=None

        if assign_results_to_member_variables:
            # Unwrap the returned variables from the output dictionary and assign them to the member variables:        
            self.curr_seconds_occupancy = computed_out_results.seconds_occupancy
            self.curr_num_pos_samples_occupancy_map = computed_out_results.num_position_samples_occupancy
            self.curr_spikes_maps_matrix = computed_out_results.spikes_maps_matrix
            self.curr_smoothed_spikes_maps_matrix = computed_out_results.smoothed_spikes_maps_matrix
            self.curr_occupancy_weighted_tuning_maps_matrix = computed_out_results.occupancy_weighted_tuning_maps_matrix
            
            self.last_t = end_time ## TODO: note that there is no notion of a start_time later than the start of the session for this class!
        else:
            # if assign_results_to_member_variables is False, don't update any of the member variables and just return the wrapped result.
            return computed_out_results        
        
        
        
        
        
    
    @classmethod    
    def perform_time_range_computation(cls, spikes_df, pos_df, position_srate, xbin, ybin, start_time, end_time, included_neuron_IDs, active_computation_config=None, override_smooth=None):
        """ This method performs complete calculation witihin a single function. 
        
        Inputs:
        
        Note that active_computation_config can be None IFF xbin, ybin, and override_smooth are provided.
        
        Usage:
            # active_pf_spikes_df = deepcopy(sess.spikes_df)
            # active_pf_pos_df = deepcopy(sess.position.to_dataframe())
            # position_srate = sess.position_sampling_rate
            # active_computation_config = curr_active_config.computation_config
            # out_dict = PfND_TimeDependent.perform_time_range_computation(spikes_df, pos_df, position_srate, xbin, ybin, start_time, end_time, included_neuron_IDs, active_computation_config)
            out_dict = PfND_TimeDependent.perform_time_range_computation(sess.spikes_df, sess.position.to_dataframe(), position_srate=sess.position_sampling_rate,
                                                             xbin=active_pf_2D.xbin, ybin=active_pf_2D.ybin,
                                                             start_time=_test_arbitrary_start_time, end_time=_test_arbitrary_end_time,
                                                             included_neuron_IDs=active_pf_2D.included_neuron_IDs, active_computation_config=curr_active_config.computation_config, override_smooth=(0.0, 0.0))
                                                             
        TODO: MAKE_1D: remove 'binned_y' references

        """
        def _build_bin_pos_counts(active_pf_pos_df, xbin_values=None, ybin_values=None, active_computation_config=active_computation_config):

            # bin_values=(None, None), position_column_names = ('x', 'y'), binned_column_names = ('binned_x', 'binned_y'),
            # active_pf_pos_df, (xbin, ybin), bin_info = build_df_discretized_binned_position_columns(active_pf_pos_df.copy(), bin_values=bin_values, active_computation_config=active_computation_config, force_recompute=False, debug_print=False)   

            ## This version was brought in from PfND.perform_time_range_computation(...):
            # If xbin_values is not None and ybin_values is None, assume 1D
            # if xbin_values is not None and ybin_values is None:
            if 'y' not in active_pf_pos_df.columns:
                # Assume 1D:
                ndim = 1
                pos_col_names = ('x',)
                binned_col_names = ('binned_x',)
                bin_values = (xbin_values,)
            else:
                # otherwise assume 2D:
                print('ERROR: 2D!!!')
                ndim = 2
                pos_col_names = ('x', 'y')
                binned_col_names = ('binned_x', 'binned_y')
                bin_values = (xbin_values, ybin_values)

            # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within.
            active_pf_pos_df, out_bins, bin_info = build_df_discretized_binned_position_columns(active_pf_pos_df.copy(), bin_values=bin_values, position_column_names=pos_col_names, binned_column_names=binned_col_names, active_computation_config=active_computation_config, force_recompute=False, debug_print=False)
            
            if ndim == 1:
                # Assume 1D:
                xbin = out_bins[0]
                ybin = None
                n_xbins = len(xbin) - 1 # the -1 is to get the counts for the centers only
                num_position_samples_occupancy = np.zeros((n_xbins, ), dtype=int) # create an initially zero position occupancy map. # TODO: should it be NaN or np.masked where we haven't visisted at all yet?
                curr_counts_df = active_pf_pos_df.value_counts(subset=['binned_x'], normalize=False, sort=False, ascending=True, dropna=True).to_frame(name='counts').reset_index()
                xbin_indicies = curr_counts_df.binned_x.values.astype('int') - 1
                num_position_samples_occupancy[xbin_indicies] = curr_counts_df.counts.values # Assignment

            else:            
                (xbin, ybin) = out_bins
                n_xbins = len(xbin) - 1 # the -1 is to get the counts for the centers only
                n_ybins = len(ybin) - 1 # the -1 is to get the counts for the centers only
                num_position_samples_occupancy = np.zeros((n_xbins, n_ybins), dtype=int) # create an initially zero position occupancy map. # TODO: should it be NaN or np.masked where we haven't visisted at all yet?
                curr_counts_df = active_pf_pos_df.value_counts(subset=['binned_x', 'binned_y'], normalize=False, sort=False, ascending=True, dropna=True).to_frame(name='counts').reset_index()
                xbin_indicies = curr_counts_df.binned_x.values.astype('int') - 1
                ybin_indicies = curr_counts_df.binned_y.values.astype('int') - 1
                num_position_samples_occupancy[xbin_indicies, ybin_indicies] = curr_counts_df.counts.values # Assignment
                # num_position_samples_occupancy[xbin_indicies, ybin_indicies] += curr_counts_df.counts.values # Additive

            return curr_counts_df, num_position_samples_occupancy

        def _build_bin_spike_counts(active_pf_spikes_df, neuron_ids=included_neuron_IDs, xbin_values=None, ybin_values=None, active_computation_config=active_computation_config):
            ## This version was brought in from PfND.perform_time_range_computation(...):
            # If xbin_values is not None and ybin_values is None, assume 1D
            # if xbin_values is not None and ybin_values is None:
            if 'y' not in active_pf_spikes_df.columns:
                # Assume 1D:
                ndim = 1
                pos_col_names = ('x',)
                binned_col_names = ('binned_x',)
                bin_values = (xbin_values,)
            else:
                # otherwise assume 2D:
                print('ERROR: 2D!!!')
                ndim = 2
                pos_col_names = ('x', 'y')
                binned_col_names = ('binned_x', 'binned_y')
                bin_values = (xbin_values, ybin_values)

            # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within.
            active_pf_spikes_df, out_bins, bin_info = build_df_discretized_binned_position_columns(active_pf_spikes_df.copy(), bin_values=bin_values, binned_column_names=binned_col_names, position_column_names=pos_col_names, active_computation_config=active_computation_config, force_recompute=False, debug_print=False) # removed , position_column_names=pos_col_names, binned_column_names=binned_col_names
            
            if ndim == 1:
                # Assume 1D:
                xbin = out_bins[0]
                ybin = None
                n_xbins = len(xbin) - 1 # the -1 is to get the counts for the centers only
                n_neuron_ids = len(neuron_ids)
                curr_spikes_map_dict = {neuron_id:np.zeros((n_xbins, ), dtype=int) for neuron_id in neuron_ids} # create an initially zero spikes map, one for each possible neruon_id, even if there aren't spikes for that neuron yet
                curr_counts_df = active_pf_spikes_df.value_counts(subset=['aclu', 'binned_x'], sort=False).to_frame(name='counts').reset_index()
                for name, group in curr_counts_df.groupby('aclu'):
                    xbin_indicies = group.binned_x.values.astype('int') - 1
                    # curr_spikes_map_dict[name][xbin_indicies, ybin_indicies] += group.counts.values # Additive
                    curr_spikes_map_dict[name][xbin_indicies] = group.counts.values # Assignment

            else:
                # Regular 2D:
                (xbin, ybin) = out_bins
                n_xbins = len(xbin) - 1 # the -1 is to get the counts for the centers only
                n_ybins = len(ybin) - 1 # the -1 is to get the counts for the centers only
                n_neuron_ids = len(neuron_ids)
                curr_spikes_map_dict = {neuron_id:np.zeros((n_xbins, n_ybins), dtype=int) for neuron_id in neuron_ids} # create an initially zero spikes map, one for each possible neruon_id, even if there aren't spikes for that neuron yet
                curr_counts_df = active_pf_spikes_df.value_counts(subset=['aclu', 'binned_x', 'binned_y'], sort=False).to_frame(name='counts').reset_index()
                for name, group in curr_counts_df.groupby('aclu'):
                    xbin_indicies = group.binned_x.values.astype('int') - 1
                    ybin_indicies = group.binned_y.values.astype('int') - 1
                    # curr_spikes_map_dict[name][xbin_indicies, ybin_indicies] += group.counts.values # Additive
                    curr_spikes_map_dict[name][xbin_indicies, ybin_indicies] = group.counts.values # Assignment

            return curr_counts_df, curr_spikes_map_dict


        ## Only the spikes_df and pos_df are required, and are not altered by the analyses:
        active_pf_spikes_df = deepcopy(spikes_df)
        active_pf_pos_df = deepcopy(pos_df)

        ## NEEDS:
        # position_srate, xbin, ybin, included_neuron_IDs, active_computation_config
       
        if override_smooth is not None:
            smooth = override_smooth
        else:
            smooth = active_computation_config.pf_params.smooth

        ## Test arbitrarily slicing by first _test_arbitrary_end_time seconds
        active_pf_spikes_df = active_pf_spikes_df.spikes.time_sliced(start_time, end_time)
        active_pf_pos_df = active_pf_pos_df.position.time_sliced(start_time, end_time)

        counts_df, num_position_samples_occupancy = _build_bin_pos_counts(active_pf_pos_df, xbin_values=xbin, ybin_values=ybin, active_computation_config=active_computation_config)
        spikes_counts_df, spikes_map_dict = _build_bin_spike_counts(active_pf_spikes_df, neuron_ids=included_neuron_IDs, xbin_values=xbin, ybin_values=ybin, active_computation_config=active_computation_config)
        # Convert curr_spikes_map_dict from a dict into the expected 3-dim matrix:
        spikes_maps_matrix = np.array([spikes_matrix for an_aclu, spikes_matrix in spikes_map_dict.items()])  # spikes_maps_matrix.shape # (40, 64, 29) (len(curr_spikes_map_dict), n_xbins, n_ybins)

        # active_computation_config.grid_bin, smooth=active_computation_config.smooth
        seconds_occupancy, normalized_occupancy = _normalized_occupancy(num_position_samples_occupancy, position_srate=position_srate)

        ## TODO: Copy the 1D Gaussian filter code here. Currently it always does 2D:
        if 'y' not in active_pf_spikes_df.columns:
            # Assume 1D:
            ndim = 1
            smooth_criteria_fn = lambda smooth: (smooth[0] > 0.0)
            occupancy_smooth_gaussian_filter_fn = lambda x, smooth: gaussian_filter1d(x, sigma=smooth[0]) 
            spikes_maps_smooth_gaussian_filter_fn = lambda x, smooth: gaussian_filter1d(x, sigma=smooth[0]) 
        else:
            # otherwise assume 2D:
            ndim = 2
            smooth_criteria_fn = lambda smooth: ((smooth[0] > 0.0) & (smooth[1] > 0.0))
            occupancy_smooth_gaussian_filter_fn = lambda x, smooth: gaussian_filter(x, sigma=(smooth[1], smooth[0])) 
            spikes_maps_smooth_gaussian_filter_fn = lambda x, smooth: gaussian_filter(x, sigma=(0, smooth[1], smooth[0])) 


        # Smooth the final tuning map if needed and valid smooth parameter. Default FALSE.
        if (cls.should_smooth_spatial_occupancy_map and (smooth is not None) and smooth_criteria_fn(smooth)):
            # num_position_samples_occupancy = gaussian_filter(num_position_samples_occupancy, sigma=(smooth[1], smooth[0])) 
            # seconds_occupancy = gaussian_filter(seconds_occupancy, sigma=(smooth[1], smooth[0])) # 2d gaussian filter
            num_position_samples_occupancy = occupancy_smooth_gaussian_filter_fn(num_position_samples_occupancy, smooth) 
            seconds_occupancy = occupancy_smooth_gaussian_filter_fn(seconds_occupancy, smooth)
            

        # Smooth the spikes maps if needed and valid smooth parameter. Default False.
        if (cls.should_smooth_spikes_map and (smooth is not None) and smooth_criteria_fn(smooth)): 
            # smoothed_spikes_maps_matrix = gaussian_filter(spikes_maps_matrix, sigma=(0, smooth[1], smooth[0])) # 2d gaussian filter
            smoothed_spikes_maps_matrix = spikes_maps_smooth_gaussian_filter_fn(spikes_maps_matrix, smooth)
            occupancy_weighted_tuning_maps_matrix = PfND_TimeDependent.compute_occupancy_weighted_tuning_map(seconds_occupancy, smoothed_spikes_maps_matrix)
        else:
            smoothed_spikes_maps_matrix = None
            occupancy_weighted_tuning_maps_matrix = PfND_TimeDependent.compute_occupancy_weighted_tuning_map(seconds_occupancy, spikes_maps_matrix)

        # Smooth the final tuning map if needed and valid smooth parameter. Default True.            
        if (cls.should_smooth_final_tuning_map and (smooth is not None) and smooth_criteria_fn(smooth)): 
            # occupancy_weighted_tuning_maps_matrix = gaussian_filter(occupancy_weighted_tuning_maps_matrix, sigma=(0, smooth[1], smooth[0])) # 2d gaussian filter
            occupancy_weighted_tuning_maps_matrix = spikes_maps_smooth_gaussian_filter_fn(occupancy_weighted_tuning_maps_matrix, smooth)



        return DynamicContainer.init_from_dict({'num_position_samples_occupancy': num_position_samples_occupancy, 'seconds_occupancy': seconds_occupancy,
         'spikes_maps_matrix': spikes_maps_matrix, 'smoothed_spikes_maps_matrix': smoothed_spikes_maps_matrix,
         'occupancy_weighted_tuning_maps_matrix':occupancy_weighted_tuning_maps_matrix})
        
        

def perform_compute_time_dependent_placefields(active_session_spikes_df, active_pos, computation_config: PlacefieldComputationParameters, active_epoch_placefields1D=None, active_epoch_placefields2D=None, included_epochs=None, should_force_recompute_placefields=True):
    """ Most general computation function. Computes both 1D and 2D time-dependent placefields.
    active_epoch_session_Neurons: 
    active_epoch_pos: a Position object
    included_epochs: a Epoch object to filter with, only included epochs are included in the PF calculations
    active_epoch_placefields1D (Pf1D, optional) & active_epoch_placefields2D (Pf2D, optional): allow you to pass already computed Pf1D and Pf2D objects from previous runs and it won't recompute them so long as should_force_recompute_placefields=False, which is useful in interactive Notebooks/scripts
    Usage:
        active_epoch_placefields1D, active_epoch_placefields2D = perform_compute_placefields(active_epoch_session_Neurons, active_epoch_pos, active_epoch_placefields1D, active_epoch_placefields2D, active_config.computation_config, should_force_recompute_placefields=True)
    """
    ## Linearized (1D) Position Placefields:
    if ((active_epoch_placefields1D is None) or should_force_recompute_placefields):
        print('Recomputing active_epoch_time_dependent_placefields...', end=' ')
        # PfND version:
        active_epoch_placefields1D = PfND_TimeDependent(deepcopy(active_session_spikes_df), deepcopy(active_pos.linear_pos_obj), epochs=included_epochs,
                                        speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                        grid_bin=computation_config.grid_bin, grid_bin_bounds=computation_config.grid_bin_bounds, smooth=computation_config.smooth)

        print('\t done.')
    else:
        print('active_epoch_placefields1D already exists, reusing it.')

    ## 2D Position Placemaps:
    if ((active_epoch_placefields2D is None) or should_force_recompute_placefields):
        print('Recomputing active_epoch_time_dependent_placefields2D...', end=' ')
        # PfND version:
        active_epoch_placefields2D = PfND_TimeDependent(deepcopy(active_session_spikes_df), deepcopy(active_pos), epochs=included_epochs,
                                        speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                        grid_bin=computation_config.grid_bin, grid_bin_bounds=computation_config.grid_bin_bounds, smooth=computation_config.smooth)

        print('\t done.')
    else:
        print('active_epoch_placefields2D already exists, reusing it.')
    
    return active_epoch_placefields1D, active_epoch_placefields2D
