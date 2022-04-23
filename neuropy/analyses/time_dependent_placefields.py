from copy import deepcopy
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, interpolation
from neuropy.analyses.placefields import PfND, PlacefieldComputationParameters
from neuropy.core.epoch import Epoch
from neuropy.core.position import Position
from neuropy.core.ratemap import Ratemap

from neuropy.analyses.placefields import _normalized_occupancy

# time_dependent_placefields


class PfND_TimeDependent(PfND):
    """A version PfND that can return the current state of placefields considering only up to a certain period of time.
    
        Represents a collection of placefields at a given time over binned, N-dimensional space. 
    """
    
    @property
    def filtered_spikes_df(self):
        """The filtered_spikes_df property."""
        return self._filtered_spikes_df
    @filtered_spikes_df.setter
    def filtered_spikes_df(self, value):
        self._filtered_spikes_df = value
    
    
    @property
    def filtered_pos_df(self):
        """The filtered_pos_df property."""
        return self._filtered_pos_df
    @filtered_pos_df.setter
    def filtered_pos_df(self, value):
        self._filtered_pos_df = value
    
    
    @property
    def smooth(self):
        """The smooth property."""
        return self.config.smooth

    
    def __init__(self, spikes_df: pd.DataFrame, position: Position, epochs: Epoch = None, frate_thresh=1, speed_thresh=5, grid_bin=(1,1), smooth=(1,1)):
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
        """
        _save_intermediate_firing_maps = True # False is not yet implemented
        # save the config that was used to perform the computations
        self.config = PlacefieldComputationParameters(speed_thresh=speed_thresh, grid_bin=grid_bin, smooth=smooth, frate_thresh=frate_thresh)
        self.position_srate = position.sampling_rate
        # Set the dimensionality of the PfND object from the position's dimensionality
        self.ndim = position.ndim
        
        pos_df = position.to_dataframe().copy()
        spk_df = spikes_df.copy()

        # filtering:
        if epochs is not None:
            # filter the spikes_df:
            self._filtered_spikes_df = spk_df.spikes.time_sliced(epochs.starts, epochs.stops)
            # filter the pos_df:
            self._filtered_pos_df = pos_df.position.time_sliced(epochs.starts, epochs.stops) # 5378 rows × 18 columns
        else:
            # if no epochs filtering, set the filtered objects to be sliced by the available range of the position data (given by position.t_start, position.t_stop)
            self._filtered_spikes_df = spk_df.spikes.time_sliced(position.t_start, position.t_stop)
            self._filtered_pos_df = pos_df.position.time_sliced(position.t_start, position.t_stop)
            
     
        # Set animal observed position member variables:
        self.t = self._filtered_pos_df.t.to_numpy()
        self.x = self._filtered_pos_df.x.to_numpy()
        self.speed = self._filtered_pos_df.speed.to_numpy()
        if (self.should_smooth_speed and (smooth is not None) and (smooth[0] > 0.0)):
            self.speed = gaussian_filter1d(self.speed, sigma=smooth[0])
        if (self.ndim > 1):
            self.y = self._filtered_pos_df.y.to_numpy()
        else:
            self.y = None

        # Add interpolated velocity information to spikes dataframe:
        self._filtered_spikes_df['speed'] = np.interp(self._filtered_spikes_df[spikes_df.spikes.time_variable_name].to_numpy(), self.t, self.speed)
        
        # Filter for speed:
        print(f'pre speed filtering: {np.shape(self._filtered_spikes_df)[0]} spikes.')
        self._filtered_spikes_df = self._filtered_spikes_df[self._filtered_spikes_df['speed'] > speed_thresh]
        print(f'post speed filtering: {np.shape(self._filtered_spikes_df)[0]} spikes.')
        
        ## Binning with Fixed Number of Bins:    
        # xbin, ybin, bin_info = PfND._bin_pos_nD(self.x, self.y, num_bins=grid_num_bins) # num_bins mode:
        self.xbin, self.ybin, self.bin_info = PfND._bin_pos_nD(self.x, self.y, bin_size=grid_bin) # bin_size mode
        
        self._filtered_pos_df.dropna(axis=0, how='any', subset=['x','y','binned_x','binned_y'], inplace=True) # dropped NaN values
        self.xbin_labels = np.arange(start=1, stop=len(self.xbin)) # bin labels are 1-indexed, thus adding 1
        self.ybin_labels = np.arange(start=1, stop=len(self.ybin))

        self.unit_ids = np.unique(self._filtered_spikes_df.unit_id) # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])
        self.n_unit_ids = len(self.unit_ids)

        ## Interpolate the spikes over positions
        self._filtered_spikes_df['x'] = np.interp(self._filtered_spikes_df[spikes_df.spikes.time_variable_name].to_numpy(), self.t, self.x)
        if 'binned_x' not in self._filtered_spikes_df:
            self._filtered_spikes_df['binned_x'] = pd.cut(self._filtered_spikes_df['x'].to_numpy(), bins=self.xbin, include_lowest=True, labels=self.xbin_labels) # same shape as the input data 
            
        # update the dataframe 'x','speed' and 'y' properties:
        # cell_df.loc[:, 'x'] = spk_x
        # cell_df.loc[:, 'speed'] = spk_spd
        if (position.ndim > 1):
            self._filtered_spikes_df['y'] = np.interp(self._filtered_spikes_df[spikes_df.spikes.time_variable_name].to_numpy(), self.t, self.y)
            if 'binned_y' not in self._filtered_spikes_df:
                self._filtered_spikes_df['binned_y'] = pd.cut(self._filtered_spikes_df['y'].to_numpy(), bins=self.ybin, include_lowest=True, labels=self.ybin_labels)
            # cell_df.loc[:, 'y'] = spk_y
    
        self.setup()
            
        # --- occupancy map calculation -----------
        # if not self.should_smooth_spatial_occupancy_map:
        #     smooth_occupancy_map = (0.0, 0.0)
        # else:
        #     smooth_occupancy_map = smooth
            
        # if (position.ndim > 1):
        #     occupancy, xedges, yedges = Pf2D._compute_occupancy(self.x, self.y, xbin, ybin, self.position_srate, smooth_occupancy_map)
        # else:
        #     occupancy, xedges = Pf1D._compute_occupancy(self.x, xbin, self.position_srate, smooth_occupancy_map[0])
        
        # # Once filtering and binning is done, apply the grouping:
        # # Group by the aclu (cluster indicator) column
        # cell_grouped_spikes_df = self.filtered_spikes_df.groupby(['aclu'])
        # cell_spikes_dfs = [cell_grouped_spikes_df.get_group(a_neuron_id) for a_neuron_id in self.filtered_spikes_df.spikes.neuron_ids] # a list of dataframes for each neuron_id

        # # NOTE: regardless of whether should_smooth_final_tuning_map is true or not, we must pass in the actual smooth value to the _compute_tuning_map(...) function so it can choose to filter its firing map or not. Only if should_smooth_final_tuning_map is enabled will the final product be smoothed.
            
        # # re-interpolate given the updated spks
        # for cell_df in cell_spikes_dfs:
        #     cell_spike_times = cell_df[spikes_df.spikes.time_variable_name].to_numpy() # not this was subbed from 't_rel_seconds' to spikes_df.spikes.time_variable_name
        #     # spk_spd = np.interp(cell_spike_times, self.t, self.speed)
        #     spk_x = np.interp(cell_spike_times, self.t, self.x)
            
        #     # update the dataframe 'x','speed' and 'y' properties:
        #     # cell_df.loc[:, 'x'] = spk_x
        #     # cell_df.loc[:, 'speed'] = spk_spd
        #     if (position.ndim > 1):
        #         spk_y = np.interp(cell_spike_times, self.t, self.y)
        #         # cell_df.loc[:, 'y'] = spk_y
        #         spk_pos.append([spk_x, spk_y])
        #         # TODO: Make "firing maps" before "tuning maps"
        #         # raw_tuning_maps = np.asarray([Pf2D._compute_tuning_map(neuron_split_spike_dfs[i].x.to_numpy(), neuron_split_spike_dfs[i].y.to_numpy(), xbin, ybin, occupancy, None, should_return_raw_tuning_map=True) for i in np.arange(len(neuron_split_spike_dfs))]) # dataframes split for each ID:
        #         # tuning_maps = np.asarray([raw_tuning_maps[i] / occupancy for i in np.arange(len(raw_tuning_maps))])
        #         # ratemap = Ratemap(tuning_maps, xbin=xbin, ybin=ybin, neuron_ids=active_epoch_session.neuron_ids)
        #         curr_cell_tuning_map, curr_cell_firing_map = Pf2D._compute_tuning_map(spk_x, spk_y, xbin, ybin, occupancy, smooth, should_also_return_intermediate_firing_map=_save_intermediate_firing_maps)
        #     else:
        #         # otherwise only 1D:
        #         spk_pos.append([spk_x])
        #         curr_cell_tuning_map, curr_cell_firing_map = Pf1D._compute_tuning_map(spk_x, xbin, occupancy, smooth[0], should_also_return_intermediate_firing_map=_save_intermediate_firing_maps)
            
        #     spk_t.append(cell_spike_times)
        #     # tuning curve calculation:               
        #     tuning_maps.append(curr_cell_tuning_map)
        #     firing_maps.append(curr_cell_firing_map)
                
        # # ---- cells with peak frate abouve thresh ------
        # filtered_tuning_maps, filter_function = PfND._filter_by_frate(tuning_maps.copy(), frate_thresh)
        # filtered_firing_maps = filter_function(firing_maps.copy())
        # filtered_neuron_ids = filter_function(self.filtered_spikes_df.spikes.neuron_ids)        
        # filtered_tuple_neuron_ids = filter_function(self.filtered_spikes_df.spikes.neuron_probe_tuple_ids) # the (shank, probe) tuples corresponding to neuron_ids
        
        # self.ratemap = Ratemap(
        #     filtered_tuning_maps, firing_maps=filtered_firing_maps, xbin=xbin, ybin=ybin, neuron_ids=filtered_neuron_ids, occupancy=occupancy, neuron_extended_ids=filtered_tuple_neuron_ids
        # )
        # self.ratemap_spiketrains = filter_function(spk_t)
        # self.ratemap_spiketrains_pos = filter_function(spk_pos)
        
        # done!
    
    # def occupancy(self, t):
    #  # returns the ocupancy at a specific time
    #  self.ratemap.occupancy
    
    
    @property
    def ratemap(self):
        """The ratemap property is computed only as needed. Note, this might be the slowest way to get this data, it's like this just for compatibility with the other display functions."""
        return Ratemap(self.curr_occupancy_weighted_tuning_maps_matrix, firing_maps=self.curr_firing_maps_matrix, xbin=self.xbin, ybin=self.ybin, neuron_ids=self.filtered_spikes_df.spikes.neuron_ids, occupancy=self.curr_seconds_occupancy, neuron_extended_ids=self.filtered_spikes_df.spikes.neuron_probe_tuple_ids)


    def setup(self):
        # Initialize for the 0th timestamp:
        n_xbins = len(self.xbin) - 1 # the -1 is to get the counts for the centers only
        n_ybins = len(self.ybin) - 1 # the -1 is to get the counts for the centers only
        self.curr_firing_maps_matrix = np.zeros((self.n_unit_ids, n_xbins, n_ybins), dtype=int) # create an initially zero occupancy map
        self.curr_smoothed_firing_maps_matrix = None
        self.curr_raw_occupancy_map = np.zeros((n_xbins, n_ybins), dtype=int) # create an initially zero occupancy map
        self.curr_raw_smoothed_occupancy_map = None
        self.last_t = 0.0
        self.curr_seconds_occupancy = self.curr_raw_occupancy_map.copy()
        self.curr_normalized_occupancy = self.curr_raw_occupancy_map.copy()
        self.curr_occupancy_weighted_tuning_maps_matrix = self.curr_firing_maps_matrix.copy()


    def update(self, t):
        """ updates all variables to the latest versions """
        self.minimal_update(t)
        self.display_update(t)


    def minimal_update(self, t):
        """ Updates the current_occupancy_map, curr_firing_maps_matrix
        # t: the "current time" for which to build the best possible placefields
        """
        # Post Initialization Update
        # t = self.last_t + 1 # add one second
        curr_t, self.curr_raw_occupancy_map = PfND_TimeDependent.update_occupancy_map(self.last_t, self.curr_raw_occupancy_map, t, self._filtered_pos_df)
        curr_t, self.curr_firing_maps_matrix = PfND_TimeDependent.update_firing_map(self.last_t, self.curr_firing_maps_matrix, t, self._filtered_spikes_df)
        self.last_t = curr_t
        
    def display_update(self, t):
        """ updates the extended variables:
        
            self.curr_raw_smoothed_occupancy_map
            self.curr_smoothed_firing_maps_matrix
            self.curr_seconds_occupancy
            self.curr_normalized_occupancy
            self.curr_occupancy_weighted_tuning_maps_matrix
        
        """
        # Smooth if needed:
        if ((self.smooth is not None) and ((self.smooth[0] > 0.0) & (self.smooth[1] > 0.0))): 
            # Smooth the occupancy map:
            self.curr_raw_smoothed_occupancy_map = gaussian_filter(self.curr_raw_occupancy_map, sigma=(self.smooth[1], self.smooth[0])) # 2d gaussian filter
            # Smooth the firing map:
            self.curr_smoothed_firing_maps_matrix = gaussian_filter(self.curr_firing_maps_matrix, sigma=(0, self.smooth[1], self.smooth[0])) # 2d gaussian filter
            self.curr_seconds_occupancy, self.curr_normalized_occupancy = _normalized_occupancy(self.curr_raw_smoothed_occupancy_map, position_srate=self.position_srate)
            self.curr_occupancy_weighted_tuning_maps_matrix = PfND_TimeDependent.compute_occupancy_weighted_tuning_map(self.curr_seconds_occupancy, self.curr_smoothed_firing_maps_matrix)

        else:
            self.curr_seconds_occupancy, self.curr_normalized_occupancy = _normalized_occupancy(self.curr_raw_occupancy_map, position_srate=self.position_srate)
            self.curr_occupancy_weighted_tuning_maps_matrix = PfND_TimeDependent.compute_occupancy_weighted_tuning_map(self.curr_seconds_occupancy, self.curr_firing_maps_matrix)
        
    
    @classmethod
    def update_occupancy_map(cls, last_t, last_occupancy_matrix, t, active_pos_df, debug_print=False):
        """ Given the last_occupancy_matrix computed at time last_t, determines the additional positional occupancy from active_pos_df and adds them producing an updated version
        Inputs:
        # t: the "current time" for which to build the best possible placefields
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
        # current_bin_counts.index.to_flat_index() # Index([ (2, 12),  (3, 11),  (3, 12),  (3, 13),  (3, 14),  (3, 15),  (3, 16),  (3, 17),   (4, 9),  (4, 10), ... (56, 17), (56, 18), (56, 19),  (57, 9), (57, 12), (57, 13), (57, 14),  (58, 9), (58, 10), (58, 12)], dtype='object', length=247)
        if debug_print:
            print(f'np.shape(current_bin_counts): {np.shape(current_bin_counts)}') # (247,)
        for (xbin_label, ybin_label), count in current_bin_counts.iteritems():
            if debug_print:
                print(f'xbin_label: {xbin_label}, ybin_label: {ybin_label}, count: {count}')
            # last_occupancy_matrix[xbin_label-1, ybin_label-1] += count
            # last_occupancy_matrix[xbin_label, ybin_label] += count
            last_occupancy_matrix[xbin_label, ybin_label] += count
            
        return t, last_occupancy_matrix

    @classmethod
    def update_firing_map(cls, last_t, last_firing_maps_matrix, t, active_spike_df, debug_print=False):
        """ Given the last_firing_maps_matrix computed at time last_t, determines the additional updates (spikes) from active_spike_df and adds them producing an updated version
        Inputs:
        # t: the "current time" for which to build the best possible placefields
        """
        active_current_spike_df = active_spike_df.spikes.time_sliced(last_t, t)
        
        # Compute the updated counts:
        current_spike_per_unit_per_bin_counts = active_current_spike_df.value_counts(subset=['unit_id', 'binned_x', 'binned_y'], normalize=False, sort=False, ascending=True, dropna=True) # dropna=True
        
        if debug_print:
            print(f'np.shape(current_spike_per_unit_per_bin_counts): {np.shape(current_spike_per_unit_per_bin_counts)}') # (247,)
        for (unit_id, xbin_label, ybin_label), count in current_spike_per_unit_per_bin_counts.iteritems():
            if debug_print:
                print(f'unit_id: {unit_id}, xbin_label: {xbin_label}, ybin_label: {ybin_label}, count: {count}')
            last_firing_maps_matrix[unit_id, xbin_label-1, ybin_label-1] += count

        return t, last_firing_maps_matrix

    @classmethod
    def compute_occupancy_weighted_tuning_map(cls, curr_seconds_occupancy_map, curr_firing_maps_matrix, debug_print=False):
        """ Given the curr_occupancy_map and curr_firing_maps_matrix for this timestamp, returns the occupancy weighted tuning map
        Inputs:
        # curr_seconds_occupancy_map: note that this is the occupancy map in seconds, not the raw counts
        """
        # occupancy[occupancy == 0.0] = np.nan # pre-set the zero occupancy locations to NaN to avoid a warning in the next step. They'll be replaced with zero afterwards anyway
        occupancy_weighted_tuning_maps_matrix = curr_firing_maps_matrix / curr_seconds_occupancy_map # dividing by positions with zero occupancy result in a warning and the result being set to NaN. Set to 0.0 instead.
        # occupancy_weighted_tuning_map = firing_map / occupancy # dividing by positions with zero occupancy result in a warning and the result being set to NaN. Set to 0.0 instead.
        occupancy_weighted_tuning_maps_matrix = np.nan_to_num(occupancy_weighted_tuning_maps_matrix, copy=True, nan=0.0) # set any NaN values to 0.0, as this is the correct weighted occupancy
        # occupancy[np.isnan(occupancy)] = 0.0 # restore these entries back to zero
        return occupancy_weighted_tuning_maps_matrix




