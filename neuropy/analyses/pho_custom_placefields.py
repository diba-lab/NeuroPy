from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d


from neuropy.analyses.placefields import Pf1D, PfnConfigMixin, PfnDMixin, PfnDPlottingMixin, PlacefieldComputationParameters, _bin_pos_nD, _filter_by_frate, Pf2D, _normalized_occupancy
# plot_placefield_occupancy, plot_occupancy_custom
from neuropy.core.epoch import Epoch
from neuropy.core.neurons import Neurons
from neuropy.core.position import Position

from neuropy.core.ratemap import Ratemap
from neuropy.utils.mixins.concatenatable import ConcatenationInitializable

# First, interested in answering the question "where did the animal spend its time on the track" to assess the relative frequency of events that occur in a given region. If the animal spends a lot of time in a certain region,
# it's more likely that any cell, not just the ones that hold it as a valid place field, will fire there.
    # this can be done by either binning (lumping close position points together based on a standardized grid), neighborhooding, or continuous smearing. 

def build_customPf2D_fromConfig(active_epoch_session, custom_computation_config):

    should_plot = False
    should_plot_multiple_occupancy_curves = False
    
    if should_plot:
        fig = Figure(figsize=(10, 6))
        ax = fig.subplots(2, 1)
    else:
        # if should_plot is False, disable all other specific plotting options.
        should_plot_multiple_occupancy_curves = False
        fig = None

    pos_df = active_epoch_session.position.to_dataframe().copy()
    laps_df = active_epoch_session.laps.to_dataframe().copy()
    spk_df = active_epoch_session.spikes_df.copy()

    ## Binning with Fixed Number of Bins:    
    xbin, ybin, bin_info = _bin_pos_nD(pos_df.x.to_numpy(), pos_df.y.to_numpy(), bin_size=custom_computation_config.grid_bin) # bin_size mode
    # print(bin_info)
    ## Binning with Fixed Bin Sizes:
    # xbin, ybin, bin_info = _bin_pos_nD(pos_df.x.to_numpy(), pos_df.y.to_numpy(), num_bins=num_bins) # num_bins mode
    # print(bin_info)

    # print('xbin: {}'.format(xbin))
    # print('ybin: {}'.format(ybin))

    # # Laps plotting:
    # # pos_df.lin_pos.plot();
    # curr_lap_id = 3
    # plt.plot(pos_df.t, pos_df.lin_pos, '*');
    # plt.xlim([laps_df.start[curr_lap_id], laps_df.stop[curr_lap_id]])
    # # pos_df.describe()
    # # pos_df.boxplot()

    raw_occupancy, xedges, yedges = Pf2D._compute_occupancy(pos_df.x.to_numpy(), pos_df.y.to_numpy(), xbin, ybin, active_epoch_session.position.sampling_rate, custom_computation_config.smooth, should_return_raw_occupancy=True)
    seconds_occupancy, normalized_occupancy = _normalized_occupancy(raw_occupancy, position_srate=active_epoch_session.position.sampling_rate)
    occupancy = seconds_occupancy
    # print(np.shape(occupancy))
    # print(occupancy)
    # plot_occupancy(occupancy)
    # plot_occupancy_custom(active_epoch_placefields2D)

    if should_plot_multiple_occupancy_curves:
        fig, ax = plot_occupancy_custom(raw_occupancy, xedges, yedges, max_normalized=False)
        ax.set_title('Custom Occupancy: Raw')
        fig, ax = plot_occupancy_custom(normalized_occupancy, xedges, yedges, max_normalized=False)
        ax.set_title('Custom Occupancy: Normalized')
        fig, ax = plot_occupancy_custom(seconds_occupancy, xedges, yedges, max_normalized=False)
        ax.set_title('Custom Occupancy: Seconds')

    # pos_df.groupby('lap').plas.hist(alpha=0.4)

    # Given a cell's last several seconds of its instantaneous firing rate at a given point in time, what's like likelihood that it's at a given position.
        # continuous position used.

    # spk_df_filtered_speed_thresh = spk_df[spk_df['speed'] >= custom_computation_config.speed_thresh].copy() # filter out the spikes below the speed_threshold
    # spk_x = spk_df_filtered_speed_thresh['x'].to_numpy()
    # spk_y = spk_df_filtered_speed_thresh['y'].to_numpy()

    spk_x = spk_df['x'].to_numpy()
    spk_y = spk_df['y'].to_numpy()
    num_spike_counts_map = Pf2D._compute_tuning_map(spk_x, spk_y, xbin, ybin, occupancy, custom_computation_config.smooth, should_return_raw_tuning_map=True)
    
    
    if should_plot:
        fig, ax[0] = plot_occupancy_custom(num_spike_counts_map, xbin, ybin, max_normalized=False, fig=fig, ax=ax[0])
        ax[0].set_title('Custom num_spike_counts_map: All Neurons')

        mpl_pane = pn.pane.Matplotlib(fig, dpi=144, height=800)
        tabs = pn.Tabs(('num_spike_counts_map', fig))
        
        
    ## This seems to be wrong, the highest spike rate is like 0.1 (in Hz)
    spike_rate_Hz_map = num_spike_counts_map / seconds_occupancy
    
    if should_plot:
        fig, ax[1] = plot_occupancy_custom(spike_rate_Hz_map, xbin, ybin, max_normalized=False, fig=fig, ax=ax[1])
        ax[1].set_title('Custom spike_rate_Hz_map [Hz]: All Neurons, Occupancy Divided')
        # Add a tab
        tabs.append(('spike_rate_Hz_map', fig))
        # # Add a tab
        # tabs.append(('Slider', pn.widgets.FloatSlider()))

    neuron_split_spike_dfs = [spk_df.groupby('aclu').get_group(neuron_id)[['t','x','y','lin_pos']] for neuron_id in active_epoch_session.neuron_ids] # dataframes split for each ID:
    raw_tuning_maps = np.asarray([Pf2D._compute_tuning_map(neuron_split_spike_dfs[i].x.to_numpy(), neuron_split_spike_dfs[i].y.to_numpy(), xbin, ybin, occupancy, custom_computation_config.smooth, should_return_raw_tuning_map=True) for i in np.arange(len(neuron_split_spike_dfs))]) # dataframes split for each ID:
    tuning_maps = np.asarray([raw_tuning_maps[i] / occupancy for i in np.arange(len(raw_tuning_maps))])
    ratemap = Ratemap(tuning_maps, xbin=xbin, ybin=ybin, neuron_ids=active_epoch_session.neuron_ids)

    # fig, ax = plot_occupancy_custom(raw_tuning_maps[0], xedges, yedges, max_normalized=False)
    # ax.set_title('Custom raw_tuning_maps: Seconds')
    firing_spike_counts_max = np.asarray([np.nanmax(raw_tuning_maps[i]) for i in np.arange(len(neuron_split_spike_dfs))]) # dataframes split for each ID:
    # print('firing_spike_counts_max: {}'.format(firing_spike_counts_max))
    firing_rate_max = np.asarray([np.nanmax(tuning_maps[i]) for i in np.arange(len(neuron_split_spike_dfs))]) # dataframes split for each ID:
    # print('firing_rate_max: {}'.format(firing_rate_max))

    filtered_tuning_maps, filter_function = _filter_by_frate(tuning_maps.copy(), custom_computation_config.frate_thresh)
    filtered_ratemap = Ratemap(filtered_tuning_maps, xbin=xbin, ybin=ybin, neuron_ids=filter_function(ratemap.neuron_ids))
    
    # outputs: filtered_ratemap, filtered_ratemap
    
    # plt.fastcolor(active_epoch_placefields1D.occupancy)
    # Convolve the location data

    # plot_occupancy(active_epoch_placefields2D)
    # pn.pane.Matplotlib(fig)
    
    return filtered_ratemap, fig

def build_customPf2D(active_epoch_session, speed_thresh=1, grid_bin=(10, 3), smooth=(0.0, 0.0), frate_thresh=0.0):
    # custom_active_config = active_config
    # note the second smoothing paramter affects the horizontal axis on the occupancy plot:
    # custom_computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=(10, 3), smooth=(2, 0.1), frate_thresh=0.0)
    # custom_computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=(10, 3), smooth=(0.0, 0.0), frate_thresh=0.0)
    custom_computation_config = PlacefieldComputationParameters(speed_thresh=speed_thresh, grid_bin=grid_bin, smooth=smooth, frate_thresh=frate_thresh)
    return build_customPf2D_fromConfig(active_epoch_session, custom_computation_config)

def build_customPf2D_separate(active_epoch_session, speed_thresh=1, grid_bin_x=10, grid_bin_y=3, smooth_x=0.0, smooth_y=0.0, frate_thresh=0.0):
    return build_customPf2D(active_epoch_session, speed_thresh=speed_thresh, grid_bin=(grid_bin_x, grid_bin_y), smooth=(smooth_x, smooth_y), frate_thresh=frate_thresh)


# build_customPf2D(active_epoch_session, speed_thresh=1, grid_bin=10, smooth=0.0, frate_thresh=0.0)
# pn.interact(build_customPf2D, active_epoch_session=fixed(active_epoch_session), speed_thresh=1, grid_bin=(10, 3), smooth=(0.0, 0.0), frate_thresh=2.0)

# pn.interact(build_customPf2D_separate, active_epoch_session=fixed(active_epoch_session), speed_thresh=(0.0, 20.0, 1.0), grid_bin_x=(0.10, 20.0, 0.5), grid_bin_y=(0.10, 20.0, 0.5), smooth_x=(0.0, 20.0, 0.25), smooth_y=(0.0, 20.0, 0.25), frate_thresh=(0.0, 20.0, 1.0))

class TimeSlicedMixin:
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
        num_slices = len(starts)
        
        for i in np.arange(num_slices):
            # curr_lap_id = laps_df.loc[i, 'lap_id']
            # curr_lap_t_start, curr_lap_t_stop = laps_df.loc[i, 'start'], laps_df.loc[i, 'stop']
            curr_slice_t_start, curr_slice_t_stop = starts[i], stops[i]
            curr_lap_position_df_is_included = self._obj[self.time_variable_name].between(curr_slice_t_start, curr_slice_t_stop, inclusive=True) # returns a boolean array indicating inclusion
            inclusion_mask[curr_lap_position_df_is_included] = True
            # position_df.loc[curr_lap_position_df_is_included, ['lap']] = curr_lap_id # set the 'lap' identifier on the object
            
        # once all slices have been computed and the inclusion_mask is complete, use it to mask the output dataframe
        return self._obj.loc[inclusion_mask, :].copy()
    
    


@pd.api.extensions.register_dataframe_accessor("position")
class PositionAccessor(TimeSlicedMixin):
    __time_variable_name = 't' # currently hardcoded
    
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        if "t" not in obj.columns:
            raise AttributeError("Must have at least one time variable: either 't' and 't_seconds', or 't_rel_seconds'.")
        if "x" not in obj.columns:
            raise AttributeError("Must have at least one position dimension column 'x'.")
        # if "lin_pos" not in obj.columns or "speed" not in obj.columns:
        #     raise AttributeError("Must have 'lin_pos' column and 'x'.")

    @property
    def time_variable_name(self):
        return PositionAccessor.__time_variable_name
    
    @property
    def ndim(self):
        # returns the count of the spatial columns that the dataframe has
        return np.sum(np.isin(['x','y','z'], self._obj.columns))
        
    @property
    def dim_columns(self):
        # returns the labels of the columns that correspond to spatial columns 
        # If ndim == 1, returns ['x'], 
        # if ndim == 2, returns ['x','y'], etc.
        spatial_column_labels = np.array(['x','y','z'])
        return list(spatial_column_labels[np.isin(spatial_column_labels, self._obj.columns)])
    
    @property
    def n_frames(self):
        return len(self._obj.index)
    
    @property
    def speed(self):
        # dt = 1 / self.sampling_rate
        if 'speed' in self._obj.columns:
            return self._obj['speed'].to_numpy()
        else:
            # dt = np.diff(self.time)
            dt = np.mean(np.diff(self.time))
            self._obj['speed'] = np.insert((np.sqrt(((np.abs(np.diff(self.traces, axis=1))) ** 2).sum(axis=0)) / dt), 0, 0.0) # prepends a 0.0 value to the front of the result array so it's the same length as the other position vectors (x, y, etc)        
        return self._obj['speed'].to_numpy()
    
    
    def compute_higher_order_derivatives(self, component_label: str):
        """Computes the higher-order positional derivatives for a single component (given by component_label) of the pos_df
        Args:
            self (pd.DataFrame): [a pos_df]
            component_label (str): [description]
        Returns:
            pd.DataFrame: The updated dataframe with the dt, velocity, and acceleration columns added.
        """
        # compute each component separately:
        velocity_column_key = f'velocity_{component_label}'
        acceleration_column_key = f'acceleration_{component_label}'
                
        dt = np.insert(np.diff(self._obj['t']), 0, np.nan)
        velocity_comp = np.insert(np.diff(self._obj[component_label]), 0, 0.0) / dt
        velocity_comp[np.isnan(velocity_comp)] = 0.0 # replace NaN components with zero
        acceleration_comp = np.insert(np.diff(velocity_comp), 0, 0.0) / dt
        acceleration_comp[np.isnan(acceleration_comp)] = 0.0 # replace NaN components with zero
        dt[np.isnan(dt)] = 0.0 # replace NaN components with zero
        
        # add the columns to the dataframe:
        self._obj['dt'] = dt
        self._obj[velocity_column_key] = velocity_comp
        self._obj[acceleration_column_key] = acceleration_comp
        
        return self._obj  
    
    
@pd.api.extensions.register_dataframe_accessor("spikes")
class SpikesAccessor(TimeSlicedMixin):
    __time_variable_name = 't_rel_seconds' # currently hardcoded
    
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude        
        if "aclu" not in obj.columns or "cell_type" not in obj.columns:
            raise AttributeError("Must have unit id column 'aclu' and 'cell_type' column.")
        if "flat_spike_idx" not in obj.columns:
            raise AttributeError("Must have 'flat_spike_idx' column.")
        if "t" not in obj.columns and "t_seconds" not in obj.columns and "t_rel_seconds" not in obj.columns:
            raise AttributeError("Must have at least one time column: either 't' and 't_seconds', or 't_rel_seconds'.")
        
    @property
    def time_variable_name(self):
        return SpikesAccessor.__time_variable_name
        
    @property
    def neuron_ids(self):
        # return the unique cell identifiers (given by the unique values of the 'aclu' column) for this DataFrame
        unique_aclus = np.unique(self._obj['aclu'].values)
        return unique_aclus
    
    
    @property
    def neuron_probe_tuple_ids(self):
        """ returns a list of tuples where the first element is the shank_id and the second is the cluster_id. Returned in the same order as self.neuron_ids """
        # groupby the multi-index [shank, cluster]:
        # shank_cluster_grouped_spikes_df = self._obj.groupby(['shank','cluster'])
        aclu_grouped_spikes_df = self._obj.groupby(['aclu'])
        shank_cluster_reference_df = aclu_grouped_spikes_df[['shank','cluster']].first() # returns a df indexed by 'aclu' with only the 'shank' and 'cluster' columns
        output_tuples_list = [(an_id.shank, an_id.cluster) for an_id in shank_cluster_reference_df.itertuples()] # returns a list of tuples where the first element is the shank_id and the second is the cluster_id. Returned in the same order as self.neuron_ids
        return output_tuples_list
        

    @property
    def n_total_spikes(self):
        return np.shape(self._obj)[0]

    @property
    def n_neurons(self):
        return len(self.neuron_ids)
    

class PfND(PfnConfigMixin, PfnDPlottingMixin):
    
    def str_for_filename(self, prefix_string=''):
        if self.ndim <= 1:
            return '-'.join(['pf1D', f'{prefix_string}{self.config.str_for_filename(False)}'])
        else:
            return '-'.join(['pf2D', f'{prefix_string}{self.config.str_for_filename(True)}'])
    
    def str_for_display(self, prefix_string=''):
        if self.ndim <= 1:
            return '-'.join(['pf1D', f'{prefix_string}{self.config.str_for_display(False)}', f'cell_{curr_cell_id:02d}'])
        else:
            return '-'.join(['pf2D', f'{prefix_string}{self.config.str_for_display(True)}', f'cell_{curr_cell_id:02d}'])
        
        
    
    
    def __init__(self,
        spikes_df: pd.DataFrame,
        position: Position,
        epochs: Epoch = None,
        frate_thresh=1,
        speed_thresh=5,
        grid_bin=(1,1),
        smooth=(1,1)):
        
        """computes 2d place field using (x,y) coordinates. It always computes two place maps with and
        without speed thresholds.

        Parameters
        ----------
        track_name : str
            name of track
        direction : forward, backward or None
            direction of running, by default None which means direction is ignored
        grid_bin : int
            bin size of position bining, by default 5
        speed_thresh : int
            speed threshold for calculating place field
        """
        
    
        # save the config that was used to perform the computations
        self.config = PlacefieldComputationParameters(speed_thresh=speed_thresh, grid_bin=grid_bin, smooth=smooth, frate_thresh=frate_thresh)
        # assert position.ndim < 2, "Only 2+ dimensional position are acceptable"
        # spiketrains = neurons.spiketrains
        # neuron_ids = neurons.neuron_ids
        # n_neurons = neurons.n_neurons
        self.position_srate = position.sampling_rate
        # Set the dimensionality of the PfND object from the position's dimensionality
        self.ndim = position.ndim
        
        
        # Don't set these properties prematurely, filter first if needed       
        # self.t = position.time
        # t_start = position.t_start
        # t_stop = position.t_stop
        
        # self.x = position.x
        # if (position.ndim > 1):
        #     self.y = position.y
        
        
        # Output lists, for compatibility with Pf1D and Pf2D:
        spk_pos, spk_t, tuning_maps = [], [], []

        pos_df = position.to_dataframe().copy()
        # laps_df = active_epoch_session.laps.to_dataframe().copy()
        # spk_df = neurons.spikes_df.copy()
        spk_df = spikes_df.copy()

        # filtering:
        if epochs is not None:
            # filter the spikes_df:
            filtered_spikes_df = spk_df.spikes.time_sliced(epochs.starts, epochs.stops)
            # filter the pos_df:
            filtered_pos_df = pos_df.position.time_sliced(epochs.starts, epochs.stops) # 5378 rows × 18 columns
        else:
            # if no epochs filtering, set the filtered objects to be sliced by the available range of the position data (given by position.t_start, position.t_stop)
            filtered_spikes_df = spk_df.spikes.time_sliced(position.t_start, position.t_stop)
            filtered_pos_df = pos_df.position.time_sliced(position.t_start, position.t_stop)

        # Set animal observed position member variables:
        self.t = filtered_pos_df.t.to_numpy()
        self.x = filtered_pos_df.x.to_numpy()
        self.speed = filtered_pos_df.speed.to_numpy()
        if ((smooth is not None) and (smooth[0] > 0.0)):
            self.speed = gaussian_filter1d(self.speed, sigma=smooth[0])
        
        
        if (self.ndim > 1):
            self.y = filtered_pos_df.y.to_numpy()
        else:
            self.y = None
        
        
        ## Binning with Fixed Number of Bins:    
        # xbin, ybin, bin_info = _bin_pos_nD(self.x, self.y, num_bins=grid_num_bins) # num_bins mode:
        xbin, ybin, bin_info = _bin_pos_nD(self.x, self.y, bin_size=grid_bin) # bin_size mode
        
        # --- occupancy map calculation -----------
        if (position.ndim > 1):
            occupancy, xedges, yedges = Pf2D._compute_occupancy(self.x, self.y, xbin, ybin, self.position_srate, smooth)
        else:
            occupancy, xedges = Pf1D._compute_occupancy(self.x, xbin, self.position_srate, smooth[0])
        
        
        # Once filtering and binning is done, apply the grouping:
        # Group by the aclu (cluster indicator) column
        cell_grouped_spikes_df = filtered_spikes_df.groupby(['aclu'])
        cell_spikes_dfs = [cell_grouped_spikes_df.get_group(a_neuron_id) for a_neuron_id in filtered_spikes_df.spikes.neuron_ids] # a list of dataframes for each neuron_id

        # re-interpolate given the updated spks
        for cell_df in cell_spikes_dfs:
            cell_spike_times = cell_df['t_rel_seconds'].to_numpy()
            spk_spd = np.interp(cell_spike_times, self.t, self.speed)
            spk_x = np.interp(cell_spike_times, self.t, self.x)
            
            # update the dataframe 'x','speed' and 'y' properties:
            # cell_df.loc[:, 'x'] = spk_x
            # cell_df.loc[:, 'speed'] = spk_spd
            if (position.ndim > 1):
                spk_y = np.interp(cell_spike_times, self.t, self.y)
                # cell_df.loc[:, 'y'] = spk_y
                spk_pos.append([spk_x, spk_y])
                curr_cell_tuning_map = Pf2D._compute_tuning_map(spk_x, spk_y, xbin, ybin, occupancy, smooth)
            else:
                # otherwise only 1D:
                spk_pos.append([spk_x])
                curr_cell_tuning_map = Pf1D._compute_tuning_map(spk_x, xbin, occupancy, smooth[0])
            
            spk_t.append(cell_spike_times)
            # tuning curve calculation:               
            tuning_maps.append(curr_cell_tuning_map)
                
        # ---- cells with peak frate abouve thresh ------
        filtered_tuning_maps, filter_function = _filter_by_frate(tuning_maps.copy(), frate_thresh)

        filtered_neuron_ids = filter_function(filtered_spikes_df.spikes.neuron_ids)        
        filtered_tuple_neuron_ids = filter_function(filtered_spikes_df.spikes.neuron_probe_tuple_ids) # the (shank, probe) tuples corresponding to neuron_ids
        
        self.ratemap = Ratemap(
            filtered_tuning_maps, xbin=xbin, ybin=ybin, neuron_ids=filtered_neuron_ids
        )
        self.tuple_neuron_ids = filtered_tuple_neuron_ids
        
        self.ratemap_spiketrains = filter_function(spk_t)
        self.ratemap_spiketrains_pos = filter_function(spk_pos)
        self.occupancy = occupancy
        self.frate_thresh = frate_thresh
        self.speed_thresh = speed_thresh
        # done!
            