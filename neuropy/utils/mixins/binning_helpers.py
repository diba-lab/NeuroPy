import numpy as np


class BinnedPositionsMixin(object):
    """ Adds common accessors for convenince properties such as *bin_centers/*bin_labels
    
    Requires (Implementor Must Provide):
        self.xbin
        self.ybin
    
    Provides:
        Provided Properties:
            xbin_centers
            ybin_centers
            xbin_labels
            ybin_labels
        
    """
    @property
    def xbin_centers(self):
        """ the x-position of the centers of each xbin. Note that there is (n_xbins - 1) of these. """
        return self.xbin[:-1] + np.diff(self.xbin) / 2

    @property
    def ybin_centers(self):
        """ the y-position of the centers of each xbin. Note that there is (n_ybins - 1) of these. """
        if self.ybin is None:
            return None
        else:
            return self.ybin[:-1] + np.diff(self.ybin) / 2

    @property
    def xbin_labels(self):
        """ the labels of each xbin center. Starts at 1!"""
        return np.arange(start=1, stop=len(self.xbin)) # bin labels are 1-indexed, thus adding 1

    @property
    def ybin_labels(self):
        """ the labels of each ybin center. Starts at 1!"""
        if self.ybin is None:
            return None
        else:
            return np.arange(start=1, stop=len(self.ybin))




## Add Binned Position Columns to spikes_df:
def build_df_discretized_binned_position_columns(active_df, active_computation_config, xbin_values=None, ybin_values=None, force_recompute=False, debug_print=False):
    """ Adds the 'binned_x' and 'binned_y' columns to the passed-in dataframe
    Requires that the passed in dataframe has at least the 'x' column (1D) and optionally the 'y' column.
    Works for both position_df and spikes_df
    
    Usage:
        active_df, xbin, ybin, bin_info = build_df_discretized_binned_position_columns(active_pf_2D.filtered_spikes_df.copy(), active_computation_config, xbin_values=active_pf_2D.xbin, ybin_values=active_pf_2D.ybin, force_recompute=False, debug_print=True)
        active_df
    
    ## TODO: Move into perminant location and replace duplicated/specific implementations with this more general version.
        Known Reimplementations:
            neuropy.analyses.time_dependent_placefields.__init__(...)
            General\Decoder\decoder_result.py - build_position_df_discretized_binned_positions(...)
    """
    # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within.
    if (xbin_values is None) or (ybin_values is None):
        # determine the correct bins to use from active_computation_config.grid_bin:
        if debug_print:
            print(f'active_grid_bin: {active_computation_config.grid_bin}')

        if 'y' in active_df.columns:
            # 2D case:
            # if (('binned_x' not in active_df.columns) or ('binned_y' not in active_df.columns)) and not force_recompute:
            xbin, ybin, bin_info = PfND._bin_pos_nD(active_df['x'].values, active_df['y'].values, bin_size=active_computation_config.grid_bin) # bin_size mode            
        else:
            # 1D case:
            # if ('binned_x' not in active_df.columns) and not force_recompute:
            xbin, ybin, bin_info = PfND._bin_pos_nD(active_df['x'].values, None, bin_size=active_computation_config.grid_bin) # bin_size mode
    else:
        # use the extant values passed in:
        if debug_print:
            print(f'using extant bins passed as arguments: xbin_values.shape: {xbin_values.shape}, ybin_values.shape: {ybin_values.shape}')
        xbin = xbin_values
        ybin = ybin_values
        bin_info = None

    if ('binned_x' not in active_df.columns) and not force_recompute:
        active_df['binned_x'] = pd.cut(active_df['x'].to_numpy(), bins=xbin, include_lowest=True, labels=np.arange(start=1, stop=len(xbin))) # same shape as the input data 
    if 'y' in active_df.columns:
        # Only do the y-variables in the 2D case.
        if ('binned_y' not in active_df.columns) and not force_recompute:
            active_df['binned_y'] = pd.cut(active_df['y'].to_numpy(), bins=ybin, include_lowest=True, labels=np.arange(start=1, stop=len(ybin))) 

    return active_df, xbin, ybin, bin_info


# active_df, xbin, ybin, bin_info = build_df_discretized_binned_position_columns(active_pf_2D.filtered_spikes_df.copy(), active_computation_config, xbin_values=active_pf_2D.xbin, ybin_values=active_pf_2D.ybin, force_recompute=False, debug_print=True)
# active_df

