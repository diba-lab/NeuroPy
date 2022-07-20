import numpy as np
import pandas as pd


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


def bin_pos_nD(x: np.ndarray, y: np.ndarray, num_bins=None, bin_size=None):
        """ Spatially bins the provided x and y vectors into position bins based on either the specified num_bins or the specified bin_size
        Usage:
            ## Binning with Fixed Number of Bins:    
            xbin, ybin, bin_info = _bin_pos(pos_df.x.to_numpy(), pos_df.y.to_numpy(), bin_size=active_config.computation_config.grid_bin) # bin_size mode
            print(bin_info)
            ## Binning with Fixed Bin Sizes:
            xbin, ybin, bin_info = _bin_pos(pos_df.x.to_numpy(), pos_df.y.to_numpy(), num_bins=num_bins) # num_bins mode
            print(bin_info)
            
            
        TODO: 2022-04-22 - Note that I discovered that the bins generated here might cause an error when used with Pandas .cut function, which does not include the left (most minimum) values by default. This would cause the minimumal values not to be included.
        2022-07-20 - Extracted from PfND
        
        
        """
        assert (num_bins is None) or (bin_size is None), 'You cannot constrain both num_bins AND bin_size. Specify only one or the other.'
        assert (num_bins is not None) or (bin_size is not None), 'You must specify either the num_bins XOR the bin_size.'
        
        bin_info_out_dict = dict()
        
        if num_bins is not None:
            ## Binning with Fixed Number of Bins:
            mode = 'num_bins'
            if np.isscalar(num_bins):
                num_bins = [num_bins]
            
            xnum_bins = num_bins[0]
            xbin, xstep = np.linspace(np.nanmin(x), np.nanmax(x), num=xnum_bins, retstep=True)  # binning of x position

            if y is not None:
                ynum_bins = num_bins[1]
                ybin, ystep = np.linspace(np.nanmin(y), np.nanmax(y), num=ynum_bins, retstep=True)  # binning of y position       
                
        elif bin_size is not None:
            ## Binning with Fixed Bin Sizes:
            mode = 'bin_size'
            if np.isscalar(bin_size):
                print(f'np.isscalar(bin_size): {bin_size}')
                bin_size = [bin_size]
                
            xstep = bin_size[0]
            xbin = np.arange(np.nanmin(x), (np.nanmax(x) + xstep), xstep)  # binning of x position
            xnum_bins = len(xbin)

            if y is not None:
                ystep = bin_size[1]
                ybin = np.arange(np.nanmin(y), (np.nanmax(y) + ystep), ystep)  # binning of y position
                ynum_bins = len(ybin)
                
        # print('xbin: {}'.format(xbin))
        # print('ybin: {}'.format(ybin))
        bin_info_out_dict = {'mode':mode, 'xstep':xstep, 'xnum_bins':xnum_bins}
        if y is not None:
            # if at least 2D output, add the y-axis properties to the info dictionary
            bin_info_out_dict['ystep'], bin_info_out_dict['ynum_bins']  = ystep, ynum_bins
        else:
            ybin = None
            
        return xbin, ybin, bin_info_out_dict # {'mode':mode, 'xstep':xstep, 'ystep':ystep, 'xnum_bins':xnum_bins, 'ynum_bins':ynum_bins}



## Add Binned Position Columns to spikes_df:
def build_df_discretized_binned_position_columns(active_df, bin_values=(None, None), position_column_names = ('x', 'y'), binned_column_names = ('binned_x', 'binned_y'),
                                                 active_computation_config=None,
                                                 force_recompute=False, debug_print=False):
    """ Adds the columns specified in binned_column_names (e.g. ('binned_x', 'binned_y') columns to the passed-in dataframe
    Requires that the passed in dataframe has at least the 'x' column (1D) and optionally the 'y' column.
    Works for both position_df and spikes_df
    
    Should work for n-dimensional data
    
    Inputs:
        active_df: the dataframe to use
        bin_values: a tuple of independent np.arrays (e.g. (xbin_values, ybin_values)) specifying the complete bin_edges for both the x and y position spaces. If not provided, active_computation_config will be used to compute appropriate ones.
        position_column_names: a tuple of the independent position column names to be binned
        binned_column_names: a tuple of the output binned column names that will be added to the dataframe
        force_recompute: if True, the columns with names binned_column_names will be overwritten even if they already exist.
        
    Usage:
        active_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(active_pf_2D.filtered_spikes_df.copy(), bin_values=(active_pf_2D.xbin, active_pf_2D.ybin), active_computation_config=active_computation_config, force_recompute=False, debug_print=True)
        active_df
    
    ## TODO: Move into perminant location and replace duplicated/specific implementations with this more general version.
        Known Reimplementations:
            neuropy.analyses.time_dependent_placefields.__init__(...)
            General\Decoder\decoder_result.py - build_position_df_discretized_binned_positions(...)
    """
    ndim = len(bin_values)
    assert len(bin_values) == len(position_column_names) == len(binned_column_names), f"all input tuples should be of equal length (of the n position dimension dimensions to bin, e.g. ('x', 'y')) but len(bin_values): {len(bin_values)} == len(position_column_names): {len(position_column_names)} == len(binned_column_names): {len(binned_column_names)}"
    # See if we need any bin_values computed for any dimension:
    updated_bin_values = []
    updated_bin_infos = []
    for i, curr_dim_bin_values, curr_dim_position_col_name, curr_dim_binned_col_name  in zip(np.arange(ndim), bin_values, position_column_names, binned_column_names):
        if curr_dim_bin_values is None:
            # Compute needed:
            assert active_computation_config is not None, f"active_computation_config is required for its grid_bin parameter if incomplete bin_values are passed, but it is None!"
            curr_bins, _, bin_info = bin_pos_nD(active_df[curr_dim_position_col_name].values, None, bin_size=active_computation_config.grid_bin[i]) # bin_size mode, 1D
            if debug_print:
                print(f'computed new bins for dim {i}: bin_values[{i}].shape: {curr_bins.shape}')
        else:
            # Use the extant provided values:
            curr_bins = curr_dim_bin_values
            bin_info = None  # bin_info is None for pre-computed values
            if debug_print:
                print(f'using extant bins passed as arguments: bin_values[{i}].shape: {curr_bins.shape}')
            
        # Now we have the bin values in curr_bins:
        updated_bin_values.append(curr_bins)
        updated_bin_infos.append(bin_info)
        # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within:    
        if (curr_dim_binned_col_name not in active_df.columns) and not force_recompute:
            active_df[curr_dim_binned_col_name] = pd.cut(active_df[curr_dim_position_col_name].to_numpy(), bins=curr_bins, include_lowest=True, labels=np.arange(start=1, stop=len(curr_bins))) # same shape as the input data 
        
    return active_df, updated_bin_values, updated_bin_infos




# ==================================================================================================================== #


# def build_2D_df_discretized_binned_position_columns(active_df, xbin_values=None, ybin_values=None, active_computation_config=None,
#                                                  position_column_names = ('x', 'y'), binned_column_names = ('binned_x', 'binned_y'),
#                                                  force_recompute=False, debug_print=False):
#     """ Adds the 'binned_x' and 'binned_y' columns to the passed-in dataframe
#     Requires that the passed in dataframe has at least the 'x' column (1D) and optionally the 'y' column.
#     Works for both position_df and spikes_df
    
#     TODO: currently requires 2D positions (doesn't work for 1D)
    
#     Inputs:
    
#         xbin_values, ybin_values: np.arrays specifying the complete bin_edges for both the x and y position spaces. If not provided, active_computation_config will be used to compute appropriate ones.
        
#         position_column_names: a tuple of the independent position column names to be binned
#         binned_column_names: a tuple of the output binned column names that will be added to the dataframe
#         force_recompute: if True, the columns with names binned_column_names will be overwritten even if they already exist.
        
#     Usage:
#         active_df, xbin, ybin, bin_info = build_df_discretized_binned_position_columns(active_pf_2D.filtered_spikes_df.copy(), active_computation_config, xbin_values=active_pf_2D.xbin, ybin_values=active_pf_2D.ybin, force_recompute=False, debug_print=True)
#         active_df
    
#     ## TODO: Move into perminant location and replace duplicated/specific implementations with this more general version.
#         Known Reimplementations:
#             neuropy.analyses.time_dependent_placefields.__init__(...)
#             General\Decoder\decoder_result.py - build_position_df_discretized_binned_positions(...)
#     """
#     # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within.
#     if (xbin_values is None) or (ybin_values is None):
#         # determine the correct bins to use from active_computation_config.grid_bin:
#         if debug_print:
#             print(f'active_grid_bin: {active_computation_config.grid_bin}')

#         if position_column_names[1] in active_df.columns:
#             # 2D case:
#             # if ((binned_column_names[0] not in active_df.columns) or (binned_column_names[1] not in active_df.columns)) and not force_recompute:
#             xbin, ybin, bin_info = PfND._bin_pos_nD(active_df[position_column_names[0]].values, active_df[position_column_names[1]].values, bin_size=active_computation_config.grid_bin) # bin_size mode            
#         else:
#             # 1D case:
#             # if (binned_column_names[0] not in active_df.columns) and not force_recompute:
#             xbin, ybin, bin_info = PfND._bin_pos_nD(active_df[position_column_names[0]].values, None, bin_size=active_computation_config.grid_bin) # bin_size mode
#     else:
#         # use the extant values passed in:
#         if debug_print:
#             print(f'using extant bins passed as arguments: xbin_values.shape: {xbin_values.shape}, ybin_values.shape: {ybin_values.shape}')
#         xbin = xbin_values
#         ybin = ybin_values
#         bin_info = None

#     if (binned_column_names[0] not in active_df.columns) and not force_recompute:
#         active_df[binned_column_names[0]] = pd.cut(active_df[position_column_names[0]].to_numpy(), bins=xbin, include_lowest=True, labels=np.arange(start=1, stop=len(xbin))) # same shape as the input data 
#     if position_column_names[1] in active_df.columns:
#         # Only do the y-variables in the 2D case.
#         if (binned_column_names[1] not in active_df.columns) and not force_recompute:
#             active_df[binned_column_names[1]] = pd.cut(active_df[position_column_names[1]].to_numpy(), bins=ybin, include_lowest=True, labels=np.arange(start=1, stop=len(ybin))) 

#     return active_df, xbin, ybin, bin_info


# active_df, xbin, ybin, bin_info = build_df_discretized_binned_position_columns(active_pf_2D.filtered_spikes_df.copy(), active_computation_config, xbin_values=active_pf_2D.xbin, ybin_values=active_pf_2D.ybin, force_recompute=False, debug_print=True)
# active_df

