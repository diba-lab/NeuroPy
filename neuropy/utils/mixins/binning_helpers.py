from typing import Optional
import numpy as np
import pandas as pd

from dataclasses import dataclass # for BinningInfo


def get_bin_centers(bin_edges):
    """ For a series of 1D bin edges given by bin_edges, returns the center of the bins. Output will have one less element than bin_edges. """
    return (bin_edges[:-1] + np.diff(bin_edges) / 2.0)
    
def get_bin_edges(bin_centers):
    """ For a series of 1D bin centers given by bin_centers, returns the edges of the bins. Output will have one more element than bin_centers
        Reciprocal of get_bin_centers(bin_edges)
    """
    bin_width = float((bin_centers[1] - bin_centers[0]))
    half_bin_width = bin_width / 2.0 # TODO: assumes fixed bin width
    bin_start_edges = bin_centers - half_bin_width
    last_edge_bin = bin_centers[-1] + half_bin_width # the last edge bin is one half_bin_width beyond the last bin_center
    out = bin_start_edges.tolist()
    out.append(last_edge_bin) # append the last_edge_bin to the bins.
    return np.array(out)

@dataclass
class BinningInfo(object):
    """ Factored out of pyphocorehelpers.indexing_helpers.BinningInfo """
    variable_extents: tuple
    step: float
    num_bins: int
    bin_indicies: np.ndarray
    

class BinningContainer(object):
    """A container that allows accessing either bin_edges (self.edges) or bin_centers (self.centers) 
    Factored out of pyphocorehelpers.indexing_helpers.BinningContainer
    """
    edges: np.ndarray
    centers: np.ndarray
    
    edge_info: BinningInfo
    center_info: BinningInfo
    
    def __init__(self, edges: Optional[np.ndarray]=None, centers: Optional[np.ndarray]=None, edge_info: Optional[BinningInfo]=None, center_info: Optional[BinningInfo]=None):
        super(BinningContainer, self).__init__()
        assert (edges is not None) or (centers is not None) # Require either centers or edges to be provided
        if edges is not None:
            self.edges = edges
        else:
            # Compute from edges
            self.edges = get_bin_edges(centers)
            
        if centers is not None:
            self.centers = centers
        else:
            self.centers = get_bin_centers(edges)
            
            
        if edge_info is not None:
            self.edge_info = edge_info
        else:
            # Otherwise try to reverse engineer edge_info:
            self.edge_info = BinningContainer.build_edge_binning_info(self.edges)
            
        if center_info is not None:
            self.center_info = center_info
        else:
            self.center_info = BinningContainer.build_center_binning_info(self.centers, self.edge_info.variable_extents)
            
            
    @classmethod
    def build_edge_binning_info(cls, edges):
        # Otherwise try to reverse engineer edge_info            
        actual_window_size = edges[2] - edges[1]
        variable_extents = [edges[0], edges[-1]] # get first and last values as the extents
        return BinningInfo(variable_extents, actual_window_size, len(edges), np.arange(len(edges)))
    
    @classmethod
    def build_center_binning_info(cls, centers, variable_extents):
        # Otherwise try to reverse engineer center_info
        actual_window_size = centers[2] - centers[1]
        return BinningInfo(variable_extents, actual_window_size, len(centers), np.arange(len(centers)))
    
    
    def setup_from_edges(self, edges: np.ndarray, edge_info: Optional[BinningInfo]=None):
        # Set the edges first:
        self.edges = edges
        if edge_info is not None:
            self.edge_info = edge_info # valid edge_info provided, use that
        else:
            # Otherwise try to reverse engineer edge_info:
            self.edge_info = BinningContainer.build_edge_binning_info(self.edges)
        
        ## Build the Centers from the new edges:
        self.centers = get_bin_centers(edges)
        self.center_info = BinningContainer.build_center_binning_info(self.centers, self.edge_info.variable_extents)
            
        

def compute_spanning_bins(variable_values, num_bins:int=None, bin_size:float=None, variable_start_value:float=None, variable_end_value:float=None):
    """Extracted from pyphocorehelpers.indexing_helpers import compute_position_grid_size for use in BaseDataSessionFormats


    Args:
        variable_values ([type]): The variables to be binned, used to determine the start and end edges of the returned bins.
        num_bins (int, optional): The total number of bins to create. Defaults to None.
        bin_size (float, optional): The size of each bin. Defaults to None.
        variable_start_value (float, optional): The minimum value of the binned variable. If specified, overrides the lower binned limit instead of computing it from variable_values. Defaults to None.
        variable_end_value (float, optional): The maximum value of the binned variable. If specified, overrides the upper binned limit instead of computing it from variable_values. Defaults to None.
        debug_print (bool, optional): Whether to print debug messages. Defaults to False.

    Raises:
        ValueError: [description]

    Returns:
        np.array<float>: The computed bins
        BinningInfo: information about how the binning was performed
        
    Usage:
        ## Binning with Fixed Number of Bins:    
        xbin_edges, xbin_edges_binning_info = compute_spanning_bins(pos_df.x.to_numpy(), bin_size=active_config.computation_config.grid_bin[0]) # bin_size mode
        print(xbin_edges_binning_info)
        ## Binning with Fixed Bin Sizes:
        xbin_edges_edges, xbin_edges_binning_info = compute_spanning_bins(pos_df.x.to_numpy(), num_bins=num_bins) # num_bins mode
        print(xbin_edges_binning_info)
        
    """
    assert (num_bins is None) or (bin_size is None), 'You cannot constrain both num_bins AND bin_size. Specify only one or the other.'
    assert (num_bins is not None) or (bin_size is not None), 'You must specify either the num_bins XOR the bin_size.'
    if variable_start_value is not None:
        curr_variable_min_extent = variable_start_value
    else:
        curr_variable_min_extent = np.nanmin(variable_values)
        
    if variable_end_value is not None:
        curr_variable_max_extent = variable_end_value
    else:
        curr_variable_max_extent = np.nanmax(variable_values)
        
    curr_variable_extents = (curr_variable_min_extent, curr_variable_max_extent)
    
    if num_bins is not None:
        ## Binning with Fixed Number of Bins:
        mode = 'num_bins'
        xnum_bins = num_bins
        xbin, xstep = np.linspace(curr_variable_extents[0], curr_variable_extents[1], num=num_bins, retstep=True)  # binning of x position
        
    elif bin_size is not None:
        ## Binning with Fixed Bin Sizes:
        mode = 'bin_size'
        xstep = bin_size
        xbin = np.arange(curr_variable_extents[0], (curr_variable_extents[1] + xstep), xstep, )  # binning of x position
        # the interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.
        xnum_bins = len(xbin)
        
    else:
        raise ValueError
    
    return xbin, BinningInfo(curr_variable_extents, xstep, xnum_bins, np.arange(xnum_bins))
      

def build_spanning_grid_matrix(x_values, y_values, debug_print=False):
    """ builds a 2D matrix with entries spanning x_values across axis 0 and spanning y_values across axis 1.
        
        For example, used to build a grid of position points from xbins and ybins.
    Usage:
        all_positions_matrix, flat_all_positions_matrix, original_data_shape = build_all_positions_matrix(active_one_step_decoder.xbin_centers, active_one_step_decoder.ybin_centers)
    """
    num_rows = len(y_values)
    num_cols = len(x_values)

    original_data_shape = (num_cols, num_rows) # original_position_data_shape: (64, 29)
    if debug_print:
        print(f'original_position_data_shape: {original_data_shape}')
    x_only_matrix = np.repeat(np.expand_dims(x_values, 1).T, num_rows, axis=0).T
    # np.shape(x_only_matrix) # (29, 64)
    flat_x_only_matrix = np.reshape(x_only_matrix, (-1, 1))
    if debug_print:
        print(f'np.shape(x_only_matrix): {np.shape(x_only_matrix)}, np.shape(flat_x_only_matrix): {np.shape(flat_x_only_matrix)}') # np.shape(x_only_matrix): (64, 29), np.shape(flat_x_only_matrix): (1856, 1)
    y_only_matrix = np.repeat(np.expand_dims(y_values, 1), num_cols, axis=1).T
    # np.shape(y_only_matrix) # (29, 64)
    flat_y_only_matrix = np.reshape(y_only_matrix, (-1, 1))

    # flat_all_positions_matrix = np.array([np.append(an_x, a_y) for (an_x, a_y) in zip(flat_x_only_matrix, flat_y_only_matrix)])
    flat_all_entries_matrix = [tuple(np.append(an_x, a_y)) for (an_x, a_y) in zip(flat_x_only_matrix, flat_y_only_matrix)] # a list of position tuples (containing two elements)
    # reconsitute its shape:
    all_entries_matrix = np.reshape(flat_all_entries_matrix, (original_data_shape[0], original_data_shape[1], 2))
    if debug_print:
        print(f'np.shape(all_positions_matrix): {np.shape(all_entries_matrix)}') # np.shape(all_positions_matrix): (1856, 2) # np.shape(all_positions_matrix): (64, 29, 2)
        print(f'flat_all_positions_matrix[0]: {flat_all_entries_matrix[0]}\nall_positions_matrix[0,0,:]: {all_entries_matrix[0,0,:]}')

    return all_entries_matrix, flat_all_entries_matrix, original_data_shape


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
            General.Decoder.decoder_result.py - build_position_df_discretized_binned_positions(...)
    """
    ndim = len(bin_values)
    assert len(bin_values) == len(position_column_names) == len(binned_column_names), f"all input tuples should be of equal length (of the n position dimension dimensions to bin, e.g. ('x', 'y')) but len(bin_values): {len(bin_values)} == len(position_column_names): {len(position_column_names)} == len(binned_column_names): {len(binned_column_names)}"
    # See if we need any bin_values computed for any dimension:
    updated_bin_values = []
    # updated_combined_bin_infos = {'mode': '', 'step': [], 'num_bins': []} # used to be 'xstep', 'ystep', 'xnum_bins', 'ynum_bins'
    updated_combined_bin_infos = {'mode': '', 'step': [], 'num_bins': [], 'xstep': None, 'ystep': None, 'xnum_bins': None, 'ynum_bins': None} # used to be 'xstep', 'ystep', 'xnum_bins', 'ynum_bins'
    
    for i, curr_dim_bin_values, curr_dim_position_col_name, curr_dim_binned_col_name in zip(np.arange(ndim), bin_values, position_column_names, binned_column_names):
        if curr_dim_bin_values is None:
            # Compute needed:
            assert active_computation_config is not None, f"active_computation_config is required for its grid_bin parameter if incomplete bin_values are passed, but it is None!"
            curr_bins, _, bin_info = bin_pos_nD(active_df[curr_dim_position_col_name].values, None, bin_size=active_computation_config.grid_bin[i]) # bin_size mode, 1D
            if debug_print:
                print(f'computed new bins for dim {i}: bin_values[{i}].shape: {curr_bins.shape}')
                
            updated_combined_bin_infos['mode'] = bin_info['mode']
            updated_combined_bin_infos['step'].append(bin_info['xstep'])
            updated_combined_bin_infos['num_bins'].append(bin_info['xnum_bins'])
            
        else:
            # Use the extant provided values:
            curr_bins = curr_dim_bin_values
            # bin_info = None  # bin_info is None for pre-computed values
            updated_combined_bin_infos['mode'] = 'provided'
            updated_combined_bin_infos['step'].append((curr_bins[1]-curr_bins[0]))
            updated_combined_bin_infos['num_bins'].append(len(curr_bins))
            
            if debug_print:
                print(f'using extant bins passed as arguments: bin_values[{i}].shape: {curr_bins.shape}')
        
        # Now we have the bin values in curr_bins:
        updated_bin_values.append(curr_bins)
        # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within:    
        if (curr_dim_binned_col_name not in active_df.columns) and not force_recompute:
            active_df[curr_dim_binned_col_name] = pd.cut(active_df[curr_dim_position_col_name].to_numpy(), bins=curr_bins, include_lowest=True, labels=np.arange(start=1, stop=len(curr_bins))) # same shape as the input data 
        
    ## Compatibility with prev implementations:
    # for compatibility, add 'xstep', 'ystep', 'xnum_bins', 'ynum_bins' for compatibility (for the first two variables being 'x' and 'y'
    if len(updated_combined_bin_infos['num_bins']) > 0:
        updated_combined_bin_infos['xstep'] = updated_combined_bin_infos['step'][0]
        updated_combined_bin_infos['xnum_bins'] = updated_combined_bin_infos['num_bins'][0]
    else:
        updated_combined_bin_infos['xstep'] = None
        updated_combined_bin_infos['xnum_bins'] = None
        
    if len(updated_combined_bin_infos['num_bins']) > 1:
        updated_combined_bin_infos['ystep'] = updated_combined_bin_infos['step'][1]
        updated_combined_bin_infos['ynum_bins'] = updated_combined_bin_infos['num_bins'][1]
    else:
        updated_combined_bin_infos['ystep'] = None
        updated_combined_bin_infos['ynum_bins'] = None
        
        
    return active_df, updated_bin_values, updated_combined_bin_infos

