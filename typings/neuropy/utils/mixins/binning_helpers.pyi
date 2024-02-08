"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import Optional
from nptyping import NDArray
from dataclasses import dataclass

def find_minimum_time_bin_duration(epoch_durations: NDArray) -> float:
    """ determines the minimum time bin size that can be used to bin epochs with the provided durations.
    2024-01-25 - Used to require that the epoch was divisible into at least two bins. With my updated code it can handle the case where it's divisible into a single bin.
    
    Usage:
        from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
        min_possible_time_bin_size: float = find_minimum_time_bin_duration(global_replays['duration'].to_numpy())
        min_possible_time_bin_size
    """
    ...

def get_bin_centers(bin_edges):
    """ For a series of 1D bin edges given by bin_edges, returns the center of the bins. Output will have one less element than bin_edges. """
    ...

def get_bin_edges(bin_centers): # -> NDArray[Any]:
    """ For a series of 1D bin centers given by bin_centers, returns the edges of the bins. Output will have one more element than bin_centers
        Reciprocal of get_bin_centers(bin_edges)
    """
    ...

@dataclass
class BinningInfo:
    """ Factored out of pyphocorehelpers.indexing_helpers.BinningInfo """
    variable_extents: tuple
    step: float
    num_bins: int
    bin_indicies: np.ndarray
    ...


class BinningContainer:
    """A container that allows accessing either bin_edges (self.edges) or bin_centers (self.centers) 
    Factored out of pyphocorehelpers.indexing_helpers.BinningContainer
    """
    edges: np.ndarray
    centers: np.ndarray
    edge_info: BinningInfo
    center_info: BinningInfo
    @property
    def num_bins(self) -> int:
        ...
    
    def __init__(self, edges: Optional[np.ndarray] = ..., centers: Optional[np.ndarray] = ..., edge_info: Optional[BinningInfo] = ..., center_info: Optional[BinningInfo] = ...) -> None:
        ...
    
    @classmethod
    def build_edge_binning_info(cls, edges): # -> BinningInfo:
        ...
    
    @classmethod
    def build_center_binning_info(cls, centers, variable_extents): # -> BinningInfo:
        ...
    
    def setup_from_edges(self, edges: np.ndarray, edge_info: Optional[BinningInfo] = ...): # -> None:
        ...
    


def compute_spanning_bins(variable_values, num_bins: int = ..., bin_size: float = ..., variable_start_value: float = ..., variable_end_value: float = ...): # -> tuple[NDArray[floating[Any]], BinningInfo]:
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
    ...

def build_spanning_grid_matrix(x_values, y_values, debug_print=...): # -> tuple[NDArray[Any], list[tuple[Any, ...]], tuple[int, int]]:
    """ builds a 2D matrix with entries spanning x_values across axis 0 and spanning y_values across axis 1.
        
        For example, used to build a grid of position points from xbins and ybins.
    Usage:
        all_positions_matrix, flat_all_positions_matrix, original_data_shape = build_all_positions_matrix(active_one_step_decoder.xbin_centers, active_one_step_decoder.ybin_centers)
    """
    ...

class BinnedPositionsMixin:
    """ Adds common accessors for convenince properties such as *bin_centers/*bin_labels
    
    Requires (Implementor Must Provide):
        self.ndim
        self.xbin
        self.ybin
    
    Provides:
        Provided Properties:
            xbin_centers
            ybin_centers
            xbin_labels
            ybin_labels
            dims_coord_tuple
        
    """
    @property
    def xbin_centers(self):
        """ the x-position of the centers of each xbin. Note that there is (n_xbins - 1) of these. """
        ...
    
    @property
    def ybin_centers(self): # -> None:
        """ the y-position of the centers of each xbin. Note that there is (n_ybins - 1) of these. """
        ...
    
    @property
    def xbin_labels(self): # -> NDArray[signedinteger[Any]]:
        """ the labels of each xbin center. Starts at 1!"""
        ...
    
    @property
    def ybin_labels(self): # -> NDArray[signedinteger[Any]] | None:
        """ the labels of each ybin center. Starts at 1!"""
        ...
    
    @property
    def dims_coord_tuple(self): # -> tuple[int, int] | tuple[int]:
        """Returns a tuple containing the number of bins in each dimension. For 1D it will be (n_xbins,) for 2D (n_xbins, n_ybins) 
        TODO 2023-03-08 19:31: - [ ] Add to parent class (PfND) since it should have the same implementation.
        """
        ...
    


def bin_pos_nD(x: np.ndarray, y: np.ndarray, num_bins=..., bin_size=...): # -> tuple[Any | NDArray[Any], Any | NDArray[Any] | None, dict[str, Any]]:
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
    ...

def build_df_discretized_binned_position_columns(active_df, bin_values=..., position_column_names=..., binned_column_names=..., active_computation_config=..., force_recompute=..., debug_print=...): # -> tuple[Any, list[Any], dict[str, Any]]:
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
    ...

def transition_matrix(state_sequence, markov_order: int = ..., max_state_index: int = ...): # -> NDArray[Any]:
    """" Computes the transition matrix from Markov chain sequence of order `n`.
    See https://stackoverflow.com/questions/58048810/building-n-th-order-markovian-transition-matrix-from-a-given-sequence

    :param state_sequence: Discrete Markov chain state sequence in discrete time with states in 0, ..., N
    :param markov_order: Transition order

    :return: Transition matrix

    Usage:
        from neuropy.utils.mixins.binning_helpers import transition_matrix

        pf1D = deepcopy(curr_active_pipeline.computation_results['maze1'].computed_data['pf1D'])
        num_position_states = len(pf1D.xbin_labels)
        binned_x = pf1D.filtered_pos_df['binned_x'].to_numpy()
        binned_x_indicies = binned_x - 1
        binned_x_transition_matrix = transition_matrix(deepcopy(binned_x_indicies), markov_order=1, max_state_index=num_position_states)
        binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix, transition_matrix(deepcopy(binned_x_indicies), markov_order=2, max_state_index=num_position_states), transition_matrix(deepcopy(binned_x_indicies), markov_order=3, max_state_index=num_position_states)]

        ## Old method without using markov_order parameter:
        binned_x_transition_matrix[np.isnan(binned_x_transition_matrix)] = 0.0
        binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix, np.linalg.matrix_power(binned_x_transition_matrix, 2), np.linalg.matrix_power(binned_x_transition_matrix, 3)]

        ## Visualize Result:
        from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
        out = BasicBinnedImageRenderingWindow(binned_x_transition_matrix, pf1D.xbin_labels, pf1D.xbin_labels, name='binned_x_transition_matrix', title="Transition Matrix for binned x (from, to)", variable_label='Transition Matrix', scrollability_mode=LayoutScrollability.NON_SCROLLABLE)

        ## print the entries in the transition matrix:
        for row in binned_x_transition_matrix: print(' '.join(f'{x:.2f}' for x in row))


    TODO 2023-03-08 13:49: - [ ] 2D Placefield Position Transition Matrix
        pf2D = deepcopy(curr_active_pipeline.computation_results['maze1'].computed_data['pf2D'])
        # try to get the position sizes for the 2D placefields:
        original_position_data_shape = np.shape(pf2D.occupancy) # (63, 16)
        flat_position_size = np.shape(np.reshape(deepcopy(pf2D.occupancy), (-1, 1)))[0] # 1008
        print(f'{original_position_data_shape = }, {flat_position_size = }')
        num_position_states = int(float(len(pf2D.xbin_labels)) * float(len(pf2D.ybin_labels)))
        # num_position_states = flat_position_size
        print(f'{num_position_states = }')
        binned_x = pf2D.filtered_pos_df['binned_x'].to_numpy()
        binned_y = pf2D.filtered_pos_df['binned_y'].to_numpy()

        binned_x_indicies = binned_x - 1
        binned_y_indicies = binned_y - 1
        ## Reference: Method of getting all combinations from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldDensityAnalysisComputationFunctions.PlacefieldDensityAnalysisComputationFunctions._perform_placefield_overlap_computation
        ```python
            all_pairwise_neuron_IDs_combinations = np.array(list(itertools.combinations(computation_result.computed_data['pf2D_Decoder'].neuron_IDs, 2)))
            list_of_unit_pfs = [computation_result.computed_data['pf2D_Decoder'].pf.ratemap.normalized_tuning_curves[i,:,:] for i in computation_result.computed_data['pf2D_Decoder'].neuron_IDXs]
            all_pairwise_pfs_combinations = np.array(list(itertools.combinations(list_of_unit_pfs, 2)))
            # np.shape(all_pairwise_pfs_combinations) # (903, 2, 63, 63)
            all_pairwise_overlaps = np.squeeze(np.prod(all_pairwise_pfs_combinations, axis=1)) # multiply over the dimension containing '2' (multiply each pair of pfs).
        ```
        for a_row in pf2D.filtered_pos_df[['binned_x', 'binned_y']].itertuples():
            x_bin_idx, y_bin_idx = (a_row.binned_x-1), (a_row.binned_y-1)
            print(f'')

        # binned_x_transition_matrix = transition_matrix(deepcopy(binned_x))
        binned_x_transition_matrix = transition_matrix(deepcopy(binned_x_indicies), markov_order=1)
        binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix, transition_matrix(deepcopy(binned_x_indicies), markov_order=2), transition_matrix(deepcopy(binned_x_indicies), markov_order=3)]

    """
    ...
