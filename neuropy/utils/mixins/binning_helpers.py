from typing import Dict, List, Tuple, Optional, Callable, Union, Any, NewType, TypeVar
from typing_extensions import TypeAlias
from copy import deepcopy
import numpy as np
from nptyping import NDArray
import pandas as pd
import attrs
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field, keys_only_repr, SimpleFieldSizesReprMixin
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin

    
def find_minimum_time_bin_duration(epoch_durations: NDArray) -> float:
    """ determines the minimum time bin size that can be used to bin epochs with the provided durations.
    2024-01-25 - Used to require that the epoch was divisible into at least two bins. With my updated code it can handle the case where it's divisible into a single bin.
    
    Usage:
        from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
        min_possible_time_bin_size: float = find_minimum_time_bin_duration(global_replays['duration'].to_numpy())
        min_possible_time_bin_size
    """
    # return float(int((np.nanmin(epoch_durations)/2.0) * 1000) / 1000.0) # rounded_down_value: ensure that the size is rounded down
    return float(int((np.nanmin(epoch_durations)/1.0) * 1000) / 1000.0) # rounded_down_value: ensure that the size is rounded down



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

@custom_define(slots=False, repr=False, eq=False)
class BinningInfo(SimpleFieldSizesReprMixin, HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ Factored out of pyphocorehelpers.indexing_helpers.BinningInfo
    
    2024-08-07: refactored from `dataclass` to attrs
    
    Removed `bin_indicies`
    
    Usage:
        from neuropy.utils.mixins.binning_helpers import BinningInfo, BinningContainer
    
    """
    variable_extents: Tuple = serialized_field(is_computable=False) # serialized_field
    step: float = serialized_attribute_field(is_computable=False)
    num_bins: int = serialized_attribute_field(is_computable=False)
    bin_indicies: NDArray = serialized_field(init=False, repr=False, is_computable=True, metadata={'shape':('num_bins',)}) # , eq=attrs.cmp_using(eq=np.array_equal)
    
    def _validate_variable_extents(self):
        variable_extents = deepcopy(self.variable_extents)
        if isinstance(variable_extents, NDArray):
            variable_extents_length = np.shape(variable_extents)[0]
        else:
            variable_extents_length = len(variable_extents)
        # assert variable_extents_length == 2, f"variable_extents should be of length 2 (start, end) but is instead:\n\t variable_extents: {variable_extents}\n\t np.shape(variable_extents): {np.shape(variable_extents)}" # find where the bad extents are being introduced here!
        assert variable_extents_length >= 2 , f"variable_extents should be of length 2 (start, end) but is instead:\n\t variable_extents: {variable_extents}\n\t variable_extents_length: {variable_extents_length}" # don't allow any less than 2
            
        if (variable_extents_length > 2):
            print(f'WARNING: fixing invalid variable_extents!\n\t invalid variable_extents: {variable_extents}\n\tupdated self.variable_extents: {self.variable_extents}')
            self.variable_extents = (variable_extents[0], variable_extents[-1]) # first and last values
            print(f'\t VARIABLE EXTENTS CHANGED!')
            variable_extents_length = len(self.variable_extents) # new length
        
        assert variable_extents_length == 2, f"variable_extents should be of length 2 (start, end) but is instead:\n\t variable_extents: {self.variable_extents}\n\t variable_extents_length: {variable_extents_length}"
        
        
    def __attrs_post_init__(self):
        """ validate and build bin_indicies """
        self._validate_variable_extents()
        self.bin_indicies = np.arange(self.num_bins)
        assert len(self.bin_indicies) == self.num_bins, f"len(self.bin_indicies): {len(self.bin_indicies)} does not equal self.num_bins: {self.num_bins}!! Something is wrong"
        # self.num_bins = len(self.bin_indicies) # update from bin_indicies
    
    
    # @property
    # def bin_indicies(self) -> NDArray:
    #     """bin_indicies - np.ndarray of time_bin_indicies that will be used for the produced output dataframe of the binned spike counts for each unit. (e.g. time_bin_indicies = time_window_edges_binning_info.bin_indicies[1:])."""
    #     # return np.arange(self.num_bins)
    #     return self.bin_indicies
    # @bin_indicies.setter
    # def bin_indicies(self, value: NDArray):
    #     self._bin_indicies = value
    
    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # super(BinningInfo, self).__init__() # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        self.__dict__.update(state)



@custom_define(slots=False, repr=False, eq=False)
class BinningContainer(SimpleFieldSizesReprMixin, HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """A container that allows accessing either bin_edges (self.edges) or bin_centers (self.centers) 
    Factored out of pyphocorehelpers.indexing_helpers.BinningContainer
    
    #TODO 2024-08-07 16:12: - [ ] Observing inconsistent values:
        a_time_bin_container: neuropy.utils.mixins.binning_helpers.BinningContainer
        │   ├── edges: numpy.ndarray  = [678.314 678.315 678.316 678.317 678.318 678.319 678.32 678.321 678.322 678.323 678.324 678.325 678.326 678.327 678.328 678.329 678.33 678.331 678.332 678.333 678.334 678.335 678.336 678.337 678.338 678.339 678.34 678.341 678.342 678.343 678.344 678.345 678.346 678.347 678.348... - (283,)
        │   ├── centers: numpy.ndarray  = [678.314] - (1,)
        │   ├── edge_info: neuropy.utils.mixins.binning_helpers.BinningInfo  = BinningInfo(variable_extents=[678.3138549999567, 678.59585499995], step=0.0009999999999763531, num_bins=283, bin_indicies=array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29...
            │   ├── variable_extents: list  = [678.3138549999567, 678.59585499995] - (2,)
            │   ├── step: numpy.float64  = 0.0009999999999763531
            │   ├── num_bins: int  = 283
            │   ├── bin_indicies: numpy.ndarray  = [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68 ... - (283,)
        │   ├── center_info: neuropy.utils.mixins.binning_helpers.BinningInfo  = BinningInfo(variable_extents=array([678.314, 678.315, 678.316, 678.317, 678.318, 678.319, 678.32, 678.321, 678.322, 678.323, 678.324, 678.325, 678.326, 678.327, 678.328, 678.329, 678.33, 678.331, 678.332, 678.333, 678.334, 678.335, 678.336, 678.337, 678.338, 678.339, 678.34, 6...
            │   ├── variable_extents: numpy.ndarray  = [678.314 678.315 678.316 678.317 678.318 678.319 678.32 678.321 678.322 678.323 678.324 678.325 678.326 678.327 678.328 678.329 678.33 678.331 678.332 678.333 678.334 678.335 678.336 678.337 678.338 678.339 678.34 678.341 678.342 678.343 678.344 678.345 678.346 678.347 678.348... - (283,)
            │   ├── step: float  = 0.0009999999999763531
            │   ├── num_bins: int  = 1
            │   ├── bin_indicies: numpy.ndarray  = [0] - (1,)
        

    Observations:
        clearly initialized from edges, since edge_info.variable_extents is correct and center_info.variable_extents is so messed up. I didn't know it could get that way!
    

    """
    edges: NDArray = serialized_field(repr=False, is_computable=True, metadata={'shape':('num_bins+1',)})
    centers: NDArray = serialized_field(repr=False, is_computable=True, metadata={'shape':('num_bins',)})
    
    edge_info: BinningInfo = serialized_field(is_computable=True)
    center_info: BinningInfo = serialized_field(is_computable=False)
    
    @property
    def num_bins(self) -> int:
        return self.center_info.num_bins
        # return len(self.centers)`


    @property
    def left_edges(self) -> NDArray:
        """ the left edges of the bins. len(right_edges) == len(centers) """
        return self.centers - (self.center_info.step/2.0)

    @property
    def right_edges(self) -> NDArray:
        """ the right edges of the bins. len(right_edges) == len(centers) """
        return self.centers + (self.center_info.step/2.0)

        
    

    def __init__(self, edges: Optional[NDArray]=None, centers: Optional[NDArray]=None, edge_info: Optional[BinningInfo]=None, center_info: Optional[BinningInfo]=None):
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
            
        ## build infos:
        if edge_info is not None:
            self.edge_info = edge_info
        else:
            # Otherwise try to reverse engineer edge_info:
            self.edge_info = BinningContainer.build_edge_binning_info(self.edges)
            
        if center_info is not None:
            self.center_info = center_info
        else:
            self.center_info = BinningContainer.build_center_binning_info(self.centers, variable_extents=self.edge_info.variable_extents)
            
            
    @classmethod
    def build_edge_binning_info(cls, edges: NDArray):
        # Otherwise try to reverse engineer edge_info
        try:
            actual_window_size = edges[2] - edges[1] # if at least 3 bins long, safer to use the 2nd and 3rd bin to determine the actual_window_size
        except IndexError:
            # If edges is smaller than size 3, use the only two we have
            assert len(edges) == 2
            actual_window_size = edges[1] - edges[0]
        except Exception as e:
            raise
        variable_extents = (edges[0], edges[-1]) # get first and last values as the extents
        return BinningInfo(variable_extents=variable_extents, step=actual_window_size, num_bins=len(edges))
    
    @classmethod
    def build_center_binning_info(cls, centers: NDArray, variable_extents):
        # Otherwise try to reverse engineer center_info
        assert len(centers) > 1, f"centers must be of at least length 2 to re-derive center_info, but it is of length {len(centers)}. centers: {centers}\n\tCannot continue!"
            
        try:
            # The very end bins can be slightly different sizes occasionally, so if our list is longer than length 2 use the differences in the points after the left end.
            actual_window_size = centers[2] - centers[1]
        except IndexError as e:
            # For lists of length 2, use the only bins we have
            actual_window_size = None
            assert len(centers) == 2, f"centers must be of at least length 2 to re-derive center_info, but it is of length {len(centers)}. centers: {centers}\n\tIndexError e: {e}"
            actual_window_size = centers[1] - centers[0]
        except Exception as e:
            raise
            # step = variable_extents.step # could use
    
        return BinningInfo(variable_extents=deepcopy(variable_extents), step=actual_window_size, num_bins=len(centers))
    
    @classmethod
    def init_from_edges(cls, edges: NDArray, edge_info: Optional[BinningInfo]=None) -> "BinningContainer":
        """ initializes from edges, overwritting everything else
        """
        # Set the edges first:
        edges = deepcopy(edges)
        if edge_info is not None:
            edge_info = deepcopy(edge_info) # valid edge_info provided, use that
        else:
            # Otherwise try to reverse engineer edge_info:
            edge_info = cls.build_edge_binning_info(edges)
        
        ## Build the Centers from the new edges:
        centers = get_bin_centers(edges)
        if (len(edges) == 1):
            assert (edge_info is not None), f"need `edge_info` to get extents for (len(edges) == 1) case"
            # Built using `edge_info.variable_extents` - have to manually build center_info from subsampled `bins` because it doesn't work with two or less entries.
            variable_extents = deepcopy(edge_info.variable_extents)
            center_info = BinningInfo(variable_extents=variable_extents, step=edge_info.step, num_bins=1) # num_bins == 1, just like when (len(reduced_time_bin_edges) == 2)                  
        elif len(edges) == 2:
            # have to manually build center_info from subsampled `bins` because it doesn't work with two or less entries.
            # And the bin center is just the middle of the epoch
            actual_window_size = float(edges[1] - edges[0]) # the actual (variable) bin size... #TODO 2024-08-07 18:50: - [ ] this might be the subsampled bin size
            center_info = BinningInfo(variable_extents=(edges[0], edges[-1]), step=actual_window_size, num_bins=1)
        else:
            # can do it like normal by calling `.build_center_binning_info(...)`:
            ## automatically computes reduced_time_bin_centers and both infos:
            center_info = cls.build_center_binning_info(centers, variable_extents=deepcopy(edge_info.variable_extents)) # BinningContainer.build_center_binning_info(centers, variable_extents=self.edge_info.variable_extents)
        
        return cls(edges=edges, edge_info=edge_info, centers=centers, center_info=center_info)

def compute_spanning_bins(variable_values, num_bins:int=None, bin_size:float=None, variable_start_value:float=None, variable_end_value:float=None) -> Tuple[NDArray, BinningInfo]:
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
    
    return xbin, BinningInfo(variable_extents=curr_variable_extents, step=xstep, num_bins=xnum_bins)
      

def build_spanning_grid_matrix(x_values, y_values, debug_print=False):
    """ builds a 2D matrix with entries spanning x_values across axis 0 and spanning y_values across axis 1.
        
        For example, used to build a grid of position points from xbins and ybins.
    Usage:
        from neuropy.utils.mixins.binning_helpers import build_spanning_grid_matrix
        all_positions_matrix, flat_all_positions_matrix, original_data_shape = build_all_positions_matrix(active_one_step_decoder.xbin_centers, active_one_step_decoder.ybin_centers)
    
    Outputs:
        all_positions_matrix: a 3D matrix # .shape # (num_cols, num_rows, 2)
        flat_all_positions_matrix: a list of 2-tuples of length num_rows * num_cols
        original_data_shape: a tuple containing the shape of the original data (num_cols, num_rows)
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
    def xbin_centers(self) -> NDArray:
        """ the x-position of the centers of each xbin. Note that there is (n_xbins - 1) of these. """
        return self.xbin[:-1] + np.diff(self.xbin) / 2

    @property
    def ybin_centers(self) -> Optional[NDArray]:
        """ the y-position of the centers of each xbin. Note that there is (n_ybins - 1) of these. """
        if self.ybin is None:
            return None
        else:
            return self.ybin[:-1] + np.diff(self.ybin) / 2

    @property
    def xbin_labels(self) -> NDArray:
        """ the labels of each xbin center. Starts at 1!"""
        return np.arange(start=1, stop=len(self.xbin)) # bin labels are 1-indexed, thus adding 1

    @property
    def ybin_labels(self) -> Optional[NDArray]:
        """ the labels of each ybin center. Starts at 1!"""
        if self.ybin is None:
            return None
        else:
            return np.arange(start=1, stop=len(self.ybin))

    @property
    def n_xbin_edges(self) -> int:
        """ the number of xbin edges. """
        return len(self.xbin) 

    @property
    def n_ybin_edges(self) -> Optional[int]:
        """ the number of ybin edges. """
        if self.ybin is None:
            return None
        else:
             return len(self.ybin)

    @property
    def n_xbin_centers(self) -> int:
        """ the number of xbin (centers). Note that there is (n_xbin_edges - 1) of these. """
        return (len(self.xbin) - 1) # the -1 is to get the counts for the centers only

    @property
    def n_ybin_centers(self) -> Optional[int]:
        """ the number of ybin (centers). Note that there is (n_ybin_edges - 1) of these. """
        if self.ybin is None:
            return None
        else:
             return (len(self.ybin) - 1) # the -1 is to get the counts for the centers only

            
    
    @property
    def dims_coord_tuple(self):
        """Returns a tuple containing the number of bins in each dimension. For 1D it will be (n_xbins,) for 2D (n_xbins, n_ybins) 
        TODO 2023-03-08 19:31: - [ ] Add to parent class (PfND) since it should have the same implementation.
        """
        n_xbins = len(self.xbin) - 1 # the -1 is to get the counts for the centers only
        if (self.ndim > 1):
            n_ybins = len(self.ybin) - 1 # the -1 is to get the counts for the centers only
            dims_coord_ist = (n_xbins, n_ybins)
        else:
            # 1D Only
            n_ybins = None # singleton dimension along this axis. Decide how we want to shape it.
            dims_coord_ist = (n_xbins,)
        return dims_coord_ist




def bin_pos_nD(x: NDArray, y: NDArray, num_bins=None, bin_size=None):
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
                # print(f'np.isscalar(bin_size): {bin_size}')
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
            bin_info_out_dict['ystep'], bin_info_out_dict['ynum_bins'] = ystep, ynum_bins
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
        from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
        
        active_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(active_pf_2D.filtered_spikes_df.copy(), bin_values=(active_pf_2D.xbin, active_pf_2D.ybin), active_computation_config=active_computation_config, force_recompute=False, debug_print=True)
        active_df
    
        
    Usage 1D (x-only):
        epochs_track_identity_marginal_df, (xbin, ), bin_infos = build_df_discretized_binned_position_columns(deepcopy(epochs_track_identity_marginal_df), bin_values=(deepcopy(active_pf_2D.xbin),),
                                                                                                            position_column_names = ('x_meas',),  binned_column_names = ('binned_x', ),
                                                                                                            force_recompute=False, debug_print=True)
    
    ## TODO: Move into perminant location and replace duplicated/specific implementations with this more general version.
        Known Reimplementations:
            neuropy.analyses.time_dependent_placefields.__init__(...)
            General.Decoder.decoder_result.py - build_position_df_discretized_binned_positions(...)
    """
    ndim = len(bin_values)
    if debug_print:
        print(f'ndim: {ndim}, position_column_names: {position_column_names}, binned_column_names: {binned_column_names}')
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
            updated_combined_bin_infos['step'].append((curr_bins[1]-curr_bins[0])) # IndexError: index 1 is out of bounds for axis 0 with size 1 -- only has 1 bin ... [0, ] - 2025-01-15 06:32 - SOLVED: this occured when I was accidentally specifying grid_bin_bounds as a single tuple instead of a pair of two tuples (for the x & y))
            updated_combined_bin_infos['num_bins'].append(len(curr_bins))
            
            if debug_print:
                print(f'using extant bins passed as arguments: bin_values[{i}].shape: {curr_bins.shape}')
        
        # Now we have the bin values in curr_bins:
        updated_bin_values.append(curr_bins)
        # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within:
        needs_update_bin_col: bool = (curr_dim_binned_col_name not in active_df.columns)
        if (force_recompute or needs_update_bin_col):
            if debug_print:
                print(f'\tadding binned column: "{curr_dim_binned_col_name}"')
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


## Transition Matrix Computations
def transition_matrix(state_sequence, markov_order:int=1, max_state_index:int=None, nan_entries_replace_value:Optional[float]=None, should_validate_normalization:bool=False):
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
    from sklearn.preprocessing import normalize
    
    assert markov_order > 0
    
    if max_state_index is None:
        print(f'WARNING: `max_state_index` is not provided, guessing from maximimum observed state in sequence!')
        max_state_index = max(state_sequence)
        num_states: int = 1 + max_state_index #number of states
    else:
        # use user-provided max_state_index:
        num_states: int = max_state_index + 1

    assert max(state_sequence) <= max_state_index, f"VIOLATED! max(state_sequence): {max(state_sequence)} <= max_state_index: {max_state_index}"
    assert max(state_sequence) < num_states, f"VIOLATED! max(state_sequence): {max(state_sequence)} < num_states: {num_states}"
    # assert 0 in state_sequence, f"does not contain zero! Make sure that it is not a 1-indexed sequence!"
    
    offset_idx: int = markov_order # the markov_order is how many elements ahead we should look

    # Note that zippping with an unequal number of elements means that the number of iterations will be limited to the minimum number of elements:
    # offset_idx: int = 2
    # np.shape(state_sequence):  (4769,)
    # np.shape(state_sequence[offset_idx:]):  (4767,)
    # len(list(zip(state_sequence, state_sequence[offset_idx:]))): 4767
    M = np.zeros(shape=(num_states, num_states))
    for (i, j) in zip(state_sequence, state_sequence[offset_idx:]):
        M[i, j] += 1
        
    # now convert to probabilities:
    ## NOTE: NaNs will occur when there are rows of all zeros, as when we compute the sum of the row it becomes zero, and thus we divide the whole row by zero giving a whole row of NaNs
    # Normalize matrix by rows
    T = normalize(M, axis=1, norm='l1')
    
    if nan_entries_replace_value is not None:
        # replace NaN entries in final output
        T[np.isnan(T)] = float(nan_entries_replace_value)
    
    # ## Check:
    if should_validate_normalization:
        ## test row normalization (only considering non-zero entries):
        _check_row_normalization_sum = np.sum(T, axis=1)
        _is_row_normalization_all_valid = np.allclose(_check_row_normalization_sum[np.nonzero(_check_row_normalization_sum)], 1.0)
        assert _is_row_normalization_all_valid, f"not row normalized!\n\t_is_row_normalization_all_valid: {_is_row_normalization_all_valid}\n\t_check_row_normalization_sum: {_check_row_normalization_sum}"

    return T
