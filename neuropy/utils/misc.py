import types
from collections import namedtuple
from enum import Enum, IntEnum, auto, unique
from itertools import islice
from typing import Optional, Tuple
import numpy as np
from nptyping import NDArray
import pandas as pd
from collections.abc import Iterable   # import directly from collections for Python < 3.3

import collections
import _collections_abc as cabc
import abc

from datetime import datetime
from enum import unique, Enum



## Solution from Alexander McFarlane, https://stackoverflow.com/questions/1055360/how-to-tell-a-variable-is-iterable-but-not-a-string. answered Jun 30 '20 at 13:25
class NonStringIterable(metaclass=abc.ABCMeta):
    __slots__ = ()

    @abc.abstractmethod
    def __iter__(self):
        while False:
            yield None

    @classmethod
    def __subclasshook__(cls, c):
        if cls is NonStringIterable:
            if issubclass(c, str):
                return False
            return cabc._check_methods(c, "__iter__")
        return NotImplemented
    

def is_iterable(value):
    """Returns true if the value is iterable but not a string.
    Args:
        value ([type]): [description]
    Returns:
        [type]: [description]
    
    Usage:
        from neuropy.utils.misc import is_iterable

    """
    return isinstance(value, NonStringIterable) # use Alexander McFarlane's solution.
    # return isinstance(value, Iterable) # this version works but neglects classes that are iterable through __getitem__




class AutoNameEnum(Enum):
    """ Inheriting enums will be able to auto generate their name from a string value.

    Usage:
        class Ordinal(AutoNameEnum):
            NORTH = auto()
            SOUTH = auto()
            EAST = auto()
            WEST = auto()
    """
    def _generate_next_value_(name, start, count, last_values):
        return name


def chunks(iterable, size=10):
    """[summary]

    Args:
        iterable ([type]): [description]
        size (int, optional): [description]. Defaults to 10.

    Usage:
        laps_pages = [list(chunk) for chunk in _chunks(sess.laps.lap_id, curr_num_subplots)]
    """
    iterator = iter(iterable)
    for first in iterator:    # stops when iterator is depleted
        def chunk():          # construct generator for next chunk
            yield first       # yield element from for loop
            for more in islice(iterator, size - 1):
                yield more    # yield more elements from the iterator
        yield chunk()         # in outer generator, yield next chunk


RowColTuple = namedtuple('RowColTuple', 'num_rows num_columns')
PaginatedGridIndexSpecifierTuple = namedtuple('PaginatedGridIndexSpecifierTuple', 'linear_idx row_idx col_idx data_idx')
RequiredSubplotsTuple = namedtuple('RequiredSubplotsTuple', 'num_required_subplots num_columns num_rows combined_indicies')

def compute_paginated_grid_config(num_required_subplots, max_num_columns, max_subplots_per_page, data_indicies=None, last_figure_subplots_same_layout=True, debug_print=False):
    """ Fills row-wise first 

    Args:
        num_required_subplots ([type]): [description]
        max_num_columns ([type]): [description]
        max_subplots_per_page ([type]): [description]
        data_indicies ([type], optional): your indicies into your original data that will also be accessible in the main loop. Defaults to None, in which case they will be the same as the linear indicies unless otherwise specified
    """
    
    def _compute_subplots_grid_layout(num_page_required_subplots, page_max_num_columns):
        """ For a single page """
        fixed_columns = min(page_max_num_columns, num_page_required_subplots) # if there aren't enough plots to even fill up a whole row, reduce the number of columns
        needed_rows = int(np.ceil(num_page_required_subplots / fixed_columns))
        return RowColTuple(needed_rows, fixed_columns)
    
    
    def _compute_num_subplots(num_required_subplots, max_num_columns, data_indicies=None):
        linear_indicies = np.arange(num_required_subplots)
        if data_indicies is None:
            data_indicies = np.arange(num_required_subplots) # the data_indicies are just the same as the linear indicies unless otherwise specified
        (total_needed_rows, fixed_columns) = _compute_subplots_grid_layout(num_required_subplots, max_num_columns)
        all_row_column_indicies = np.unravel_index(linear_indicies, (total_needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
        all_combined_indicies = [PaginatedGridIndexSpecifierTuple(linear_indicies[i], all_row_column_indicies[0][i], all_row_column_indicies[1][i], data_indicies[i]) for i in np.arange(len(linear_indicies))]
        return RequiredSubplotsTuple(num_required_subplots, fixed_columns, total_needed_rows, all_combined_indicies)

    subplot_no_pagination_configuration = _compute_num_subplots(num_required_subplots, max_num_columns=max_num_columns, data_indicies=data_indicies)
    included_combined_indicies_pages = [list(chunk) for chunk in chunks(subplot_no_pagination_configuration.combined_indicies, max_subplots_per_page)]
    
    if last_figure_subplots_same_layout:
        page_grid_sizes = [RowColTuple(subplot_no_pagination_configuration.num_rows, subplot_no_pagination_configuration.num_columns) for a_page in included_combined_indicies_pages]
        
    else:
        page_grid_sizes = [_compute_subplots_grid_layout(len(a_page), subplot_no_pagination_configuration.num_columns) for a_page in included_combined_indicies_pages]

    if debug_print:
        print(f'page_grid_sizes: {page_grid_sizes}')
    return subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes



def get_interval(self, period, nwindows):
import math
from collections.abc import Iterable
from itertools import chain


def find_nearest(array, value):
    """https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return array[idx - 1]
    else:
        return array[idx]


def arg_find_nearest(array, value):
    """https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def get_interval(period, nwindows):

    interval = np.linspace(period[0], period[1], nwindows + 1)
    interval = [[interval[i], interval[i + 1]] for i in range(nwindows)]
    return interval


def print_seconds_human_readable(seconds):
    """ prints the seconds arguments as a human-redable HH::MM:SS.FRACTIONAL time. """
    if isinstance(seconds, int):
        whole_seconds = seconds
        fractional_seconds = None
    else:    
        whole_seconds = int(seconds)
        fractional_seconds = seconds - whole_seconds
    
    m, s = divmod(whole_seconds, 60)
    h, m = divmod(m, 60)
    timestamp = '{0:02}:{1:02}:{2:02}'.format(h, m, s)
    if fractional_seconds is not None:
        frac_seconds_string = ('%f' % fractional_seconds).rstrip('0').rstrip('.').lstrip('0').lstrip('.') # strips any insignficant zeros from the right, and then '0.' string from the left.        
        timestamp = '{}:{}'.format(timestamp, frac_seconds_string) # append the fracitonal seconds string to the timestamp string
    print(timestamp) # print the timestamp
    return h, m, s, fractional_seconds



def copy_if_not_none(val):
    """ solves the problem of AttributeError: 'NoneType' object has no attribute 'copy', gracefully passing None through if the value is None and copying it otherwise. """
    if val is not None:
        return val.copy()
    else:
        return None
    
def shuffle_ids(neuron_ids, seed:Optional[int]=None):
    """ Shuffles the neuron_ids list, and returns the shuffled list and the shuffle indicies. The shuffle indicies can be used to shuffle other lists in the same way. 

    Input:

        neuron_ids: a list of neuron ids to shuffle
    
    Usage:
        from neuropy.utils.misc import shuffle_ids
        shuffled_aclus, shuffle_IDXs = shuffle_ids(original_1D_decoder.neuron_IDs)
    """
    import random
    if not isinstance(neuron_ids, np.ndarray):
        neuron_ids = np.array(neuron_ids)
    shuffle_IDXs = list(range(len(neuron_ids)))
    random.Random(seed).shuffle(shuffle_IDXs) # shuffle the list of indicies
    shuffle_IDXs = np.array(shuffle_IDXs)
    return neuron_ids[shuffle_IDXs], shuffle_IDXs



def build_shuffled_ids(neuron_ids, num_shuffles: int = 1000, seed:Optional[int]=None, debug_print=False) -> Tuple[np.ndarray, np.ndarray]:
	""" Builds `num_shuffles` of the neuron_ids and returns both shuffled_aclus and shuffled_IDXs
	
	Uses numpy 2023-10-20 best practices for random number generation.
	
	Shuffled.
    
    Returns:
        shuffled_aclus.shape # .shape: (num_shuffles, n_neurons)
        shuffled_IDXs.shape # .shape: (num_shuffles, n_neurons)
        
        
    Usage:
        from neuropy.utils.misc import build_shuffled_ids

        num_shuffles = 1000
        shuffled_aclus, shuffled_IDXs = build_shuffled_ids(shared_aclus_only_neuron_IDs, num_shuffles=num_shuffles, seed=1337) # .shape: ((num_shuffles, n_neurons), (num_shuffles, n_neurons))

	"""
	rng = np.random.default_rng(seed=seed)
	
	shuffled_IDXs = np.tile(np.arange(len(neuron_ids)), (num_shuffles, 1)) # not shuffled yet, just duplicated because shuffling a multidim array only occurs along the first axis.
	shuffled_aclus = np.tile(neuron_ids, (num_shuffles, 1)) # not shuffled yet, just duplicated because shuffling a multidim array only occurs along the first axis.
	for i in np.arange(num_shuffles):
		# shuffle in place
		rng.permuted(shuffled_IDXs[i], axis=0, out=shuffled_IDXs[i])
		shuffled_aclus[i,:] = shuffled_aclus[i,:][shuffled_IDXs[i]] # sort the row's aclus by the shuffled indicies

	if debug_print:
		# shuffled_aclus.shape # .shape: (num_shuffles, n_neurons)
		print(f'shuffled_IDXs.shape: {np.shape(shuffled_IDXs)}')
	return shuffled_aclus, shuffled_IDXs



# ==================================================================================================================== #
# Dictionary Helpers                                                                                                   #
# ==================================================================================================================== #
def split_list_of_dicts(list_of_dicts: list) -> dict:
    """ Converts of a list<dict> (a list of dictionaries) where each element dictionary has the same keys to a dictionary of equal-length lists.
    
    Input:
        [{'smooth': (None, None), 'grid_bin': (0.5, 0.5)},
         {'smooth': (None, None), 'grid_bin': (1.0, 1.0)},
         {'smooth': (None, None), 'grid_bin': (2.0, 2.0)},
         {'smooth': (None, None), 'grid_bin': (5.0, 5.0)},
         {'smooth': (0.5, 0.5), 'grid_bin': (0.5, 0.5)},
         {'smooth': (0.5, 0.5), 'grid_bin': (1.0, 1.0)},
         {'smooth': (0.5, 0.5), 'grid_bin': (2.0, 2.0)},
         {'smooth': (0.5, 0.5), 'grid_bin': (5.0, 5.0)},
         {'smooth': (1.0, 1.0), 'grid_bin': (0.5, 0.5)},
         {'smooth': (1.0, 1.0), 'grid_bin': (1.0, 1.0)},
         {'smooth': (1.0, 1.0), 'grid_bin': (2.0, 2.0)},
         {'smooth': (1.0, 1.0), 'grid_bin': (5.0, 5.0)}]

    from neuropy.utils.misc import split_list_of_dicts
    split_list_of_dicts(all_param_sweep_options)

    Output:
        {'smooth': [(None, None), (None, None), (None, None), (None, None), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)], 
         'grid_bin': [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
         }

    """
    # Extract the keys from the first dictionary in the list
    keys = list(list_of_dicts[0].keys())

    # ALTERNATIVE: Use a list comprehension to extract the values for each key and zip them together
    # list_of_lists = [list(x) for x in zip(*[d.values() for d in list_of_dicts])]
    
    # Initialize the output list of lists
    list_of_lists = [[] for _ in keys]

    # Loop over the dictionaries in the input list
    for d in list_of_dicts:
        # Loop over the keys and append the corresponding value to the appropriate list
        for i, k in enumerate(keys):
            list_of_lists[i].append(d[k])

    dict_of_lists = dict(zip(keys, list_of_lists))
    return dict_of_lists




# ==================================================================================================================== #
# Numpy Helpers                                                                                                        #
# ==================================================================================================================== #
def safe_item(arr: np.ndarray, *args, default=None):
    """ a version of .item() for ndarrays that returns the scalar if there's a single item in a list, otherwise returns default_value
    Usage:
        safe_item(np.array([0]), default=None) # 0
        safe_item(np.array([]), default=-1) # -1
    """
    try:
        return arr.item(*args)  #@IgnoreException 
    except ValueError as e:
        return default


def split_array(arr: np.ndarray, sub_element_lengths: np.ndarray) -> list:
    """ 2023-03-25 - Takes a numpy array `arr` of length N and splits it into len(sub_element_lengths) pieces where piece i has length sub_element_lengths[i].
    
    Args:
        arr (np.ndarray): Input numpy array of length N.
        sub_element_lengths (np.ndarray): Array of integers indicating the length of each sub-element.
        
    Returns:
        np.ndarray: A numpy array of shape (len(sub_element_lengths), sub_element_lengths[i]) containing the sub-elements.
        
    Raises:
        ValueError: If the sum of sub_element_lengths is not equal to N.

    Usage:
        from neuropy.utils.misc import split_array
        

    """
    if sum(sub_element_lengths) != len(arr):
        raise ValueError("Sum of sub-element lengths must be equal to the length of input array.")
    split_arr = []
    start_index = 0
    for length in sub_element_lengths:
        split_arr.append(arr[start_index:start_index+length])
        start_index += length
    return split_arr

def numpyify_array(sequences) -> NDArray:
    """
    Convert a list of sequences to a list of NumPy arrays. If the sequence
    is already a NumPy array, it is left as-is.

    Usage:

    from neuropy.utils.misc import numpyify_array

    
    """
    return np.array([np.array(s) if not isinstance(s, np.ndarray) else s for s in sequences])




# ==================================================================================================================== #
# Pandas Helpers                                                                                                       #
# ==================================================================================================================== #
def safe_pandas_get_group(dataframe_group, key):
    """ returns an empty dataframe if the key isn't found in the group.
    Usage:
        from neuropy.utils.misc import safe_pandas_get_group
        safe_pandas_get_group(grouped_rdf, False)
    """
    if key in dataframe_group.groups.keys():
        return dataframe_group.get_group(key)
    else:
        original_df = dataframe_group.obj
        return original_df.drop(original_df.index)
    

# ==================================================================================================================== #
# Date/Time Helpers                                                                                                    #
# ==================================================================================================================== #

@unique
class DateTimeFormat(Enum):
    """Converts between datetime and string
    
    Usage:
    
        from neuropy.utils.misc import DateTimeFormat
        
        now = datetime.now()

        # Convert datetime to string
        s = DateTimeFormat.WHOLE_SECONDS.datetime_to_string(now)
        print(s)

        # Convert string back to datetime
        dt = DateTimeFormat.WHOLE_SECONDS.string_to_datetime(s)
        print(dt)

    """
    WHOLE_SECONDS = "%Y-%m-%dT%H-%M-%S" # Format the date and time in ISO 8601 format, without fractional seconds
    FRACTIONAL_SECONDS = "%Y-%m-%dT%H-%M-%S.%f" # Format the date and time in ISO 8601 format, but replace the ':' (which is illegal in filenames) with '-'

    def datetime_to_string(self, dt: datetime) -> str:
        return dt.strftime(self.value)

    def string_to_datetime(self, s: str) -> datetime:
        return datetime.strptime(s, self.value)

    @property
    def now_string(self) -> str:
        """Get the current date and time as an appropriately formatted string
        Usage:
            from neuropy.utils.misc import DateTimeFormat
            DateTimeFormat.WHOLE_SECONDS.now_string
        """
        return self.datetime_to_string(datetime.now())
        
def flatten(list_in):
    """Flatten a ragged list of different sized lists into one continuous list.
    Unlike `flatten_all` this only flattens the top level."""
    return list(chain.from_iterable(list_in))

def flatten_all(xs):
    """Completely flattens an iterable of iterables into one long generator"""
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_all(x)
        else:
            yield x


def nan_helper(y: np.ndarray):
    """Helper to handle indices and logical indices of NaNs.
    From https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interp_nans(y: np.ndarray):
    """interpolate nans based on values on either side of them in an array! In case of 2d array
    will move along rows"""

    if y.ndim == 2:
        for idr, yrow in enumerate(y):
            y[idr] = interp_nans(yrow)
    else:

        nans, x = nan_helper(y)
        if not np.all(np.isnan(y)):
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y
