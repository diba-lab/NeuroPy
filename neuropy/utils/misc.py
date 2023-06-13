import types
from collections import namedtuple
from enum import Enum, IntEnum, auto, unique
from itertools import islice
import numpy as np
import pandas as pd
from collections.abc import Iterable   # import directly from collections for Python < 3.3

import collections
import _collections_abc as cabc
import abc




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
    
def shuffle_ids(neuron_ids, seed:int=1337):
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
        return arr.item(*args)
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
    
def convert_dataframe_columns_to_datatype_if_possible(df: pd.DataFrame, datatype_str_column_names_list_dict, debug_print=False):
    """ If the columns specified in datatype_str_column_names_list_dict exist in the dataframe df, their type is changed to the key of the dict. See usage example below:
    
    Inputs:
        df: Pandas.DataFrame 
        datatype_str_column_names_list_dict: {'int':['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']}

    Usage:
        from neuropy.utils.misc import convert_dataframe_columns_to_datatype_if_possible
        convert_dataframe_columns_to_datatype_if_possible(curr_active_pipeline.sess.spikes_df, {'int':['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']})
    """
    # spikes_df[['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']] = spikes_df[['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']].astype('int') # convert integer calumns to correct datatype
    for a_datatype_name, a_column_names_list in datatype_str_column_names_list_dict.items():
        is_included = np.isin(a_column_names_list, df.columns)
        if debug_print:
            print(f'datatype: {a_datatype_name}: {a_column_names_list}:{is_included}')
        a_column_names_list = np.array(a_column_names_list)
        subset_extant_columns = a_column_names_list[is_included] # Get only the column names thare are included in the dataframe
        if debug_print:
            print(f'\t subset_extant_columns: {subset_extant_columns}')
        # subset_extant_columns = a_column_names_list
        df[subset_extant_columns] = df[subset_extant_columns].astype(a_datatype_name) # convert integer calumns to correct datatype


def add_explicit_dataframe_columns_from_lookup_df(df, lookup_properties_map_df, join_column_name='aclu'):
    """ Uses a value (specified by `join_column_name`) in each row of `df` to lookup the appropriate values in `lookup_properties_map_df` to be explicitly added as columns to `df`
    df: a dataframe. Each row has a join_column_name value (e.g. 'aclu')
    
    lookup_properties_map_df: a dataframe with one row for each `join_column_name` value (e.g. one row for each 'aclu', describing various properties of that neuron)
    
    
    By default lookup_properties_map_df can be obtained from curr_active_pipeline.sess.neurons._extended_neuron_properties_df and has the columns:
        ['aclu', 'qclu', 'cell_type', 'shank', 'cluster']
    Which will be added to the spikes_df
    
    WARNING: the df will be unsorted after this operation, and you'll need to sort it again if you want it sorted
    
    
    Usage:
        curr_active_pipeline.sess.flattened_spiketrains._spikes_df = add_explicit_dataframe_columns_from_lookup_df(curr_active_pipeline.sess.spikes_df, curr_active_pipeline.sess.neurons._extended_neuron_properties_df)
        curr_active_pipeline.sess.spikes_df.sort_values(by=['t_seconds'], inplace=True) # Need to re-sort by timestamps once done
        curr_active_pipeline.sess.spikes_df

    """
    ## only find the columns in lookup_properties_map_df that are NOT in df. e.g. ['qclu', 'cluster']
    missing_spikes_df_columns = list(lookup_properties_map_df.columns[np.logical_not(np.isin(lookup_properties_map_df.columns, df.columns))]) 
    missing_spikes_df_columns.insert(0, join_column_name) # Insert 'aclu' into the list so we can join on it
    subset_neurons_properties_df = lookup_properties_map_df[missing_spikes_df_columns] # get the subset dataframe with only the requested columns
    ## Merge the result:
    # df = pd.merge(subset_neurons_properties_df, df, on=join_column_name, how='outer', suffixes=('_neurons_properties', '_spikes_df'))
    return pd.merge(df, subset_neurons_properties_df, on=join_column_name, how='left', suffixes=('_neurons_properties', '_spikes_df'), copy=False) # avoids copying if possible

