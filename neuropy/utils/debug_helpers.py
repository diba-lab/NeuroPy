from copy import deepcopy
import itertools # required for parameter_sweeps
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Can safely include using
from typing import Callable
from neuropy.utils.mixins.dict_representable import overriding_dict_with # required for safely_accepts_kwargs

def inspect_callable_arguments(a_callable: Callable, debug_print=False):
    """ Not yet validated/implemented
    Progress:
        import inspect
        from neuropy.plotting.ratemaps import plot_ratemap_1D, plot_ratemap_2D

        fn_spec = inspect.getfullargspec(plot_ratemap_2D)
        fn_sig = inspect.signature(plot_ratemap_2D)
        ?fn_sig

            # fn_sig
        dict(fn_sig.parameters)
        # fn_sig.parameters.values()

        fn_sig.parameters['plot_mode']
        # fn_sig.parameters
        fn_spec.args # all kwarg arguments: ['x', 'y', 'num_bins', 'debug_print']

        fn_spec.defaults[-2].__class__.__name__ # a tuple of default values corresponding to each argument in args; ((64, 64), False)
    """
    import inspect
    full_fn_spec = inspect.getfullargspec(a_callable) # FullArgSpec(args=['item1', 'item2', 'item3'], varargs=None, varkw=None, defaults=(None, '', 5.0), kwonlyargs=[], kwonlydefaults=None, annotations={})
    # fn_sig = inspect.signature(compute_position_grid_bin_size)
    if debug_print:
        print(f'fn_spec: {full_fn_spec}')
    # fn_spec.args # ['item1', 'item2', 'item3']
    # fn_spec.defaults # (None, '', 5.0)

    num_positional_args = len(full_fn_spec.args) - len(full_fn_spec.defaults) # all kwargs have a default value, so if there are less defaults than args, than the first args must be positional args.
    positional_args_names = full_fn_spec.args[:num_positional_args] # [fn_spec.args[i] for i in np.arange(num_positional_args, )] np.arange(num_positional_args)
    kwargs_names = full_fn_spec.args[num_positional_args:] # [fn_spec.args[i] for i in np.arange(num_positional_args, )]
    if debug_print:
        print(f'fn_spec_positional_args_list: {positional_args_names}\nfn_spec_kwargs_list: {kwargs_names}')
    default_kwargs_dict = {argname:v for argname, v in zip(kwargs_names, full_fn_spec.defaults)} # {'item1': None, 'item2': '', 'item3': 5.0}

    return full_fn_spec, positional_args_names, kwargs_names, default_kwargs_dict

def safely_accepts_kwargs(fn):
    """ builds a wrapped version of fn that only takes the kwargs that it can use, and shrugs the rest off 
    Can be used as a decorator to make any function gracefully accept unhandled kwargs

    Can be used to conceptually "splat" a configuration dictionary of properties against a function that only uses a subset of them, such as might need to be done for plotting, etc)
    
    Usage:
        @safely_accepts_kwargs
        def _test_fn_with_limited_parameters(item1=None, item2='', item3=5.0):
            print(f'item1={item1}, item2={item2}, item3={item3}')
            
            
    TODO: Tests:
        from neuropy.utils.debug_helpers import safely_accepts_kwargs

        # def _test_fn_with_limited_parameters(newitem, item1=None, item2='', item3=5.0):
        #     print(f'item1={item1}, item2={item2}, item3={item3}')

        @safely_accepts_kwargs
        def _test_fn_with_limited_parameters(item1=None, item2='', item3=5.0):
            print(f'item1={item1}, item2={item2}, item3={item3}')

        @safely_accepts_kwargs
        def _test_fn2_with_limited_parameters(itemA=None, itemB='', itemC=5.0):
            print(f'itemA={itemA}, itemB={itemB}, itemC={itemC}')
            
        def _test_outer_fn(**kwargs):
            _test_fn_with_limited_parameters(**kwargs)
            _test_fn2_with_limited_parameters(**kwargs)
            # _test_fn_with_limited_parameters(**overriding_dict_with(lhs_dict=fn_spec_default_arg_dict, **kwargs))
            # _test_fn2_with_limited_parameters(**overriding_dict_with(lhs_dict=fn_spec_default_arg_dict, **kwargs))
            
            # Build safe versions of the functions
            # _safe_test_fn_with_limited_parameters = _build_safe_kwargs(_test_fn_with_limited_parameters)
            # _safe_test_fn2_with_limited_parameters = _build_safe_kwargs(_test_fn2_with_limited_parameters)
            # Call the safe versions:
            # _safe_test_fn_with_limited_parameters(**kwargs)
            # _safe_test_fn2_with_limited_parameters(**kwargs)
            
            
        # _test_outer_fn()
        _test_outer_fn(itemB=15) # TypeError: _test_fn_with_limited_parameters() got an unexpected keyword argument 'itemB'

    """
    full_fn_spec, positional_args_names, kwargs_names, default_kwargs_dict = inspect_callable_arguments(fn)
    def _safe_kwargs_fn(*args, **kwargs):
        return fn(*args, **overriding_dict_with(lhs_dict=default_kwargs_dict, **kwargs))
    return _safe_kwargs_fn



## Debug Printing:
def debug_print_ratemap(ratemap):
    # Get the cell IDs that have a good place field mapping:
    good_placefield_neuronIDs = np.array(ratemap.neuron_ids) # in order of ascending ID
    print('good_placefield_neuronIDs: {}; ({} good)'.format(good_placefield_neuronIDs, len(good_placefield_neuronIDs)))
    
    
def debug_print_placefield(active_epoch_placefield, short=True):
    # Get the cell IDs that have a good place field mapping:
    good_placefield_neuronIDs = np.array(active_epoch_placefield.ratemap.neuron_ids) # in order of ascending ID
    num_spikes_per_spiketrain = np.array([np.shape(a_spk_train)[0] for a_spk_train in active_epoch_placefield.spk_t])
    if short:
        print('good_placefield_neuronIDs: ({} good)'.format(len(good_placefield_neuronIDs)), end='\n')
        print('num_spikes: ({} total spikes)'.format(np.sum(num_spikes_per_spiketrain)), end='\n')
    else:
        print('good_placefield_neuronIDs: {}; ({} good)'.format(good_placefield_neuronIDs, len(good_placefield_neuronIDs)), end='\n')
        print('num_spikes: {}; ({} total spikes)'.format(num_spikes_per_spiketrain, np.sum(num_spikes_per_spiketrain)), end='\n')
    return pd.DataFrame({'neuronID':good_placefield_neuronIDs, 'num_spikes':num_spikes_per_spiketrain}).T

def debug_print_spike_counts(session):
    uniques, indicies, inverse_indicies, count_arr = np.unique(session.spikes_df['aclu'].values, return_index=True, return_inverse=True, return_counts=True)
    # count_arr = np.bincount(active_epoch_session.spikes_df['aclu'].values)
    print('active_epoch_session.spikes_df unique aclu values: {}'.format(uniques))
    print('active_epoch_session.spikes_df unique aclu value counts: {}'.format(count_arr))
    print(len(uniques)) # 69 
    uniques, indicies, inverse_indicies, count_arr = np.unique(session.spikes_df['fragile_linear_neuron_IDX'].values, return_index=True, return_inverse=True, return_counts=True)
    # count_arr = np.bincount(active_epoch_session.spikes_df['fragile_linear_neuron_IDX'].values)
    print('active_epoch_session.spikes_df unique fragile_linear_neuron_IDX values: {}'.format(uniques))
    print('active_epoch_session.spikes_df unique fragile_linear_neuron_IDX value counts: {}'.format(count_arr))
    print(len(uniques)) # 69 
    
    
def debug_plot_2d_binning(xbin, ybin, xbin_center, ybin_center):
    """ Displays the locations of the x & y bins and their center equivalents on a 2D plot
    Usage: 
        fig = debug_plot_2d_binning(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.xbin_centers, active_epoch_placefields2D.ratemap.ybin_centers)
    """
    fig = plt.Figure(figsize=(10, 10))
    xmin = min(np.min(xbin), np.min(xbin_center))
    xmax = max(np.max(xbin), np.max(xbin_center))
    ymin = min(np.min(ybin), np.min(ybin_center))
    ymax = max(np.max(ybin), np.max(ybin_center))
    
    plt.hlines(ybin, xmin, xmax, colors='b', linestyles='solid', label='ybin')
    plt.hlines(ybin_center, xmin, xmax, colors='cyan', linestyles='dashed', label='ybin_center')
    plt.vlines(xbin, ymin, ymax, colors='r', linestyles='solid', label='xbin')
    plt.vlines(xbin_center, ymin, ymax, colors='pink', linestyles='dashed', label='xbin_center')
    
    plt.xticks(xbin)
    plt.yticks(ybin)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    fig = plt.gcf()
    fig.set_dpi(180)
    return fig


def debug_print_subsession_neuron_differences(prev_session_Neurons, subsession_Neurons):
    num_original_neurons = prev_session_Neurons.n_neurons
    num_original_total_spikes = np.sum(prev_session_Neurons.n_spikes)
    num_subsession_neurons = subsession_Neurons.n_neurons
    num_subsession_total_spikes = np.sum(subsession_Neurons.n_spikes)
    print('{}/{} total spikes spanning {}/{} units remain in subsession'.format(num_subsession_total_spikes, num_original_total_spikes, num_subsession_neurons, num_original_neurons))


def print_aligned_columns(column_labels, column_values, pad_fill_str:str = ' ', enable_print:bool = True, enable_string_return:bool = False, enable_checking_all_values_width:bool=False):
    """ Prints a text representation of a table of values. All columns must have the same number of rows.

    enable_checking_all_values_width: if True, will compute the explicit width of the string representation of each element in the table and use the maximum of those widths as the width for that column. If False, will use the width of the column label as the width for that column.

    Usage:
        pad_fill_str = ' ' # the string to pad with
        list_of_names = ['frate_thresh', 'num_good_neurons', 'num_total_spikes']
        list_of_values = [frate_thresh_options, num_good_placefield_neurons_list, num_total_spikes_list]
        print_aligned_columns(list_of_names, list_of_values, pad_fill_str = ' ')

        Example Output:

        frate_thresh  num_good_neurons  num_total_spikes  
        0.0           70                58871             
        0.1           70                58871             
        1.0           65                57937             
        5.0           35                38800             
        10.0          14                20266             
        100.0         0                 0.0               


    """

    # align_command = lambda x, col_width, pad_fill_str: x.center(col_width, pad_fill_str)
    align_command = lambda x, col_width, pad_fill_str: x.ljust(col_width, pad_fill_str)
    extra_column_padding = 2 # add 2 spaces between columns

    num_rows_list = [len(v) for v in column_values]
    assert np.all(np.array(num_rows_list) == num_rows_list[0]), f"all lists must be the same length, but row lengths equal: {num_rows_list}"
    num_rows = num_rows_list[0] # all the same, so get the first one
    column_widths = [len(n)+extra_column_padding for n in column_labels] # add 1 to the length of each list name to get that column's width
    if enable_checking_all_values_width:
        max_col_values_width = [max(col_width, np.max([len(str(col_val[row_i]))+extra_column_padding for row_i in np.arange(num_rows)])) for col_width, col_val in zip(column_widths, column_values)]
        # print(f'{max_col_values_width = }')
        column_widths = max_col_values_width # update the column widths

    column_header_strings = [align_command(col_str, col_width, pad_fill_str) for col_width, col_str in zip(column_widths, column_labels)]
    aligned_header = ''.join(column_header_strings)
    if enable_print:
        print(aligned_header)
    if enable_string_return:
        aligned_row_strings_list = []
    for row_i in np.arange(num_rows):
        aligned_row = ''.join([align_command(str(col_val[row_i]), col_width, pad_fill_str) for col_width, col_val in zip(column_widths, column_values)])
        if enable_string_return:
            aligned_row_strings_list.append(aligned_row)
        if enable_print:
            print(aligned_row)
    if enable_string_return:
        return aligned_header, aligned_row_strings_list


def compare_placefields_info(output_pfs):
    """Compares a list of placefields

    Args:
        output_pfs (_type_): _description_

    Returns:
        _type_: _description_

    Usage:

        num_good_placefield_neurons_list, num_total_spikes_list, num_spikes_per_spiketrain_list = compare_placefields_info(output_pfs)

    """
    num_good_placefield_neurons_list = []
    num_spikes_per_spiketrain_list = []
    num_total_spikes_list = []

    for output_pf in output_pfs.values():
        # Get the cell IDs that have a good place field mapping:
        good_placefield_neuronIDs = np.array(output_pf.ratemap.neuron_ids) # in order of ascending ID
        num_good_placefield_neurons = len(good_placefield_neuronIDs)
        num_spikes_per_spiketrain = np.array([np.shape(a_spk_train)[0] for a_spk_train in output_pf.spk_t])
        num_total_spikes = np.sum(num_spikes_per_spiketrain)
        num_good_placefield_neurons_list.append(num_good_placefield_neurons)
        num_spikes_per_spiketrain_list.append(num_spikes_per_spiketrain)
        num_total_spikes_list.append(num_total_spikes)
        # debug_print_placefield(output_pf)

    return num_good_placefield_neurons_list, num_total_spikes_list, num_spikes_per_spiketrain_list




# ==================================================================================================================== #
# Parameter Sweeps                                                                                                     #
# ==================================================================================================================== #

def parameter_sweeps(**kwargs):
    """ Returns every unique combination of the passed in parameters. Superior to cartesian_product as it preserves the labels (returning a flat list of dicts) and accepts more than 2 inputs.
    
    Usage:
        from neuropy.utils.debug_helpers import parameter_sweeps
        all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(smooth=[(None, None), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)], grid_bin=[(1,1),(5,5),(10,10)])
        >> all_param_sweep_options:  [{'smooth': (None, None), 'grid_bin': (1, 1)}, {'smooth': (None, None), 'grid_bin': (5, 5)}, {'smooth': (None, None), 'grid_bin': (10, 10)},
        {'smooth': (0.5, 0.5), 'grid_bin': (1, 1)},  {'smooth': (0.5, 0.5), 'grid_bin': (5, 5)},  {'smooth': (0.5, 0.5), 'grid_bin': (10, 10)},
        {'smooth': (1.0, 1.0), 'grid_bin': (1, 1)}, ...]
        >> param_sweep_option_n_values: {'smooth': 5, 'grid_bin': 3}

        # !! SEE EXAMPLE BELOW `_compute_parameter_sweep` for usage of the returned values
        
    NOTE:
        Replaces:
            all_param_sweep_options = cartesian_product(grid_bin_options, smooth_options)
            param_sweep_option_n_values = dict(grid_bin=len(grid_bin_options), smooth=len(smooth_options)) 

    """
    all_param_sweep_options = []
    param_sweep_option_n_values = {k:len(v) for k, v in kwargs.items()}
    for values in itertools.product(*kwargs.values()):
        all_param_sweep_options.append(dict(zip(kwargs.keys(), values))) # Output dictionary
    return all_param_sweep_options, param_sweep_option_n_values
                      

""" USAGE EXAMPLE of `_compute_parameter_sweep`

def _compute_parameter_sweep(all_param_sweep_options: dict) -> dict:
    ''' Computes the PfNDs for all the swept parameters (combinations of grid_bin, smooth, etc)
    
    Usage:
        smooth_options = [(None, None), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
        grid_bin_options = [(1,1),(5,5),(10,10)]
        all_param_sweep_options = cartesian_product(smooth_options, grid_bin_options)
        param_sweep_option_n_values = dict(smooth=len(smooth_options), grid_bin=len(grid_bin_options)) 
        output_pfs = _compute_parameter_sweep(all_param_sweep_options)

    '''
    output_pfs = {} # empty dict

    for a_sweep_dict in all_param_sweep_options:
        a_sweep_tuple = frozenset(a_sweep_dict.items())
        output_pfs[a_sweep_tuple] = PfND(deepcopy(spikes_df).spikes.sliced_by_neuron_type('pyramidal'), deepcopy(active_pos.linear_pos_obj), **a_sweep_dict) # grid_bin=, etc
        
    return output_pfs


"""

def _plot_parameter_sweep(output_pfs, param_sweep_option_n_values, debug_print=False):
    """ Sweeps a 1D parameter for the placefields and plots it
    
    Usage:
        fig, axs = _plot_parameter_sweep(output_pfs, param_sweep_option_n_values)
        
    """
    if len(output_pfs)>0:        
        # remove any singleton variables
        formatting_included_items_list = [k for k, v in param_sweep_option_n_values.items() if v>1]
        
        if len(formatting_included_items_list) > 1:
            # more than one variable
            num_rows = list(param_sweep_option_n_values.values())[1] # get the first length TODO: check
        else:
            # only one variable
            num_rows = 1
        
        num_columns = len(output_pfs) // num_rows

        def _plot_title_formatter(x):
            printable_dict = {k:v for k, v in x if k in formatting_included_items_list}
            return f"{printable_dict}"
        
        plot_title_formatter = _plot_title_formatter
        
    else:
        return # empty
    
    if debug_print:
        print(f'{num_rows = }, {num_columns = }')
    fig, axs = plt.subplots(num_rows, num_columns, sharex=True);
    plt.subplots_adjust(top=0.968,bottom=0.05,left=0.021,right=0.993,hspace=0.2,wspace=0.116)
    # Flatten the axs array
    axs = axs.ravel()
    for i, (param_sweep_tuple, output_pf) in enumerate(output_pfs.items()):
        output_pf.plot_ratemaps_1D(ax=axs[i])
        axs[i].set_title(plot_title_formatter(param_sweep_tuple)) # TODO: display the parameter value without losing the number of good cells for each.
        if debug_print:
            debug_print_placefield(output_pf)
    return fig, axs

