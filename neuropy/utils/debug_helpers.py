from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Can safely include using
from typing import Callable
from neuropy.utils.dynamic_container import overriding_dict_with # required for safely_accepts_kwargs

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

