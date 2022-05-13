from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Can safely include using

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

