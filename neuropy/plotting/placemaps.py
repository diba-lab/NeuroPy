import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from neuropy.plotting.figure import pretty_plot



    

def plot_all_placefields(active_placefields1D, active_placefields2D, active_config, variant_identifier_label=None, should_save_to_disk=True):
    """ Main function to plot all aspects of 1D and 2D placefields
    active_placefields1D: (Pf1D)
    active_placefields2D: (Pf2D)
    active_config:
    Usage:
        ax_pf_1D, occupancy_fig, active_pf_2D_figures = plot_all_placefields(active_epoch_placefields1D, active_epoch_placefields2D, active_config)
    """
    active_epoch_name = active_config.active_epochs.name
    common_parent_foldername = active_config.computation_config.str_for_filename(True)
    
    ## Linearized (1D) Position Placefields:
    if active_placefields1D is not None:
        ax_pf_1D = active_placefields1D.plot_ratemaps_1D(sortby='id') # by passing a string ('id') the plot_ratemaps_1D function chooses to use the normal range as the sort index (instead of sorting by the default)
        active_pf_1D_identifier_string = '1D Placefields - {}'.format(active_epoch_name)
        if variant_identifier_label is not None:
            active_pf_1D_identifier_string = ' - '.join([active_pf_1D_identifier_string, variant_identifier_label])

        # plt.title(active_pf_1D_identifier_string)
        # active_pf_1D_output_filename = '{}.pdf'.format(active_pf_1D_identifier_string)
        # active_pf_1D_output_filepath = active_config.plotting_config.active_output_parent_dir.joinpath(active_pf_1D_output_filename)
        
        title_string = ' '.join([active_pf_1D_identifier_string])
        subtitle_string = ' '.join([f'{active_placefields1D.config.str_for_display(False)}'])
        
        plt.gcf().suptitle(title_string, fontsize='14')
        plt.gca().set_title(subtitle_string, fontsize='10')
        # plt.title(active_pf_1D_identifier_string, fontsize=22)
        # common_parent_basename = active_placefields1D.config.str_for_filename(False)
        
        if should_save_to_disk:
            active_pf_1D_filename_prefix_string = f'Placefield1D-{active_epoch_name}'
            if variant_identifier_label is not None:
                active_pf_1D_filename_prefix_string = '-'.join([active_pf_1D_filename_prefix_string, variant_identifier_label])
            active_pf_1D_filename_prefix_string = f'{active_pf_1D_filename_prefix_string}-' # it always ends with a '-' character
            common_basename = active_placefields1D.str_for_filename(prefix_string=active_pf_1D_filename_prefix_string)
            active_pf_1D_output_filepath = active_config.plotting_config.get_figure_save_path(common_parent_foldername, common_basename).with_suffix('.png')
            print('Saving 1D Placefield image out to "{}"...'.format(active_pf_1D_output_filepath), end='')
            plt.savefig(active_pf_1D_output_filepath)
            print('\t done.')
            
    else:
        print('plot_all_placefields(...): active_epoch_placefields1D does not exist. Skipping it.')
        ax_pf_1D = None

    ## 2D Position Placemaps:
    if active_placefields2D is not None:
        # active_pf_occupancy_2D_identifier_string = '2D Occupancy - {}'.format(active_epoch_name)
        # if variant_identifier_label is not None:
        #     active_pf_occupancy_2D_identifier_string = ' - '.join([active_pf_occupancy_2D_identifier_string, variant_identifier_label])
                    
        # title_string = ' '.join([active_pf_occupancy_2D_identifier_string])
        # subtitle_string = ' '.join([f'{active_placefields2D.config.str_for_display(True)}'])
        # occupancy_fig, occupancy_ax = plot_placefield_occupancy(active_placefields2D)
        # occupancy_fig.suptitle(title_string, fontsize='14')
        # occupancy_ax.set_title(subtitle_string, fontsize='10')
        active_2D_occupancy_variant_identifier_list = [active_epoch_name]
        if variant_identifier_label is not None:
            active_2D_occupancy_variant_identifier_list.append(variant_identifier_label)
        occupancy_fig, occupancy_ax = active_placefields2D.plot_occupancy(identifier_details_list=active_2D_occupancy_variant_identifier_list)
        
        # Save ocupancy figure out to disk:
        if should_save_to_disk:
            active_2D_occupancy_filename_prefix_string = f'Occupancy-{active_epoch_name}'
            if variant_identifier_label is not None:
                active_2D_occupancy_filename_prefix_string = '-'.join([active_2D_occupancy_filename_prefix_string, variant_identifier_label])
            active_2D_occupancy_filename_prefix_string = f'{active_2D_occupancy_filename_prefix_string}-' # it always ends with a '-' character
            common_basename = active_placefields2D.str_for_filename(prefix_string=active_2D_occupancy_filename_prefix_string)
            active_pf_occupancy_2D_output_filepath = active_config.plotting_config.get_figure_save_path(common_parent_foldername, common_basename).with_suffix('.png')
            print('Saving 2D Placefield image out to "{}"...'.format(active_pf_occupancy_2D_output_filepath), end='')
            occupancy_fig.savefig(active_pf_occupancy_2D_output_filepath)
            print('\t done.')
            
        ## 2D Tuning Curves Figure:
        active_pf_2D_identifier_string = '2D Placefields - {}'.format(active_epoch_name)
        if variant_identifier_label is not None:
            active_pf_2D_identifier_string = ' - '.join([active_pf_2D_identifier_string, variant_identifier_label])
        title_string = ' '.join([active_pf_2D_identifier_string])
        subtitle_string = ' '.join([f'{active_placefields2D.config.str_for_display(True)}'])
        
        active_pf_2D_figures, active_pf_2D_gs = active_placefields2D.plot_ratemaps_2D(subplots=(80, 3), figsize=(30, 30))        
        # occupancy_fig.suptitle(title_string, fontsize='22')
        # occupancy_ax.set_title(subtitle_string, fontsize='16')

        if should_save_to_disk:
            active_pf_2D_filename_prefix_string = f'Placefields-{active_epoch_name}'
            if variant_identifier_label is not None:
                active_pf_2D_filename_prefix_string = '-'.join([active_pf_2D_filename_prefix_string, variant_identifier_label])
            active_pf_2D_filename_prefix_string = f'{active_pf_2D_filename_prefix_string}-' # it always ends with a '-' character
            common_basename = active_placefields2D.str_for_filename(prefix_string=active_pf_2D_filename_prefix_string)
            active_pf_2D_output_filepath = active_config.plotting_config.get_figure_save_path(common_parent_foldername, common_basename).with_suffix('.png')
            print('Saving 2D Placefield image out to "{}"...'.format(active_pf_2D_output_filepath), end='')
            for aFig in active_pf_2D_figures:
                aFig.savefig(active_pf_2D_output_filepath)
            print('\t done.')
    else:
        print('plot_all_placefields(...): active_epoch_placefields2D does not exist. Skipping it.')
        occupancy_fig = None
        active_pf_2D_figures = None
    
    return ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs


def plot_placefield_occupancy(active_epoch_placefields2D):
    return plot_occupancy_custom(active_epoch_placefields2D.occupancy, active_epoch_placefields2D.ratemap.xbin_centers, active_epoch_placefields2D.ratemap.ybin_centers, max_normalized=True, drop_below_threshold=1E-16)

def plot_occupancy_custom(occupancy, xbin, ybin, max_normalized: bool, drop_below_threshold: float=None, fig=None, ax=None):
    """ Plots a 2D Heatmap of the animal's occupancy (the amount of time the animal spent in each posiution bin)

    Args:
        occupancy ([type]): [description]
        xbin ([type]): [description]
        ybin ([type]): [description]
        max_normalized (bool): [description]
        drop_below_threshold (float, optional): [description]. Defaults to None.
        fig ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
    """
    if fig is None:
        occupancy_fig = plt.figure()
    else:
        occupancy_fig = fig
    
    if ax is None:
        occupancy_ax = occupancy_fig.gca()
    else:
        occupancy_ax = ax
        
    only_visited_occupancy = occupancy.copy()
    # print('only_visited_occupancy: {}'.format(only_visited_occupancy))
    if drop_below_threshold is not None:
        only_visited_occupancy[np.where(only_visited_occupancy < drop_below_threshold)] = np.nan
    if max_normalized:
        only_visited_occupancy = only_visited_occupancy / np.nanmax(only_visited_occupancy)
    im = occupancy_ax.pcolorfast(
        xbin,
        ybin,
        np.rot90(np.fliplr(only_visited_occupancy)),
        cmap="jet", vmin=0.0
    )  # rot90(flipud... is necessary to match plotRaw configuration.
    occupancy_ax.set_title('Custom Occupancy')
    occupancy_cbar = occupancy_fig.colorbar(im, ax=occupancy_ax, location='right')
    occupancy_cbar.minorticks_on()
    return occupancy_fig, occupancy_ax

def plot_occupancy_1D(active_epoch_placefields1D, max_normalized, drop_below_threshold=None, fig=None, ax=None):
    """ Draws an occupancy curve showing the relative proprotion of the recorded positions that occured in a given position bin. """
    should_fill = False
    
    if fig is None:
        occupancy_fig = plt.figure()
    else:
        occupancy_fig = fig
    
    if ax is None:
        occupancy_ax = occupancy_fig.gca()
    else:
        occupancy_ax = ax

    only_visited_occupancy = active_epoch_placefields1D.occupancy.copy()
    # print('only_visited_occupancy: {}'.format(only_visited_occupancy))
    if drop_below_threshold is not None:
        only_visited_occupancy[np.where(only_visited_occupancy < drop_below_threshold)] = np.nan
    
    if max_normalized:
        only_visited_occupancy = only_visited_occupancy / np.nanmax(only_visited_occupancy)
        
    if should_fill:
        occupancy_ax.plot(active_epoch_placefields1D.ratemap.xbin_centers, only_visited_occupancy)
        occupancy_ax.scatter(active_epoch_placefields1D.ratemap.xbin_centers, only_visited_occupancy, color='r')
    
    occupancy_ax.stairs(only_visited_occupancy, active_epoch_placefields1D.ratemap.xbin, fill=False, label='1D Placefield Occupancy', hatch='//') # can also use: , orientation='horizontal'
    
    occupancy_ax.set_ylim([0, np.nanmax(only_visited_occupancy)])
    
    # specify bin_size, etc
    
    occupancy_ax.set_title('Occupancy 1D')
    
    
    return occupancy_fig, occupancy_ax
