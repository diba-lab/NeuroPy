from __future__ import annotations # otherwise have to do type like 'Ratemap'

import enum
from ipywidgets import widgets
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

# from https://www.stefaanlippens.net/circular-imports-type-hints-python.html to avoid circular import issues
# also you must add the following line to the beginning of this file:
#   from __future__ import annotations # otherwise have to do type like 'Ratemap'
#
from typing import TYPE_CHECKING
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum
if TYPE_CHECKING:
    from neuropy.core.neuron_identities import NeuronExtendedIdentityTuple
    from neuropy.core.ratemap import Ratemap
    
from neuropy.utils import mathutil
from neuropy.utils.misc import RowColTuple, safe_item
from neuropy.utils.colors_util import get_neuron_colors
from neuropy.utils.matplotlib_helpers import _build_variable_max_value_label, add_inner_title, enumTuningMap2DPlotMode, _build_square_checkerboard_image, enumTuningMap2DPlotVariables, _determine_best_placefield_2D_layout, _scale_current_placefield_to_acceptable_range, _build_neuron_identity_label
from neuropy.utils.debug_helpers import safely_accepts_kwargs
from .figure import Fig

def _add_points_to_plot(curr_ax, overlay_points, plot_opts=None, scatter_opts=None):
    """ Adds the overlay points to the image plot with the specified axis. 
    
    Usage:
        spike_overlay_points, spike_overlay_sc = _add_points_to_plot(curr_ax, spike_overlay_spikes[neuron_IDX], plot_opts={'markersize': 2, 'marker': ',', 'markeredgecolor': 'red', 'linestyle': 'none', 'markerfacecolor': 'red', 'alpha': 0.1, 'label': 'spike_overlay_points'},
                                                                             scatter_opts={'s': 2, 'c': 'white', 'alpha': 0.1, 'marker': ',', 'label': 'spike_overlay_sc'})
        
    """
    if plot_opts is None:
        plot_opts = {}
    if scatter_opts is None:
        scatter_opts = {}
        
    spike_overlay_points = curr_ax.plot(overlay_points[0], overlay_points[1], **({'markersize': 2, 'marker': ',', 'markeredgecolor': 'red', 'linestyle': 'none', 'markerfacecolor': 'red', 'alpha': 0.1, 'label': 'UNKNOWN_overlay_points'} | plot_opts))                
    spike_overlay_sc = curr_ax.scatter(overlay_points[0], overlay_points[1], **({'s': 2, 'c': 'white', 'alpha': 0.1, 'marker': ',', 'label': 'UNKNOWN_overlay_sc'} | scatter_opts))
    return spike_overlay_points, spike_overlay_sc


class BackgroundRenderingOptions(enum.Enum):
    PATTERN_CHECKERBOARD = 1
    SOLID_COLOR = 2
    EMPTY = 3


def _help_plot_ratemap_neuronIDs(ratemap: Ratemap, included_unit_indicies=None, included_unit_neuron_IDs=None, plot_variable: enumTuningMap2DPlotVariables=enumTuningMap2DPlotVariables.TUNING_MAPS, debug_print=False):
    """ Builds shared neuron_IDs
    Factored out of `plot_ratemap_2D(...) on 2022-11-22. 
    Usage:
        active_maps, title_substring, included_unit_indicies = _help_plot_ratemap_neuronIDs(ratemap, included_unit_indicies=included_unit_indicies, included_unit_neuron_IDs=included_unit_neuron_IDs, plot_variable=plot_variable, debug_print=debug_print)
    """
    ## Brought in from display_all_pf_2D_pyqtgraph_binned_image_rendering:
    if included_unit_neuron_IDs is not None:
        if debug_print:
            print(f'included_unit_neuron_IDs: {included_unit_neuron_IDs}')
        if not isinstance(included_unit_neuron_IDs, np.ndarray):
            included_unit_neuron_IDs = np.array(included_unit_neuron_IDs) # convert to np.array if needed

        n_neurons = np.size(included_unit_neuron_IDs)
        if debug_print:
            print(f'\t n_neurons: {n_neurons}')

        shared_IDXs_map = [safe_item(np.squeeze(np.argwhere(aclu == ratemap.neuron_ids)), default=None) for aclu in included_unit_neuron_IDs] # [0, 1, None, 2, 3, 4, 5, None, 6, 7, 8, None, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]

        if plot_variable.name is enumTuningMap2DPlotVariables.TUNING_MAPS.name:
            active_maps = ratemap.tuning_curves
            title_substring = 'Placemaps'
        elif plot_variable.name == enumTuningMap2DPlotVariables.SPIKES_MAPS.name:
            active_maps = ratemap.spikes_maps
            title_substring = 'Spikes Maps'
        else:
            raise ValueError

        ## Non-pre-build method where shared_IDXs_map is directly passed as included_unit_indicies so it's returned in the main loop:
        included_unit_indicies = shared_IDXs_map
        if debug_print:
            print(f'active_maps.shape: {np.shape(active_maps)}, type: {type(active_maps)}') # _local_active_maps.shape: (70, 63, 16), type: <class 'numpy.ndarray'>

    else:
        ## normal (non-shared mode)
        shared_IDXs_map = None
        active_maps = None

        if included_unit_indicies is None:
            included_unit_indicies = np.arange(ratemap.n_neurons) # include all unless otherwise specified
        
        ## Get Data to plot:
        if plot_variable.name is enumTuningMap2DPlotVariables.TUNING_MAPS.name:
            active_maps = ratemap.tuning_curves[included_unit_indicies]
            title_substring = 'Placemaps'
        elif plot_variable.name == enumTuningMap2DPlotVariables.SPIKES_MAPS.name:
            active_maps = ratemap.spikes_maps[included_unit_indicies]
            title_substring = 'Spikes Maps'
        else:
            raise ValueError

    return active_maps, title_substring, included_unit_indicies #, shared_IDXs_map

def _plot_single_tuning_map_2D(xbin, ybin, pfmap, occupancy, final_title_str=None, drop_below_threshold: float=0.0000001,
                              plot_mode: enumTuningMap2DPlotMode=None, ax=None, brev_mode=PlotStringBrevityModeEnum.CONCISE, max_value_formatter=None, use_special_overlayed_title:bool=True, bg_rendering_mode=BackgroundRenderingOptions.PATTERN_CHECKERBOARD):
    """Plots a single tuning curve Heatmap using matplotlib

    Args:
        xbin ([type]): [description]
        ybin ([type]): [description]
        pfmap ([type]): [description]
        occupancy ([type]): [description]
        drop_below_threshold (float, optional): [description]. Defaults to 0.0000001.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if plot_mode is None:
        plot_mode = enumTuningMap2DPlotMode.IMSHOW
    assert plot_mode is enumTuningMap2DPlotMode.IMSHOW, f"Plot mode should not be specified to anything other than None or enumTuningMap2DPlotMode.IMSHOW as of 2022-08-15 but value was: {plot_mode}"
    
    # use_special_overlayed_title = True
    
    # use_alpha_by_occupancy = False # Only supported in IMSHOW mode
    use_alpha_by_occupancy = False # Only supported in IMSHOW mode

    if ax is None:
        ax = plt.gca()
            
    curr_pfmap = _scale_current_placefield_to_acceptable_range(pfmap, occupancy=occupancy, drop_below_threshold=drop_below_threshold)     
    
    ## Seems to work:
    curr_pfmap = np.rot90(curr_pfmap, k=-1)
    curr_pfmap = np.fliplr(curr_pfmap)
        
    # # curr_pfmap = curr_pfmap / np.nanmax(curr_pfmap) # for when the pfmap already had its transpose taken

    """ https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html """
    """ TODO: Use the brightness to reflect the confidence in the outcome. Could also use opacity. """
    # mesh_X, mesh_Y = np.meshgrid(xbin, ybin)
    xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], ybin[0], ybin[-1])
    # The extent keyword arguments controls the bounding box in data coordinates that the image will fill specified as (left, right, bottom, top) in data coordinates, the origin keyword argument controls how the image fills that bounding box, and the orientation in the final rendered image is also affected by the axes limits.
    extent = (xmin, xmax, ymin, ymax)
    # vmax = np.abs(curr_pfmap).max()
            
    imshow_shared_kwargs = {
        'origin': 'lower',
        'extent': extent,
    }
    
    main_plot_kwargs = imshow_shared_kwargs | {
        # 'vmax': vmax,
        'vmin': 0,
        'cmap': 'jet',
    }
    
    if use_alpha_by_occupancy:
        # alphas = np.ones(curr_pfmap.shape)
        # alphas[:, :] = np.linspace(1, 0, curr_pfmap.shape[1]) # Test, blend transparency linearly
        # Normalize:
        # Create an alpha channel based on weight values
        # Any value whose absolute value is > .0001 will have zero transparency
        alphas = Normalize(clip=True)(np.abs(occupancy))
        # alphas = Normalize(0, .3, clip=True)(np.abs(occupancy))
        # alphas = np.clip(alphas, .4, 1)  # alpha value clipped at the bottom at .4
        main_plot_kwargs['alpha'] = alphas
        pass
    else:
        main_plot_kwargs['alpha'] = None
    
    ## Determine which background graphics to use:    
    if isinstance(bg_rendering_mode, str):
        background_rendering_mode_name = bg_rendering_mode # Already a string.
    else:
        # Otherwise assume it's the enum type and get its .name property
        background_rendering_mode_name = bg_rendering_mode.name

    if background_rendering_mode_name == BackgroundRenderingOptions.PATTERN_CHECKERBOARD.name:
        # Grey checkerboard background:
        # background_chessboard = np.add.outer(range(8), range(8)) % 2  # chessboard
        background_chessboard = _build_square_checkerboard_image(extent, num_checkerboard_squares_short_axis=8)
        bg_im = ax.imshow(background_chessboard, cmap=plt.cm.gray, alpha=0.25, interpolation='nearest', **imshow_shared_kwargs, label='background_image')
    elif background_rendering_mode_name == BackgroundRenderingOptions.SOLID_COLOR.name:
        # We'll also create a black background into which the pixels will fade
        background_black = np.full((*curr_pfmap.shape, 3), 0, dtype=np.uint8)
        bg_im = ax.imshow(background_black, **imshow_shared_kwargs, label='background_image')
    else:
        # No added background image
        bg_im = None
    
    im = ax.imshow(curr_pfmap, **main_plot_kwargs, label='main_image')
    ax.axis("off")
        
    ## Labeling:
    if final_title_str is None:
        final_title_str = 'ERR'

    if use_special_overlayed_title:
        title_anchored_text = add_inner_title(ax, final_title_str, loc='upper center', strokewidth=2, stroke_foreground='k', text_foreground='w') # loc = 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        title_anchored_text.patch.set_ec("none")
        # t.patch.set_alpha(0.5)
    else:
        # conventional way:
        ax.set_title(final_title_str) # f"Cell {ratemap.neuron_ids[cell]} - {ratemap.get_extended_neuron_id_string(neuron_i=cell)} \n{round(np.nanmax(pfmap),2)} Hz"
        title_anchored_text = None
    # always
    ax.set_label(final_title_str)
    return im, title_anchored_text
    
# all extracted from the 2D figures
@safely_accepts_kwargs
def plot_ratemap_2D(ratemap: Ratemap, computation_config=None, included_unit_indicies=None, included_unit_neuron_IDs=None, subplots:RowColTuple=(40, 3), fig_column_width:float=8.0, fig_row_height:float=1.0, resolution_multiplier:float=1.0, max_screen_figure_size=(None, None), fignum=1, fig=None,
     enable_spike_overlay=False, spike_overlay_spikes=None, extended_overlay_points_datasource_dicts=None, drop_below_threshold: float=0.0000001, brev_mode: PlotStringBrevityModeEnum=PlotStringBrevityModeEnum.CONCISE, plot_variable: enumTuningMap2DPlotVariables=enumTuningMap2DPlotVariables.TUNING_MAPS, plot_mode: enumTuningMap2DPlotMode=None, bg_rendering_mode=BackgroundRenderingOptions.PATTERN_CHECKERBOARD, use_special_overlayed_title=True, debug_print=False):
    """Plots heatmaps of placefields with peak firing rate
    
    Internally calls plot_single_tuning_map_2D(...) for each individual ratemap (regardless of the plot_mode)
    
    Parameters
    ----------
    speed_thresh : bool, optional
        [description], by default False
    subplots : tuple, optional
        number of cells within each figure window. If cells exceed the number of subplots, then cells are plotted in successive figure windows of same size, by default (10, 8)
    fignum : int, optional
        figure number to start from, by default None
    fig_subplotsize: tuple, optional
        fig_subplotsize: the size of a single subplot. used to compute the figure size
        
        
    spike_overlay_spikes: a 
    
    extended_overlay_points_datasource_dicts: a general dict of additional overlay point datasources to potentially add to the images. Each is passed to _add_points_to_plot(...)
        TODO: NOTE: currently the subplot the points are plotted on is determined by getting: `a_datasource['points_data'][neuron_IDX]`, meaning the assumption is that each datasource has one xy point to draw for every neuron. Obviously it would be better if multiple points could be provided for each neuron, so really the datasource should be re-speced to have a function that takes the neuron_id and returns the desired values (kinda like a datasource of datasources, or maybe a dataframe that it filters to get the points, that might be more 'flat' of a design. 
        
        Example:
            # overlay_points data
            peaks_overlay_points_data_dict = dict(is_enabled=True, points_data=peak_xy_points_pos_list, plot_opts={'markersize': 28, 'marker': '*', 'markeredgecolor': 'grey', 'linestyle': 'none', 'markerfacecolor': 'white', 'alpha': 0.9, 'label': 'peaks_overlay_points'},
                                                                                        scatter_opts={'s': 28, 'c': 'white', 'alpha': 0.9, 'marker': '*', 'label': 'peaks_overlay_sc'}, plots={})

            extended_overlay_points_datasource_dicts = {'peaks_overlay_points': peaks_overlay_points_data_dict}

    
    
    # TODO: maybe add a fig property: an explicit figure to use instead of fignum
    
    
    TODO: Cleaning up with  grid_layout_mode == 'imagegrid'
    plot_mode == 


    Returns:
            active_graphics_obj_dict[curr_neuron_ID] = {'axs': [curr_ax], 'image': curr_im, 'title_obj': curr_title_anchored_text}
    """
    # last_figure_subplots_same_layout = False
    last_figure_subplots_same_layout = True
    
    active_maps, title_substring, included_unit_indicies = _help_plot_ratemap_neuronIDs(ratemap, included_unit_indicies=included_unit_indicies, included_unit_neuron_IDs=included_unit_neuron_IDs, plot_variable=plot_variable, debug_print=debug_print)

    # ==================================================================================================================== #

    # Build the formatter for rendering the max values such as the peak firing rate or max spike counts:
    if brev_mode.should_show_firing_rate_label:
        max_value_formatter = _build_variable_max_value_label(plot_variable=plot_variable)
    else:
        max_value_formatter = None

    ## BEGIN FACTORING OUT:
    # nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio = _perform_plot_advanced_2D(xbin=ratemap.xbin, ybin=ratemap.ybin, included_unit_indicies=included_unit_indicies, subplots=subplots, last_figure_subplots_same_layout=last_figure_subplots_same_layout)
    ## NEW COMBINED METHOD, COMPUTES ALL PAGES AT ONCE:
    if resolution_multiplier is None:
        resolution_multiplier = 1.0
    nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes = _determine_best_placefield_2D_layout(xbin=ratemap.xbin, ybin=ratemap.ybin, included_unit_indicies=included_unit_indicies, subplots=subplots, fig_column_width=fig_column_width, fig_row_height=fig_row_height, resolution_multiplier=resolution_multiplier, max_screen_figure_size=max_screen_figure_size, last_figure_subplots_same_layout=last_figure_subplots_same_layout, debug_print=debug_print)
    
    if fignum is None:
        if f := plt.get_fignums():
            fignum = f[-1] + 1
        else:
            fignum = 1

    figures, page_gs, graphics_obj_dicts = [], [], []
    for fig_ind in range(nfigures):
        # Dynamic Figure Sizing: 
        curr_fig_page_grid_size = page_grid_sizes[fig_ind]
        active_figure_size = page_figure_sizes[fig_ind]
        
        if fig is not None:
            extant_fig = fig
            # fig = plt.figure(extant_fig)
        else:
            extant_fig = None # is this okay?
            
        if fig is not None:
            active_fig_id = fig
        else:
            if isinstance(fignum, int):
                # a numeric fignum that can be incremented
                active_fig_id = fignum + fig_ind
            elif isinstance(fignum, str):
                # a string-type fignum.
                # TODO: deal with inadvertant reuse of figure? perhaps by appending f'{fignum}[{fig_ind}]'
                if fig_ind > 0:
                    active_fig_id = f'{fignum}[{fig_ind}]'
                else:
                    active_fig_id = fignum
            else:
                raise NotImplementedError        
        
        ## Configure Colorbar options:
        ### curr_cbar_mode: 'each', 'one', None
        # curr_cbar_mode = 'each'
        curr_cbar_mode = None
        
        # grid_rect = (0.01, 0.05, 0.98, 0.9) # (left, bottom, width, height) 
        grid_rect = 111
        # fig = plt.figure(fignum + fig_ind, figsize=active_figure_size, dpi=None, clear=True, tight_layout=True)
        if extant_fig is None:
            fig = plt.figure(active_fig_id, figsize=active_figure_size, dpi=None, clear=True, tight_layout=False)
        else:
            fig = extant_fig
            
        grid = ImageGrid(fig, grid_rect,  # similar to subplot(211)
                nrows_ncols=(curr_fig_page_grid_size.num_rows, curr_fig_page_grid_size.num_columns),
                axes_pad=0.05,
                label_mode="1",
                share_all=True,
                aspect=True,
                cbar_location="top",
                cbar_mode=curr_cbar_mode,
                cbar_size="7%",
                cbar_pad="1%",
                )
        
        page_gs.append(grid)
            
        title_string = f'2D Placemaps {title_substring} ({len(ratemap.neuron_ids)} good cells)'
        
        if computation_config is not None:
            if computation_config.speed_thresh is not None:
                title_string = f'{title_string} (speed_threshold = {str(computation_config.speed_thresh)})'
            
        fig.suptitle(title_string)
        figures.append(fig)
        graphics_obj_dicts.append({}) # New empty dict

    # New page-based version:
    for page_idx in np.arange(num_pages):
        if debug_print:
            print(f'page_idx: {page_idx}')
        
        active_page_grid = page_gs[page_idx]
        active_graphics_obj_dict = graphics_obj_dicts[page_idx]
        # print(f'active_page_grid: {active_page_grid}')
            
        for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
            # Need to convert to page specific:
            curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
            curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
            curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
            # print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')
           
            if curr_included_unit_index is not None:
                # valid neuron ID, access like normal
                pfmap = active_maps[curr_included_unit_index]
                # normal (non-shared mode)
                curr_ratemap_relative_neuron_IDX = curr_included_unit_index
                curr_neuron_ID = ratemap.neuron_ids[curr_ratemap_relative_neuron_IDX]
                
                ## Labeling:
                formatted_max_value_string = None
                if brev_mode.should_show_firing_rate_label:
                    assert max_value_formatter is not None
                    ## NOTE: must set max_value_formatter on the pfmap BEFORE the `_scale_current_placefield_to_acceptable_range` is called to have it show accurate labels!
                    formatted_max_value_string = max_value_formatter(np.nanmax(pfmap))
                    
                final_title_str = _build_neuron_identity_label(neuron_extended_id=ratemap.neuron_extended_ids[curr_ratemap_relative_neuron_IDX], brev_mode=brev_mode, formatted_max_value_string=formatted_max_value_string, use_special_overlayed_title=use_special_overlayed_title)

            else:
                # invalid neuron ID, generate blank entry
                curr_ratemap_relative_neuron_IDX = None # This neuron_ID doesn't correspond to a neuron_IDX in the current ratemap, so we'll mark this value as None
                curr_neuron_ID = included_unit_neuron_IDs[a_linear_index]

                pfmap = np.zeros((np.shape(active_maps)[1], np.shape(active_maps)[2])) # fully allocated new array of zeros
                curr_extended_id_string = f'{curr_neuron_ID}' # get the aclu value (which is all that's known about the missing cell and use that as the curr_extended_id_string
                final_title_str = f'{curr_extended_id_string} <shared>'

            # Get the axis to plot on:
            curr_ax = active_page_grid[curr_page_relative_linear_index]
            
            ## Plot the main heatmap for this pfmap:
            curr_im, curr_title_anchored_text = _plot_single_tuning_map_2D(ratemap.xbin, ratemap.ybin, pfmap, ratemap.occupancy, final_title_str=final_title_str, drop_below_threshold=drop_below_threshold, brev_mode=brev_mode, plot_mode=plot_mode,
                                            ax=curr_ax, max_value_formatter=max_value_formatter, use_special_overlayed_title=use_special_overlayed_title, bg_rendering_mode=bg_rendering_mode)
            
            active_graphics_obj_dict[curr_neuron_ID] = {'axs': [curr_ax], 'image': curr_im, 'title_obj': curr_title_anchored_text}

            if curr_ratemap_relative_neuron_IDX is not None:
                # This means this neuron is included in the current ratemap:
                ## Decision: Only do these extended plotting things when the neuron_IDX is included/valid.
                if extended_overlay_points_datasource_dicts is not None:
                    for (overlay_datasource_name, overlay_datasource) in extended_overlay_points_datasource_dicts.items():
                        # There can be multiple named datasources, with either of two modes: 
                        # 1. Linear indexed list
                        if overlay_datasource.get('is_enabled', False):
                            points_data = overlay_datasource.get('points_data', None)
                            if points_data is not None:
                                if debug_print:
                                    print(f'overlay_datasource_name: {overlay_datasource_name} looks good. Trying to add.')
                                curr_overlay_points, curr_overlay_sc = _add_points_to_plot(curr_ax, points_data[curr_ratemap_relative_neuron_IDX], plot_opts=overlay_datasource.get('plot_opts', None), scatter_opts=overlay_datasource.get('scatter_opts', None))
                                overlay_datasource['plots'] = dict(points=curr_overlay_points, sc=curr_overlay_sc)
                        else:
                            # 2. ACLU indexed dict
                            curr_neuron_ID = ratemap.neuron_ids[curr_ratemap_relative_neuron_IDX]
                            found_neuron_aclu_datasource = overlay_datasource.get(curr_neuron_ID, None) 
                            if found_neuron_aclu_datasource is not None:
                                if found_neuron_aclu_datasource.get('is_enabled', False):
                                    points_data = found_neuron_aclu_datasource.get('points_data', None)
                                    if points_data is not None:
                                        if debug_print:
                                            print(f'overlay_datasource_name: {overlay_datasource_name} looks good. Trying to add.')
                                        curr_overlay_points, curr_overlay_sc = _add_points_to_plot(curr_ax, points_data.T, plot_opts=found_neuron_aclu_datasource.get('plot_opts', None), scatter_opts=found_neuron_aclu_datasource.get('scatter_opts', None))
                                        found_neuron_aclu_datasource['plots'] = dict(points=curr_overlay_points, sc=curr_overlay_sc)
                        
                                
                if enable_spike_overlay:
                    spike_overlay_points, spike_overlay_sc = _add_points_to_plot(curr_ax, spike_overlay_spikes[curr_ratemap_relative_neuron_IDX], plot_opts={'markersize': 2, 'marker': ',', 'markeredgecolor': 'red', 'linestyle': 'none', 'markerfacecolor': 'red', 'alpha': 0.1, 'label': 'spike_overlay_points'},
                                                                                scatter_opts={'s': 2, 'c': 'white', 'alpha': 0.1, 'marker': ',', 'label': 'spike_overlay_sc'})
                    active_graphics_obj_dict[curr_neuron_ID] = active_graphics_obj_dict[curr_neuron_ID] | {'spike_overlay_points': spike_overlay_points, 'spike_overlay_sc': spike_overlay_sc} # Add in the spike_overlay_points and spike_overlay_sc

        ## Remove the unused axes if there are any:
        # Note that this makes use of the fact that curr_page_relative_linear_index is left maxed-out after the above loop finishes executing.
        num_axes_to_remove = (len(active_page_grid) - 1) - curr_page_relative_linear_index
        if (num_axes_to_remove > 0):
            for a_removed_linear_index in np.arange(curr_page_relative_linear_index+1, len(active_page_grid)):
                removal_ax = active_page_grid[a_removed_linear_index]
                fig.delaxes(removal_ax)

        # Apply subplots adjust to fix margins:
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        
    return figures, page_gs, graphics_obj_dicts

@safely_accepts_kwargs
def plot_ratemap_1D(ratemap: Ratemap, normalize_xbin=False, ax=None, pad=2, normalize_tuning_curve=False, sortby=None, cmap=None):
    """Plot 1D place fields stacked

    Parameters
    ----------
    ax : [type], optional
        [description], by default None
    speed_thresh : bool, optional
        [description], by default False
    pad : int, optional
        [description], by default 2
    normalize : bool, optional
        [description], by default False
    sortby : bool, optional
        [description], by default True
    cmap : str, optional
        [description], by default "tab20b"

    Returns
    -------
    [type]
        [description]


    Notes:
    Unlike the plot_ratemap_2D(...), this version seems to plot all the cells on a single axis: using `ax.set_yticklabels(list(sorted_neuron_id_labels))` to label each cell's tuning curve and offsets to plot them.
    """
    
    
    # cmap = mpl.cm.get_cmap(cmap)

    tuning_curves = ratemap.tuning_curves
    n_neurons = ratemap.n_neurons
    bin_cntr = ratemap.xbin_centers
    if normalize_xbin:
        bin_cntr = (bin_cntr - np.min(bin_cntr)) / np.ptp(bin_cntr)

    if ax is None:
        _, gs = Fig().draw(grid=(1, 1), size=(5.5, 11))
        ax = plt.subplot(gs[0])

    if normalize_tuning_curve:
        tuning_curves = mathutil.min_max_scaler(tuning_curves)
        pad = 1

    if sortby is None:
        # sort by the location of the placefield's maximum
        sort_ind = np.argsort(np.argmax(tuning_curves, axis=1))
    elif isinstance(sortby, (list, np.ndarray)):
        # use the provided sort indicies
        sort_ind = sortby
    else:
        sort_ind = np.arange(n_neurons)

    # Use the get_neuron_colors function to generate colors for these neurons
    neurons_colors_array = get_neuron_colors(sort_ind, cmap=cmap)

    ## TODO: actually sort the ratemap object's neuron_ids and tuning_curves by the sort_ind
    # sorted_neuron_ids = ratemap.neuron_ids[sort_ind]
    
    sorted_neuron_ids = np.take_along_axis(np.array(ratemap.neuron_ids), sort_ind, axis=0)
    
    sorted_alt_tuple_neuron_ids = ratemap.neuron_extended_ids.copy()
    sorted_alt_tuple_neuron_ids = [sorted_alt_tuple_neuron_ids[a_sort_idx] for a_sort_idx in sort_ind]
    
    # sorted_tuning_curves = tuning_curves[sorted_neuron_ids, :]
    # sorted_neuron_id_labels = [f'Cell[{a_neuron_id}]' for a_neuron_id in sorted_neuron_ids]
    sorted_neuron_id_labels = [f'C[{sorted_neuron_ids[i]}]({sorted_alt_tuple_neuron_ids[i].shank}|{sorted_alt_tuple_neuron_ids[i].cluster})' for i in np.arange(len(sorted_neuron_ids))]
    
    # neurons_colors_array = np.zeros((4, n_neurons))
    for i, neuron_ind in enumerate(sort_ind):
        color = neurons_colors_array[:, i]
        curr_neuron_id = sorted_neuron_ids[i]

        ax.fill_between(bin_cntr, i * pad, i * pad + tuning_curves[neuron_ind], color=color, ec=None, alpha=0.5, zorder=i + 1)
        ax.plot(bin_cntr, i * pad + tuning_curves[neuron_ind], color=color, alpha=0.7)
        


    ax.set_yticks(list(np.arange(len(sort_ind)) + 0.5))
    ax.set_yticklabels(list(sorted_neuron_id_labels))
    
    ax.set_xlabel("Position")
    ax.spines["left"].set_visible(False)
    if normalize_xbin:
        ax.set_xlim([0, 1])
    ax.tick_params("y", length=0)
    ax.set_ylim([0, len(sort_ind)])
        
    title_string = f'1D Placemaps {title_substring} ({len(ratemap.neuron_ids)} good cells)'
    ax.set_title(title_string) # this doesn't appear to be visible, so what is it used for?
    # if self.run_dir is not None:
    #     ax.set_title(self.run_dir.capitalize() + " Runs only")


    return ax, sort_ind, neurons_colors_array



@safely_accepts_kwargs
def plot_raw(ratemap: Ratemap, t, x, run_dir, ax=None, subplots=(8, 9)):
    """Plot spike location on animal's path

    Parameters
    ----------
    speed_thresh : bool, optional
        [description], by default False
    ax : [type], optional
        [description], by default None
    subplots : tuple, optional
        [description], by default (8, 9)
    """

    # mapinfo = self.ratemaps
    mapinfo = ratemap
    nCells = len(mapinfo["pos"])

    def plot_(cell, ax):
        if subplots is None:
            ax.clear()
        ax.plot(x, t, color="gray", alpha=0.6)
        ax.plot(mapinfo["pos"][cell], mapinfo["spikes"][cell], ".", color="#ff5f5c")
        ax.set_title(
            " ".join(filter(None, ("Cell", str(cell), run_dir.capitalize())))
        )
        ax.invert_yaxis()
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Time (s)")

    if ax is None:

        if subplots is None:
            _, gs = Fig().draw(grid=(1, 1), size=(6, 8))
            ax = plt.subplot(gs[0])
            widgets.interact(
                plot_,
                cell=widgets.IntSlider(
                    min=0,
                    max=nCells - 1,
                    step=1,
                    description="Cell ID:",
                ),
                ax=widgets.fixed(ax),
            )
        else:
            _, gs = Fig().draw(grid=subplots, size=(10, 11))
            for cell in range(nCells):
                ax = plt.subplot(gs[cell])
                ax.set_yticks([])
                plot_(cell, ax)





