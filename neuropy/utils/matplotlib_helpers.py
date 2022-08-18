from __future__ import annotations # otherwise have to do type like 'Ratemap'

from enum import Enum, IntEnum, auto, unique
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from neuropy.utils.misc import AutoNameEnum, compute_paginated_grid_config, RowColTuple

from neuropy.plotting.figure import compute_figure_size_pixels, compute_figure_size_inches # needed for _determine_best_placefield_2D_layout(...)'s internal _perform_compute_required_figure_sizes(...) function

from typing import TYPE_CHECKING
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # needed for _build_neuron_identity_label
if TYPE_CHECKING:
    from neuropy.core.neuron_identities import NeuronExtendedIdentityTuple # needed for _build_neuron_identity_label
    

""" Note that currently the only Matplotlib-specific functions here are add_inner_title(...) and draw_sizebar(...). The rest have general uses! """

# refactored to pyPhoCoreHelpers.geometery_helpers but had to be bring back in explicitly
Width_Height_Tuple = namedtuple('Width_Height_Tuple', 'width height')
def compute_data_extent(xpoints, *other_1d_series):
    """Computes the outer bounds, or "extent" of one or more 1D data series.

    Args:
        xpoints ([type]): [description]
        other_1d_series: any number of other 1d data series

    Returns:
        xmin, xmax, ymin, ymax, imin, imax, ...: a flat list of paired min, max values for each data series provided.
        
    Usage:
        # arbitrary number of data sequences:        
        xmin, xmax, ymin, ymax, x_center_min, x_center_max, y_center_min, y_center_max = compute_data_extent(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.xbin_centers, active_epoch_placefields2D.ratemap.ybin_centers)
        print(xmin, xmax, ymin, ymax, x_center_min, x_center_max, y_center_min, y_center_max)

        # simple 2D extent:
        extent = compute_data_extent(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin)
        print(extent)
    """
    num_total_series = len(other_1d_series) + 1 # + 1 for the x-series
    # pre-allocate output:     
    extent = np.empty(((2 * num_total_series),))
    # Do first-required series:
    xmin, xmax = min_max_1d(xpoints)
    extent[0], extent[1] = [xmin, xmax]
    # finish remaining series passed as inputs.
    for (i, a_series) in enumerate(other_1d_series):
        curr_min, curr_xmax = min_max_1d(a_series)
        curr_start_idx = 2 * (i + 1)
        extent[curr_start_idx] = curr_min
        extent[curr_start_idx+1] = curr_xmax
    return extent
def compute_data_aspect_ratio(xbin, ybin, sorted_inputs=True):
    """Computes the aspect ratio of the provided data

    Args:
        xbin ([type]): [description]
        ybin ([type]): [description]
        sorted_inputs (bool, optional): whether the input arrays are pre-sorted in ascending order or not. Defaults to True.

    Returns:
        float: The aspect ratio of the data such that multiplying any height by the returned float would result in a width in the same aspect ratio as the data.
    """
    if sorted_inputs:
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], ybin[0], ybin[-1]) # assumes-pre-sourced events, which is valid for bins but not general
    else:
        xmin, xmax, ymin, ymax = compute_data_extent(xbin, ybin) # more general form.

    # The extent keyword arguments controls the bounding box in data coordinates that the image will fill specified as (left, right, bottom, top) in data coordinates, the origin keyword argument controls how the image fills that bounding box, and the orientation in the final rendered image is also affected by the axes limits.
    # extent = (xmin, xmax, ymin, ymax)
    
    width = xmax - xmin
    height = ymax - ymin
    
    aspect_ratio = width / height
    return aspect_ratio, Width_Height_Tuple(width, height)

@unique
class enumTuningMap2DPlotMode(AutoNameEnum):
    PCOLORFAST = auto() # DEFAULT prior to 2021-12-24
    PCOLORMESH = auto() # UNTESTED
    PCOLOR = auto() # UNTESTED
    IMSHOW = auto() # New Default as of 2021-12-24

@unique
class enumTuningMap2DPlotVariables(AutoNameEnum):
    TUNING_MAPS = auto() # DEFAULT
    SPIKES_MAPS = auto() 
    
    
    
def _build_neuron_identity_label(neuron_extended_id: NeuronExtendedIdentityTuple=None, brev_mode=PlotStringBrevityModeEnum.CONCISE, formatted_max_value_string=None, use_special_overlayed_title=True):
    """ builds the subplot title for 2D PFs that displays the neuron identity and other important info. """
    if neuron_extended_id is not None:    
        full_extended_id_string = brev_mode.extended_identity_formatting_string(neuron_extended_id)
    else:
        full_extended_id_string = ''
    
    final_string_components = [full_extended_id_string]
    
    if formatted_max_value_string is not None:
        final_string_components.append(formatted_max_value_string)
    
    if use_special_overlayed_title:
        final_title = ' - '.join(final_string_components)
    else:
        # conventional way:
        final_title = '\n'.join(final_string_components) # f"Cell {ratemap.neuron_ids[cell]} - {ratemap.get_extended_neuron_id_string(neuron_i=cell)} \n{round(np.nanmax(pfmap),2)} Hz"
    return final_title
    
    
def _build_variable_max_value_label(plot_variable: enumTuningMap2DPlotVariables):
    """  Builds a label that displays the max value with the appropriate unit suffix for the title
    if brev_mode.should_show_firing_rate_label:
        pf_firing_rate_string = f'{round(np.nanmax(pfmap),2)} Hz'
        final_string_components.append(pf_firing_rate_string)
    """
    if plot_variable.name is enumTuningMap2DPlotVariables.TUNING_MAPS.name:
        return lambda value: f'{round(value,2)} Hz'
    elif plot_variable.name == enumTuningMap2DPlotVariables.SPIKES_MAPS.name:
        return lambda value: f'{round(value,2)} Spikes'
    else:
        raise NotImplementedError


def _determine_best_placefield_2D_layout(xbin, ybin, included_unit_indicies, subplots:RowColTuple=(40, 3), fig_column_width:float=8.0, fig_row_height:float=1.0, resolution_multiplier:float=1.0, max_screen_figure_size=(None, None), last_figure_subplots_same_layout=True, debug_print:bool=False):
    """ Computes the optimal sizes, number of rows and columns, and layout of the individual 2D placefield subplots in terms of the overarching pf_2D figure
    
    Known Uses:
        plot_advanced_2D
    
    Major outputs:
    
    
    (curr_fig_page_grid_size.num_rows, curr_fig_page_grid_size.num_columns)
    
    
    Usage Example:
        nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes = _final_wrapped_determine_placefield_2D_layout(xbin=active_pf_2D.xbin, ybin=active_pf_2D.ybin, included_unit_indicies=np.arange(active_pf_2D.ratemap.n_neurons), subplots=(40, 3), fig_column_width=8.0, fig_row_height=1.0, resolution_multiplier=1.0, max_screen_figure_size=(None, None), last_figure_subplots_same_layout=True, debug_print=True)
        
        print(f'nfigures: {nfigures}\ndata_aspect_ratio: {data_aspect_ratio}')
        # Loop through each page/figure that's required:
        for page_fig_ind, page_fig_size, page_grid_size in zip(np.arange(nfigures), page_figure_sizes, page_grid_sizes):
            print(f'\tpage_fig_ind: {page_fig_ind}, page_fig_size: {page_fig_size}, page_grid_size: {page_grid_size}')
               
        
    """
    def _perform_compute_optimal_paginated_grid_layout(xbin, ybin, included_unit_indicies, subplots:RowColTuple=(40, 3), last_figure_subplots_same_layout=True, debug_print:bool=False):
        if not isinstance(subplots, RowColTuple):
            subplots = RowColTuple(subplots[0], subplots[1])
        
        nMapsToShow = len(included_unit_indicies)
        data_aspect_ratio = compute_data_aspect_ratio(xbin, ybin)
        if debug_print:
            print(f'data_aspect_ratio: {data_aspect_ratio}')
        
        if (subplots.num_columns is None) or (subplots.num_rows is None):
            # This will disable pagination by setting an arbitrarily high value
            max_subplots_per_page = nMapsToShow
            if debug_print:
                print('Pagination is disabled because one of the subplots values is None. Output will be in a single figure/page.')
        else:
            # valid specified maximum subplots per page
            max_subplots_per_page = int(subplots.num_columns * subplots.num_rows)
        
        if debug_print:
            print(f'nMapsToShow: {nMapsToShow}, subplots: {subplots}, max_subplots_per_page: {max_subplots_per_page}')
            
        # Paging Management: Constrain the subplots values to just those that you need
        subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=subplots.num_columns, max_subplots_per_page=max_subplots_per_page, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=last_figure_subplots_same_layout)
        num_pages = len(included_combined_indicies_pages)
        nfigures = num_pages
        # nfigures = nMapsToShow // np.prod(subplots) + 1 # "//" is floor division (rounding result down to nearest whole number)
        return nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio
    
    def _perform_compute_required_figure_sizes(curr_fig_page_grid_size, data_aspect_ratio, fig_column_width:float=None, fig_row_height:float=None, resolution_multiplier:float=1.0, max_screen_figure_size=(None, None), debug_print:bool=False):
        if resolution_multiplier is None:
            resolution_multiplier = 1.0
        if (fig_column_width is not None) and (fig_row_height is not None):
            desired_single_map_width = fig_column_width * resolution_multiplier
            desired_single_map_height = fig_row_height * resolution_multiplier
        else:
            ## TODO: I think this hardcoded 4.0 should be set to data_aspect_ratio: (1.0167365776358197 for square maps)
            desired_single_map_width = data_aspect_ratio[0] * resolution_multiplier
            desired_single_map_height = 1.0 * resolution_multiplier
            
        # Computes desired_single_map_width and desired_signle_map_height
            
        ## Figure size should be (Width, height)
        required_figure_size = ((float(curr_fig_page_grid_size.num_columns) * float(desired_single_map_width)), (float(curr_fig_page_grid_size.num_rows) * float(desired_single_map_height))) # (width, height)
        required_figure_size_px = compute_figure_size_pixels(required_figure_size)
        if debug_print:
            print(f'resolution_multiplier: {resolution_multiplier}, required_figure_size: {required_figure_size}, required_figure_size_px: {required_figure_size_px}') # this is figure size in inches

        active_figure_size = required_figure_size
        
        # If max_screen_figure_size is not None (it should be a two element tuple, specifying the max width and height in pixels for the figure:
        if max_screen_figure_size is not None:
            required_figure_size_px = list(required_figure_size_px) # convert to a list instead of a tuple to make it mutable
            if max_screen_figure_size[0] is not None:
                required_figure_size_px[0] = min(required_figure_size_px[0], max_screen_figure_size[0])
            if max_screen_figure_size[1] is not None:
                required_figure_size_px[1] = min(required_figure_size_px[1], max_screen_figure_size[1])

        required_figure_size_px = tuple(required_figure_size_px)
        # convert back to inches from pixels to constrain the figure size:
        required_figure_size = compute_figure_size_inches(required_figure_size_px) # Convert back from pixels to inches when done
        # Update active_figure_size again:
        active_figure_size = required_figure_size
        
        # active_figure_size=figsize
        # active_figure_size=required_figure_size
        if debug_print:
            print(f'final active_figure_size: {active_figure_size}, required_figure_size_px: {required_figure_size_px} (after constraining by max_screen_figure_size, etc)')

        return active_figure_size

    # BEGIN MAIN FUNCTION BODY ___________________________________________________________________________________________ #
    nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio = _perform_compute_optimal_paginated_grid_layout(xbin=xbin, ybin=ybin, included_unit_indicies=included_unit_indicies, subplots=subplots, last_figure_subplots_same_layout=last_figure_subplots_same_layout, debug_print=debug_print)
    if resolution_multiplier is None:
        resolution_multiplier = 1.0

    page_figure_sizes = []
    for fig_ind in range(nfigures):
        # Dynamic Figure Sizing: 
        curr_fig_page_grid_size = page_grid_sizes[fig_ind]
        ## active_figure_size is the primary output
        active_figure_size = _perform_compute_required_figure_sizes(curr_fig_page_grid_size, data_aspect_ratio=data_aspect_ratio, fig_column_width=fig_column_width, fig_row_height=fig_row_height, resolution_multiplier=resolution_multiplier, max_screen_figure_size=max_screen_figure_size, debug_print=debug_print)
        page_figure_sizes.append(active_figure_size)
        
    return nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes


def _scale_current_placefield_to_acceptable_range(image, occupancy, drop_below_threshold: float=0.0000001):
    """ Universally used to prepare the pfmap to be displayed (across every plot time)
    
    Input:
        drop_below_threshold: if None, no indicies are dropped. Otherwise, values of occupancy less than the threshold specified are used to build a mask, which is subtracted from the returned image (the image is NaN'ed out in these places.

    Known Uses:
            NeuroPy.neuropy.plotting.ratemaps.plot_single_tuning_map_2D(...)
            pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields.pyqtplot_plot_image_array(...)
            
     # image = np.squeeze(images[a_linear_index,:,:])
    """
    # Pre-filter the data:
    with np.errstate(divide='ignore', invalid='ignore'):
        image = np.array(image.copy()) / np.nanmax(image) # note scaling by maximum here!
        if drop_below_threshold is not None:
            image[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy
        return image # return the modified and masked image


    
def _build_square_checkerboard_image(extent, num_checkerboard_squares_short_axis:int=10, debug_print=False):
    """ builds a background checkerboard image used to indicate opacity
    Usage:
    # Updating Existing:
    background_chessboard = _build_square_checkerboard_image(active_ax_main_image.get_extent(), num_checkerboard_squares_short_axis=8)
    active_ax_bg_image.set_data(background_chessboard) # updating mode
    
    # Creation:
    background_chessboard = _build_square_checkerboard_image(active_ax_main_image.get_extent(), num_checkerboard_squares_short_axis=8)
    bg_im = ax.imshow(background_chessboard, cmap=plt.cm.gray, interpolation='nearest', **imshow_shared_kwargs, label='background_image')
    
    """
    left, right, bottom, top = extent
    width = np.abs(left - right)
    height = np.abs(top - bottom) # width: 241.7178791533281, height: 30.256480996256016
    if debug_print:
        print(f'width: {width}, height: {height}')
    
    if width >= height:
        short_axis_length = float(height)
        long_axis_length = float(width)
    else:
        short_axis_length = float(width)
        long_axis_length = float(height)
    
    checkerboard_square_side_length = short_axis_length / float(num_checkerboard_squares_short_axis) # checkerboard_square_side_length is the same along all axes
    frac_num_checkerboard_squares_long_axis = long_axis_length / float(checkerboard_square_side_length)
    num_checkerboard_squares_long_axis = int(np.round(frac_num_checkerboard_squares_long_axis))
    if debug_print:
        print(f'checkerboard_square_side: {checkerboard_square_side_length}, num_checkerboard_squares_short_axis: {num_checkerboard_squares_short_axis}, num_checkerboard_squares_long_axis: {num_checkerboard_squares_long_axis}')
    # Grey checkerboard background:
    background_chessboard = np.add.outer(range(num_checkerboard_squares_short_axis), range(num_checkerboard_squares_long_axis)) % 2  # chessboard
    return background_chessboard





# ==================================================================================================================== #
# These are the only Matplotlib-specific functions here: add_inner_title(...) and draw_sizebar(...).                              #
# ==================================================================================================================== #
def add_inner_title(ax, title, loc, strokewidth=3, stroke_foreground='w', text_foreground='k', **kwargs):
    """used to add a figure title inside the border of the figure (instead of outside)
    """
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke    
    # Afont = {'family': 'serif',
    #     'backgroundcolor': 'blue',
    #     'color':  'white',
    #     'weight': 'normal',
    #     'size': 14,
    #     }
    prop = dict(path_effects=[withStroke(foreground=stroke_foreground, linewidth=strokewidth)],
                size=plt.rcParams['legend.title_fontsize'], # 'legend.fontsize' is too small
                color=text_foreground)
    at = AnchoredText(title, loc=loc, prop=prop, pad=0., borderpad=0.5, frameon=False, **kwargs)
    ax.add_artist(at)
    return at

## TODO: Not currently used, but looks like it can add anchored scale/size bars to matplotlib figures pretty easily:
def draw_sizebar(ax):
    """
    Draw a horizontal bar with length of 0.1 in data coordinates,
    with a fixed label underneath.
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    asb = AnchoredSizeBar(ax.transData,
                          0.1,
                          r"1$^{\prime}$",
                          loc='lower center',
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)
    