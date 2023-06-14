from __future__ import annotations # otherwise have to do type like 'Ratemap'

from enum import Enum, IntEnum, auto, unique
from collections import namedtuple
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import BrokenBarHCollection # for draw_epoch_regions
from matplotlib.widgets import RectangleSelector # required for `add_rectangular_selector`
from matplotlib.widgets import SpanSelector


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
    OCCUPANCY = auto()
    
    
    
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
    
    Interally Calls:
        neuropy.utils.misc.compute_paginated_grid_config(...)


    Known Uses:
        display_all_pf_2D_pyqtgraph_binned_image_rendering
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
        drop_below_threshold: if None, no indicies are dropped. Otherwise, values of occupancy less than the threshold specified are used to build a mask, which is subtracted from the returned image (the image is NaN'ed out in these places).

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
    

def build_or_reuse_figure(fignum=1, fig=None, fig_idx:int=0, **kwargs):
    """ Reuses a Matplotlib figure if it exists, or creates a new one if needed
    Inputs:
        fignum - an int or str that identifies a figure
        fig - an existing Matplotlib figure
        fig_idx:int - an index to identify this figure as part of a series of related figures, e.g. plot_pf_1D[0], plot_pf_1D[1], ... 
        **kwargs - are passed as kwargs to the plt.figure(...) command when creating a new figure
    Outputs:
        fig: a Matplotlib figure object

    History: factored out of `plot_ratemap_2D`
    """
    if fignum is None:
        if f := plt.get_fignums():
            fignum = f[-1] + 1
        else:
            fignum = 1

    ## Figure Setup:
    if fig is not None:
        # provided figure
        extant_fig = fig
    else:
        extant_fig = None # is this okay?
        
    if fig is not None:
        # provided figure
        active_fig_id = fig
    else:
        if isinstance(fignum, int):
            # a numeric fignum that can be incremented
            active_fig_id = fignum + fig_idx
        elif isinstance(fignum, str):
            # a string-type fignum.
            # TODO: deal with inadvertant reuse of figure? perhaps by appending f'{fignum}[{fig_ind}]'
            if fig_idx > 0:
                active_fig_id = f'{fignum}[{fig_idx}]'
            else:
                active_fig_id = fignum
        else:
            raise NotImplementedError
    
    if extant_fig is None:
        fig = plt.figure(active_fig_id, **({'dpi': None, 'clear': True} | kwargs)) # , 'tight_layout': False - had to remove 'tight_layout': False because it can't coexist with 'constrained_layout'
            #  UserWarning: The Figure parameters 'tight_layout' and 'constrained_layout' cannot be used together.
    else:
        fig = extant_fig
    return fig

def scale_title_label(ax, curr_title_obj, curr_im, debug_print=False):
    """ Scales some matplotlib-based figures titles to be reasonable. I remember that this was important and hard to make, but don't actually remember what it does as of 2022-10-24. It needs to be moved in to somewhere else.
    

    History: From PendingNotebookCode's 2022-11-09 section


    Usage:

        from neuropy.utils.matplotlib_helpers import scale_title_label

        ## Scale all:
        _display_outputs = widget.last_added_display_output
        curr_graphics_objs = _display_outputs.graphics[0]

        ''' curr_graphics_objs is:
        {2: {'axs': [<Axes:label='2'>],
        'image': <matplotlib.image.AxesImage at 0x1630c4556d0>,
        'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c4559a0>},
        4: {'axs': [<Axes:label='4'>],
        'image': <matplotlib.image.AxesImage at 0x1630c455f70>,
        'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c463280>},
        5: {'axs': [<Axes:label='5'>],
        'image': <matplotlib.image.AxesImage at 0x1630c463850>,
        'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c463b20>},
        ...
        '''
        for aclu, curr_neuron_graphics_dict in curr_graphics_objs.items():
            curr_title_obj = curr_neuron_graphics_dict['title_obj'] # matplotlib.offsetbox.AnchoredText
            curr_title_text_obj = curr_title_obj.txt.get_children()[0] # Text object
            curr_im = curr_neuron_graphics_dict['image'] # matplotlib.image.AxesImage
            curr_ax = curr_neuron_graphics_dict['axs'][0]
            scale_title_label(curr_ax, curr_title_obj, curr_im)

    
    """
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    transform = ax.transData
    curr_label_extent = curr_title_obj.get_window_extent(ax.get_figure().canvas.get_renderer()) # Bbox([[1028.49144862968, 2179.0555555555566], [1167.86644862968, 2193.0555555555566]])
    curr_label_width = curr_label_extent.width # 139.375
    # curr_im.get_extent() # (-85.75619321393464, 112.57838773103435, -96.44772761274268, 98.62205280781535)
    img_extent = curr_im.get_window_extent(ax.get_figure().canvas.get_renderer()) # Bbox([[1049.76842294452, 2104.7727272727284], [1146.5894743148401, 2200.000000000001]])
    curr_img_width = img_extent.width # 96.82105137032022
    needed_scale_factor = curr_img_width / curr_label_width
    if debug_print:
        print(f'curr_label_width: {curr_label_width}, curr_img_width: {curr_img_width}, needed_scale_factor: {needed_scale_factor}')
    needed_scale_factor = min(needed_scale_factor, 1.0) # Only scale up, don't scale down
    
    curr_font_props = curr_title_obj.prop # FontProperties
    curr_font_size_pts = curr_font_props.get_size_in_points() # 10.0
    curr_scaled_font_size_pts = needed_scale_factor * curr_font_size_pts
    if debug_print:
        print(f'curr_font_size_pts: {curr_font_size_pts}, curr_scaled_font_size_pts: {curr_scaled_font_size_pts}')

    if isinstance(curr_title_obj, AnchoredText):
        curr_title_text_obj = curr_title_obj.txt.get_children()[0] # Text object
    else:
        curr_title_text_obj = curr_title_obj
    
    curr_title_text_obj.set_fontsize(curr_scaled_font_size_pts)
    font_foreground = 'white'
    # font_foreground = 'black'
    curr_title_text_obj.set_color(font_foreground)
    # curr_title_text_obj.set_fontsize(6)
    
    stroke_foreground = 'black'
    # stroke_foreground = 'gray'
    # stroke_foreground = 'orange'
    strokewidth = 4
    curr_title_text_obj.set_path_effects([withStroke(foreground=stroke_foreground, linewidth=strokewidth)])
    ## Disable path effects:
    # curr_title_text_obj.set_path_effects([])


def add_value_labels(ax, spacing=5, labels=None):
    """Add labels to the end (top) of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes of the plot to annotate.
        spacing (int): The distance between the labels and the bars.

    History:
        Factored out of `plot_short_v_long_pf1D_scalar_overlap_comparison` on 2023-03-28

    Usage:
        from neuropy.utils.matplotlib_helpers import add_value_labels
        # Call the function above. All the magic happens there.
        add_value_labels(ax, labels=x_labels) # 

    """

    # For each bar: Place a label
    for i, rect in enumerate(ax.patches):
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if labels is None:
            label = "{:.1f}".format(y_value)
            # # Use cell ID (given by x position) as the label
            label = "{}".format(x_value)
        else:
            label = str(labels[i])
            
        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va,                      # Vertically align label differently for positive and negative values.
            color=rect.get_facecolor(),
            rotation=90)                      
                                        # 


def fit_both_axes(ax_lhs, ax_rhs):
    """ 2023-05-25 - Computes the x and y bounds needed to fit all data on both axes, and the actually applies these bounds to each. """
    def _subfn_compute_fitting_both_axes(ax_lhs, ax_rhs):
        """ computes the fitting x and y bounds for both axes to fit all the data. 
        
        >>> ((0.8970694235737637, 95.79803141394544),
            (-1.3658343711184302, 32.976028484630994))
        """
        fitting_xbounds = (min(*ax_lhs.get_xbound(), *ax_rhs.get_xbound()), max(*ax_lhs.get_xbound(), *ax_rhs.get_xbound())) 
        fitting_ybounds = (min(*ax_lhs.get_ybound(), *ax_rhs.get_ybound()), max(*ax_lhs.get_ybound(), *ax_rhs.get_ybound())) 
        return (fitting_xbounds, fitting_ybounds)

    fitting_xbounds, fitting_ybounds = _subfn_compute_fitting_both_axes(ax_lhs, ax_rhs)
    ax_lhs.set_xbound(*fitting_xbounds)
    ax_lhs.set_ybound(*fitting_ybounds)
    ax_rhs.set_xbound(*fitting_xbounds)
    ax_rhs.set_ybound(*fitting_ybounds)
    return (fitting_xbounds, fitting_ybounds)






#     ## Figure computation
#     fig: plt.Figure = ax.get_figure()
#     dpi = fig.dpi
#     rect_height_inch = rect_height / dpi
#     # Initial fontsize according to the height of boxes
#     fontsize = rect_height_inch * 72
#     print(f'rect_height_inch: {rect_height_inch}, fontsize: {fontsize}')

# #     text: Annotation = ax.annotate(txt, xy, ha=ha, va=va, xycoords=transform, **kwargs)

# #     # Adjust the fontsize according to the box size.
# #     text.set_fontsize(fontsize)
#     bbox: Bbox = text.get_window_extent(fig.canvas.get_renderer())
#     adjusted_size = fontsize * rect_width / bbox.width
#     print(f'bbox: {bbox}, adjusted_size: {adjusted_size}')
#     text.set_fontsize(adjusted_size)


def plot_position_curves_figure(position_obj, include_velocity=True, include_accel=False, figsize=(24, 10)):
    """ Renders a figure with a position curve and optionally its higher-order derivatives """
    num_subplots = 1
    out_axes_list = []
    if include_velocity:
        num_subplots = num_subplots + 1
    if include_accel:
        num_subplots = num_subplots + 1
    subplots=(num_subplots, 1)
    fig = plt.figure(figsize=figsize, clear=True)
    gs = plt.GridSpec(subplots[0], subplots[1], figure=fig, hspace=0.02)
    
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(position_obj.time, position_obj.x, 'k')
    ax0.set_ylabel('pos_x')
    out_axes_list.append(ax0)
    
    if include_velocity:
        ax1 = fig.add_subplot(gs[1])
        # ax1.plot(position_obj.time, pos_df['velocity_x'], 'grey')
        # ax1.plot(position_obj.time, pos_df['velocity_x_smooth'], 'r')
        ax1.plot(position_obj.time, position_obj._data['velocity_x_smooth'], 'k')
        ax1.set_ylabel('Velocity_x')
        ax0.set_xticklabels([]) # this is intensionally ax[i-1], as we want to disable the tick labels on above plots        
        out_axes_list.append(ax1)

    if include_accel:  
        ax2 = fig.add_subplot(gs[2])
        # ax2.plot(position_obj.time, position_obj.velocity)
        # ax2.plot(position_obj.time, pos_df['velocity_x'])
        ax2.plot(position_obj.time, position_obj._data['acceleration_x'], 'k')
        # ax2.plot(position_obj.time, pos_df['velocity_y'])
        ax2.set_ylabel('Higher Order Terms')
        ax1.set_xticklabels([]) # this is intensionally ax[i-1], as we want to disable the tick labels on above plots
        out_axes_list.append(ax2)
    
    # Shared:
    # ax0.get_shared_x_axes().join(ax0, ax1)
    ax0.get_shared_x_axes().join(*out_axes_list)
    ax0.set_xticklabels([])
    ax0.set_xlim([position_obj.time[0], position_obj.time[-1]])

    return fig, out_axes_list

    


# ==================================================================================================================== #
# 2022-12-14 Batch Surprise Recomputation                                                                              #
# ==================================================================================================================== #



def _subfn_build_epoch_region_label(xy, text, ax, **labels_kwargs):
    """ places a text label inside a square area the top, just inside of it 
    the epoch at

    Used by:
        draw_epoch_regions: to draw the epoch name inside the epoch
    """
    if labels_kwargs is None:
        labels_kwargs = {}
    labels_y_offset = labels_kwargs.pop('y_offset', -0.05)
    # y = xy[1]
    y = xy[1] + labels_y_offset  # shift y-value for label so that it's below the artist
    return ax.text(xy[0], y, text, **({'ha': 'center', 'va': 'top', 'family': 'sans-serif', 'size': 14, 'rotation': 0} | labels_kwargs)) # va="top" places it inside the box if it's aligned to the top

# @function_attributes(short_name='draw_epoch_regions', tags=['epoch','matplotlib','helper'], input_requires=[], output_provides=[], uses=['BrokenBarHCollection'], used_by=[], creation_date='2023-03-28 14:23')
def draw_epoch_regions(epoch_obj, curr_ax, facecolor=('green','red'), edgecolors=("black",), alpha=0.25, labels_kwargs=None, defer_render=False, debug_print=False):
    """ plots epoch rectangles with customizable color, edgecolor, and labels on an existing matplotlib axis
    2022-12-14

    Info:
    
    https://matplotlib.org/stable/tutorials/intermediate/autoscale.html
    
    Usage:
        from neuropy.utils.matplotlib_helpers import draw_epoch_regions
        epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, defer_render=False, debug_print=False)

    Full Usage Examples:

    ## Example 1:
        active_filter_epochs = curr_active_pipeline.sess.replay
        active_filter_epochs

        if not 'stop' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
            
        if not 'label' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['label'] = active_filter_epochs['flat_replay_idx'].copy()

        active_filter_epoch_obj = Epoch(active_filter_epochs)
        active_filter_epoch_obj


        fig, ax = plt.subplots()
        ax.plot(post_update_times, flat_surprise_across_all_positions)
        ax.set_ylabel('Relative Entropy across all positions')
        ax.set_xlabel('t (seconds)')
        epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, facecolor=('red','cyan'), alpha=0.1, edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 14}, defer_render=True, debug_print=False)
        laps_epochs_collection, laps_epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.laps.as_epoch_obj(), ax, facecolor='red', edgecolors='black', labels_kwargs={'y_offset': -16.0, 'size':8}, defer_render=True, debug_print=False)
        replays_epochs_collection, replays_epoch_labels = draw_epoch_regions(active_filter_epoch_obj, ax, facecolor='orange', edgecolors=None, labels_kwargs=None, defer_render=False, debug_print=False)
        fig.show()


    ## Example 2:

        # Show basic relative entropy vs. time plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(post_update_times, flat_relative_entropy_results)
        ax.set_ylabel('Relative Entropy')
        ax.set_xlabel('t (seconds)')
        epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, defer_render=False, debug_print=False)
        fig.show()

    """
    # epoch_obj

    epoch_tuples = [(start_t, width_duration) for start_t, width_duration in zip(epoch_obj.starts, epoch_obj.durations)] # [(0.0, 1211.5580800310709), (1211.5580800310709, 882.3397767931456)]
    epoch_mid_t = [a_tuple[0]+(0.5*a_tuple[1]) for a_tuple in epoch_tuples] # used for labels

    curr_span_ymin = curr_ax.get_ylim()[0]
    curr_span_ymax = curr_ax.get_ylim()[1]
    curr_span_height = curr_span_ymax-curr_span_ymin
    # xrange: list of (float, float) The sequence of (left-edge-position, width) pairs for each bar.
    # yrange: (lower-edge, height) 
    epochs_collection = BrokenBarHCollection(xranges=epoch_tuples, yrange=(curr_span_ymin, curr_span_height), facecolor=facecolor, alpha=alpha, edgecolors=edgecolors, linewidths=(1,)) # , offset_transform=curr_ax.transData
    if debug_print:
        print(f'(curr_span_ymin, curr_span_ymax): ({curr_span_ymin}, {curr_span_ymax}), epoch_tuples: {epoch_tuples}')
    curr_ax.add_collection(epochs_collection)
    if labels_kwargs is not None:
        epoch_labels = [_subfn_build_epoch_region_label((a_mid_t, curr_span_ymax), a_label, curr_ax, **labels_kwargs) for a_label, a_mid_t in zip(epoch_obj.labels, epoch_mid_t)]
    else:
        epoch_labels = None
    if not defer_render:
        curr_ax.get_figure().canvas.draw()
    return epochs_collection, epoch_labels


def plot_overlapping_epoch_analysis_diagnoser(position_obj, epoch_obj):
    """ builds a MATPLOTLIB figure showing the position and velocity overlayed by the epoch intervals in epoch_obj. Useful for diagnosing overlapping epochs.
    Usage:
        from neuropy.utils.matplotlib_helpers import plot_overlapping_epoch_analysis_diagnoser
        fig, out_axes_list = plot_overlapping_epoch_analysis_diagnoser(sess.position, curr_active_pipeline.sess.laps.as_epoch_obj())
    """
    fig, out_axes_list = plot_position_curves_figure(position_obj, include_velocity=True, include_accel=False, figsize=(24, 10))
    for ax in out_axes_list:
        laps_epochs_collection, laps_epoch_labels = draw_epoch_regions(epoch_obj, ax, facecolor=('red','green'), edgecolors='black', labels_kwargs={'y_offset': -16.0, 'size':8, 'rotation':90}, defer_render=False, debug_print=False)
    fig.show()
    return fig, out_axes_list


# ==================================================================================================================== #
# 2023-05-09 Misc Utility Functions                                                                                    #
# ==================================================================================================================== #



def extract_figure_properties(fig):
    """ UNTESTED, UNFINISHED
    Extracts styles, formatting, and set options from a matplotlib Figure object.
    Returns a dictionary with the following keys:
        - 'title': the Figure title (if any)
        - 'xlabel': the label for the x-axis (if any)
        - 'ylabel': the label for the y-axis (if any)
        - 'xlim': the limits for the x-axis (if any)
        - 'ylim': the limits for the y-axis (if any)
        - 'xscale': the scale for the x-axis (if any)
        - 'yscale': the scale for the y-axis (if any)
        - 'legend': the properties of the legend (if any)
        - 'grid': the properties of the grid (if any)
        
    TO ADD:
        -   fig.get_figwidth()
            fig.get_figheight()
            # fig.set_figheight()

            print(f'fig.get_figwidth(): {fig.get_figwidth()}\nfig.get_figheight(): {fig.get_figheight()}')


        
        Usage:        
            curr_fig = plt.gcf()
            curr_fig = out.figures[0]
            curr_fig_properties = extract_figure_properties(curr_fig)
            curr_fig_properties

    """
    properties = {}
    
    # Extract title
    properties['title'] = fig._suptitle.get_text() if fig._suptitle else None
    
    # Extract axis labels and limits
    for ax in fig.get_axes():
        if ax.get_label() == 'x':
            properties['xlabel'] = ax.get_xlabel()
            properties['xlim'] = ax.get_xlim()
            properties['xscale'] = ax.get_xscale()
        elif ax.get_label() == 'y':
            properties['ylabel'] = ax.get_ylabel()
            properties['ylim'] = ax.get_ylim()
            properties['yscale'] = ax.get_yscale()
    
    # Extract legend properties
    if hasattr(fig, 'legend_'):
        legend = fig.legend_
        if legend:
            properties['legend'] = {
                'title': legend.get_title().get_text(),
                'labels': [t.get_text() for t in legend.get_texts()],
                'loc': legend._loc,
                'frameon': legend.get_frame_on(),
            }
    
    # Extract grid properties
    first_ax = fig.axes[0]
    grid = first_ax.get_gridlines()[0] if first_ax.get_gridlines() else None
    if grid:
        properties['grid'] = {
            'color': grid.get_color(),
            'linestyle': grid.get_linestyle(),
            'linewidth': grid.get_linewidth(),
        }
    
    return properties


# ==================================================================================================================== #
# 2023-06-05 Interactive Selection Helpers                                                                             #
# ==================================================================================================================== #
def add_range_selector(fig, ax, initial_selection=None, orientation="horizontal", on_selection_changed=None) -> SpanSelector:
    """ 2023-06-06 - a 1D version of `add_rectangular_selector` which adds a selection band to an existing axis

    from neuropy.utils.matplotlib_helpers import add_range_selector
    curr_pos = deepcopy(curr_active_pipeline.sess.position)
    curr_pos_df = curr_pos.to_dataframe()

    curr_pos_df.plot(x='t', y=['lin_pos'])
    fig, ax = plt.gcf(), plt.gca()
    range_selector, set_extents = add_range_selector(fig, ax, orientation="vertical", initial_selection=None) # (-86.91, 141.02)

    """
    assert orientation in ["horizontal", "vertical"]
    use_midline = False

    if use_midline:
        def update_mid_line(xmin, xmax):
            xmid = np.mean([xmin, xmax])
            mid_line.set_ydata(xmid)

        def on_move_callback(xmin, xmax):
            """ Callback whenever the range is moved. 

            """
            print(f'on_move_callback(xmin: {xmin}, xmax: {xmax})')
            update_mid_line(xmin, xmax)
    else:
        on_move_callback = None

    def select_callback(xmin, xmax):
        """
        Callback for range selection.
        """
        # indmin, indmax = np.searchsorted(x, (xmin, xmax))
        # indmax = min(len(x) - 1, indmax)
        print(f"({xmin:3.2f}, {xmax:3.2f})")
        if on_selection_changed is not None:
            """ call the user-provided callback """
            on_selection_changed(xmin, xmax)
        
    if initial_selection is not None:
        # convert to extents:
        (x0, x1) = initial_selection # initial_selection should be `(xmin, xmax)`
        extents = (min(x0, x1), max(x0, x1))
    else:
        extents = None
        
    props=dict(alpha=0.5, facecolor="tab:red")
    selector = SpanSelector(ax, select_callback, orientation, useblit=True, props=props, interactive=True, drag_from_anywhere=True, onmove_callback=on_move_callback) # Set useblit=True on most backends for enhanced performance.
    if extents is not None:
        selector.extents = extents
    
    ## Add midpoint line:
    if use_midline:
        mid_line = ax.axhline(linewidth=1, alpha=0.6, color='r', label='midline', linestyle="--")
        update_mid_line(*selector.extents)

    def set_extents(selection):
        """ can be called to set the extents on the selector object. Captures `selector` """
        if selection is not None:
            (x0, x1) = selection # initial_selection should be `(xmin, xmax)`
            extents = (min(x0, x1), max(x0, x1))
            selector.extents = extents
            
    return selector, set_extents

def add_rectangular_selector(fig, ax, initial_selection=None, on_selection_changed=None) -> RectangleSelector:
    """ 2023-05-16 - adds an interactive rectangular selector to an existing matplotlib figure/ax.
    
    Usage:
    
        from neuropy.utils.matplotlib_helpers import add_rectangular_selector

        fig, ax = curr_active_pipeline.computation_results['maze'].computed_data.pf2D.plot_occupancy()
        rect_selector, set_extents = add_rectangular_selector(fig, ax, initial_selection=grid_bin_bounds) # (24.82, 257.88), (125.52, 149.19)

    
    The returned RectangleSelector object can have its selection accessed via:
        rect_selector.extents # (25.508610487986658, 258.5627661142404, 128.10121504465053, 150.48449186696848)
    
    Or updated via:
        rect_selector.extents = (25, 258, 128, 150)

    """
    def select_callback(eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        print(f'({x1:3.2f}, {x2:3.2f}), ({y1:3.2f}, {y2:3.2f})')
        print(f"The buttons you used were: {eclick.button} {erelease.button}")
        if on_selection_changed is not None:
            """ call the user-provided callback """
            extents = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            on_selection_changed(extents)

        
    if initial_selection is not None:
        # convert to extents:
        (x0, x1), (y0, y1) = initial_selection # initial_selection should be `((xmin, xmax), (ymin, ymax))`
        extents = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    else:
        extents = None
        
    # ax = axs[0]
    # props = dict(facecolor='blue', alpha=0.5)
    props=None
    selector = RectangleSelector(ax, select_callback, useblit=True, button=[1, 3], minspanx=5, minspany=5, spancoords='data', interactive=True, ignore_event_outside=True, props=props) # spancoords='pixels', button=[1, 3]: disable middle button 
    if extents is not None:
        selector.extents = extents
    # fig.canvas.mpl_connect('key_press_event', toggle_selector)
    def set_extents(selection):
        """ can be called to set the extents on the selector object. Captures `selector` """
        if selection is not None:
            (x0, x1), (y0, y1) = selection # initial_selection should be `((xmin, xmax), (ymin, ymax))`
            extents = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            selector.extents = extents
            
    return selector, set_extents


# grid_bin_bounds updating versions __________________________________________________________________________________ #

def interactive_select_grid_bin_bounds_1D(curr_active_pipeline, epoch_name='maze'):
    """ allows the user to interactively select the grid_bin_bounds for the pf1D
    
    Usage:
        from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_1D
        fig, ax, range_selector, set_extents = interactive_select_grid_bin_bounds_1D(curr_active_pipeline, epoch_name='maze')
    """
    # from neuropy.utils.matplotlib_helpers import add_range_selector
    computation_result = curr_active_pipeline.computation_results[epoch_name]
    grid_bin_bounds_1D = computation_result.computation_config['pf_params'].grid_bin_bounds_1D
    fig, ax = computation_result.computed_data.pf1D.plot_occupancy() #plot_occupancy()
    # curr_pos = deepcopy(curr_active_pipeline.sess.position)
    # curr_pos_df = curr_pos.to_dataframe()
    # curr_pos_df.plot(x='t', y=['lin_pos'])
    # fig, ax = plt.gcf(), plt.gca()

    def _on_range_changed(xmin, xmax):
        # print(f'xmin: {xmin}, xmax: {xmax}')
        # xmid = np.mean([xmin, xmax])
        # print(f'xmid: {xmid}')
        print(f'new_grid_bin_bounds_1D: ({xmin}, {xmax})')

    # range_selector, set_extents = add_range_selector(fig, ax, orientation="vertical", initial_selection=grid_bin_bounds_1D, on_selection_changed=_on_range_changed) # (-86.91, 141.02)
    range_selector, set_extents = add_range_selector(fig, ax, orientation="horizontal", initial_selection=grid_bin_bounds_1D, on_selection_changed=_on_range_changed)
    return fig, ax, range_selector, set_extents

def interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input:bool=True):
    """ allows the user to interactively select the grid_bin_bounds for the pf2D
    
    Usage:
        from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_2D
        fig, ax, rect_selector, set_extents = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze')
    """
    # from neuropy.utils.matplotlib_helpers import add_rectangular_selector # interactive_select_grid_bin_bounds_2D
    computation_result = curr_active_pipeline.computation_results[epoch_name]
    grid_bin_bounds = computation_result.computation_config['pf_params'].grid_bin_bounds
    fig, ax = computation_result.computed_data.pf2D.plot_occupancy()
    rect_selector, set_extents = add_rectangular_selector(fig, ax, initial_selection=grid_bin_bounds) # (24.82, 257.88), (125.52, 149.19)
    
    def _on_update_grid_bin_bounds(new_grid_bin_bounds):
        """ called to update the grid_bin_bounds for all filtered_epochs with the new values (new_grid_bin_bounds) 
        Captures: `curr_active_pipeline`
        """
        print(f'_on_update_grid_bin_bounds(new_grid_bin_bounds: {new_grid_bin_bounds})')
        for epoch_name, computation_result in curr_active_pipeline.computation_results.items():
            computation_result.computation_config['pf_params'].grid_bin_bounds = new_grid_bin_bounds
                
    if should_block_for_input:
        print(f'blocking and waiting for user input. Press [enter] to confirm selection change or [esc] to revert with no change.')
        # hold plot until a keyboard key is pressed
        keyboardClick = False
        while keyboardClick != True:
            keyboardClick = plt.waitforbuttonpress() # plt.waitforbuttonpress() exits the inactive state as soon as either a key is pressed or the Mouse is clicked. However, the function returns True if a keyboard key was pressed and False if a Mouse was clicked
            if keyboardClick:
                # Button was pressed
                # if plt.get_current_fig_manager().toolbar.mode == '':
                # [Enter] was pressed
                confirmed_extents = rect_selector.extents
                print(f'user confirmed extents: {confirmed_extents}')
                if confirmed_extents is not None:
                    _on_update_grid_bin_bounds(confirmed_extents) # update the grid_bin_bounds.
                    x0, x1, y0, y1 = confirmed_extents
                    print(f"Add this to `specific_session_override_dict`:\n\n{curr_active_pipeline.get_session_context().get_initialization_code_string()}:dict(grid_bin_bounds=({(x0, x1), (y0, y1)})),\n")
                    
                plt.close() # close the figure
                return confirmed_extents
                # elif plt.get_current_fig_manager().toolbar.mode == '':
                #     # [Esc] was pressed
                #     print(f'user canceled selection with [Esc].')
                #     plt.close()
                #     return grid_bin_bounds
    else:
        return fig, ax, rect_selector, set_extents





