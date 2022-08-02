from __future__ import annotations # otherwise have to do type like 'Ratemap'

from enum import Enum, IntEnum, auto, unique
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from neuropy.utils.misc import AutoNameEnum

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
    