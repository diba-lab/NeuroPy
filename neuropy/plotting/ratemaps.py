from __future__ import annotations # otherwise have to do type like 'Ratemap'

from enum import Enum, IntEnum, auto, unique
from collections import namedtuple
from ipywidgets import widgets
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


# from .. import core

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
from neuropy.utils.misc import AutoNameEnum, compute_paginated_grid_config, RowColTuple, PaginatedGridIndexSpecifierTuple, RequiredSubplotsTuple
from neuropy.utils.colors_util import get_neuron_colors


from .figure import Fig, compute_figure_size_pixels

## TODO: refactor plot_ratemap_1D and plot_ratemap_2D to a single flat function (if that's appropriate).
## TODO: refactor plot_ratemap_1D and plot_ratemap_2D to take a **kwargs and apply optional defaults (find previous code where I did that using the | and dict conversion. In my 3D code.


def add_inner_title(ax, title, loc, strokewidth=3, stroke_foreground='w', text_foreground='k', **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke    
    # Afont = {'family': 'serif',
    #     'backgroundcolor': 'blue',
    #     'color':  'white',
    #     'weight': 'normal',
    #     'size': 14,
    #     }
    prop = dict(path_effects=[withStroke(foreground=stroke_foreground, linewidth=strokewidth)],
                size=plt.rcParams['legend.fontsize'],
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
    FIRING_MAPS = auto() 
    
    
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


def plot_single_tuning_map_2D(xbin, ybin, pfmap, occupancy, neuron_extended_id: NeuronExtendedIdentityTuple=None, drop_below_threshold: float=0.0000001, plot_mode: enumTuningMap2DPlotMode=None, ax=None, brev_mode=PlotStringBrevityModeEnum.CONCISE):
    """Plots a single tuning curve Heatmap

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
        # plot_mode = enumTuningMap2DPlotMode.PCOLORFAST
        plot_mode = enumTuningMap2DPlotMode.IMSHOW
        
    use_special_overlayed_title = True
    
    # use_alpha_by_occupancy = False # Only supported in IMSHOW mode
    use_alpha_by_occupancy = False # Only supported in IMSHOW mode
    
    use_black_rendered_background_image = True 
    
    if ax is None:
        ax = plt.gca()
            
    curr_pfmap = np.array(pfmap.copy()) / np.nanmax(pfmap)
    if drop_below_threshold is not None:
        curr_pfmap[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy
    
    # curr_pfmap = np.rot90(np.fliplr(curr_pfmap)) ## Bug was introduced here! At least with pcolorfast, this order of operations is wrong!
    # curr_pfmap = np.rot90(curr_pfmap)
    # curr_pfmap = np.fliplr(curr_pfmap) # I thought stopping after this was sufficient, as the values lined up with the 1D placefields... but it seems to be flipped vertically now!
    
    ## Seems to work:
    curr_pfmap = np.rot90(curr_pfmap, k=-1)
    curr_pfmap = np.fliplr(curr_pfmap)
        
    # # curr_pfmap = curr_pfmap / np.nanmax(curr_pfmap) # for when the pfmap already had its transpose taken

    if plot_mode is enumTuningMap2DPlotMode.PCOLORFAST:
        im = ax.pcolorfast(
            xbin,
            ybin,
            curr_pfmap,
            cmap="jet", vmin=0.0
        )
        
    elif plot_mode is enumTuningMap2DPlotMode.PCOLORMESH:
        raise DeprecationWarning # 'Code not maintained, may be out of date'  
        mesh_X, mesh_Y = np.meshgrid(xbin, ybin)
        ax.pcolormesh(mesh_X, mesh_Y, curr_pfmap, cmap='jet', vmin=0, edgecolors='k', linewidths=0.1)
        # ax.pcolormesh(mesh_X, mesh_Y, curr_pfmap, cmap='jet', vmin=0)
        
    elif plot_mode is enumTuningMap2DPlotMode.PCOLOR: 
        raise DeprecationWarning # 'Code not maintained, may be out of date'
        im = ax.pcolor(
            xbin,
            ybin,
            np.rot90(np.fliplr(pfmap)) / np.nanmax(pfmap),
            cmap="jet",
            vmin=0,
        )    
    elif plot_mode is enumTuningMap2DPlotMode.IMSHOW:
        """ https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html """
        """ Use the brightness to reflect the confidence in the outcome. Could also use opacity. """
        # mesh_X, mesh_Y = np.meshgrid(xbin, ybin)
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], ybin[0], ybin[-1])
        # The extent keyword arguments controls the bounding box in data coordinates that the image will fill specified as (left, right, bottom, top) in data coordinates, the origin keyword argument controls how the image fills that bounding box, and the orientation in the final rendered image is also affected by the axes limits.
        extent = (xmin, xmax, ymin, ymax)
        # print(f'extent: {extent}')
        # extent = None

        vmax = np.abs(curr_pfmap).max()
                
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
        
        if use_black_rendered_background_image:
            # We'll also create a black background into which the pixels will fade
            # background_black = np.full((*curr_pfmap.shape, 3), 0, dtype=np.uint8)
            # bg_im = ax.imshow(background_black, **imshow_shared_kwargs, label='background_image')
            
            # Grey checkerboard background:
            # background_chessboard = np.add.outer(range(8), range(8)) % 2  # chessboard
            background_chessboard = _build_square_checkerboard_image(extent, num_checkerboard_squares_short_axis=8)
            bg_im = ax.imshow(background_chessboard, cmap=plt.cm.gray, alpha=0.25, interpolation='nearest', **imshow_shared_kwargs, label='background_image')

        else:
            bg_im = None
        
        im = ax.imshow(curr_pfmap, **main_plot_kwargs, label='main_image')
        ax.axis("off")
        
    else:
        raise NotImplementedError   
    
    # ax.vlines(200, 'ymin'=0, 'ymax'=1, 'r')
    # ax.set_xticks([25, 50])
    # ax.vline(50, 'r')
    # ax.vlines([50], 0, 1, transform=ax.get_xaxis_transform(), colors='r')
    # ax.vlines([50], 0, 1, colors='r')
    # brev_mode = PlotStringBrevityModeEnum.MINIMAL

    if neuron_extended_id is not None:    
        # old way independent of brev_mode:
        # extended_id_string = f'(shank {neuron_extended_id.shank}, cluster {neuron_extended_id.cluster})'
        # full_extended_id_string = f"Cell {neuron_extended_id.id} - {extended_id_string}"
        # new brev_mode dependent way:
        full_extended_id_string = brev_mode.extended_identity_formatting_string(neuron_extended_id)
    else:
        full_extended_id_string = ''
    
    final_string_components = [full_extended_id_string]
    if brev_mode.should_show_firing_rate_label:
        pf_firing_rate_string = f'{round(np.nanmax(pfmap),2)} Hz'
        final_string_components.append(pf_firing_rate_string)
    
    if use_special_overlayed_title:
        final_title = ' - '.join(final_string_components)
        # t = add_inner_title(ax, final_title, loc='upper left', strokewidth=1.0)
        t = add_inner_title(ax, final_title, loc='upper center', strokewidth=3.0, stroke_foreground='k', text_foreground='w') # loc = 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        t.patch.set_ec("none")
        # t.patch.set_alpha(0.5)
    else:
        # conventional way:
        final_title = '\n'.join(final_string_components)
        ax.set_title(final_title) # f"Cell {ratemap.neuron_ids[cell]} - {ratemap.get_extended_neuron_id_string(neuron_i=cell)} \n{round(np.nanmax(pfmap),2)} Hz"
    
    ax.set_label(final_title)
    return im
    

# all extracted from the 2D figures
def plot_ratemap_2D(ratemap: Ratemap, computation_config=None, included_unit_indicies=None, subplots:RowColTuple=(40, 3), fig_column_width:float=8.0, fig_row_height:float=1.0, resolution_multiplier:float=1.0, max_screen_figure_size=(None, None), fignum=1, enable_spike_overlay=False, spike_overlay_spikes=None, drop_below_threshold: float=0.0000001, brev_mode: PlotStringBrevityModeEnum=PlotStringBrevityModeEnum.CONCISE, plot_variable: enumTuningMap2DPlotVariables=enumTuningMap2DPlotVariables.TUNING_MAPS, plot_mode: enumTuningMap2DPlotMode=None, debug_print=False):
    """Plots heatmaps of placefields with peak firing rate
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
    """
    
    # grid_layout_mode = 'gridspec'
    grid_layout_mode = 'imagegrid'
    # grid_layout_mode = 'subplot'
    
    # last_figure_subplots_same_layout = False
    last_figure_subplots_same_layout = True
    
    if not isinstance(subplots, RowColTuple):
        subplots = RowColTuple(subplots[0], subplots[1])
    
    if included_unit_indicies is None:
        included_unit_indicies = np.arange(ratemap.n_neurons) # include all unless otherwise specified
    
    if plot_variable.name is enumTuningMap2DPlotVariables.TUNING_MAPS.name:
        active_maps = ratemap.tuning_curves[included_unit_indicies]
        title_substring = 'Placemaps'
    elif plot_variable.name == enumTuningMap2DPlotVariables.FIRING_MAPS.name:
        active_maps = ratemap.firing_maps[included_unit_indicies]
        title_substring = 'Firing Maps'
    else:
        raise ValueError

    nMapsToShow = len(active_maps)
    
    data_aspect_ratio = compute_data_aspect_ratio(ratemap.xbin, ratemap.ybin)
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
    
    
    # Paging Management: Constrain the subplots values to just those that you need
    subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=subplots.num_columns, max_subplots_per_page=max_subplots_per_page, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=last_figure_subplots_same_layout)
    num_pages = len(included_combined_indicies_pages)

    nfigures = num_pages
    # nfigures = nMapsToShow // np.prod(subplots) + 1 # "//" is floor division (rounding result down to nearest whole number)

    if fignum is None:
        if f := plt.get_fignums():
            fignum = f[-1] + 1
        else:
            fignum = 1

    figures, page_gs = [], []
    
    if grid_layout_mode == 'subplot':
        page_axes = []
        
    for fig_ind in range(nfigures):
        # Dynamic Figure Sizing: 
        curr_fig_page_grid_size = page_grid_sizes[fig_ind]
        if resolution_multiplier is None:
            resolution_multiplier = 1.0
        if (fig_column_width is not None) and (fig_row_height is not None):
            desired_single_map_width = fig_column_width * resolution_multiplier
            desired_single_map_height = fig_row_height * resolution_multiplier
        else:
            desired_single_map_width = 4.0 * resolution_multiplier
            desired_single_map_height = 1.0 * resolution_multiplier
         
        ## Figure size should be (Width, height)
        required_figure_size = ((float(curr_fig_page_grid_size.num_columns) * float(desired_single_map_width)), (float(curr_fig_page_grid_size.num_rows) * float(desired_single_map_height))) # (width, height)
       
        # max_screen_figure_size
        
        required_figure_size_px = compute_figure_size_pixels(required_figure_size)
        if debug_print:
            print(f'resolution_multiplier: {resolution_multiplier}, required_figure_size: {required_figure_size}, required_figure_size_px: {required_figure_size_px}') # this is figure size in inches

        active_figure_size = required_figure_size
        
        # If max_screen_figure_size is not None (it should be a two element tuple, specifying the max width and height in pixels for the figure:
        if max_screen_figure_size is not None:
            active_figure_size = list(active_figure_size) # convert to a list instead of a tuple to make it mutable
            if max_screen_figure_size[0] is not None:
                active_figure_size[0] = min(active_figure_size[0], max_screen_figure_size[0])
            if max_screen_figure_size[1] is not None:
                active_figure_size[1] = min(active_figure_size[1], max_screen_figure_size[1])
  
        active_figure_size = tuple(active_figure_size)              
        # active_figure_size=figsize
        # active_figure_size=required_figure_size
    
        if grid_layout_mode == 'gridspec':
            fig = plt.figure(fignum + fig_ind, figsize=active_figure_size, clear=True)
            if last_figure_subplots_same_layout:
                page_gs.append(GridSpec(subplot_no_pagination_configuration.num_rows, subplot_no_pagination_configuration.num_columns, figure=fig))
            else:
                # print(f'fig_ind {fig_ind}: curr_fig_page_grid_size: {curr_fig_page_grid_size}')
                page_gs.append(GridSpec(curr_fig_page_grid_size.num_rows, curr_fig_page_grid_size.num_columns, figure=fig))
            
            fig.subplots_adjust(hspace=0.2)
            
        elif grid_layout_mode == 'imagegrid':
            ## Configure Colorbar options:
            ### curr_cbar_mode: 'each', 'one', None
            # curr_cbar_mode = 'each'
            curr_cbar_mode = None
            
            # grid_rect = (0.01, 0.05, 0.98, 0.9) # (left, bottom, width, height) 
            grid_rect = 111
            # fig = plt.figure(fignum + fig_ind, figsize=active_figure_size, dpi=None, clear=True, tight_layout=True)
            fig = plt.figure(fignum + fig_ind, figsize=active_figure_size, dpi=None, clear=True, tight_layout=False)
            
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
            
        elif grid_layout_mode == 'subplot':
            # otherwise uses subplots mode:
            fig, axs = plt.subplots(ncols=curr_fig_page_grid_size.num_columns, nrows=curr_fig_page_grid_size.num_rows, figsize=active_figure_size, clear=True, constrained_layout=True)
            page_axes.append(axs)
            
        title_string = f'2D Placemaps {title_substring} ({len(ratemap.neuron_ids)} good cells)'
        
        if computation_config is not None:
            if computation_config.speed_thresh is not None:
                title_string = f'{title_string} (speed_threshold = {str(computation_config.speed_thresh)})'
            
        fig.suptitle(title_string)
        figures.append(fig)

    # New page-based version:
    for page_idx in np.arange(num_pages):
        if debug_print:
            print(f'page_idx: {page_idx}')
        if grid_layout_mode == 'imagegrid':
            active_page_grid = page_gs[page_idx]
            # print(f'active_page_grid: {active_page_grid}')
            
        for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
            # Need to convert to page specific:
            curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
            curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
            curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
            # print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')
            
            cell_idx = curr_included_unit_index
            pfmap = active_maps[a_linear_index]
            # Get the axis to plot on:
            if grid_layout_mode == 'gridspec':
                curr_ax = figures[page_idx].add_subplot(page_gs[page_idx][a_linear_index])
            elif grid_layout_mode == 'imagegrid':
                curr_ax = active_page_grid[curr_page_relative_linear_index]
            elif grid_layout_mode == 'subplot':
                curr_ax = page_axes[page_idx][curr_page_relative_row, curr_page_relative_col]
            
            # Plot the main heatmap for this pfmap:
            im = plot_single_tuning_map_2D(ratemap.xbin, ratemap.ybin, pfmap, ratemap.occupancy, neuron_extended_id=ratemap.neuron_extended_ids[cell_idx], drop_below_threshold=drop_below_threshold, brev_mode=brev_mode, plot_mode=plot_mode, ax=curr_ax)
            
            if enable_spike_overlay:
                spike_overlay_points = curr_ax.plot(spike_overlay_spikes[cell_idx][0], spike_overlay_spikes[cell_idx][1], markersize=2, marker=',', markeredgecolor='red', linestyle='none', markerfacecolor='red', alpha=0.10, label='spike_overlay_points')                
                spike_overlay_sc = curr_ax.scatter(spike_overlay_spikes[cell_idx][0], spike_overlay_spikes[cell_idx][1], s=2, c='white', alpha=0.10, marker=',', label='spike_overlay_sc')
            
            # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
            # cbar = fig.colorbar(im, cax=cbar_ax)
            # cbar.set_label("firing rate (Hz)")

        # Remove the unused axes if there are any:
        if grid_layout_mode == 'imagegrid':
            num_axes_to_remove = (len(active_page_grid) - 1) - curr_page_relative_linear_index
            if (num_axes_to_remove > 0):
                for a_removed_linear_index in np.arange(curr_page_relative_linear_index+1, len(active_page_grid)):
                    removal_ax = active_page_grid[a_removed_linear_index]
                    fig.delaxes(removal_ax)

            # Apply subplots adjust to fix margins:
            plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        
    return figures, page_gs

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
    # sorted_alt_tuple_neuron_ids = ratemap.metadata['tuple_neuron_ids'].copy()
    sorted_alt_tuple_neuron_ids = [sorted_alt_tuple_neuron_ids[a_sort_idx] for a_sort_idx in sort_ind]
    
    # sorted_tuning_curves = tuning_curves[sorted_neuron_ids, :]
    # sorted_neuron_id_labels = [f'Cell[{a_neuron_id}]' for a_neuron_id in sorted_neuron_ids]
    sorted_neuron_id_labels = [f'C[{sorted_neuron_ids[i]}]({sorted_alt_tuple_neuron_ids[i].shank}|{sorted_alt_tuple_neuron_ids[i].cluster})' for i in np.arange(len(sorted_neuron_ids))]
    
    # neurons_colors_array = np.zeros((4, n_neurons))
    for i, neuron_ind in enumerate(sort_ind):
        # color = cmap(i / len(sort_ind))
        # neurons_colors_array[:, i] = color
        color = neurons_colors_array[:, i]
        curr_neuron_id = sorted_neuron_ids[i]

        ax.fill_between(
            bin_cntr,
            i * pad,
            i * pad + tuning_curves[neuron_ind],
            color=color,
            ec=None,
            alpha=0.5,
            zorder=i + 1,
        )
        ax.plot(
            bin_cntr,
            i * pad + tuning_curves[neuron_ind],
            color=color,
            alpha=0.7,
        )
        ax.set_title('Cell[{}]'.format(curr_neuron_id)) # this doesn't appear to be visible, so what is it used for?

    # ax.set_yticks(list(range(len(sort_ind)) + 0.5))
    ax.set_yticks(list(np.arange(len(sort_ind)) + 0.5))
    # ax.set_yticklabels(list(sort_ind))
    ax.set_yticklabels(list(sorted_neuron_id_labels))
    
    ax.set_xlabel("Position")
    ax.spines["left"].set_visible(False)
    if normalize_xbin:
        ax.set_xlim([0, 1])
    ax.tick_params("y", length=0)
    ax.set_ylim([0, len(sort_ind)])
    # if self.run_dir is not None:
    #     ax.set_title(self.run_dir.capitalize() + " Runs only")

    return ax, sort_ind, neurons_colors_array


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
