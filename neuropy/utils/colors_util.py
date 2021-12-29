## Plotting Colors:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, ColorConverter
from matplotlib.colors import Normalize, to_rgba_array, to_hex


class ColorsUtil:
    """ A convenience class for managing good combinations of colors for data visualization and extending colormaps dynamically """
    class Colors:
        @property
        def active_rgba_colors(self):
            return to_rgba_array(ColorsUtil.Colors.extended_tab20b_main_colors_hex()) # np.shape(tab20b_main_rgba_colors) # (5, 4)

        @property
        def active_rgb_colors(self):
             return self.active_rgba_colors[:,:-1] # np.shape(tab20b_main_rgb_colors) # (5, 3)

        @property
        def active_cmap(self):
             return matplotlib.colors.ListedColormap(self.active_rgba_colors)

        @staticmethod
        def extended_tab20b_main_colors_hex():
            c_weird_bright_orange = '#b25809'
            # c_weird_orange = '#846739'
            c_dark_teal = '#397084'
            pho_modified_tab20b_main_colors_hex = ['#843c39', c_weird_bright_orange, '#8c6d31', '#637939', c_dark_teal, '#393b79', '#7b4173']
            return pho_modified_tab20b_main_colors_hex

    colors = Colors()
    
    @staticmethod
    def compute_needed_single_hue_variations(n_needed_colors, colors):
        """ Computes the number of subcategories needed to build a colormap that fits n_needed_colors. Determines the base size of the extant colormap automatically from the colors variable. """
        if isinstance(colors, str):
            # if it's a string, treat it as a colormap name
            cmap_name = colors # it's actually a name of a cmap, not a cmap object
            cmap = plt.get_cmap(cmap_name) # get the object
            base_colormap_n_colors = cmap.N
        elif isinstance(colors, np.ndarray):
            colors_shape = np.shape(colors) # colors_shape: (5, 3)
            if ((colors_shape[0] != 3) and (colors_shape[0] != 4)):
                # expected form
                single_color_axis = 1
                data_axis = 0
                base_colormap_n_colors = colors_shape[data_axis]
            else:
                assert ((colors_shape[1] == 3) or (colors_shape[1] == 4)), "No dimension of colors array is of length 3 or 4. This should be RGB or RGBA data."
                single_color_axis = 0
                data_axis = 1
                base_colormap_n_colors = colors_shape[data_axis]
        else:
            # assume it's a matplotlib colormap:
            base_colormap_n_colors = colors.N
            
        needed_single_hue_variations = int(np.ceil(n_needed_colors / base_colormap_n_colors)) # for n_colors = 40, needed_repeats = 2
        
        num_needed_categories = min(base_colormap_n_colors, n_needed_colors)
        num_needed_subcategories = needed_single_hue_variations
        return num_needed_categories, num_needed_subcategories
        
        
    @staticmethod
    def pho_categorical_colormap(n_needed_colors, colors: np.array, debug_print=False):
        """ Builds a larger colormap with lumance adjusted variations of the colors in the provided colors array
        Inputs:
            colors should have two axis: the single_color_axis (of size 3 or 4) and the data_axis (of size N) 
        
        Usage:
            PhoColors.pho_categorical_colormap(40, PhoColors.colors.active_rgba_colors)
        """
        colors_shape = np.shape(colors) # colors_shape: (5, 3)
        if debug_print:
            print(f'colors_shape: {np.shape(colors)}')
        if ((colors_shape[0] != 3) and (colors_shape[0] != 4)):
            # expected form
            pass
        else:
            assert ((colors_shape[0] == 3) or (colors_shape[0] == 4)), "No dimension of colors array is of length 3 or 4. This should be RGB or RGBA data."
            colors = colors.T # transpose the colors so they're in the correct form:
            colors_shape = np.shape(colors)

        single_color_axis = 1
        data_axis = 0
        base_colormap_n_colors = colors_shape[data_axis]
        needed_single_hue_variations = int(np.ceil(n_needed_colors / base_colormap_n_colors)) # for n_colors = 40, needed_repeats = 2
        if debug_print:
            print(f'needed_single_hue_variations: {needed_single_hue_variations}, base_colormap_n_colors: {base_colormap_n_colors}, n_needed_colors: {n_needed_colors}')
        # cmap = categorical_cmap(base_colormap_n_colors, needed_single_hue_variations, cmap=cmap, continuous=False)
        return ColorsUtil.categorical_cmap_from_colors(base_colormap_n_colors, needed_single_hue_variations, ccolors=colors)

    
    @staticmethod
    def _categorical_subdivide_colors(ccolors, nc, nsc):
        cols = np.zeros((nc*nsc, 3))
        for i, c in enumerate(ccolors):
            chsv = matplotlib.colors.rgb_to_hsv(c[:3])
            arhsv = np.tile(chsv,nsc).reshape(nsc,3)
            arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
            arhsv[:,2] = np.linspace(chsv[2],1,nsc)
            rgb = matplotlib.colors.hsv_to_rgb(arhsv)
            cols[i*nsc:(i+1)*nsc,:] = rgb       
        return cols

    @staticmethod
    def categorical_cmap(nc, nsc, cmap='tab20b', continuous=False):
        """ takes as input the number of categories (nc) and the number of subcategories (nsc) and returns a colormap with nc*nsc different colors, where for each category there are nsc colors of same hue.
            From https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10

        """
        if isinstance(cmap, str):
            cmap_name = cmap.copy() # it's actually a name of a cmap, not a cmap object
            cmap = plt.get_cmap(cmap_name) # get the object

        if nc > cmap.N:
            raise ValueError("Too many categories for colormap.")
        if continuous:
            ccolors = cmap(np.linspace(0,1,nc))
        else:
            ccolors = cmap(np.arange(nc, dtype=int))
        cols = ColorsUtil._categorical_subdivide_colors(ccolors, nc, nsc)
        cmap = matplotlib.colors.ListedColormap(cols)
        return cmap


    @staticmethod
    def categorical_cmap_from_colors(nc, nsc, ccolors):
        """ takes as input the number of categories (nc) and the number of subcategories (nsc) and returns a colormap with nc*nsc different colors, where for each category there are nsc colors of same hue.
            From https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
        """
        cols = ColorsUtil._categorical_subdivide_colors(ccolors, nc, nsc)
        cmap = matplotlib.colors.ListedColormap(cols)
        return cmap



def get_neuron_colors(sort_indicies, cmap=None):
    # returns the list of colors, an RGBA np.array of shape: 4 x n_neurons. 
    if cmap is None:
        # when none, get Pho's defaults which look good for large number of neurons.
        cmap = ColorsUtil.colors.active_cmap.copy()
    elif isinstance(cmap, str):
        cmap_name = cmap
        cmap = mpl.cm.get_cmap(cmap_name)
    else:
        # otherwise it's just a cmap object
        pass
    n_neurons = len(sort_indicies)
    
    if (cmap.N < n_neurons):
        print(f'The specified cmap supports less colors than n_neurons (supports {cmap.N}, n_neurons: {n_neurons}). An extended colormap will be built.')
        # Extend the cmap if there are more neurons in n_neurons than the original colormap supports:
        num_needed_categories, num_needed_subcategories = ColorsUtil.compute_needed_single_hue_variations(n_neurons, cmap)
        extended_cmap = ColorsUtil.categorical_cmap(num_needed_categories, num_needed_subcategories, cmap=cmap)
    
    colors_array = np.zeros((4, n_neurons))
    for i, neuron_ind in enumerate(sort_indicies):
        colors_array[:, i] = extended_cmap(i / len(sort_indicies))
        
    return colors_array
