"""
This type stub file was generated by pyright.
"""

import numpy as np

class ColorsUtil:
    """ A convenience class for managing good combinations of colors for data visualization and extending colormaps dynamically """
    class Colors:
        @property
        def active_rgba_colors(self): # -> ndarray[Any, Any]:
            ...
        
        @property
        def active_rgb_colors(self): # -> ndarray[Any, Any]:
            ...
        
        @property
        def active_cmap(self): # -> ListedColormap:
            ...
        
        @staticmethod
        def extended_tab20b_main_colors_hex(): # -> list[str]:
            ...
        
    
    
    colors = ...
    @staticmethod
    def compute_needed_single_hue_variations(n_needed_colors, colors): # -> tuple[int | Any, int]:
        """ Computes the number of subcategories needed to build a colormap that fits n_needed_colors. Determines the base size of the extant colormap automatically from the colors variable. """
        ...
    
    @staticmethod
    def pho_categorical_colormap(n_needed_colors, colors: np.array, debug_print=...): # -> ListedColormap:
        """ Builds a larger colormap with lumance adjusted variations of the colors in the provided colors array
        Inputs:
            colors should have two axis: the single_color_axis (of size 3 or 4) and the data_axis (of size N) 
        
        Usage:
            PhoColors.pho_categorical_colormap(40, PhoColors.colors.active_rgba_colors)
        """
        ...
    
    @staticmethod
    def categorical_cmap(nc, nsc, cmap=..., continuous=...): # -> ListedColormap:
        """ takes as input the number of categories (nc) and the number of subcategories (nsc) and returns a colormap with nc*nsc different colors, where for each category there are nsc colors of same hue.
            From https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10

        """
        ...
    
    @staticmethod
    def categorical_cmap_from_colors(nc, nsc, ccolors): # -> ListedColormap:
        """ takes as input the number of categories (nc) and the number of subcategories (nsc) and returns a colormap with nc*nsc different colors, where for each category there are nsc colors of same hue.
            From https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
        """
        ...
    


def get_neuron_colors(sort_indicies, cmap=..., debug_print=...) -> np.ndarray:
    """ returns the list of colors, an RGBA np.array of shape: 4 x n_neurons. 
    
    colors_array = get_neuron_colors(sort_indicies, cmap=None, debug_print=False)
    
    """
    ...
