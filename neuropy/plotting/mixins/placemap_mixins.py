from copy import deepcopy
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum
from neuropy.plotting.placemaps import plot_placefield_occupancy
from neuropy.plotting.ratemaps import enumTuningMap2DPlotVariables, plot_ratemap_1D, plot_ratemap_2D, plot_ratemap_2D
from neuropy.utils.debug_helpers import safely_accepts_kwargs
from neuropy.utils.matplotlib_helpers import FormattedFigureText, FigureMargins

class PfnD_PlotOccupancy_Mixin:
    @safely_accepts_kwargs
    def plot_occupancy(self, identifier_details_list=[], fig=None, ax=None, active_context=None, **kwargs):
        """ the actually used plotting function. 
        Calls `plot_placefield_occupancy` to do the real plotting. Mostly just sets the title, subtitle, etc.
        #TODO 2023-06-13 19:25: - [ ] Fix `self.config.str_for_display(is_2D)` to enable excluding irrelevant items by includelist
        """
        if self.ndim > 1:
            active_pf_occupancy_identifier_string = '2D Occupancy'
            is_2D = True
        else:
            active_pf_occupancy_identifier_string = '1D Occupancy'
            is_2D = False
            

        use_flexitext_titles = kwargs.get('use_flexitext_titles', True) # use fancy flexitext titles if this is true
        
        active_pf_occupancy_identifier_string = ' - '.join([active_pf_occupancy_identifier_string] + identifier_details_list)
        title_string = ' '.join([active_pf_occupancy_identifier_string])
        subtitle_string = ' '.join([f'{self.config.str_for_display(is_2D)}'])    
        
        occupancy_fig, occupancy_ax = plot_placefield_occupancy(self, fig=fig, ax=ax, **kwargs)

        occupancy_fig.canvas.manager.set_window_title(title_string) # sets the window's title
        
        if (active_context is None) or (not use_flexitext_titles):
            occupancy_fig.suptitle(title_string, fontsize='14', wrap=True)
            occupancy_ax.set_title(subtitle_string, fontsize='10', wrap=True)
        else:
            from flexitext import flexitext ## flexitext version
            # Clear the normal text:
            occupancy_fig.suptitle('')
            occupancy_ax.set_title('')
            text_formatter = FormattedFigureText()
            text_formatter.setup_margins(occupancy_fig)
            active_config = deepcopy(self.config)
            # active_config.float_precision = 1
            
            subtitle_string = '\n'.join([f'{active_config.str_for_display(is_2D)}'])
            header_text_obj = flexitext(text_formatter.left_margin, 0.90, f'<size:22><weight:bold>{title_string}</></>\n<size:10>{subtitle_string}</>', va="bottom", xycoords="figure fraction")
            footer_text_obj = text_formatter.add_flexitext_context_footer(active_context=active_context) # flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")

        return occupancy_fig, occupancy_ax
    
    

class PfnDPlottingMixin(PfnD_PlotOccupancy_Mixin):
    # Extracted fro the 1D figures:
    # @safely_accepts_kwargs
    def plot_ratemaps_1D(self, ax=None, pad=2, normalize=True, sortby=None, cmap=None, active_context=None, **kwargs):
        """ Note that normalize is required to fit all of the plots on this kind of stacked figure. """
        # returns: ax , sort_ind, colors
        return plot_ratemap_1D(self.ratemap, ax=ax, pad=pad, normalize_tuning_curve=normalize, sortby=sortby, cmap=cmap, active_context=active_context, **kwargs)
    
    # all extracted from the 2D figures
    def plot_ratemaps_2D(self, **kwargs):
        """Plots heatmaps of placefields with peak firing rate
        
        Wraps neuropy.plotting.ratemaps.plot_ratemap_2D
        
        Defaults: 
        **({'subplots': (10, 8), 'resolution_multiplier': 2.0, 'fignum': None, 'enable_spike_overlay': True, 'drop_below_threshold': 1e-07, 'brev_mode': PlotStringBrevityModeEnum.CONCISE, 'plot_variable': enumTuningMap2DPlotVariables.TUNING_MAPS} | kwargs)
        """
        return plot_ratemap_2D(self.ratemap, computation_config=self.config, **({'subplots': (10, 8), 'resolution_multiplier': 2.0, 'fignum': None, 'enable_spike_overlay': True, 'spike_overlay_spikes':self.spk_pos, 'extended_overlay_points_datasource_dicts':None, 'drop_below_threshold': 1e-07,
                         'brev_mode': PlotStringBrevityModeEnum.CONCISE, 'plot_variable': enumTuningMap2DPlotVariables.TUNING_MAPS, 'use_special_overlayed_title': True} | kwargs))
        # return plot_ratemap_2D(self.ratemap, computation_config=self.config, **({'subplots': (10, 8), 'resolution_multiplier': 2.0, 'fignum': None, 'enable_spike_overlay': True, 'spike_overlay_spikes':self.spk_pos, 'extended_overlay_points_datasource_dicts':None, 'drop_below_threshold': 1e-07, 'brev_mode': PlotStringBrevityModeEnum.CONCISE, 'plot_variable': enumTuningMap2DPlotVariables.TUNING_MAPS} | kwargs))
        

        