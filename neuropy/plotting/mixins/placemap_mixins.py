from neuropy.core.neuron_identities import PlotStringBrevityModeEnum
from neuropy.plotting.placemaps import plot_placefield_occupancy
from neuropy.plotting.ratemaps import enumTuningMap2DPlotVariables, plot_ratemap_1D, plot_ratemap_2D, plot_advanced_2D

class PfnD_PlotOccupancy_Mixin:
    
    def plot_occupancy(self, identifier_details_list=[]):
        active_pf_occupancy_2D_identifier_string = '2D Occupancy'
        active_pf_occupancy_2D_identifier_string = ' - '.join([active_pf_occupancy_2D_identifier_string] + identifier_details_list)
        title_string = ' '.join([active_pf_occupancy_2D_identifier_string])
        subtitle_string = ' '.join([f'{self.config.str_for_display(True)}'])
        occupancy_fig, occupancy_ax = plot_placefield_occupancy(self)
        occupancy_fig.suptitle(title_string, fontsize='14')
        occupancy_ax.set_title(subtitle_string, fontsize='10')
        return occupancy_fig, occupancy_ax
    
    

class PfnDPlottingMixin(PfnD_PlotOccupancy_Mixin):
    # Extracted fro the 1D figures:
    def plot_ratemaps_1D(self, ax=None, pad=2, normalize=True, sortby=None, cmap=None):
        """ Note that normalize is required to fit all of the plots on this kind of stacked figure. """
        # returns: ax , sort_ind, colors
        return plot_ratemap_1D(self.ratemap, ax=ax, pad=pad, normalize_tuning_curve=normalize, sortby=sortby, cmap=cmap)
    
    # all extracted from the 2D figures
    def plot_ratemaps_2D(self, **kwargs):
        """Plots heatmaps of placefields with peak firing rate
        
        Wraps neuropy.plotting.ratemaps.plot_ratemap_2D
        
        Defaults: 
        **({'subplots': (10, 8), 'resolution_multiplier': 2.0, 'fignum': None, 'enable_spike_overlay': True, 'drop_below_threshold': 1e-07, 'brev_mode': PlotStringBrevityModeEnum.CONCISE, 'plot_variable': enumTuningMap2DPlotVariables.TUNING_MAPS} | kwargs)
        """
        return plot_advanced_2D(self.ratemap, computation_config=self.config, **({'subplots': (10, 8), 'resolution_multiplier': 2.0, 'fignum': None, 'enable_spike_overlay': True, 'spike_overlay_spikes':self.spk_pos, 'extended_overlay_points_datasource_dicts':None, 'drop_below_threshold': 1e-07, 'brev_mode': PlotStringBrevityModeEnum.CONCISE, 'plot_variable': enumTuningMap2DPlotVariables.TUNING_MAPS} | kwargs))
        # return plot_ratemap_2D(self.ratemap, computation_config=self.config, **({'subplots': (10, 8), 'resolution_multiplier': 2.0, 'fignum': None, 'enable_spike_overlay': True, 'spike_overlay_spikes':self.spk_pos, 'extended_overlay_points_datasource_dicts':None, 'drop_below_threshold': 1e-07, 'brev_mode': PlotStringBrevityModeEnum.CONCISE, 'plot_variable': enumTuningMap2DPlotVariables.TUNING_MAPS} | kwargs))
        
        
        