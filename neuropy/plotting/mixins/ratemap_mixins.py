

from neuropy.plotting.ratemaps import enumTuningMap2DPlotVariables, enumTuningMap2DPlotMode, plot_ratemap_1D, plot_ratemap_2D


class RatemapPlottingMixin:
	def plot(self, computation_config=None, included_unit_indicies=None, subplots=(10, 8), figsize=(6, 10), fignum=None, enable_spike_overlay=False, spike_overlay_spikes=None, drop_below_threshold: float=0.0000001, plot_variable: enumTuningMap2DPlotVariables=enumTuningMap2DPlotVariables.TUNING_MAPS, plot_mode: enumTuningMap2DPlotMode=None, **kwargs):
		if self.n_dim == 1:
			return plot_ratemap_1D(self, **kwargs) # TODO: allow passing options
		elif self.n_dim == 2:
			return plot_ratemap_2D(self, computation_config=computation_config, included_unit_indicies=included_unit_indicies, subplots=subplots, figsize=figsize, fignum=fignum, enable_spike_overlay=enable_spike_overlay, spike_overlay_spikes=spike_overlay_spikes, drop_below_threshold=drop_below_threshold, plot_variable=plot_variable, plot_mode=plot_mode, **kwargs)
		else:
			raise ValueError

