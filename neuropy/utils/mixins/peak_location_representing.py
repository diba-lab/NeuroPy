import numpy as np


class PeakLocationRepresentingMixin:
    """ Implementor provides peaks.
    requires: .xbin_centers, .ybin_centers
    requires .tuning_curves
    
	Example:

		from neuropy.utils.mixins.peak_location_representing import PeakLocationRepresentingMixin

		class AClass:		
			...
			# PeakLocationRepresentingMixin conformances:
			@property
			def PeakLocationRepresentingMixin_peak_curves_variable(self) -> np.array:
				return self.ratemap.PeakLocationRepresentingMixin_peak_curves_variable
			

    """
    @property
    def PeakLocationRepresentingMixin_peak_curves_variable(self) -> np.array:
        """ the variable that the peaks are calculated and returned for """
        return self.tuning_curves

    @property
    def peak_indicies(self) -> np.array:
        if self.ndim > 1:
            original_data_shape = np.shape(self.PeakLocationRepresentingMixin_peak_curves_variable[0])
            return np.array([np.unravel_index(np.argmax(a_curve.flatten(), axis=0), original_data_shape) for a_curve in self.PeakLocationRepresentingMixin_peak_curves_variable]) # .shape: (79, 2)
            
        else:
            # 1D Decoder case:
            return np.array([np.argmax(x) for x in self.PeakLocationRepresentingMixin_peak_curves_variable])

    @property
    def peak_locations(self) -> np.array:
        """ returns the peak locations using self.xbin_centers and self.peak_indicies """
        if self.ndim > 1:
            return  np.vstack([(self.xbin_centers[idx[0]], self.ybin_centers[idx[1]]) for idx in self.peak_indicies]) # (79, 2)
        else:
            # 1D Decoder case:
            return np.array([self.xbin_centers[idx] for idx in self.peak_indicies])
        
