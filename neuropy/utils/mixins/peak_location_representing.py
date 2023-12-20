from typing import Optional, List, Dict, Tuple
from nptyping import NDArray
import numpy as np
from scipy import ndimage # used for `compute_placefield_center_of_masses`

def compute_placefield_center_of_mass_coord_indicies(tuning_curves: NDArray) -> NDArray:
    """ returns the coordinates (index-space, not track-position space) of the center of mass for each of the tuning_curves. """
    return np.squeeze(np.array([ndimage.center_of_mass(x) for x in tuning_curves]))




def _subfn_interpolate_coord_indicies_to_positions_1D(tuning_curve_CoM_coordinates: NDArray, xbin: NDArray) -> NDArray:
    """ for a single 1D array `tuning_curve_CoM_coordinates` of coordinates (index-space, not track-position space), returns the locations of the center of mass for each of the tuning_curves. 
    
        tuning_curve_CoM_coordinates: in coordinate (index) space

    """
    xbin_edge_labels = np.arange(len(xbin)) # the index range spanning all x-bins
    assert np.all(np.diff(xbin) > 0), f"requires monotonically increasing bins"
    assert np.allclose(np.diff(xbin), np.full_like(np.diff(xbin), np.diff(xbin)[0])), f'Requres equally spaced bins'
    tuning_curve_CoM_positions = np.interp(tuning_curve_CoM_coordinates, xp=xbin_edge_labels, fp=xbin) # in position space
    return tuning_curve_CoM_positions

def compute_placefield_center_of_mass_positions(tuning_curves: NDArray, xbin: NDArray, ybin: Optional[NDArray]=None) -> NDArray:
    """ returns the locations of the center of mass for each of the tuning_curves. """
    tuning_curve_CoM_coordinates = compute_placefield_center_of_mass_coord_indicies(tuning_curves)
    
    if ybin is not None:
        # 2D Case
        assert np.ndim(tuning_curves) == 2, f"{np.shape(tuning_curves)} is not 2D?"
        tuning_curve_x_CoM_positions = _subfn_interpolate_coord_indicies_to_positions_1D(np.squeeze(tuning_curve_CoM_coordinates[:, 0]), xbin) # in position space
        tuning_curve_y_CoM_positions = _subfn_interpolate_coord_indicies_to_positions_1D(np.squeeze(tuning_curve_CoM_coordinates[:, 1]), ybin)
        tuning_curve_CoM_positions = np.stack((tuning_curve_x_CoM_positions, tuning_curve_y_CoM_positions), axis=-1) # (79, 2)
            
    else:
        # 1D Case
        tuning_curve_CoM_positions = _subfn_interpolate_coord_indicies_to_positions_1D(tuning_curve_CoM_coordinates, xbin) # in position space
        
    return tuning_curve_CoM_positions


class ContinuousPeakLocationRepresentingMixin:
    """ Implementors provides peaks in position-space (e.g. a location on the maze) which are computed from a `ContinuousPeakLocationRepresentingMixin_peak_curves_variable` it provides, such as the turning curves.
        
    from neuropy.utils.mixins.peak_location_representing import ContinuousPeakLocationRepresentingMixin
    
    Provides:
        peak_tuning_curve_center_of_mass_bin_coordinates
        peak_tuning_curve_center_of_masses
        
    """
    @property
    def ContinuousPeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
        """ the variable that the peaks are calculated and returned for """
        return self.pdf_normalized_tuning_curves
    
    @property
    def peak_tuning_curve_center_of_mass_bin_coordinates(self) -> NDArray:
        """ returns the coordinates (in bin-index space) of the center of mass of each of the tuning curves."""
        return compute_placefield_center_of_mass_coord_indicies(self.ContinuousPeakLocationRepresentingMixin_peak_curves_variable) # in coordinate (index) space
    
    @property
    def peak_tuning_curve_center_of_masses(self) -> NDArray:
        """ returns the locations of the center of mass of each of the tuning curves."""
        return compute_placefield_center_of_mass_positions(self.ContinuousPeakLocationRepresentingMixin_peak_curves_variable, xbin=self.xbin, ybin=self.ybin)
    
    
    
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
			def PeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
				return self.ratemap.PeakLocationRepresentingMixin_peak_curves_variable
			

    """
    @property
    def PeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
        """ the variable that the peaks are calculated and returned for """
        return self.tuning_curves

    @property
    def peak_indicies(self) -> NDArray:
        if self.ndim > 1:
            original_data_shape = np.shape(self.PeakLocationRepresentingMixin_peak_curves_variable[0])
            return np.array([np.unravel_index(np.argmax(a_curve.flatten(), axis=0), original_data_shape) for a_curve in self.PeakLocationRepresentingMixin_peak_curves_variable]) # .shape: (79, 2)
            
        else:
            # 1D Decoder case:
            return np.array([np.argmax(x) for x in self.PeakLocationRepresentingMixin_peak_curves_variable])

    @property
    def peak_locations(self) -> NDArray:
        """ returns the peak locations using self.xbin_centers and self.peak_indicies """
        if self.ndim > 1:
            return  np.vstack([(self.xbin_centers[idx[0]], self.ybin_centers[idx[1]]) for idx in self.peak_indicies]) # (79, 2)
        else:
            # 1D Decoder case:
            return np.array([self.xbin_centers[idx] for idx in self.peak_indicies])
        
