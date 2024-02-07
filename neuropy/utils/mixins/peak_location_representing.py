from typing import Optional, List, Dict, Tuple, Union
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


def _subfn_compute_general_positions_from_peak_indicies(peak_coordinate_indicies, xbin: NDArray, ybin: Optional[NDArray]=None) -> NDArray:
    """ returns the locations in position space for each of provided coordinates in bin space. """
    
    if ybin is not None:
        # 2D Case
        # assert np.ndim(tuning_curves) == 2, f"{np.shape(tuning_curves)} is not 2D?"
        tuning_curve_x_CoM_positions = _subfn_interpolate_coord_indicies_to_positions_1D(np.squeeze(peak_coordinate_indicies[:, 0]), xbin) # in position space
        tuning_curve_y_CoM_positions = _subfn_interpolate_coord_indicies_to_positions_1D(np.squeeze(peak_coordinate_indicies[:, 1]), ybin)
        tuning_curve_CoM_positions = np.stack((tuning_curve_x_CoM_positions, tuning_curve_y_CoM_positions), axis=-1) # (79, 2)
            
    else:
        # 1D Case
        tuning_curve_CoM_positions = _subfn_interpolate_coord_indicies_to_positions_1D(peak_coordinate_indicies, xbin) # in position space
        
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
    
                
    def get_tuning_curve_peaks_bin_coordinates(self, peak_mode='peaks', **find_peaks_kwargs) -> Union[List,NDArray]:
        """ returns the peaks in coordinate bin space 

        peak_mode: str: either ['CoM', 'peaks']
        find_peaks_kwargs: only used if `peak_mode == 'peaks'`

        """
        if peak_mode == 'CoM':
            return self.peak_tuning_curve_center_of_mass_bin_coordinates
        elif peak_mode == 'peaks':
            # This mode can return multiple peaks for each aclu:
            from scipy.signal import find_peaks
            n_curves: int = np.shape(self.ContinuousPeakLocationRepresentingMixin_peak_curves_variable)[0] # usually n_neurons
            # Convert to unit max first:
            unit_max_tuning_curves = np.array([a_tuning_curve / np.nanmax(a_tuning_curve) for a_tuning_curve in self.ContinuousPeakLocationRepresentingMixin_peak_curves_variable])
            # active_ratemap.tuning_curves.shape # (73, 56) - (n_neurons, n_pos_bins)
            find_peaks_kwargs = ({'height': 0.2, 'width': 2} | find_peaks_kwargs) # for raw tuning_curves. height=0.25 requires that the secondary peaks are at least 25% the height of the main peak
            # print(f'find_peaks_kwargs: {find_peaks_kwargs}')
            # peaks_results_list = [find_peaks(unit_max_tuning_curves[i,:], **find_peaks_kwargs) for i in np.arange(n_curves)]
            peaks_coordinates_list = [np.array(find_peaks(unit_max_tuning_curves[i,:], **find_peaks_kwargs)[0]) for i in np.arange(n_curves)]
            # peaks_results_dict = dict(zip(self.neuron_ids, peaks_results_list))
            # peaks_dict = {k:v[0] for k,v in peaks_results_dict.items()} # [0] outside the find_peaks function gets the location of the peak
            # aclu_n_peaks_dict = {k:len(v) for k,v in peaks_dict.items()} # number of peaks ("models" for each aclu)
            # unimodal_peaks_dict = {k:v for k,v in peaks_dict.items() if len(v) < 2}
            return peaks_coordinates_list
        else:
            raise NotImplementedError(f"Unknown peak_mode: '{peak_mode}' specified. Known modes: ['CoM', 'peaks']")


    def get_tuning_curve_peak_positions(self, peak_mode='peaks', **find_peaks_kwargs) -> Union[List,NDArray]:
        """ returns the peaks in position space. """
        if peak_mode == 'CoM':
            return self.peak_tuning_curve_center_of_masses
        elif peak_mode == 'peaks':
            peaks_coordinates_list = self.get_tuning_curve_peaks_bin_coordinates(peak_mode=peak_mode, **find_peaks_kwargs)
            # This mode can return multiple peaks for each aclu:
            peaks_positions_list = []
            for coordinates in peaks_coordinates_list:
                # num_peaks = len(coordinates)
                positions = _subfn_compute_general_positions_from_peak_indicies(coordinates, xbin=self.xbin, ybin=self.ybin)
                peaks_positions_list.append(positions)
                
            return peaks_positions_list
        else:
            raise NotImplementedError(f"Unknown peak_mode: '{peak_mode}' specified. Known modes: ['CoM', 'peaks']")
    

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
        
