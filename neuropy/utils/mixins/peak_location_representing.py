from typing import Optional, List, Dict, Tuple, Union
from nptyping import NDArray
import numpy as np
import pandas as pd
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
    
                


    def get_tuning_curve_peaks_all_info_dict(self, peak_mode='peaks', enable_sort_subpeaks_by_peak_heights: bool = True, **find_peaks_kwargs) -> Dict:
        """ returns the peaks in coordinate bin space 

        peak_mode: str: either ['CoM', 'peaks']
        enable_sort_subpeaks_by_peak_heights: bool = True: if True, the subpeaks returned are sorted by their height
        find_peaks_kwargs: only used if `peak_mode == 'peaks'`

        """
        if peak_mode == 'CoM':
            raise NotImplementedError
        elif peak_mode == 'peaks':
            # This mode can return multiple peaks for each aclu:
            from scipy.signal import find_peaks
            n_curves: int = np.shape(self.ContinuousPeakLocationRepresentingMixin_peak_curves_variable)[0] # usually n_neurons
            # Convert to unit max first:
            unit_max_tuning_curves = np.array([a_tuning_curve / np.nanmax(a_tuning_curve) for a_tuning_curve in self.ContinuousPeakLocationRepresentingMixin_peak_curves_variable])
            # active_ratemap.tuning_curves.shape # (73, 56) - (n_neurons, n_pos_bins)
            find_peaks_kwargs = ({'height': 0.2, 'width': 2} | find_peaks_kwargs) # for raw tuning_curves. height=0.25 requires that the secondary peaks are at least 25% the height of the main peak
            peaks_results_list = [find_peaks(unit_max_tuning_curves[i,:], **find_peaks_kwargs) for i in np.arange(n_curves)]
            # peaks_coordinates_list = [np.array(a_result[0]) for a_result in peaks_results_list]
            # peaks_positions_list = [_subfn_compute_general_positions_from_peak_indicies(coordinates, xbin=self.xbin, ybin=self.ybin) for coordinates in peaks_coordinates_list]

            series_idx_list = []
            peak_values_list = []
            peak_subpeak_index_list = []
            peak_info_optional_columns = {'series_idx': [], 'subpeak_idx': [], 'pos': [], 'bin_index': []}

            # max_num_included_subpeaks: int = 5

            # for aclu, peaks in peaks_dict.items():
            for series_idx, (peak_indicies, peak_info_dict) in enumerate(peaks_results_list):
                ## Enable sorting each subpeak by height:
                if enable_sort_subpeaks_by_peak_heights:
                    peak_heights = np.array(peak_info_dict['peak_heights'])
                    assert len(peak_heights) == len(peak_indicies)
                    # Get the indices to sort the array in descending order directly
                    height_sort_idxs = np.argsort(-peak_heights) # the `-` sign here indicates reversed order
                    peak_indicies = peak_indicies[height_sort_idxs]
                    peak_heights = peak_heights[height_sort_idxs]
                    peak_info_dict = {info_k:np.array(info_v)[height_sort_idxs] for info_k, info_v in peak_info_dict.items()}
                    ## should now be sorted.

                curr_peak_bin_coordinates = np.array(peak_indicies)
                curr_peak_positions = _subfn_compute_general_positions_from_peak_indicies(curr_peak_bin_coordinates, xbin=self.xbin, ybin=self.ybin)
                peak_info_optional_columns['pos'].extend(curr_peak_positions)
                for i, a_subpeak in enumerate(curr_peak_bin_coordinates):
                    peak_info_optional_columns['bin_index'].append(curr_peak_bin_coordinates[i])
                    for info_k, info_v in peak_info_dict.items():
                        if info_k not in peak_info_optional_columns:
                            peak_info_optional_columns[info_k] = []
                        peak_info_optional_columns[info_k].append(info_v[i])

                    series_idx_list.append(series_idx)
                    peak_subpeak_index_list.append(i)
                    peak_values_list.append(a_subpeak)

            peak_info_optional_columns['series_idx'] = np.array(series_idx_list)
            peak_info_optional_columns['subpeak_idx'] = np.array(peak_subpeak_index_list)

            return peak_info_optional_columns
        else:
            raise NotImplementedError(f"Unknown peak_mode: '{peak_mode}' specified. Known modes: ['CoM', 'peaks']")
        

    def get_tuning_curve_peaks_bin_coordinates(self, peak_mode='peaks', enable_sort_subpeaks_by_peak_heights: bool = True, **find_peaks_kwargs) -> Union[List,NDArray]:
        """ returns the peaks in coordinate bin space 

        peak_mode: str: either ['CoM', 'peaks']
        enable_sort_subpeaks_by_peak_heights: bool = True: if True, the subpeaks returned are sorted by their height
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
             
            if enable_sort_subpeaks_by_peak_heights:
                peaks_results_list = [find_peaks(unit_max_tuning_curves[i,:], **find_peaks_kwargs) for i in np.arange(n_curves)]
                peaks_coordinates_list = [np.array(peak_indicies)[np.argsort(-np.array(peak_info_dict['peak_heights']))] for (peak_indicies, peak_info_dict) in peaks_results_list]
            else:
                peaks_coordinates_list = [np.array(find_peaks(unit_max_tuning_curves[i,:], **find_peaks_kwargs)[0]) for i in np.arange(n_curves)]
            
            return peaks_coordinates_list
        else:
            raise NotImplementedError(f"Unknown peak_mode: '{peak_mode}' specified. Known modes: ['CoM', 'peaks']")


    def get_tuning_curve_peak_positions(self, peak_mode='peaks', enable_sort_subpeaks_by_peak_heights: bool = True, **find_peaks_kwargs) -> Union[List,NDArray]:
        """ returns the peaks in position space. """
        if peak_mode == 'CoM':
            return self.peak_tuning_curve_center_of_masses
        elif peak_mode == 'peaks':
            peaks_coordinates_list = self.get_tuning_curve_peaks_bin_coordinates(peak_mode=peak_mode, enable_sort_subpeaks_by_peak_heights=enable_sort_subpeaks_by_peak_heights, **find_peaks_kwargs)
            # This mode can return multiple peaks for each aclu:
            peaks_positions_list = []
            for coordinates in peaks_coordinates_list:
                # num_peaks = len(coordinates)
                positions = _subfn_compute_general_positions_from_peak_indicies(coordinates, xbin=self.xbin, ybin=self.ybin)
                peaks_positions_list.append(positions)
                
            return peaks_positions_list
        else:
            raise NotImplementedError(f"Unknown peak_mode: '{peak_mode}' specified. Known modes: ['CoM', 'peaks']")
    
    def get_tuning_curve_peak_df(self, peak_mode='peaks', enable_sort_subpeaks_by_peak_heights: bool = True, **find_peaks_kwargs) -> pd.DataFrame:
        """ returns a dataframe containing all info about the peaks.
        
        Usage:
        
            peaks_results_df = active_ratemap.get_tuning_curve_peak_df(height=0.2, width=None)
            peaks_results_df['aclu'] = peaks_results_df.series_idx.map(lambda x: active_ratemap.neuron_ids[x]) # Can add in an 'aclu' column like so
            peaks_results_df

        """
        if peak_mode == 'CoM':
            raise NotImplementedError
        elif peak_mode == 'peaks':
            return pd.DataFrame(self.get_tuning_curve_peaks_all_info_dict(peak_mode='peaks', enable_sort_subpeaks_by_peak_heights=enable_sort_subpeaks_by_peak_heights, **find_peaks_kwargs))
        else:
            raise NotImplementedError(f"Unknown peak_mode: '{peak_mode}' specified. Known modes: ['CoM', 'peaks']")
    
    @classmethod
    def peaks_dict_to_df(cls, peaks_dict: Dict, peaks_results_dict: Dict) -> pd.DataFrame:
        # peaks_dict, peaks_results_dict
        aclus_list = []
        peak_values_list = []
        peak_subpeak_index_list = []
        peak_info_optional_columns = {'bin_index': []}

        for aclu, peaks in peaks_dict.items():
            peak_info_tuple = peaks_results_dict.get(aclu, None)
            for i, a_subpeak in enumerate(peaks):
                if peak_info_tuple is not None:
                    peak_indicies, peak_info_dict = peak_info_tuple
                    peak_info_optional_columns['bin_index'].append(peak_indicies[i])
                    for info_k, info_v in peak_info_dict.items():
                        if info_k not in peak_info_optional_columns:
                            peak_info_optional_columns[info_k] = []
                        peak_info_optional_columns[info_k].append(info_v[i])

                aclus_list.append(aclu)
                peak_subpeak_index_list.append(i)
                peak_values_list.append(a_subpeak)

        return pd.DataFrame({'aclu': aclus_list, 'peak_position': peak_values_list, 'subpeak_index': peak_subpeak_index_list, **peak_info_optional_columns})


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
        
