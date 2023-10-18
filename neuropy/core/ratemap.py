from copy import deepcopy
from warnings import warn
import numpy as np
import h5py
from neuropy.core.neuron_identities import NeuronIdentitiesDisplayerMixin
from neuropy.utils.mixins.binning_helpers import BinnedPositionsMixin
from neuropy.plotting.mixins.ratemap_mixins import RatemapPlottingMixin
from neuropy.utils import mathutil
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin, HDF_Converter
from . import DataWriter

class Ratemap(HDFMixin, NeuronIdentitiesDisplayerMixin, RatemapPlottingMixin, BinnedPositionsMixin, DataWriter):
    """A Ratemap holds information about each unit's firing rate across binned positions. 
        In addition, it also holds (tuning curves).
        
        
    Internal:
        # Map Properties:
        self.occupancy 
        self.spikes_maps
        self.tuning_curves
        self.unsmoothed_tuning_maps

        # Neuron Identity:
        self._neuron_ids
        self._neuron_extended_ids

        # Position Identity:
        self.xbin
        self.ybin
        
        # Other:
        self.metadata
        
    Args:
        NeuronIdentitiesDisplayerMixin (_type_): _description_
        RatemapPlottingMixin (_type_): _description_
        DataWriter (_type_): _description_
    """
    def __init__(self, tuning_curves, unsmoothed_tuning_maps=None, spikes_maps=None, 
        xbin=None, ybin=None, occupancy=None,
        neuron_ids=None, neuron_extended_ids=None, metadata=None) -> None:
        
        super().__init__()

        self.spikes_maps = np.asarray(spikes_maps)
        self.tuning_curves = np.asarray(tuning_curves)
        if unsmoothed_tuning_maps is not None:
            self.unsmoothed_tuning_maps = np.asarray(unsmoothed_tuning_maps)
        else:
            self.unsmoothed_tuning_maps = None
        
        if neuron_ids is not None:
            assert len(neuron_ids) == self.tuning_curves.shape[0]
            self._neuron_ids = neuron_ids
        if neuron_extended_ids is not None:
            assert len(neuron_extended_ids) == self.tuning_curves.shape[0]
            assert len(neuron_extended_ids) == len(self._neuron_ids)
            # NeuronExtendedIdentityTuple objects
            self._neuron_extended_ids = neuron_extended_ids   
        
        self.xbin = xbin
        self.ybin = ybin
        self.occupancy = occupancy

        self.metadata = metadata
    
    # NeuronIdentitiesDisplayerMixin requirements
    @property
    def neuron_ids(self):
        """The neuron_ids property."""
        return self._neuron_ids
    @neuron_ids.setter
    def neuron_ids(self, value):
        self._neuron_ids = value
       
       
    @property
    def neuron_extended_ids(self):
        """The neuron_extended_ids property."""
        return self._neuron_extended_ids
    @neuron_extended_ids.setter
    def neuron_extended_ids(self, value):
        self._neuron_extended_ids = value
     

    @property
    def n_neurons(self):
        return self.tuning_curves.shape[0]

    @property
    def ndim(self):
        return self.tuning_curves.ndim - 1
    
    
    @property
    def normalized_tuning_curves(self):
        return self.pdf_normalized_tuning_curves
        

    # ---------------------- occupancy properties -------------------------
    @property
    def never_visited_occupancy_mask(self):
        """ a boolean mask that's True everyhwere the animal has never visited according to self.occupancy, and False everyhwere else. """
        return Ratemap.build_never_visited_mask(self.occupancy)
    
    
    @property
    def nan_never_visited_occupancy(self):
        """ returns the self.occupancy after replacing all never visited locations, indicated by a zero occupancy, by NaNs for the purpose of building visualizations. """
        return Ratemap.nan_never_visited_locations(self.occupancy)
    
    # --------------------- Normalization and Scaling Helpers -------------------- #
    @property
    def pdf_normalized_tuning_curves(self):
        """ AOC (area-under-curve) normalization for tuning curves. """
        return Ratemap.perform_AOC_normalization(self.tuning_curves)
        
    @property
    def tuning_curve_peak_firing_rates(self):
        """ the non-normalized peak location of each tuning curve. Represents the peak firing rate of that curve. """
        warn('tuning_curve_peak_firing_rates: was accessed, but does not give the actual cell firing rate because of the smoothing. Use Ratemap.tuning_curve_unsmoothed_peak_firing_rates for accurate firing rates in Spikes / Second ')
        return np.array([np.nanmax(a_tuning_curve) for a_tuning_curve in self.tuning_curves])
    
    @property
    def tuning_curve_unsmoothed_peak_firing_rates(self):
        """ the non-normalized and unsmoothed value of the maximum firing rate at the peak of each tuning curve in NumSpikes/Second. Represents the peak firing rate of that curve. """
        assert self.unsmoothed_tuning_maps is not None, "self.unsmoothed_tuning_maps is None! Did you pass it in while building the Ratemap?"
        return np.array([np.nanmax(a_tuning_curve) for a_tuning_curve in self.unsmoothed_tuning_maps])
    
        
    @property
    def unit_max_tuning_curves(self):
        """ tuning curves normalized by scaling their max value down to 1.0.
            The peak of each placefield will have height 1.0.
        """
        unit_max_tuning_curves = [a_tuning_curve / np.nanmax(a_tuning_curve) for a_tuning_curve in self.tuning_curves]
        validate_unit_max = [np.nanmax(a_unit_max_tuning_curve) for a_unit_max_tuning_curve in unit_max_tuning_curves]
        # print(f'validate_unit_max: {validate_unit_max}')
        assert np.allclose(validate_unit_max, np.full_like(validate_unit_max, 1.0), equal_nan=True), f"unit_max_tuning_curves doesn't have a max==1.0 after scaling!!! Maximums: {validate_unit_max}"
        return np.array(unit_max_tuning_curves)
    
    
    @property
    def minmax_normalized_tuning_curves(self):
        """ tuning curves normalized by scaling their min/max values down to the range (0, 1).
            The peak of each placefield will have height 1.0.
        """
        return Ratemap.nanmin_nanmax_scaler(self.tuning_curves)

    # Other ______________________________________________________________________________________________________________ #

    def get_sort_indicies(self, sortby=None):
        # curr_tuning_curves = self.normalized_tuning_curves
        # ind = np.unravel_index(np.argsort(curr_tuning_curves, axis=None), curr_tuning_curves.shape)
        
        if sortby is None:
            sort_ind = np.argsort(np.argmax(self.normalized_tuning_curves, axis=1))
        elif isinstance(sortby, (list, np.ndarray)):
            sort_ind = sortby
        else:
            print(f'WARNING: get_sort_indicies(sortby={sortby}) is not a known value type. using np.arange(self.n_neurons)...')
            sort_ind = np.arange(self.n_neurons)
        return sort_ind

    def to_1D_maximum_projection(self):
        return Ratemap.build_1D_maximum_projection(self)


    # ----------------------  Static Methods -------------------------:
    @staticmethod
    def nan_ptp(a, **kwargs):
        return np.ptp(a[np.isfinite(a)], **kwargs)

    @staticmethod
    def nanmin_nanmax_scaler(x, axis=-1, **kwargs):
        """Scales the values x to lie between 0 and 1 along the specfied axis, ignoring NaNs!
        Parameters
        ----------
        x : np.array
            numpy ndarray
        Returns
        -------
        np.array
            scaled array
        """
        try:
            return (x - np.nanmin(x, axis=axis, keepdims=True)) / Ratemap.nan_ptp(x, axis=axis, keepdims=True, **kwargs)
        except ValueError:  #raised if `y` is empty.
            # Without this try-except we encountered "ValueError: zero-size array to reduction operation minimum which has no identity" when x was empty.
            return x # just return the raw x-value, as it's empty and doesn't need scaling


    @staticmethod    
    def NormalizeData(data):
        """ Simple alternative to the mathutil.min_max_scalar that doesn't produce so man NaN values. """
        data[np.isnan(data)] = 0.0 # Set NaN values to 0.0
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


    @classmethod
    def perform_AOC_normalization(cls, active_tuning_curves, debug_print=False):
    # def perform_AOC_normalization(cls, ratemap: Ratemap, debug_print=True):
        """ Normalizes each cell's tuning map in ratemap by dividing by each cell's area under the curve (AOC). The resultant tuning maps are therefore converted into valid PDFs 
        
        Inputs:
            active_tuning_curves: nd.array
        """
        # active_tuning_curves = ratemap.normalized_tuning_curves
        # active_tuning_curves = ratemap.tuning_curves
        tuning_curves_ndim = active_tuning_curves.ndim - 1
        
        if tuning_curves_ndim == 1:
            ## 1D normalization:
            _test_1D_normalization_constants = 1.0/np.sum(active_tuning_curves, 1) # normalize by summing over all 1D positions for each cell
            ## Compute the area-under-the-curve normalization by dot-dividing each cell's PF by the normalization constant
            _test_1D_AOC_normalized_pdf = (active_tuning_curves.transpose() * _test_1D_normalization_constants).transpose() # (39, 59)
            ## Test success by summing (all should be nearly 1.0):
            is_valid_normalized_pf = np.logical_not(np.isnan(np.sum(_test_1D_AOC_normalized_pdf, 1))) # The True entries are non-Nan and should be equal to 1.0, the other elements are NaN and have no valid pf yet.
            if debug_print:
                print(f'is_valid_normalized_pf: {is_valid_normalized_pf}')
                
            assert np.isclose(np.sum(_test_1D_AOC_normalized_pdf[is_valid_normalized_pf,:], 1), 1.0).all(), f"After AOC normalization the sum over each cell should be 1.0, but it is not! {np.sum(_test_1D_AOC_normalized_pdf, 1)}"
            return _test_1D_AOC_normalized_pdf
        elif tuning_curves_ndim == 2:
            ## 2D normalization
            _test_2D_normalization_constants = 1.0/np.sum(active_tuning_curves, (1,2)) # normalize by summing over all 1D positions for each cell
            _test_2D_AOC_normalized_pdf = (active_tuning_curves.transpose(1,2,0) * _test_2D_normalization_constants).transpose(2,0,1) # (39, 59) # (59, 21, 39) prior to second transpose
            is_valid_normalized_pf = np.logical_not(np.isnan(np.sum(_test_2D_AOC_normalized_pdf, (1,2)))) # The True entries are non-Nan and should be equal to 1.0, the other elements are NaN and have no valid pf yet.
            if debug_print:
                print(f'is_valid_normalized_pf: {is_valid_normalized_pf}')
                            
            ## Test success by summing (all should be nearly 1.0):
            assert np.isclose(np.sum(_test_2D_AOC_normalized_pdf[is_valid_normalized_pf,:,:], (1,2)), 1.0).all(), f"After AOC normalization the sum over each cell should be 1.0, but it is not! {np.sum(_test_2D_AOC_normalized_pdf, (1,2))}"
            return _test_2D_AOC_normalized_pdf
        else:
            print(f'tuning_curves_ndim: {tuning_curves_ndim} not implemented!')
            raise NotImplementedError

 
    @staticmethod           
    def build_never_visited_mask(occupancy):
        """ returns a mask of never visited locations for the provided occupancy """
        return (occupancy == 0) # return locations with zero occupancy

    @staticmethod
    def nan_never_visited_locations(occupancy):
        """ replaces all never visited locations, indicated by a zero occupancy, by NaNs for the purpose of building visualizations. """
        nan_never_visited_occupancy = occupancy.copy()
        nan_never_visited_occupancy[nan_never_visited_occupancy == 0] = np.nan # all locations with zeros, replace them with NaNs
        return nan_never_visited_occupancy

    @classmethod
    def build_1D_maximum_projection(cls, ratemap_2D: "Ratemap") -> "Ratemap":
        """ builds a 1D ratemap from a 2D ratemap
        creation_date='2023-04-05 14:02'

        Usage:
            ratemap_1D = build_1D_maximum_projection(ratemap_2D)
        """
        assert ratemap_2D.ndim > 1, f"ratemap_2D ndim must be greater than 1 (usually 2) but ndim: {ratemap_2D.ndim}."
        ratemap_1D_spikes_maps = np.nanmax(ratemap_2D.spikes_maps, axis=-1) #.shape (n_cells, n_xbins)
        ## 2023-04-07 - This isn't good enough, we need to recompute the tuning_curves from the maximum spikes_maps bin. The excessive occupancy at the end-caps is already diminishing the tuning curves here, right?
        ratemap_1D_tuning_curves = np.nanmax(ratemap_2D.tuning_curves, axis=-1) #.shape (n_cells, n_xbins)
        ratemap_1D_unsmoothed_tuning_maps = np.nanmax(ratemap_2D.unsmoothed_tuning_maps, axis=-1) #.shape (n_cells, n_xbins)
        ratemap_1D_occupancy = np.sum(ratemap_2D.occupancy, axis=-1) #.shape (n_xbins,)

        ratemap_1D = Ratemap(ratemap_1D_tuning_curves, unsmoothed_tuning_maps=ratemap_1D_unsmoothed_tuning_maps, spikes_maps=ratemap_1D_spikes_maps, xbin=ratemap_2D.xbin, ybin=None, occupancy=ratemap_1D_occupancy, neuron_ids=deepcopy(ratemap_2D.neuron_ids), neuron_extended_ids=deepcopy(ratemap_2D.neuron_extended_ids), metadata=ratemap_2D.metadata)
        return ratemap_1D



    def spatial_sparcity(self) -> np.ndarray:
        """ computes the sparcity as a measure of spatial selectivity as in Silvia et al. 2015
        
        Sparcity = \frac{ <f>^2 }{ <f^2> }
        
        """
        assert self.unsmoothed_tuning_maps is not None, "self.unsmoothed_tuning_maps is None! Did you pass it in while building the Ratemap?"
        # Average over positions:
        expected_f = np.array([np.nanmean(a_tuning_curve) for a_tuning_curve in ratemap.unsmoothed_tuning_maps]) # .shape
        expected_f_squared = np.array([np.nanmean(a_tuning_curve**2) for a_tuning_curve in ratemap.unsmoothed_tuning_maps]) # .shape
        return (expected_f**2) / expected_f_squared # sparcity.shape # (n_neurons,)
        

    # ----------------------  HDF5 Serialization -------------------------:
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
        """
        # Open the file with h5py to add attributes to the group. The pandas.HDFStore object doesn't provide a direct way to manipulate groups as objects, as it is primarily intended to work with datasets (i.e., pandas DataFrames)
        with h5py.File(file_path, kwargs.pop('file_mode', 'a')) as f:
            ## Unfortunately, you cannot directly assign a dictionary to the attrs attribute of an h5py group or dataset. The attrs attribute is an instance of a special class that behaves like a dictionary in some ways but not in others. You must assign attributes individually
            group = f.require_group(key)
            # group = f[key]
            group.attrs['n_neurons'] = self.n_neurons
            group.attrs['ndim'] = self.ndim

            group['occupancy'] = self.occupancy
            group['tuning_curves'] = self.tuning_curves
            group['spikes_maps'] = self.spikes_maps
            if self.unsmoothed_tuning_maps is not None:
                group['unsmoothed_tuning_maps'] = self.unsmoothed_tuning_maps
            
            group['neuron_ids'] = self._neuron_ids
            group['neuron_ids'].make_scale('neuron_ids name')
            group['xbin'] = self.xbin
            group['xbin'].make_scale('xbin name')

            group['xbin_centers'] = self.xbin_centers
            group['xbin_centers'].make_scale('xbin_centers name')

            if self.ybin is not None:
                group['ybin'] = self.ybin
                group['ybin'].make_scale('ybin name')

                group['ybin_centers'] = self.ybin_centers
                group['ybin_centers'].make_scale('ybin_centers name')

            # Attach scales:
            group['tuning_curves'].dims[0].label = 'neuron_id'
            group['spikes_maps'].dims[0].label = 'neuron_id'
            if self.unsmoothed_tuning_maps is not None:
                group['unsmoothed_tuning_maps'].dims[0].label = 'neuron_id'

            # group['tuning_curves'].dims[0].attach_scale(group['neuron_ids'])
            # group['occupancy'].dims[0].attach_scale(group['xbin_centers'])
            group['occupancy'].dims[0].label = 'x'
            group['tuning_curves'].dims[1].label = 'x'
            group['spikes_maps'].dims[1].label = 'x'
            if self.unsmoothed_tuning_maps is not None:
                group['unsmoothed_tuning_maps'].dims[1].label = 'x'

            if self.ybin is not None:
                # group['occupancy'].dims[1].attach_scale(group['ybin_centers'])
                group['occupancy'].dims[1].label = 'y'
                group['tuning_curves'].dims[2].label = 'y'
                group['spikes_maps'].dims[2].label = 'y'
                if self.unsmoothed_tuning_maps is not None:
                    group['unsmoothed_tuning_maps'].dims[2].label = 'y'

