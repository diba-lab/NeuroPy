import numpy as np
from neuropy.core.neuron_identities import NeuronIdentitiesDisplayerMixin
from neuropy.utils import mathutil
from . import DataWriter


class Ratemap(NeuronIdentitiesDisplayerMixin, DataWriter):
    def __init__(
        self,
        tuning_curves,
        firing_maps=None,
        xbin=None,
        ybin=None,
        occupancy=None,
        neuron_ids=None,
        neuron_extended_ids=None,
        metadata=None,
    ) -> None:
        super().__init__()

        self.tuning_curves = np.asarray(tuning_curves)
        self.firing_maps = np.asarray(firing_maps)
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

    @property
    def xbin_centers(self):
        return self.xbin[:-1] + np.diff(self.xbin) / 2

    @property
    def ybin_centers(self):
        return self.ybin[:-1] + np.diff(self.ybin) / 2
    
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
        # return self.metadata['tuple_neuron_ids']
    @neuron_extended_ids.setter
    def neuron_extended_ids(self, value):
        self._neuron_extended_ids = value
     

    @property
    def n_neurons(self):
        return self.tuning_curves.shape[0]

    def ndim(self):
        return self.tuning_curves.ndim - 1
    
    
    @property
    def normalized_tuning_curves(self):
        return Ratemap.nanmin_nanmax_scaler(self.tuning_curves)
        # return mathutil.min_max_scaler(self.tuning_curves)
        # return Ratemap.NormalizeData(self.tuning_curves)

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
        return (x - np.nanmin(x, axis=axis, keepdims=True)) / Ratemap.nan_ptp(x, axis=axis, keepdims=True, **kwargs)
    
    
    @staticmethod    
    def NormalizeData(data):
        """ Simple alternative to the mathutil.min_max_scalar that doesn't produce so man NaN values. """
        data[np.isnan(data)] = 0.0 # Set NaN values to 0.0
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    
    def get_sort_indicies(self, sortby=None):
        curr_tuning_curves = self.normalized_tuning_curves
        ind = np.unravel_index(np.argsort(curr_tuning_curves, axis=None), curr_tuning_curves.shape)
        
        if sortby is None:
            sort_ind = np.argsort(np.argmax(self.normalized_tuning_curves, axis=1))
        elif isinstance(sortby, (list, np.ndarray)):
            sort_ind = sortby
        else:
            sort_ind = np.arange(self.n_neurons) 