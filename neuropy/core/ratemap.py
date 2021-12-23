import numpy as np
from neuropy.utils import mathutil
from . import DataWriter


class Ratemap(DataWriter):
    def __init__(
        self,
        tuning_curves,
        xbin=None,
        ybin=None,
        occupancy=None,
        neuron_ids=None,
        metadata=None,
    ) -> None:
        super().__init__()

        self.tuning_curves = np.asarray(tuning_curves)
        if neuron_ids is not None:
            assert len(neuron_ids) == self.tuning_curves.shape[0]
            self.neuron_ids = neuron_ids
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