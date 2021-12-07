import numpy as np
from neuropy.utils import mathutil
from . import DataWriter


class Ratemap(DataWriter):
    def __init__(
        self,
        tuning_curves: np.ndarray,
        tuning_curves,
        xbin=None,
        ybin=None,
        occupancy=None,
        neuron_ids=None,
        metadata=None,
    ) -> None:
        super().__init__()

        assert tuning_curves.ndim <= 3, "tuning curves shape should be <= 3"
        self.tuning_curves = np.asarray(tuning_curves)
        if neuron_ids is not None:
            assert len(neuron_ids) == self.tuning_curves.shape[0]
            self.neuron_ids = np.asarray(neuron_ids)
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

    def peak_firing_rate(self):
        return np.max(self.tuning_curves, axis=1)

    def get_field_locations(self):
        pass
    
    
    @property
    def normalized_tuning_curves(self):
        return mathutil.min_max_scaler(self.tuning_curves)
    
    def get_sort_indicies(self, sortby=None):
        curr_tuning_curves = self.normalized_tuning_curves
        ind = np.unravel_index(np.argsort(curr_tuning_curves, axis=None), curr_tuning_curves.shape)
        
        if sortby is None:
            sort_ind = np.argsort(np.argmax(self.normalized_tuning_curves, axis=1))
        elif isinstance(sortby, (list, np.ndarray)):
            sort_ind = sortby
        else:
            sort_ind = np.arange(n_neurons) 
            sort_ind = np.arange(n_neurons) 
