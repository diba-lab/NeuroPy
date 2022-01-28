import numpy as np
from . import DataWriter
from scipy import stats


class Ratemap(DataWriter):
    def __init__(
        self,
        tuning_curves: np.ndarray,
        xbin=None,
        ybin=None,
        occupancy=None,
        neuron_ids=None,
        metadata=None,
    ) -> None:
        super().__init__(metadata=metadata)

        assert tuning_curves.ndim <= 3, "tuning curves shape should be <= 3"
        self.tuning_curves = np.asarray(tuning_curves)
        if neuron_ids is not None:
            assert len(neuron_ids) == self.tuning_curves.shape[0]
            self.neuron_ids = np.asarray(neuron_ids)
        self.xbin = xbin
        self.ybin = ybin
        self.occupancy = occupancy

    @property
    def xbin_centers(self):
        return self.xbin[:-1] + np.diff(self.xbin) / 2

    @property
    def xbin_size(self):
        return np.diff(self.xbin)[0]

    @property
    def ybin_centers(self):
        return self.ybin[:-1] + np.diff(self.ybin) / 2

    @property
    def n_neurons(self):
        return self.tuning_curves.shape[0]

    def ndim(self):
        return self.tuning_curves.ndim - 1

    def get_normalized(self):
        pass

    def get_peak_locations(self, by="index"):
        sort_ind = np.argmax(stats.zscore(self.tuning_curves, axis=1), axis=1)
        if by == "index":
            return sort_ind
        if by == "position":
            return self.xbin[sort_ind]

    def get_sort_order(self, by="index"):
        """Return sorting order tuning curves by position in ascending order

        Parameters
        ----------
        by : str, optional
            'index' returns row-index location of tuning_curves, 'neuron_id' returns id of neurons that will sort the location of peaks tuning curves, by default "index"

        Returns
        -------
        [type]
            [description]
        """
        sort_ind = np.argsort(self.get_peak_locations(by="index"))
        if by == "neuron_id":
            return self.neuron_ids[sort_ind]
        if by == "index":
            return sort_ind

    def peak_firing_rate(self):
        return np.max(self.tuning_curves, axis=1)

    def get_field_locations(self):
        pass
