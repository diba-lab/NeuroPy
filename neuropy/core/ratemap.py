import numpy as np


class Ratemap:
    def __init__(
        self,
        tuning_curves,
        xbin=None,
        ybin=None,
        neuron_ids=None,
    ) -> None:
        self.tuning_curves = np.asarray(tuning_curves)
        if neuron_ids is not None:
            assert len(neuron_ids) == self.tuning_curves.shape[0]
            self.neuron_ids = neuron_ids
        self.xbin = xbin
        self.ybin = ybin

    @property
    def xbin_centers(self):
        return self.xbin[:-1] + np.diff(self.xbin) / 2

    @property
    def ybin_centers(self):
        return self.ybin[:-1] + np.diff(self.ybin) / 2

    @property
    def n_neurons(self):
        return self.tuning_curves.shape[0]
