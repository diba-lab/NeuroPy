import numpy as np


class Ratemap:
    def __init__(
        self,
        tuning_curves,
        xbin=None,
        ybin=None,
        neuron_ids=None,
    ) -> None:
        self.tuning_curves = tuning_curves
        self.neuron_ids = neuron_ids
        self.xbin = xbin
        self.ybin = ybin

    def calculate(self):
        pass

    @property
    def xbin_centers(self):
        pass

    @property
    def ybin_centers(self):
        pass
