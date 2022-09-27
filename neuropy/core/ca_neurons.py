import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from .datawriter import DataWriter
from copy import deepcopy


class CaNeurons(DataWriter):
    """Class to hold calcium imaging data and their labels, raw traces, etc."""

    def __init__(
        self,
        S: np.ndarray,
        C: np.ndarray,
        A: np.ndarray,
        YrA: np.ndarray,
        t=None,
        sampling_rate=15,
        neuron_ids=None,
        neuron_type=None,
        metadata=None,
    ) -> None:
        super().__init__(metadata=metadata)

        self.S = S
        self.C = C
        self.A = A
        self.YrA = YrA
        self.sampling_rate = sampling_rate
        self.t = t
        self.neuron_ids = neuron_ids
        self.neuron_type = neuron_type

    def plot_ROIs(self, neuronsIDs):
        pass

    def plot_traces(self):
        pass

    def plot_traces_and_ROIs(self):
        pass
