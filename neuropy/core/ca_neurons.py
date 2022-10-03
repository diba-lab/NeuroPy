import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from .datawriter import DataWriter
from copy import deepcopy
from skimage import feature


class CaNeurons(DataWriter):
    """Class to hold calcium imaging data and their labels, raw traces, etc."""

    def __init__(
        self,
        S: np.ndarray,
        C: np.ndarray,
        A: np.ndarray,
        YrA: np.ndarray,
        trim: dict or None = None,
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
        self.trim = trim
        self.sampling_rate = sampling_rate
        self.t = t
        self.neuron_ids = neuron_ids
        self.neuron_type = neuron_type

    def plot_ROIs(self, neuronIDs: list or np.ndarray or None = None, label: bool = False, ax: plt.Axes or None = None):
        if ax is None:
            fig, ax = plt.subplots()
        elif isinstance(ax, plt.Axes):
            fig = ax.figure

        neuronIDs = np.arange(self.A.shape[0]) if neuronIDs is None else neuronIDs
        Aplot = self.A[neuronIDs]
        ax.imshow((Aplot > 0).sum(axis=0))

        if label:
            for id, roi in zip(neuronIDs, Aplot):
                xedges, yedges = detect_roi_edges(roi > 0)
                hl, = ax.plot(xedges, yedges, linewidth=0.7)
                ax.text(xedges.mean(), yedges.mean(), str(id), color='w')

        return fig, ax, xedges, yedges

    def plot_traces(self):
        pass

    def plot_traces_and_ROIs(self):
        pass


def detect_roi_edges(roi_binary):
    """Detect roi edges and organize them nicely in CW/CCW fashion"""
    edges = feature.canny(roi_binary)  # detect edges
    inds = np.where(edges) # Get edge locations in pixels
    isort = np.argsort(np.arctan2(inds[1] - inds[1].mean(), inds[0] - inds[0].mean()))  # Sort by angle from center

    xedges = np.append(inds[1][isort], inds[1][isort[0]])
    yedges = np.append(inds[0][isort], inds[0][isort[0]])

    return xedges, yedges