import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from .datawriter import DataWriter
from copy import deepcopy
from skimage import feature
from pathlib import Path

from neuropy.utils.minian_util import load_variable


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
        basedir: str or Path = None,
    ) -> None:
        super().__init__(metadata=metadata)

        self.basedir = basedir
        self.S = S
        self.C = C
        self.A = A
        self.YrA = YrA
        self.trim = trim
        self.sampling_rate = sampling_rate
        self.t = t
        self.neuron_ids = neuron_ids
        self.neuron_type = neuron_type

    def plot_ROIs(
        self,
        neuronIDs: list or np.ndarray or None = None,
        label: bool = False,
        ax: plt.Axes or None = None,
        plot_masks: bool = True,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        elif isinstance(ax, plt.Axes):
            fig = ax.figure

        neuronIDs = np.arange(self.A.shape[0]) if neuronIDs is None else neuronIDs
        Aplot = self.A[neuronIDs]
        if plot_masks:
            ax.imshow((Aplot > 0).sum(axis=0))

        if label:
            for id, roi in zip(neuronIDs, Aplot):
                xedges, yedges = detect_roi_edges(roi > 0)
                (hl,) = ax.plot(xedges, yedges, linewidth=0.7)
                ax.text(xedges.mean(), yedges.mean(), str(id), color="w")

        return fig, ax, xedges, yedges

    @property
    def max_proj(self):
        max_proj_dir = sorted(self.basedir.glob("**/max_proj*.*"))[0].parent
        return load_variable(max_proj_dir, "max_proj")

    def min_proj(self, proj_type: str in ["", "crop", "mc_crop"] = "mc_crop"):
        min_proj_dir = sorted(self.basedir.glob("**/min_proj*.*"))[
            0
        ].parent  # get min_proj directory

        # Get name for min projection you want to grab
        var_name = "min_proj" if proj_type == "" else "min_proj_crop"
        var_name = "_".join(["mc", var_name]) if proj_type == "mc_crop" else var_name

        return load_variable(min_proj_dir, var_name)

    def plot_proj(self, proj_type, ax=None, flip_yaxis=False):
        """Plot max, min, min_crop, or mc_min_crop projections"""

        if not isinstance(ax, plt.Axes):
            _, ax = plt.subplots()

        # Grab appropriate projection
        assert proj_type in ["max", "min", "min_crop", "mc_min_crop"]
        if "max" in proj_type:
            proj_plot = self.max_proj
        else:
            proj_plot = self.min_proj(
                proj_type="_".join(sorted(proj_type.split("_"))[-2::-1])
            )

        # Plot
        ax.imshow(proj_plot)
        ax.invert_yaxis() if flip_yaxis else None  # invert axis if specified

    def plot_traces(self):
        pass

    def plot_traces_and_ROIs(self):
        pass


def detect_roi_edges(roi_binary):
    """Detect roi edges and organize them nicely in CW/CCW fashion"""
    edges = feature.canny(roi_binary)  # detect edges
    inds = np.where(edges)  # Get edge locations in pixels
    isort = np.argsort(
        np.arctan2(inds[1] - inds[1].mean(), inds[0] - inds[0].mean())
    )  # Sort by angle from center

    xedges = np.append(inds[1][isort], inds[1][isort[0]])
    yedges = np.append(inds[0][isort], inds[0][isort[0]])

    return xedges, yedges
