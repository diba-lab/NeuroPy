import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from .datawriter import DataWriter
from copy import deepcopy
from skimage import feature
from skimage.measure import regionprops
from pathlib import Path

from neuropy.utils.minian_util import load_variable


class CaNeurons(DataWriter):
    """Class to hold calcium imaging data and their labels, raw traces, etc.
    NOTE: neuron_IDs or neuron_ids always refers back to original #s from minian output.
          neuron_inds refers to indices in CaNeurons class after trimming bad neurons.
          So max(neuron_ids) >= #neurons - 1, while max(neuron_inds) = #neurons - 1"""

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

    def plot_rois(
        self,
        neuron_inds: list or np.ndarray or None = None,
        label: bool = False,
        plot_cell_id: bool = True,
        ax: plt.Axes or None = None,
        plot_masks: bool = False,
        plot_max: bool = True,
        highlight_inds: list = [],
    ):
        # Set up axes
        if ax is None:
            fig, ax = plt.subplots()
        elif isinstance(ax, plt.Axes):
            fig = ax.figure

        # Get rois
        neuron_inds = np.arange(self.A.shape[0]) if neuron_inds is None else neuron_inds
        Aplot = self.A[neuron_inds]

        # Plot masks if specified
        if plot_masks:
            ax.imshow((Aplot > 0).sum(axis=0))

        # Plot max projection if specified
        if plot_max:
            ax.imshow(self.max_proj)

        # Plot roi outlines with numbers if specified
        lines = []
        if label:
            for id, roi in zip(neuron_inds, Aplot):
                xedges, yedges = detect_roi_edges(roi > 0)
                (hl,) = ax.plot(xedges, yedges, linewidth=0.7)
                lines.append(hl)
                if plot_cell_id:
                    ax.text(xedges.mean(), yedges.mean(), str(id), color="w")
        else:
            lines = None

        if len(highlight_inds) > 0:
            # [lines[hid].set_linewidth(4) for hid in highlight_inds]
            [lines[hid].set(linewidth=3, color="r") for hid in highlight_inds]

        return fig, ax, lines

    @property
    def max_proj(self):
        """Grab max projection for session. Motion-corrected and cropped only."""
        max_proj_dir = sorted(self.basedir.glob("**/max_proj*.*"))[0].parent
        return load_variable(max_proj_dir, "max_proj")

    def roi_com(self, neuron_inds=None):
        """Get center-of-mass of all neuron rois"""
        return np.array([detect_roi_centroid(roi > 0) for roi in self.A[neuron_inds]])

    def min_proj(self, proj_type: str in ["", "crop", "mc_crop"] = "mc_crop"):
        """Grab appropriate minimum projection"""
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
        ax.imshow(proj_plot, cmap="gray")
        ax.invert_yaxis() if flip_yaxis else None  # invert axis if specified

    def plot_rois_with_min_proj(
        self, plot_max=True, min_type="mc_min_crop", ax=None, label=True, **kwargs
    ):
        """plots ROIs over max projection alongside min projection. Great for data QC purposes.
        **kwargs inputs go to .plot_ROIs"""

        # Set up axes
        if not isinstance(ax, np.ndarray):
            _, ax = plt.subplots(1, 2)
        assert isinstance(ax, np.ndarray) and len(ax) == 2, "ax must be shape (2,)"

        # Plot min
        self.plot_proj(min_type, ax=ax[0])

        # Plot rois over max
        self.plot_rois(plot_max=plot_max, ax=ax[1], label=label, **kwargs)
        pass

    def plot_traces(self):
        pass

    def plot_traces_and_ROIs(self):
        pass


class CaNeuronReg:
    """All things related to tracking neurons across days"""

    def __init__(self, caneurons: list, alias: list or None = None):
        self.caneurons = caneurons
        self.alias = alias
        self.nsessions = len(caneurons)
        pass

    def get_session(self, sesh_alias: str):
        """grab a session from its alias"""
        assert sesh_alias in self.alias, "'sesh_alias' must be in self.alias"
        sesh_ind = np.where([sesh_alias == sesh for sesh in self.alias])[0][0]

        return self.caneurons[sesh_ind]

    def plot_rois_across_sessions(
        self, fig_title="", sesh_plot: list or None = None, **kwargs
    ):
        """Plot rois over max proj with min proj also across days.
        **kwargs to CaNeurons.plot_rois_with_min_proj"""

        # Recursively plot a subset of sessions if specified in sesh_plot
        if sesh_plot is not None:
            use_alias = isinstance(sesh_plot[0], str)
            if use_alias:
                caneuro_use = [self.get_session(alias) for alias in sesh_plot]
                alias_use = sesh_plot
            else:
                assert isinstance(
                    sesh_plot[0], int
                ), "sesh_plot must be a list or int or str"
                caneuro_use = [self.caneurons[id] for id in sesh_plot]
                alias_use = [f"Session {sesh}" for sesh in sesh_plot]
            CaNeuronReg(caneuro_use, alias=alias_use).plot_rois_across_sessions()
        elif sesh_plot is None:
            # Set up plots
            fig, ax = plt.subplots(
                2,
                self.nsessions,
                figsize=(5.67 * self.nsessions, 12.6),
                sharex=True,
                sharey=True,
            )
            fig.suptitle(fig_title)

            # Get names for each session - either by session alias or overall session number
            names = (
                self.alias
                if self.alias is not None
                else [f"Session {n+1}" for n in range(self.nsessions)]
            )

            # plot rois
            for a, caneurons, name in zip(ax.T, self.caneurons, names):
                caneurons.plot_rois_with_min_proj(ax=a, **kwargs)
                a[0].set_title(name)

            return fig, ax

    @staticmethod
    def shift_roi_edges(
        roi_edges: plt.Line2D, delta_com: np.ndarray, use_mean: bool = True
    ):
        """Adjust plotted roi edges by amount in delta_com"""
        assert delta_com.shape[1] == 2
        ncells = len(roi_edges)

        if use_mean:  # Adjust ALL rois by the mean x/y value in delta_com
            delta_com = np.ones((ncells, 2)) * delta_com.mean(axis=0)

        # Adjust all roi edges by amount specifiy and keep original color
        for dcom, roi_edge in zip(delta_com, roi_edges):
            roi_color = roi_edge.get_color()
            roi_edge.set_xdata(roi_edge.get_xdata() + dcom[0])
            roi_edge.set_ydata(roi_edge.get_ydata() + dcom[1])
            roi_edge.set_color(roi_color)

        return None

    @staticmethod
    def load_pairwise_map(map_filename):
        """Load a pairwise map into a pandas dataframe"""
        map_filename = Path(map_filename)
        map_stem = map_filename.stem
        assert (
            len(map_stem.split("_")) == 3 and map_stem.split("_")[0] == "map"
        ), 'map file must be of the format "map_sesh1identifier_sesh2identifier.npy"'
        _, sesh1id, sesh2id = map_stem.split("_")

        map_array = np.load(map_filename)

        return pd.DataFrame({sesh1id: map_array[:, 0], sesh2id: map_array[:, 1]})


def id_and_plot_reference_cells(caneurons1: CaNeurons, caneurons2: CaNeurons, **kwargs):
    """identify and clearly plot reference cells active across both session to aid
    in manual cell registration.
    **kwargs to CaNeurons.plot_rois"""

    # Plot out cells from both sessions
    careg = CaNeuronReg([caneurons1, caneurons2])
    fig, ax = careg.plot_rois_across_sessions()

    # Select 3-4 reference cells that are clearly active and the same in both sessions
    sesh1_inds = list(
        map(
            int,
            input(
                "\nEnter (co-active) reference neurons from session 1 (space between) : "
            )
            .strip()
            .split(),
        )
    )
    nrefs = len(sesh1_inds)
    sesh2_inds = list(
        map(
            int,
            input(
                "\nEnter (co-active) reference neurons from session 2 (space between) : "
            )
            .strip()
            .split(),
        )
    )[:nrefs]

    # Replot cells with reference cells highlighted to check your work
    caneurons1.plot_rois(
        plot_max=True,
        ax=ax[1][0],
        label=True,
        plot_masks=False,
        highlight_inds=sesh1_inds,
        **kwargs,
    )
    caneurons2.plot_rois(
        plot_max=True,
        ax=ax[1][1],
        label=True,
        plot_masks=False,
        highlight_inds=sesh2_inds,
        **kwargs,
    )

    # Step through and match up each neuron from session1 to session2

    return fig, ax


def detect_roi_edges(roi_binary, **kwargs):
    """Detect roi edges and organize them nicely in CW/CCW fashion"""
    edges = feature.canny(roi_binary, **kwargs)  # detect edges
    inds = np.where(edges)  # Get edge locations in pixels
    isort = np.argsort(
        np.arctan2(inds[1] - inds[1].mean(), inds[0] - inds[0].mean())
    )  # Sort by angle from center

    xedges = np.append(inds[1][isort], inds[1][isort[0]])
    yedges = np.append(inds[0][isort], inds[0][isort[0]])

    return xedges, yedges


def detect_roi_centroid(roi_binary, **kwargs):
    """Detect centroid of a cell roi, input = binary (boolean) mask"""
    props = regionprops(roi_binary.astype(np.uint8))
    com = props[0].centroid

    com_y, com_x = com

    return com_x, com_y
