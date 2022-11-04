import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from neuropy.core.datawriter import DataWriter
from copy import deepcopy
from skimage import feature
from skimage.measure import regionprops
from pathlib import Path

from neuropy.utils.minian_util import load_variable
from neuropy.utils.misc import flatten


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
        posthocmerge: dict or None = None,
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
        self.posthocmerge = posthocmerge

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
                    jitter = np.random.randint(
                        -5, 5, size=2
                    )  # prevents numbers from being overlaid if rois are merged
                    ax.text(
                        xedges.mean() + jitter[0],
                        yedges.mean() + jitter[1],
                        str(id),
                        color="w",
                    )
        else:
            lines = None

        if len(highlight_inds) > 0:
            # [lines[hid].set_linewidth(4) for hid in highlight_inds]
            [
                lines[np.where(hid == np.array(neuron_inds))[0][0].astype(int)].set(
                    linewidth=3, color="r"
                )
                for hid in highlight_inds
            ]

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
            _, ax = plt.subplots(1, 2, sharex=True, sharey=True)
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

    def quick_merge(self, unit_ids: list, method="max"):
        """Quickly merge units post-hoc - will make copies of merged units at each index
        noted in merge. Again make this recursive to accommodate list of lists (multiple
        sets of neurons to merge). NOTE each nested pair of neuron in unit_ids can only be two units.
        usage: CaNeuron.quick_merge([1, 3]) will merge units 1 and 3
               CaNeuron.quick_merge([[1, 3], [2, 4]]) will merge 1 & 3 and 2 & 4 separately"""

        if (
            isinstance(unit_ids[0], list)
            and len(unit_ids[0]) >= 2
            and len(unit_ids) > 1
        ):  # If first entry is a list, merge and then proceed recursively
            caneuron_merge = self.quick_merge(unit_ids=unit_ids[0], method=method)
            # Account for last entry in unit_ids being a neuron pair
            new_unit_ids = (
                unit_ids[1:][0]
                if isinstance(unit_ids[1:], list) and len(unit_ids[1:]) == 1
                else unit_ids[1:]
            )
            return caneuron_merge.quick_merge(
                unit_ids=new_unit_ids,
                method=method,
            )

        if len(unit_ids) == 1:
            unit_ids = list(flatten(unit_ids))

        # First mush unit ROIs into one
        Amush = np.zeros_like(self.A[0])
        for A in self.A[unit_ids]:
            Amush += A
        Amerge = self.A.copy()
        Amerge[unit_ids] = Amush

        # Now iterate through unit activity time-series
        Cmerge = self.C.copy()
        Smerge = self.S.copy()
        YrAmerge = self.YrA.copy()
        for var_merge, var_use in zip([Cmerge, Smerge, YrAmerge], ["C", "S", "YrA"]):
            if method == "max":
                var_merge[unit_ids] = getattr(self, var_use)[unit_ids].max(axis=0)
            else:
                print("Methods other than 'max' not yet implemented")
                return None

        # Last return object
        caneurons_merge = CaNeurons(
            S=Smerge,
            C=Cmerge,
            A=Amerge,
            YrA=YrAmerge,
            trim=self.trim,
            t=self.t,
            sampling_rate=self.sampling_rate,
            neuron_ids=self.neuron_ids,
            neuron_type=self.neuron_type,
            metadata=self.metadata,
            basedir=self.basedir,
            posthocmerge={"method": method, "unit_ids": unit_ids},
        )

        return caneurons_merge


class CaNeuronReg:
    """All things related to tracking neurons across days"""

    def __init__(self, caneurons: list, alias: list or None = None):
        self.caneurons = caneurons
        self.alias = alias
        self.nsessions = len(caneurons)
        pass

    def quick_merge(self, sesh_alias, in_place=False, **kwargs):
        """Copy of CaNeurons.quick_merge() - see for inputs."""
        sesh_ind = self.get_alias_index(sesh_alias)

        caneurons_merge = self.caneurons[sesh_ind].quick_merge(**kwargs)
        if in_place:
            self.caneurons[sesh_ind] = caneurons_merge
        return caneurons_merge

    def get_session(self, sesh_alias: str):
        """grab data from a recording session from its alias or by index"""

        sesh_ind = self.get_alias_index(sesh_alias)

        return self.caneurons[sesh_ind]

    def get_alias_index(self, sesh_alias: str):
        """Get index for sesh_alias in .caneurons field"""

        assert sesh_alias in self.alias or (
            isinstance(sesh_alias, int)
        ), "'sesh_alias' must be in self.alias or be int"

        if sesh_alias in self.alias:
            sesh_ind = np.where([sesh_alias == sesh for sesh in self.alias])[0][0]
        else:
            sesh_ind = sesh_alias

        return sesh_ind

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
            CaNeuronReg(caneuro_use, alias=alias_use).plot_rois_across_sessions(
                **kwargs
            )
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

    def plot_reg_neurons_overlaid(
        self,
        sesh1: str or int,
        sesh2: str or int,
        map1_2: pd.DataFrame,
        plot_max=True,
        ax=None,
        color=None,
    ):
        """Plots neurons from two sessions overlaid upon each other"""

        if ax is None:
            _, ax = plt.subplots()

        # Get map of coactive neurons only
        coactive_bool = (map1_2 > -1).all(axis=1)
        coactive_map = map1_2[coactive_bool]

        # Calculate delta center-of-mass for all mapped neurons
        delta_com = self.get_session(sesh1).roi_com(
            neuron_inds=coactive_map[sesh1]
        ) - self.get_session(sesh2).roi_com(neuron_inds=coactive_map[sesh2])

        # Plot cells overlaid on top of one another with shift applied
        _, _, roi_edges1 = self.get_session(sesh1).plot_rois(
            neuron_inds=coactive_map[sesh1], label=True, ax=ax, plot_max=plot_max
        )
        ax.set_title(f"{sesh1} w/{sesh2} overlaid")
        ax.set_prop_cycle(
            None
        )  # restart automatic coloring so that session 2 colors match session 1 colors
        _, _, roi_edges2 = self.get_session(sesh2).plot_rois(
            neuron_inds=coactive_map[sesh2],
            plot_max=False,
            plot_cell_id=False,
            label=True,
            ax=ax,
        )  # Plot session 2 rois
        self.shift_roi_edges(
            roi_edges2, delta_com
        )  # shift roi edges from session 2 to match session1

        if color is not None:  # Make all rois the same color if specified
            [roi_edge.set_color(color) for roi_edge in roi_edges1]
            [roi_edge.set_color(color) for roi_edge in roi_edges2]

        return delta_com

    def overlay_base_roi_on_reg(
        self,
        sesh1: str or int,
        sesh2: str or int,
        sesh1_ind: int,
        delta_com: np.ndarray,
        highlight_inds1: np.ndarray or list or None = None,
        highlight_inds2: np.ndarray or list or None = None,
        plot_max=True,
        ax=None,
    ):
        """Reverse of above with a tweak - plot a single base session ROI overlaid on ALL reg neuron ROIs.
        Good for hand registering neurons...output 'delta_com' above should be input as a negative here."""
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(9, 4))

        self.get_session(sesh1).plot_rois(
            neuron_inds=[sesh1_ind], label=True, ax=ax[0], plot_max=plot_max
        )
        ax[0].set_title(f"{sesh1} ROI #{sesh1_ind}")
        if highlight_inds1 is not None:
            self.get_session(sesh1).plot_rois(
                neuron_inds=highlight_inds1,
                highlight_inds=highlight_inds1,
                label=True,
                plot_cell_id=False,
                ax=ax[0],
                plot_max=False,
            )

        self.get_session(sesh2).plot_rois(
            label=True, ax=ax[1], plot_max=plot_max, highlight_inds=highlight_inds2
        )
        _, _, roi_edges1 = self.get_session(sesh1).plot_rois(
            neuron_inds=[sesh1_ind],
            plot_max=False,
            plot_cell_id=False,
            label=True,
            ax=ax[1],
        )

        # Shift edges to overlay properly
        self.shift_roi_edges(roi_edges1, delta_com)

        # Make new roi stand out
        [roi_edge.set_color("w") for roi_edge in roi_edges1]
        [roi_edge.set_linewidth(3) for roi_edge in roi_edges1]

        # Automatically zoom into the area around the roi which has already been shifteg
        roi_mid = [
            np.mean(roi_edges1[0].get_xdata()),
            np.mean(roi_edges1[0].get_ydata()),
        ]

        # Zoom into reg axes.
        zoom_amt = 100
        ax[1].set_xlim(roi_mid[0] + np.array([-zoom_amt, zoom_amt]))
        ax[1].set_ylim(roi_mid[1] + np.array([zoom_amt, -zoom_amt]))

        # ax[0].set_xlim(roi_mid[0] - delta_com[0] + np.array([-60, 60]), auto=True)
        # ax[0].set_ylim(roi_mid[1] - delta_com[1] + np.array([-60, 60]), auto=True)

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
    def load_pairwise_np_map(map_np_filename):
        """Load a pairwise map into a pandas dataframe from an npy file."""
        map_filename = Path(map_np_filename)
        map_stem = map_filename.stem
        assert (
            len(map_stem.split("_")) == 3 and map_stem.split("_")[0] == "map"
        ), 'map file must be of the format "map_sesh1identifier_sesh2identifier.npy"'
        _, sesh1id, sesh2id = map_stem.split("_")

        map_array = np.load(map_filename)

        return pd.DataFrame({sesh1id: map_array[:, 0], sesh2id: map_array[:, 1]})

    def load_pairwise_map(self, sesh1, sesh2):
        savename = self.get_session(sesh1).basedir / f"map_{sesh1}_{sesh2}.pwmap.npy"

        pwmap = np.load(savename, allow_pickle=True).item()

        assert (
            pwmap.trim1 == self.get_session(sesh1).trim
        ), "Session1 trim dictionary does not match loaded map"
        assert (
            pwmap.trim2 == self.get_session(sesh2).trim
        ), "Session2 trim dictionary does not match loaded map"

        try:
            if pwmap.merge1 != self.get_session(sesh1).posthocmerge:
                self.quick_merge(
                    sesh1,
                    in_place=True,
                    method=pwmap.merge1["method"],
                    unit_ids=pwmap.merge1["unit_ids"],
                )
            if pwmap.merge2 != self.get_session(sesh2).posthocmerge:
                self.quick_merge(
                    sesh2,
                    in_place=True,
                    method=pwmap.merge2["method"],
                    unit_ids=pwmap.merge2["unit_ids"],
                )
            assert (
                pwmap.merge1 == self.get_session(sesh1).posthocmerge
            ), "pairwise map merge params don't match sesh1 merge params, double-check! May need to run .quick_merge before loading map"
            assert (
                pwmap.merge2 == self.get_session(sesh2).posthocmerge
            ), "pairwise map merge params don't match sesh2 merge params, double-check! May need to run .quick_merge before loading map"
        except AttributeError:
            print(
                "Merge params not found in either PairwiseMap or CaNeuronReg. Double check merging ok before proceeding!"
            )
        return pwmap


class PairwiseMap:
    """Class to save pairwise neuron registration maps"""

    def __init__(
        self, caregobj: CaNeuronReg, map_df, animal: str, sesh1: str, sesh2: str
    ):
        self.map = map_df
        self.animal = animal
        self.sesh1 = sesh1
        self.sesh2 = sesh2
        self.trim1 = caregobj.get_session(sesh1).trim
        self.trim2 = caregobj.get_session(sesh2).trim
        try:
            self.merge1 = caregobj.get_session(sesh1).posthocmerge
            self.merge2 = caregobj.get_session(sesh2).posthocmerge
        except AttributeError:
            self.merge1, self.merge2 = None, None
            print(
                "Can't find .posthocmerge attribue in CaNeuronReg - make sure you haven't merged neurons before proceeding"
            )

        self.savename = (
            caregobj.get_session(sesh1).basedir / f"map_{sesh1}_{sesh2}.pwmap.npy"
        )

    def to_numpy(self, savename=None):
        if savename is None:
            savename = self.savename
        np.save(savename, self, allow_pickle=True)


class MultiSessionMap:
    """Class to load in maps for more than one pair of sessions"""

    def __init__(self, pwmaps: list, sesh_order: list):
        self.maps = pwmaps
        self.multi_sesh_map = None
        sesh_names_check = []

        # Perform integrity checks before you do anything to make sure your inputs are compatible
        nsessions = len(pwmaps)
        for pwmap in pwmaps:
            sesh_names_check.append(pwmap.sesh1)
            sesh_names_check.append(pwmap.sesh2)
        sesh_names_check = [name for name in np.unique(sesh_names_check)]
        assert np.all(
            [sesh in sesh_names_check for sesh in sesh_order]
        ), 'all sessions in "sesh_order" must be in provided PairwiseMap objects'
        self.sesh_order = sesh_order

        # for idm, map_use in enumerate(
        #     pwmaps
        # ):  # Make sure sessions in sesh_names are legit
        #     assert (
        #         map_use.sesh1 == self.sesh_names[idm]
        #         and map_use.sesh2 == self.sesh_names[idm + 1]
        #     ), "map sessions do not match the order of 'sesh_name' input"

    def grab_map(self, sesh_pair_names, coactive_only: bool = False):
        """Grabs the appropriate pairwise map from the class. Returns none if that session pair doesn't exist"""
        assert (
            sesh_pair_names[0] != sesh_pair_names[1]
        ), "Self-mapping not yet enabled, sesh1 must not be equal to sesh2"
        try:
            pair_idx = np.where(
                [
                    (
                        mapp.sesh1 == sesh_pair_names[0]
                        and mapp.sesh2 == sesh_pair_names[1]
                    )
                    or (
                        mapp.sesh1 == sesh_pair_names[1]
                        and mapp.sesh2 == sesh_pair_names[0]
                    )
                    for mapp in self.maps
                ]
            )[0][0]
            map_use = self.maps[pair_idx].map
            if coactive_only:
                map_use = map_use[(map_use >= 0).all(axis=1)]
            return map_use
        except IndexError:
            return None

    def get_reverse_map(self, sesh_pair_names):
        """Gets session maps going backwards in time."""
        pass

    def stepwise_reg(self, **kwargs):
        """Step through and register each session in order to the next
        Current functionality is only for 3 sessions, trying to make recursive to
        easily add in however many sessions you want.
        :param **kwargs to .add_third_session()"""

        # Initialize multi_sesh_map with last two columns of input map
        if len(self.sesh_order) > 3:
            # Code to recursively define multi_map here
            print("more than one 3 session registration not yet implemented")
            multi_sesh_map = MultiSessionMap(
                self.maps[:-1], self.sesh_order[:-1]
            ).stepwise_reg()
            pass
        elif len(self.sesh_order) == 3:
            sesh1, sesh2, sesh3 = self.sesh_order
            # Grab map from session 1->2 and session 2->3 (and 1->3 if there)
            map1_2 = self.grab_map([sesh1, sesh2])
            map2_3 = self.grab_map([sesh2, sesh3])
            map1_3 = self.grab_map([sesh1, sesh3])
            multi_sesh_map = map1_2.iloc[:, -2:].copy()
        elif (
            len(self.sesh_order) == 2
        ):  # Just return map1_2 if only two sessions specified
            sesh1, sesh2 = self.sesh_order
            multi_sesh_map = self.grab_map([sesh1, sesh2])

            self.multi_sesh_map = multi_sesh_map

            return multi_sesh_map

        multi_sesh_map.insert(
            2, sesh3, np.ones_like(map1_2[sesh2]) * -1
        )  # Add in session 3 column

        # Identify mapped and unmapped neurons from session 2 to session 3
        sesh2neurons = map2_3[sesh2]
        mapped_neurons = map1_2.iloc[multi_sesh_map[sesh2].values > -1]
        unmapped_neurons = sesh2neurons[
            [n2 not in mapped_neurons[sesh2].values for n2 in sesh2neurons]
        ]

        # Next add in any neurons from session 2 that didn't appear in session 1
        rownum = multi_sesh_map.shape[0]  # starting row for new neurons
        for idn, neuron in enumerate(unmapped_neurons):
            multi_sesh_map.loc[rownum + idn] = [-1, neuron, -1]

        # Assign correct id to session 3 for neurons mapped from session 1 to session 2
        sesh2idx_reorder = [
            np.where(nid == multi_sesh_map[sesh2])[0][0] for nid in sesh2neurons
        ]  # Get id of each neuron in session 2 in multi_sesh_map
        multi_sesh_map[sesh3].iloc[sesh2idx_reorder] = map2_3[sesh3].iloc[sesh2neurons]

        # Last add in any neurons in session3 that don't map to other neurons.

        # Optional: add in a map from sesh1 to sesh3 to fill in any neurons that are only active in sesh1 and 3
        if map1_3 is not None:
            multi_sesh_map = self.add_third_session(multi_sesh_map, map1_3, **kwargs)
        else:
            print(
                "Map from session 1 to session 3 not found, cannot fill neurons active in sessions 1 and 3 only"
            )

        self.multi_sesh_map = multi_sesh_map

        return multi_sesh_map

    @staticmethod
    def add_third_session(
        multi_sesh_map: pd.DataFrame,
        sesh1_3map: pd.DataFrame,
        overwrite_indirect: bool = False,
    ) -> pd.DataFrame:
        """Adds in any neurons mapped from session 1 to session 3 that went silent in session 2 and could not be mapped
        indirectly through session 2."""

        sesh1, sesh3 = sesh1_3map.keys()

        sesh1_multi_idx = np.array(
            [
                multi_sesh_map[sesh1]
                .iloc[(n1 == multi_sesh_map[sesh1]).values]
                .index[0]
                for n1 in sesh1_3map[sesh1]
            ]
        )  # get index for all session1 neurons in multi_sesh_map

        # Identify new cells
        sesh2offbool = np.bitwise_and(
            multi_sesh_map.iloc[sesh1_multi_idx, 0] >= 0,
            multi_sesh_map.iloc[sesh1_multi_idx, 1] == -1,
        )
        sesh3newbool = np.bitwise_and(sesh2offbool, sesh1_3map[sesh3] >= 0).values
        sesh3new_multi_idx = sesh1_multi_idx[
            sesh3newbool
        ]  # index for new cells in multi_map

        # Assign sesh1 on - sesh2 off - sesh3 on cells the appropriate neuron id
        multi_sesh_map[sesh3].iloc[sesh3new_multi_idx] = sesh1_3map[sesh3].iloc[
            sesh3newbool
        ]

        ## NRK todo
        if overwrite_indirect:
            # Identify mappings directly and indirectly (through session 2)
            sesh3_directreg = sesh1_3map[sesh3]
            sesh3_indirectreg = multi_sesh_map[sesh3].iloc[sesh1_multi_idx]
            multi_sesh_map[sesh3].iloc[sesh1_multi_idx] = sesh3_directreg

        return multi_sesh_map

    @staticmethod
    def transitive_test(multi_sesh_map, sesh1_3_map):
        """Checks that all pairwise registrations (1->2->3) pass the transitive test and match 1->3 directly.
        Also adds in any neurons that went silent on session 2 but re-appeared in session 3.
        :param keep_only_passes: True (default) only keep neurons that pass the transitive test. Sets entire row to -1.
        :param_overwrite_transitive_fails = False (default), True will overwrite any mappings
        that don't pass the transitive test"""

        # Not yet implemented - only pairwise registrations considered.
        pass


def load_pairwise_map(map_path):

    return np.load(map_path, allow_pickle=True).item()


def id_and_plot_reference_cells(
    caneurons1: CaNeurons, caneurons2: CaNeurons, return_inds=False, **kwargs
):
    """identify and clearly plot reference cells active across both session to aid
    in manual cell registration.
    **kwargs to CaNeurons.plot_rois"""

    # Plot out cells from both sessions
    careg = CaNeuronReg([caneurons1, caneurons2])
    fig, ax = careg.plot_rois_across_sessions()
    fig.canvas.draw()  # This magical line makes the plot actually appear mid function run!

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
        plot_cell_id=False,
        plot_masks=False,
        highlight_inds=sesh1_inds,
        **kwargs,
    )
    caneurons2.plot_rois(
        plot_max=True,
        ax=ax[1][1],
        label=True,
        plot_cell_id=False,
        plot_masks=False,
        highlight_inds=sesh2_inds,
        **kwargs,
    )

    if not return_inds:
        return fig, ax
    else:
        return fig, ax, sesh1_inds, sesh2_inds


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


if __name__ == "__main__":
    import session_directory as sd

    # from neuropy.io.minianio import MinianIO
    #
    session_list = ["Habituation2", "Training", "Recall1"]
    animal = "Jyn"
    basedir = sd.get_session_dir(animal, session_list[0])
    # careg = CaNeuronReg(
    #     [
    #         MinianIO(basedir=sd.get_session_dir(animal, session)).trim_neurons(
    #             keep=["good", "maybe_interneurons"]
    #         )
    #         for session in session_list
    #     ],
    #     alias=session_list,
    # )
