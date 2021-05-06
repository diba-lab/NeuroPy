"""Class to deal with DeepLabCut data"""

from movie import tracking_movie
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd
import scipy
import seaborn as sns
from parsePath import Recinfo
import re


class DLC:
    """Import and parse DeepLabCut tracking data. Will import all files into a list unless you specify a
    `search_str` list which will only include files with ALL the strings listed."""

    def __init__(self, base_path, search_str=None, movie_type=".avi", pix2cm=1):
        # if isinstance(basepath, Recinfo):
        #     self._obj = basepath
        # else:
        #     self._obj = Recinfo(basepath)
        #
        # # ------- defining file names ---------
        # filePrefix = self._obj.files.filePrefix
        #
        # @dataclass
        # class files:
        #     DLC: str = filePrefix.with_suffix(".DLC.npy")
        #
        # self.files = files()

        # fix poorly annotated movie types
        if movie_type[0] != ".":
            movie_type = "." + movie_type

        # Make search_str into a list if not already one
        search_str = [search_str] if isinstance(search_str, str) else search_str

        # Make base directory into a pathlib object for easy manipulation
        base_dir = Path(base_path)

        # Grab tracking files and corresponding movie file.
        tracking_files = sorted(base_dir.glob("*.h5"))
        movie_files = [
            Path(str(file)[0 : str(file).find("DLC")] + movie_type)
            for file in tracking_files
        ]
        # Pull out only the specific file you want if specified with a search_str above, otherwise grab first one only!
        # NRK todo: use walrus operator here to check and make sure output file list len == 1?
        if search_str is not None:
            self.tracking_file = str(
                get_matching_files(tracking_files, match_str=search_str)[0]
            )
            self.movie_file = str(
                get_matching_files(movie_files, match_str=search_str)[0]
            )
        else:
            self.tracking_file = str(tracking_files[0])
            self.movie_file = str(movie_files[0])
        print("Using tracking file " + str(self.tracking_file))
        print("Using movie file " + str(self.movie_file))
        self.pix2cm = pix2cm

        # First import position data as-is
        pos_data = import_tracking_data(self.tracking_file)

        # Assume you've only run one scorer and that bodyparts are the same for all files
        self.scorername = scorername(pos_data)
        self.bodyparts = bodyparts(pos_data)

        # Now grab the data for the scorer identified above - easier access later!
        self.pos_data = pos_data[scorername]

        # Initialize movie
        if self.movie_file is not None:
            self.movie = tracking_movie(self.movie_file)
            self.SampleRate = self.movie.get_sample_rate()
        else:
            SampleRate = None

    def get_on_maze(self, bodypart="neck_base", likelihood_thresh=0.9):

        # Find first time the animal's body-part likelihood jumps above the threshold for an extended period of time
        # (3.33 seconds)
        on_maze_ind = np.where(
            self.pos_data[self.scorer_name][bodypart][["likelihood"]]
            .squeeze()
            .rolling(np.round(3.33 * SampleRate))
            .mean()
            > likelihood_thresh
        )[0][0]

        return on_maze_ind

    def smooth_pos(
        self,
        bodyparts=None,
        lcutoff=0.9,
        std=1 / 15,
    ):

        if bodyparts is None:
            bodyparts = self.bodyparts

        self.pos_smooth = smooth_pos(
            self.pos_data,
            bodyparts,
            lcutoff=lcutoff,
            std=std,
            SampleRate=self.SampleRate,
        )

    def plot1d(
        self,
        bodyparts=None,
        data_type="raw",
        feature="x",
        lcutoff=0,
        plot_style="-",
        ax=None,
    ):
        """Plot the feature ('x', 'y', or post-processed 'speed') for a given bodypar vs. time. """
        # Make sure you have a sample rate to calculate time from
        assert (
            self.SampleRate is not None
        ), "No video SampleRate found. Specify manually, e.g. DLC.SampleRate = 30"

        # plot 1st body-part by default
        if bodyparts is None:
            bodyparts = [self.bodyparts[0]]
        else:
            # Make into a list if only one part specified
            if not isinstance(bodyparts, list):
                bodyparts = [bodyparts]

        # Plot all lines the same if only only one plot style listed
        if not isinstance(plot_style, list) or len(plot_style) == 1:
            plot_style = [plot_style for _ in bodyparts]

        if data_type == "raw":
            data_plot = self.pos_data
        elif data_type == "smooth":
            data_plot = self.pos_smooth

        for bodypart, style in zip(bodyparts, plot_style):
            ax = plot_1d(
                data_plot,
                body_part=bodypart,
                feature=feature,
                ax=ax,
                SR=self.SampleRate,
                likelihood_cutoff=lcutoff,
                plot_style=style,
            )

        # Add legend for multiple bodyparts
        if len(bodyparts) > 1:
            ax.set_title("Multiple bodyparts")
            ax.legend(bodyparts)

        return ax

    def plot2d(
        self,
        data_type="raw",
        bodyparts=None,
        plot_style="-",
        lcutoff=0,
        ax=None,
    ):

        # Make sure you have a sample rate to calculate time from
        assert (
            self.SampleRate is not None
        ), "No video SampleRate found. Specify manually, e.g. DLC.SampleRate = 30"

        # plot 1st body-part by default
        if bodyparts is None:
            bodyparts = [self.bodyparts[0]]
        else:
            # Make into a list if only one part specified
            if not isinstance(bodyparts, list):
                bodyparts = [bodyparts]

        if data_type == "raw":
            data_plot = self.pos_data
        elif data_type == "smooth":
            data_plot = self.pos_smooth

        for bodypart in bodyparts:
            ax = plot_2d(
                data_plot,
                body_part=bodypart,
                likelihood_cutoff=lcutoff,
                pix2cm=self.pix2cm,
                plot_style=plot_style,
                ax=ax,
            )

        # Add legend for multiple bodyparts
        if len(bodyparts) > 1:
            ax.set_title("Multiple bodyparts")
            ax.legend(bodyparts)

        return ax


def _as_array(pos_data):
    """Make first entry of a list into an array"""
    if isinstance(pos_data, list):
        pos_array = pos_data[0]


def get_matching_files(files, match_str=["habituation", "Camera 1"]):
    """Grab only the files that contain the strings in `match_str`"""
    find_bool = []
    for fstr in match_str:
        find_bool.extend([re.search(fstr, str(file)) is not None for file in files])

    match_ind = np.where(np.asarray(find_bool).all(axis=0))[0]

    return [files[ind] for ind in match_ind]


def import_tracking_data(DLC_h5filename):
    return pd.read_hdf(DLC_h5filename)


def scorername(pos_data):
    """Get DLC scorername - assumes only 1 used."""
    scorername = pos_data.columns.levels[
        np.where([name == "scorer" for name in pos_data.columns.names])[0][0]
    ][0]

    return scorername


def bodyparts(pos_data):
    """Get names of bodyparts"""
    bodyparts = pos_data.columns.levels[
        np.where([name == "bodyparts" for name in pos_data.columns.names])[0][0]
    ]

    return bodyparts


def smooth_pos(pos_data, bodyparts, lcutoff=0.9, std=1 / 15, SampleRate=30):
    """Smooth data using a gaussian window. Default parameters work well for SampleRate=30, but verify yourself using
    using plot1d(pos_data, style='.'), plot1d(pos_smooth, style='-')."""
    pos_smooth = {}
    for bodypart in bodyparts:
        data_use = pos_data[bodypart]
        good_bool = data_use["likelihood"] > lcutoff
        data_use["x"][~good_bool] = np.nan
        data_use["y"][~good_bool] = np.nan
        for idc, coord in enumerate(["x", "y"]):
            data_use[coord] = (
                data_use[coord]
                .interpolate("linear", limit=interp_limit, limit_direction="both")
                .rolling(8 * std * SampleRate, win_type="gaussian", center=True)
                .mean(std=std * SampleRate)
            )

        # Dump into pos_smooth keeping likelihood!
        pos_smooth[bodypart] = data_use

    return pos_smooth


def plot_1d(
    DLCtable, body_part, feature, SR=30, ax=None, likelihood_cutoff=0, plot_style="-"
):
    """:param DLCtable: multi-dimensional pandas dataframe loaded in from DLC output .h5 file for a given scorer, e.g.
    pos_data = pd.read_h5(file_name)
    DLCtable = pos_data[scorer_name]"""

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches([23.5, 5.4])

    good_bool = DLCtable[body_part]["likelihood"] > likelihood_cutoff
    data_plot = DLCtable[body_part][feature][good_bool]

    time = np.arange(1, len(DLCtable[body_part]["likelihood"]) + 1)

    ax.plot(time[good_bool], data_plot, plot_style)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(feature)
    ax.set_title(body_part)
    sns.despine(ax=ax)

    return ax


def plot_2d(
    DLCtable, body_part, likelihood_cutoff=0, pix2cm=1, ax=None, plot_style="-"
):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches([15.2, 11.5])

    good_bool = DLCtable[body_part]["likelihood"] > likelihood_cutoff
    xpos = DLCtable[body_part]["x"][good_bool] * pix2cm
    ypos = DLCtable[body_part]["y"][good_bool] * pix2cm

    ax.plot(xpos, ypos, plot_style)
    ax.set_title(body_part)
    ax.set_xlabel("lcutoff = " + str(likelihood_cutoff))

    return ax


if __name__ == "__main__":
    dlc = DLC(
        "/data2/Trace_FC/Pilot1/Rat700/2021_02_23_training",
        search_str="Camera 1",
        pix2cm=0.13,
    )
    pass
