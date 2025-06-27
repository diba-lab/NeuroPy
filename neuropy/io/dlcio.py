"""Class to deal with DeepLabCut data"""

from neuropy.io.movie import tracking_movie, deduce_starttime_from_file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd
import scipy
import seaborn as sns
import re
from pickle import dump, load
import scipy.ndimage as simage

from neuropy.io.miniscopeio import MiniscopeIO

from neuropy.io.optitrackio import getStartTime, posfromCSV


class DLC:
    """Import and parse DeepLabCut tracking data. Will import all files into a list unless you specify a
    `search_str` list which will only include files with ALL the strings listed. Infers session and animal name
    from directory structure in base_path, e.g. base_path = ...../Animal_Name/2021_02_22_training"""

    def __init__(self, base_path, search_str=None, movie_type=".mp4", pix2cm=1, get_timestamps=True,
                 camera_type='optitrack'):

        # fix poorly annotated movie_type input
        if movie_type[0] != ".":
            movie_type = "." + movie_type

        # Make search_str into a list if not already one
        search_str = [search_str] if isinstance(search_str, str) else search_str

        # Make base directory into a pathlib object for easy manipulation
        self.base_dir = Path(base_path)

        # Infer animal name and session from base_path directory structure.
        self.session = self.base_dir.parts[-1]
        self.animal_name = self.base_dir.parts[-2]

        # Grab tracking files and corresponding movie file.
        tracking_files = sorted(self.base_dir.glob("**/*.h5"))
        movie_files = sorted(self.base_dir.glob("**/*" + movie_type))
        # if movie_type == ".avi":  # Not preferred
        #     movie_files = [
        #         Path(str(file)[0 : str(file).find("DLC")] + movie_type)
        #         for file in tracking_files
        #     ]
        # elif movie_type == ".mp4":  # Better, more consistent naming
        #     movie_files = sorted(self.base_dir.glob("*.mp4"))

        # Pull out only the specific file you want if specified with a search_str above, otherwise grab first one only!
        # NRK todo: use walrus operator here to check and make sure output file list len == 1?
        if search_str is not None:
            self.tracking_files = Path(
                get_matching_files(tracking_files, match_str=search_str)
            )

        else:
            self.tracking_files = [Path(file) for file in tracking_files]
            self.movie_files = [tracking_file.with_suffix(movie_type) for tracking_file in self.tracking_files]

        self.movies = []
        # self.nframes = self.nframes
        # self.SampleRate = []
        self.timestamps = []

        pos_list, scorernames = [], []
        for idf, file in enumerate(self.tracking_files):
            print(f"Using tracking file #{idf+1}: {str(file)}")
            # Import position data as-is
            pos_data = import_tracking_data(file)
            scorername = get_scorername(pos_data)
            scorernames.append(scorername)
            # Now grab the data for the scorer identified above - easier access later!
            pos_data = pos_data[scorername]
            pos_list.append(pos_data)
        self.pos_data = pd.concat(pos_list, axis=0)
        self.pix2cm = pix2cm

        # Assume you've only run one scorer and that bodyparts are the same for all files
        self.scorernames = scorername
        self.bodyparts = get_bodyparts(pos_data)

        # # Now grab the data for the scorer identified above - easier access later!
        # self.pos_data = pos_data[self.scorername]

        # Now convert from pixels to centimeters
        self.to_cm()

        # Grab metadata for later access
        self.meta = []
        self.get_metadata()
        self.nframes = self.get_nframes()
        self.SampleRate = self.get_SampleRate()

        # Initialize other fields
        self.timestamp_type = None
        self.speed = None

    def get_SampleRate(self):
        """Get mean sample rate across all movies"""
        SampleRate = []
        for idm, movie_file in enumerate(self.movie_files):
            try:
                SampleRate.append(self.meta[idm]["data"]["fps"])
                meta_found = True
            except KeyError:  # try to import from movie directly
                meta_found = False
            # Initialize movie
                if movie_file.is_file():
                    self.movies[idm] = tracking_movie(movie_file)
                    SampleRate.append(self.movies[idm].get_sample_rate())
                else:
                    SampleRate.append(None)

        SampleRate = np.mean(SampleRate)
        if not meta_found:
            print("No metadata found - taking mean sample rate from all raw movies in self.movie_files")
        if idm > 0:
            print("Multiple videos found - taking mean sample rate from all videos")

        return SampleRate

    def get_nframes(self):

        nframes = []
        for idm, movie_file in enumerate(self.movie_files):
            try:
                nframes.append(self.meta[idm]["data"]["nframes"])
                meta_found = True
            except KeyError:  # try to import from movie directly
                meta_found = False
                # Initialize movie
                if movie_file.is_file():
                    self.movies[idm] = tracking_movie(movie_file)
                    nframes.append(self.movies[idm].get_nframes())
                else:
                    nframes.append(None)

        if not meta_found:
            print("No metadata found - getting nframes directly from all raw movies in self.movie_files")

        return nframes

    def get_metadata(self):
        """Load in meta-data corresponding to tracking file"""
        self.meta = []
        for tracking_file in self.tracking_files:
            meta_file = tracking_file.parent / (tracking_file.stem + "_meta.pickle")

            with open(meta_file, "rb") as f:
                self.meta.append(load(f))

    def get_timestamps(self, camera_type: str in ['optitrack', 'ms_webcam', 'ms_webcam1', 'ms_webcam2'] = 'optitrack',
                       exclude_str: str or None = None, include_str: str or None = None):
        """Tries to import timestamps from CSV file from optitrack, if not, infers it from timestamp in filename,
        sample rate, and nframes
        :param camera_type: 'optitrack' looks for optitrack csv file with tracking data, other options look for
        :params exclude_str, include_str: see MiniscopeIO.load_all_timestamps - can include or exclude folders
        UCLA miniscope webcam files"""

        assert camera_type in ['optitrack', 'ms_webcam', 'ms_webcam1', 'ms_webcam2']
        self.timestamp_type = camera_type
        self.timestamps = []
        if camera_type == 'optitrack':
            for idt, tracking_file in enumerate(self.tracking_files):
                opti_file = tracking_file.parent / (
                    tracking_file.stem[: tracking_file.stem.find("-Camera")] + ".csv"
                )
                if opti_file.is_file():
                    start_time = getStartTime(opti_file)
                    _, _, _, t = posfromCSV(opti_file)

                    self.timestamps.append(start_time + pd.to_timedelta(t, unit="sec"))
                else:
                    print(f"No Optitrack csv file found at {tracking_file.stem}.")
                    print("Inferring start time from file name. SECOND PRECISION IN START TIME!!!")
                    start_time = deduce_starttime_from_file(tracking_file)
                    print(start_time)
                    time_deltas = pd.to_timedelta(
                        np.arange(self.nframes[idt]) / self.SampleRate, unit="sec"
                    )
                    self.timestamps.append(start_time + time_deltas)
            try:
                self.timestamps = pd.concat(self.timestamps, axis=0)
            except TypeError:
                self.timestamps = pd.concat([pd.Series(tstamps) for tstamps in self.timestamps], axis=0)

            # Sort appropriately
            isort = self.timestamps.argsort().values
            self.timestamps = self.timestamps.iloc[isort]

            self.timestamps = pd.DataFrame({"Timestamps": self.timestamps})

        else:
            mio = MiniscopeIO(self.base_dir)
            webcam_flag = True if camera_type == "ms_webcam" else int(camera_type[-1])
            self.timestamps = mio.load_all_timestamps(webcam=webcam_flag, exclude_str=exclude_str,
                                                      include_str=include_str)
            # Sort appropriately
            isort = self.timestamps.iloc["Timestamps"].argsort()
            self.timestamps = self.timestamps.iloc[isort, :]

        # Reorder timestamps for position data if necessary
        if not np.all(isort == np.arange(self.timestamps.shape[0])):
            print(
                "WARNING: Timestamps imported out of order. Resorting position data and setting .speed and .pos_smooth to None")
            self.pos_data = self.pos_data.iloc[isort, :]
            self.pos_smooth = None
            self.speed = None


    def to_cm(self):
        """Convert pixels to centimeters in pos_data"""
        idx = pd.IndexSlice
        self.pos_data.loc[:, idx[:, ("x", "y")]] = (
            self.pos_data.loc[:, idx[:, ("x", "y")]] * self.pix2cm
        )

    def get_on_maze(self, bodypart="neck_base", likelihood_thresh=0.9):

        # Find first time the animal's body-part likelihood jumps above the threshold for an extended period of time
        # (3.33 seconds)
        on_maze_ind = np.where(
            self.pos_data[self.scorer_name][bodypart][["likelihood"]]
            .squeeze()
            .rolling(np.round(3.33 * self.SampleRate))
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
        self.smooth_lcutoff = lcutoff

        if bodyparts is None:
            bodyparts = self.bodyparts

        self.pos_smooth = smooth_pos(
            self.pos_data,
            bodyparts,
            lcutoff=lcutoff,
            std=std,
            SampleRate=self.SampleRate,
        )

    def get_all_speed(self):
        """Get speed of all bodyparts"""
        df_list = []
        for bodypart in self.bodyparts:
            df_list.append(pd.DataFrame({bodypart: self.get_speed(bodypart)}))

        self.speed = pd.concat(df_list, axis=1)

        return self.speed

    def get_speed(self, bodypart):
        """Get speed of animal"""
        assert self.pos_smooth is not None, "You must smooth data first"
        assert self.timestamps is not None, "You must get timestamps first"
        if self.timestamp_type == "optitrack":
            print('get_speed not yet tested for optitrack data, use with caution')

        # Get position of bodypart
        data_use = self.pos_smooth[bodypart]
        x = data_use["x"]
        y = data_use["y"]
        total_seconds = (self.timestamps.reset_index()['Timestamps'] - self.timestamps.reset_index()['Timestamps'][0]).dt.total_seconds()

        # Get delta in position and time from frame-to-frame
        delta_pos = np.sqrt(np.square(np.diff(x)) + np.square(np.diff(y)))
        delta_t = np.diff(total_seconds)

        # Calculate speed
        speed = delta_pos / delta_t
        speed = np.hstack((speed, np.nan))  # add in Nan at end to match pos data shape
        return speed


    def plot1d(
        self,
        bodyparts=None,
        data_type="raw",
        feature="x",
        lcutoff=0,
        plot_style="-",
        ax=None,
    ):
        """Plot the feature ('x', 'y', or post-processed 'speed') for a given bodypar vs. time.
        See function dlc.plot1d for docstring"""

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
                title_use=self.animal_name + ": " + bodypart,
            )

        # Add legend for multiple bodyparts
        if len(bodyparts) > 1:
            ax.set_title("Multiple bodyparts lcutoff = " + str(lcutoff))
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
        """Plot the feature ('x', 'y', or post-processed 'speed') for a given bodypart vs. time.
        See function dlc.plot2d for docstring"""

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
                title_use=self.animal_name + ": " + bodypart,
            )

        # Add legend for multiple bodyparts
        if len(bodyparts) > 1:
            ax.set_title("Multiple bodyparts")
            ax.legend(bodyparts)

        return ax

    def calculate_speed(self):
        calculate_speed(self.pos_smooth, bodyparts=self.bodyparts, SR=self.SampleRate)


def _as_array(pos_data):
    """Make first entry of a list into an array"""
    if isinstance(pos_data, list):
        pos_array = pos_data[0]


def get_matching_files(files, match_str=["habituation", "Camera 1"], exclude_str=None):
    """Grab only the files that contain the strings in `match_str` and DON'T include strings in `exclude_str`"""

    # Fix inputs
    if isinstance(match_str, str):
        match_str = [match_str]
    if isinstance(exclude_str, str):
        exclude_str = [exclude_str]

    find_bool = []
    for fstr in match_str:
        find_bool.append(
            [re.search(fstr, str(file.parts[-1])) is not None for file in files]
        )

    if exclude_str is not None:
        exclude_bool = []
        for estr in exclude_str:
            exclude_bool.append(
                [re.search(estr, str(file.parts[-1])) is not None for file in files]
            )
    else:
        exclude_bool = np.zeros_like(find_bool)

    find_bool_array = np.asarray(find_bool).squeeze()
    if find_bool_array.ndim == 2:
        find_bool_array = find_bool_array.all(axis=0)

    exclude_bool_array = np.asarray(exclude_bool).squeeze()
    if exclude_bool_array.ndim == 2:
        exclude_bool_array = exclude_bool_array.any(axis=0)
    match_ind = np.where(
        np.bitwise_and(find_bool_array, np.bitwise_not(exclude_bool_array))
    )[0]

    matching_files = [files[ind] for ind in match_ind]
    if len(matching_files) == 0:
        matching_files = [None]

    return matching_files


def import_tracking_data(DLC_h5filename):
    return pd.read_hdf(DLC_h5filename)


def get_scorername(pos_data):
    """Get DLC scorername - assumes only 1 used."""
    scorername = pos_data.columns.levels[
        np.where([name == "scorer" for name in pos_data.columns.names])[0][0]
    ][0]

    return scorername


def get_bodyparts(pos_data):
    """Get names of bodyparts"""
    bodyparts = pos_data.columns.levels[
        np.where([name == "bodyparts" for name in pos_data.columns.names])[0][0]
    ]

    return bodyparts


def smooth_pos(
    pos_data, bodyparts, lcutoff=0.9, std=1 / 15, interp_limit=15, SampleRate=30
):
    """
    Smooth data using a gaussian window. Default parameters work well for SampleRate=30, but verify yourself using
    using plot1d(pos_data, style='.'), plot1d(pos_smooth, style='-').
    :param pos_data: DLCtable
    :param bodyparts: list
    :param lcutoff: likelihood threshold - all points with likelihood less than this not within interp_limit are
    sent to Nan.
    :param std: # frames for std of gaussian smoother applied
    :param interp_limit: max # frames to interpolate if nan or below likelihood threshold
    :param SampleRate: fps
    :return:
    """
    pos_smooth = pos_data.copy()
    for bodypart in bodyparts:
        data_use = pos_data[bodypart].copy()
        data_copy = data_use.copy()
        good_bool = data_use["likelihood"] > lcutoff
        data_use["x"][~good_bool] = np.nan
        data_use["y"][~good_bool] = np.nan
        for idc, coord in enumerate(["x", "y"]):
            mask = data_copy[coord] >= 0
            data_copy.loc[mask, coord] = (
                data_use.loc[mask, coord]
                .interpolate("linear", limit=interp_limit, limit_direction="both")
                .rolling(
                    np.round(8 * std * SampleRate).astype("int"),
                    win_type="gaussian",
                    center=True,
                )
                .mean(std=std * SampleRate)
            )

        # Dump into pos_smooth keeping likelihood!
        pos_smooth[bodypart] = data_copy

    return pos_smooth


def calculate_speed(pos_smooth, bodyparts=None, SR=1, pix2cm=1):
    """Calculate speed from smoothed position data and dump as a column into the dataframe"""

    if bodyparts is None:  # Calculate on all bodyparts by default.
        bodyparts = get_bodyparts(pos_smooth)

    for idb, bodypart in enumerate(bodyparts):
        pos_bodypart = pos_smooth[bodypart].copy()
        # calculate velocity
        pos_bodypart["time"] = pos_bodypart.index / SR
        dist = pos_bodypart.diff()
        dist["Dist"] = np.sqrt(dist.x ** 2 + dist.y ** 2)
        dist["Speed"] = dist.Dist * pix2cm / pos_bodypart["time"].diff()

        # pos_bodypart.loc[:, "speed"] = dist.Speed
        pos_smooth.loc[:, (bodypart, "time")] = pos_bodypart["time"]
        pos_smooth.loc[:, (bodypart, "speed")] = dist.Speed

    # This return should theoretically be un-needed since pos_smooth is mutable...
    return pos_smooth


def plot_1d(
    DLCtable,
    body_part,
    feature,
    SR=30,
    ax=None,
    likelihood_cutoff=0,
    plot_style="-",
    title_use=None,
):
    """:param DLCtable: multi-dimensional pandas dataframe loaded in from DLC output .h5 file for a given scorer, e.g.
    pos_data = pd.read_h5(file_name)
    DLCtable = pos_data[scorer_name]"""

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches([23.5, 5.4])

    good_bool = DLCtable[body_part]["likelihood"] > likelihood_cutoff
    data_plot = DLCtable[body_part][feature][good_bool]

    time = np.arange(1, len(DLCtable[body_part]["likelihood"]) + 1) / SR

    ax.plot(time[good_bool], data_plot, plot_style)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(feature)
    if title_use is None:
        ax.set_title(body_part)
    else:
        ax.set_title(title_use)
    sns.despine(ax=ax)

    return ax


def plot_2d(
    DLCtable,
    body_part,
    likelihood_cutoff=0,
    pix2cm=1,
    ax=None,
    plot_style="-",
    title_use=None,
):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches([15.2, 11.5])

    good_bool = DLCtable[body_part]["likelihood"] > likelihood_cutoff
    xpos = DLCtable[body_part]["x"][good_bool] * pix2cm
    ypos = DLCtable[body_part]["y"][good_bool] * pix2cm

    ax.plot(xpos, ypos, plot_style)
    ax.set_xlabel("lcutoff = " + str(likelihood_cutoff))

    if title_use is None:
        ax.set_title(body_part)
    else:
        ax.set_title(title_use)

    return ax


if __name__ == "__main__":
    # dlc_path = '/data2/Trace_FC/Recording_Rats/Finn2/2023_05_08_training'
    # dlc_path = '/data2/Trace_FC/Recording_Rats/Han/2022_08_03_training'
    dlc_path = '/data2/Trace_FC/Recording_Rats/Rose/2022_06_22_training'
    arena_side_pix = 60  # Keep this
    arena_side_cm = 25.4  # Update this after measuring!!!
    pix2cm = arena_side_cm / arena_side_pix

    # Read in DLC data
    dlc = DLC(dlc_path, pix2cm=pix2cm)

    # Smooth position, get timestamps, and get speed
    dlc.get_timestamps('ms_webcam', include_str="/2_training/")
    dlc.smooth_pos(bodyparts=["crown_middle", "back_middle"])
    dlc.get_all_speed()
    pass
