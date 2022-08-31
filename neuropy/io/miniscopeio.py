import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pathlib
import json
import pickle
import re
import math

from neuropy.utils.manipulate_files import prepend_time_from_folder_to_file


class MiniscopeIO:
    def __init__(self, basedir) -> None:
        self.basedir = Path(basedir)
        self.times_all = None
        self.orient_all = None
        pass

    def load_minian(self, minian_folder=None):

        # Try to autodetect minian folder
        if minian_folder is None:
            assert (
                len(minian_folder := sorted(self.basedir.glob("**/minian"))) == 1
            ), "More than one minian folder found, fill in directory manually"
            self.minian_folder = Path(minian_folder[0])

        self.minian = {
            "A": np.load(self.minian_folder / "A.npy"),
            "C": np.load(self.minian_folder / "C.npy"),
            "S": np.load(self.minian_folder / "S.npy"),
            "unit_id": None,
            "YrA": None,
            "curated_neurons": None,
            "good_bool": None,
        }

        # Load in YrA (raw traces) if saved
        try:
            self.minian["YrA"] = np.load(self.minian_folder / "YrA.npy")
        except FileNotFoundError:
            print('No raw traces found. "YrA" variable not loaded')

        # Load in unit ids if saved and warn if not!
        try:
            self.minian["unit_id"] = np.load(self.minian_folder / "unit_id.npy")
        except FileNotFoundError:
            print(
                "No unit_id file found! Unit id references might not work later! Check and save this variable!"
            )

        # try to load in curated neurons if you have done so in the NRK modified minian pipeline
        try:
            with open(self.minian_folder / "curated_neurons.pkl", "rb") as f:
                self.minian["curated_neurons"] = pickle.load(f)
        except:
            print("No curated neurons file found")

    def load_all_timestamps(self, format="UCLA", exclude_str: str = "WebCam"):
        """Loads multiple timestamps from multiple videos in the UCLA miniscope software file format
        (folder = ""hh_mm_ss")

        :param format: str, either 'UCLA' (default) to use the UCLA miniscope folder format or a regex if you are
        using a different folder naming convention.
        :param exclude_str: exclude any folders containing this string from being loaded in.
        :return:
        """

        # Get recording folders, sorted by time of acquisition
        if format == "UCLA":
            search_regex = "**/[0-1][0-9]_[0-6][0-9]_[0-6][0-9]"
        self.rec_folders = sorted(self.basedir.glob(search_regex))

        # Exclude any folders containing the specified "exclude_str" parameter
        if exclude_str is not None:
            rec_folder2 = []
            for folder in self.rec_folders:
                if re.search(exclude_str, str(folder)) is None:
                    print("including folder " + str(folder))
                    rec_folder2.append(folder)
            self.rec_folders = rec_folder2

        # Loop through and load all timestamps, then concatenate
        times_list = []
        for rec_folder in self.rec_folders:
            times_temp, _, _, _ = load_timestamps(
                rec_folder, corrupted_videos="from_file"
            )
            times_list.append(times_temp)

        self.times_all = pd.concat(times_list)

        # Remove any timestamps corresponding to frames you've removed.
        try:
            good_frames_bool = np.load(self.minian_folder / "good_frame_bool.npy")
            print(
                "Keeping "
                + str(good_frames_bool.sum())
                + ' good frames found in "good_frame_bool.npy" file'
            )
            self.times_all = self.times_all[good_frames_bool]
        except:
            print(
                "no " + str(self.minian_folder / "good_frame_bool.npy") + " file found"
            )
            print("Check and make sure frames in data and timestamps match!")
            pass

        return self.times_all

    def load_all_orientation(self, format="UCLA", exclude_str: str = "WebCam"):
        """Loads head orientation data from multiple videos in the UCLA miniscope
        software file format (folder = ""hh_mm_ss")

        :param format: str, either 'UCLA' (default) to use the UCLA miniscope folder format or a regex if you are
        using a different folder naming convention.
        :param exclude_str: exclude any folders containing this string from being loaded in.
        :return:
        """

        # Get recording folders, sorted by time of acquisition
        if format == "UCLA":
            search_regex = "**/[0-1][0-9]_[0-6][0-9]_[0-6][0-9]"
        self.rec_folders = sorted(self.basedir.glob(search_regex))

        # Exclude any folders containing the specified "exclude_str" parameter
        if exclude_str is not None:
            rec_folder2 = []
            for folder in self.rec_folders:
                if re.search(exclude_str, str(folder)) is None:
                    print("including folder " + str(folder))
                    rec_folder2.append(folder)
            self.rec_folders = rec_folder2

        # Loop through and load all timestamps, then concatenate
        orient_list = []
        for rec_folder in self.rec_folders:
            orient_data, _, _, _ = load_orientation(
                rec_folder, corrupted_videos="from_file"
            )
            orient_list.append(orient_data)

        self.orient_all = pd.concat(orient_list)


def get_recording_metadata(rec_folder: pathlib.Path):
    """Get and return relevant metadata from a recording specified in rec_folder"""

    assert isinstance(rec_folder, pathlib.Path)

    # Get video folder name from metaData.json file
    with open(rec_folder / "metaData.json", "rb") as f:
        rec_metadata = json.load(f)
    assert (
        len(rec_metadata["miniscopes"]) == 1
    ), "More/less than one miniscope detected in this recording"
    ms_folder_name = rec_metadata["miniscopes"][0].replace(" ", "_")

    # Auto detect folder where videos live
    assert (
        len(vid_folders := sorted(rec_folder.glob("*" + ms_folder_name))) == 1
    ), "Multiple Miniscope folders present, fix folders or code"
    vid_folder = vid_folders[0]

    # Get info about frame rate, gain, frames per file, etc.
    with open(str(vid_folder / "metaData.json"), "rb") as f:
        vid_metadata = json.load(f)

    return rec_metadata, vid_metadata, vid_folder


def load_timestamps(rec_folder, corrupted_videos=None):
    """Loads in timestamps corresponding to all videos in rec_folder.

    :param rec_folder: str or path
    :param corrupted_videos: (default) list of indices of corrupted files (e.g. [4, 6] would mean the 5th and 7th videos
    were corrupted and should be skipped - their timestamps will be omitted from the output.
    'from_file' will automatically grab any video indices provided as a csv file named 'corrupted_videos.csv'
    :return:
    """

    # Make Path object
    rec_folder = Path(rec_folder)

    # Grab metadata
    rec_metadata, vid_metadata, vid_folder = get_recording_metadata(rec_folder)

    # Derive start_time from rec_metadata
    rec_start = rec_metadata["recordingStartTime"]
    del rec_start["msecSinceEpoch"]
    rec_start["ms"] = rec_start["msec"]
    del rec_start["msec"]
    start_time = pd.to_datetime(pd.DataFrame(rec_start, index=["start_time"]))

    # Get timestamps, convert to datetimes
    times = pd.read_csv(str(vid_folder / "timeStamps.csv"))
    timestamps = start_time["start_time"] + pd.to_timedelta(
        times["Time Stamp (ms)"], unit="ms"
    )
    times["Timestamps"] = timestamps
    nframes = times.shape[0]

    # Last, remove any rows for corrupted videos
    good_frame_bool = np.ones(nframes, dtype=bool)
    if corrupted_videos == "from_file":
        f_per_file = vid_metadata["framesPerFile"]
        assert (
            len(corrupt_file := sorted(vid_folder.glob("corrupted_videos.csv"))) <= 1
        ), ("More than one " "corrupted_videos.csv" " file found")
        if len(corrupt_file) == 1:
            corrupt_array = (
                pd.read_csv(corrupt_file[0], header=None).to_numpy().reshape(-1)
            )
        else:
            corrupt_array = []
    elif isinstance(corrupted_videos, (list, np.ndarray)):
        corrupt_array = corrupted_videos
    elif corrupted_videos is None:
        corrupt_array = []

    for corrupt_vid in corrupt_array:
        print(
            "Eliminating timestamps from corrupted video"
            + str(corrupt_vid)
            + " in "
            + str(rec_folder.parts[-1] + " folder.")
        )
        good_frame_bool[
            range(
                corrupt_vid * f_per_file,
                np.min((f_per_file * (corrupt_vid + 1), nframes)),
            )
        ] = 0

    times = times[good_frame_bool]
    print(str(sum(good_frame_bool)) + " total good frames found")

    return times, rec_metadata, vid_metadata, rec_start


def load_orientation(rec_folder, corrupted_videos=None):
    """Loads in head orientation data and corresponding timestamps.

    :param rec_folder: str or path
    :param corrupted_videos: (default) list of indices of corrupted files (e.g. [4, 6] would mean the 5th and 7th videos
    were corrupted and should be skipped - their timestamps will be omitted from the output.
    'from_file' will automatically grab any video indices provided as a csv file named 'corrupted_videos.csv'
    :return:
    """

    # Make Path object
    rec_folder = Path(rec_folder)

    # Grab metadata
    rec_metadata, vid_metadata, vid_folder = get_recording_metadata(rec_folder)

    # Derive start_time from rec_metadata
    rec_start = rec_metadata["recordingStartTime"]
    del rec_start["msecSinceEpoch"]
    rec_start["ms"] = rec_start["msec"]
    del rec_start["msec"]
    start_time = pd.to_datetime(pd.DataFrame(rec_start, index=["start_time"]))

    # Get timestamps, convert to datetimes
    orient_data = pd.read_csv(str(vid_folder / "headOrientation.csv"))
    timestamps = start_time["start_time"] + pd.to_timedelta(
        orient_data["Time Stamp (ms)"], unit="ms"
    )
    orient_data["Timestamps"] = timestamps
    nframes = orient_data.shape[0]

    # Calculate Euler angles and dump into orient_data dataframe
    roll_x, pitch_y, yaw_z = euler_from_quaternion(
        orient_data["qx"], orient_data["qy"], orient_data["qz"], orient_data["qw"]
    )
    orient_data["roll"] = roll_x
    orient_data["pitch"] = pitch_y
    orient_data["yaw"] = yaw_z

    # Get head orientation data, convert to euler angles

    return orient_data, rec_metadata, vid_metadata, rec_start


def move_files_to_combined_folder(
    parent_folder,
    re_pattern="**/My_V4_Miniscope/*.avi",
    prepend_rec_time=True,
    copy_during_prepend=False,
):
    """Move files from home folder to miniscope combined folder with specified
    regex pattern.
    :param: prepend_rec_time = True (default) will add the folder recording time to the front
    of each file"""

    parent_folder = Path(parent_folder)
    combined_folder = parent_folder / "Miniscope_combined"

    # Get files
    movie_files = sorted(parent_folder.glob(re_pattern))
    nfiles = len(movie_files)

    # Prepend recording time to each file before moving if not done already
    if prepend_rec_time:
        file_names = [file.parts[-1].split(".")[0] for file in movie_files]
        unique_file_names = nfiles == len(
            set(file_names)
        )  # Check if all files are unique
        if unique_file_names:
            print("files already have recording time appended - moving/copying as-is")
        else:
            print("pre-pending recording time to each file before moving/copying")
            folder_names = set(
                [file.parent for file in movie_files]
            )  # get unique folder names
            for folder in folder_names:
                prepend_time_from_folder_to_file(
                    folder, ext=re_pattern.split(".")[-1], copy=copy_during_prepend
                )

        # Finally, get updated file names
        movie_files = sorted(parent_folder.glob(re_pattern))

    success = 0
    for file in movie_files:
        try:
            new_name = combined_folder / file.name
            file.rename(new_name)
            success += 1
        except:
            print("Error moving " + file.name)

    print(f"Successfully moved {success} of {nfiles} movie files to combined folder")


def euler_from_quaternion(qx, qy, qz, qw):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)


    """

    m00 = 1.0 - 2.0 * qy * qy - 2.0 * qz * qz
    m01 = 2.0 * qx * qy + 2.0 * qz * qw
    m02 = 2.0 * qx * qz - 2.0 * qy * qw
    m10 = 2.0 * qx * qy - 2.0 * qz * qw
    m11 = 1 - 2.0 * qx * qx - 2.0 * qz * qz
    m12 = 2.0 * qy * qz + 2.0 * qx * qw
    m20 = 2.0 * qx * qz + 2.0 * qy * qw
    m21 = 2.0 * qy * qz - 2.0 * qx * qw
    m22 = 1.0 - 2.0 * qx * qx - 2.0 * qy * qy

    roll_x = np.arctan2(m12, m22)
    c2 = np.sqrt(m00 * m00 + m01 * m01)
    pitch_y = np.arctan2(-m02, c2)
    s1 = np.sin(roll_x)
    c1 = np.cos(roll_x)
    yaw_z = np.arctan2(s1 * m20 - c1 * m10, c1 * m11 - s1 * m21)

    return roll_x, pitch_y, yaw_z  # in radians


if __name__ == "__main__":
    parent_folder = "/data2/Trace_FC/Recording_Rats/Rose/2022_06_20_habituation1"
    move_files_to_combined_folder(parent_folder)
