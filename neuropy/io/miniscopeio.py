import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle


class MiniscopeIO:
    def __init__(self, basedir) -> None:
        self.basedir = Path(basedir)
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
            "YrA": None,
            "curated_neurons": None,
        }

        # Load in YrA (raw traces) if saved
        try:
            self.minian["YrA"] = np.load(self.minian_folder / "YrA.npy")
        except FileNotFoundError:
            print('No raw traces found. "YrA" variable not loaded')

        # try to load in curated neurons if you have done so in the NRK modified minian pipeline
        try:
            with open(self.minian_folder / "curated_neurons.pkl", "rb") as f:
                self.minian["curated_neurons"] = pickle.load(f)
        except:
            print("No curated neurons file found")

    def load_all_timestamps(self, format="UCLA"):
        """Loads multiple timestamps from multiple videos in the UCLA miniscope software file format
        (folder = ""hh_mm_ss")

        :param basedir: two levels up from all your
        :param format: str, either 'UCLA' (default) to use the UCLA miniscope folder format or a regex if you are
        using a different folder naming convention.
        :return:
        """

        # Get recording folders, sorted by time of acquisition
        if format == "UCLA":
            search_regex = "**/[0-1][0-9]_[0-6][0-9]_[0-6][0-9]"
        self.rec_folders = sorted(self.basedir.glob(search_regex))

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
            good_frames_bool = np.load(self.minian_folder / "good_frames.npy")
            print(
                "Keeping "
                + str(good_frames_bool.sum())
                + ' good frames found in "good_frames.npy" file'
            )
            self.times_all = self.times_all[good_frames_bool]
        except:
            print("no " + str(self.minian_folder / "good_frames.npy") + " file found")
            pass

        return self.times_all


def load_timestamps(rec_folder, corrupted_videos=None):
    """Loads in timestamps corresponding to all videos in rec_folder.

    :param rec_folder: str or path
    :param corrupted_videos: (default) list of indices of corrupted files (e.g. [4, 6] would mean the 5th and 7th videos
    were corrupted and should be skipped - their timestamps will be omitted from the output.
    'from_file' will automatically grab any video indices provided as a csv file named 'corrupted_videos.csv'
    :return:
    """

    rec_folder = Path(rec_folder)

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
    if corrupted_videos == "from_file":
        good_frame_bool = np.ones(nframes, dtype=bool)
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


def move_files_to_combined_folder(parent_folder):
    """Move files from home folder to miniscope combined folder"""
    pass


if __name__ == "__main__":
    basedir = Path("/data/Working/Trace_FC/Recording_Rats/Rat698/2021_06_29_training")
    basedir2 = Path(
        "/data/Working/Trace_FC/Recording_Rats/Rat698/2021_06_30_recall/2_tone_recall"
    )
    msobj = MiniscopeIO(basedir)
    msobj.load_all_timestamps()
