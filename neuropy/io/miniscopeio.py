import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json


class MiniscopeIO:
    def __init__(self) -> None:
        pass

    def load_timestamps(self, rec_folder, corrupted_videos=None):
        timestamps, rec_metadata, vid_metadata, rec_start = load_timestamps(
            rec_folder, corrupted_videos=corrupted_videos
        )

    def load_multiple_timestamps(self, basedir):
        """Loads multiple timestamps from"""
        pass


def load_timestamps(rec_folder, corrupted_videos=None):
    """Loads in timestamps corresponding to all videos in rec_folder.

    :param rec_folder: str or path
    :param corrupted_videos: list of indices of corrupted files (e.g. [4, 6] would mean the 5th and 7th videos
    were corrupted and should be skipped - their timestamps will be ommitted from the output.
    :return:
    """

    rec_folder = Path(rec_folder)
    assert (
        len(vid_folders := sorted(rec_folder.glob("*Miniscope"))) == 1
    ), "Multiple Miniscope folders present, fix folders or code"
    vid_folder = vid_folders[0]

    # Get info about frame rate, gain, frames per file, etc.
    with open(str(vid_folder / "metaData.json"), "rb") as f:
        vid_metadata = json.load(f)

    # Get more specific info (e.g. Rat Name, baseDirectory, and recording start time parameters
    with open(str(rec_folder / "metaData.json"), "rb") as f:
        rec_metadata = json.load(f)

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
    times["Timestamps"] = times
    nframes = times.shape[0]

    # Last, remove any rows for corrupted videos
    good_frame_bool = np.ones(nframes, dtype=bool)
    f_per_file = vid_metadata["framesPerFile"]
    for corrupt_vid in corrupted_videos:
        good_frame_bool[
            range(
                corrupt_vid * f_per_file,
                np.min((f_per_file * (corrupt_vid + 1), nframes)),
            )
        ] = 0

    return timestamps, rec_metadata, vid_metadata, rec_start
