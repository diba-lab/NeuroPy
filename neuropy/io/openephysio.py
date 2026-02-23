import ast

import numpy as np
from glob import glob
import os
import re
from pathlib import Path
from xml.etree import ElementTree
import pandas as pd
from datetime import datetime, timezone
from dateutil import tz
import matplotlib.pyplot as plt
from packaging.version import Version


class OESyncIO:
    """Class to synchronize external data sources to Open-Ephys recordings."""
    def __init__(self, basepath) -> None:
        pass

    def rough_align_w_datetime(self):
        """Perform a rough first-pass alignment of recordings using datetimes"""
        pass

    def align_w_TTL(self):
        """Align TTLs in OE with external timestamps.

        :return pd.DataFrame OR np.array with OE timestamps matching external TTL events"""
        pass


def get_settings_file(experiment_or_recording_folder):
    if "Record Node" in Path(experiment_or_recording_folder).stem:
        node_folder = Path(experiment_or_recording_folder)
        fname = "settings.xml"
    elif "experiment" in Path(experiment_or_recording_folder).stem:
        node_folder = Path(experiment_or_recording_folder).parent
        fname = get_settings_filename(experiment_or_recording_folder)
    elif "recording" in Path(experiment_or_recording_folder).stem:
        node_folder = Path(experiment_or_recording_folder).parents[1]
        fname = get_settings_filename(experiment_or_recording_folder)
    else:
        assert False, "must enter any one of '.../Record Node ###', '.../experiment#', or '.../experiment#/recording#' folder"

    settings_path = sorted(node_folder.glob(f"**/{fname}"))
    return settings_path[0]


def get_recording_start(recording_folder, from_zone="UTC", to_zone="America/Detroit"):
    """Get start time of recording."""

    # Get version number
    settings_path = get_settings_file(recording_folder)
    oe_version = get_version_number(settings_path)

    #
    if Version(oe_version) < Version("0.6"):
        # Try to get microsecond start time from PhoTimestamp plugin.
        try:
            start_time = get_us_start(settings_path, from_zone=from_zone, to_zone=to_zone)
            precision = "us"

        # Otherwise get time from settings file (lower precision)
        except KeyError:
            try:
                experiment_meta = XML2Dict(settings_path)  # Get meta data
                start_time = pd.Timestamp(experiment_meta["INFO"]["DATE"]).tz_localize(
                    to_zone
                )  # get start time from meta-data
                precision = "s"

            # Last ditch get from directory structure and spit out warning.
            except FileNotFoundError:
                print(
                    "WARNING:"
                    + str(settings_path)
                    + " not found. Inferring start time from directory structure. PLEASE CHECK!"
                )
                # Find folder with timestamps
                m = re.search(
                    "[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}",
                    str(settings_path),
                )
                start_time = pd.to_datetime(
                    m.group(0), format="%Y-%m-%d_%H-%M-%S"
                ).tz_localize(to_zone)
                precision = "s"

        # Now add in lag from settings file creation (first time you hit record after hitting play)
        # Get SR and sync frame info - for syncing to time of day.
        SR, sync_frame_exp = parse_sync_file(recording_folder.parent / "recording1/sync_messages.txt")

        # sync_frame info between recordings within the same experiment folder
        _, sync_frame_rec = parse_sync_file(recording_folder / "sync_messages.txt")

        # Add in start frame lag
        start_time_rec = start_time + pd.Timedelta((sync_frame_rec - sync_frame_exp) / SR, unit="sec")

        # Try to improve precision by getting start time directly from sync_messages.txt modification time
        if precision == "s":
            # Sometimes sync_messages.txt gets modified and created anew at the beginning of the next recording.
            # So loop through both `sync_messages.txt` and 'structure.oebin` to check modification time
            for file_name in ["sync_messages.txt", "structure.oebin"]:
                try:
                    start_time_ms_precision = get_ms_start_from_creation_time(recording_folder / file_name)

                    if np.abs((start_time_ms_precision - start_time_rec).total_seconds()) < 1.9999:
                        start_time_rec = start_time_ms_precision
                        precision = "ms"
                        break
                except FileNotFoundError:
                    continue

        if precision == "s":  # Output warning if ms or better precision can't be obtained
            print(
                "Can't find PhoTimestamp plugin in settings.xml file and sync_messages.txt modification time is off"
                " from approximate start time by more than 1 second - start time will only have second precision")

    elif Version(oe_version) >= Version("0.6"):
        start_time_rec = get_ms_start(recording_folder / "sync_messages.txt")
        precision = "ms"

    return start_time_rec, precision


def get_us_start(settings_file: str, from_zone="UTC", to_zone="America/Detroit"):
    """Get microsecond time second precision start time from Pho/Timestamp plugin"""

    experiment_meta = XML2Dict(settings_file)

    start_us = experiment_meta["SIGNALCHAIN"]["PROCESSOR"][
        "Utilities/PhoStartTimestamp Processor"
    ]["PhoStartTimestampPlugin"]["RecordingStartTimestamp"]["startTime"]
    dt_start_utc = datetime.strptime(start_us[:-1], "%Y-%m-%d_%H:%M:%S.%f").replace(
        tzinfo=tz.gettz(from_zone)
    )
    to_zone = tz.gettz(to_zone)

    return dt_start_utc.astimezone(to_zone)


def get_ms_start_from_creation_time(sync_messages_file, to_zone="America/Detroit"):
    """Try to get recording start time from file modification time - only works if original file metadata is preserved,
    and will spit out the wrong time if the file was copied by apps like Dropbox which write new files instead of
    copying properly"""

    return (pd.Timestamp(datetime.fromtimestamp(os.path.getmtime(sync_messages_file))).tz_localize(to_zone))


def get_ms_start(sync_file, to_zone="America/Detroit"):
    """
    Parameters
    ----------
    sync_file: Path to sync_messages.txt file of interest
    to_zone: Set to America/Detroit

    Returns
    -------
    rec_start: Start time of recording down to millisecond precision in America/Detroit Timezone Datetimes.

    """
    # Read in file
    # sync_lines = open(sync_file).readlines()
    with open(sync_file, 'r') as sfile:
        sync_lines = sfile.readlines()

        # Read in OE.Version-only 0.6+ seems to write in a current millis into sync_messages.txt.
        oe_version = get_version_number(Path(sync_file).parents[2] / "settings.xml")

        try:
            #
            if Version(oe_version) > Version("0.6"):
                # Extract milliseconds from the first line
                ms_match = re.search(r"Software Time .*: (\d+)", sync_lines[0])
                ms = int(ms_match.group(1)) if ms_match else None

                # Convert milliseconds elapsed to UTC datetime
                utc_datetime = datetime.fromtimestamp(ms / 1000, timezone.utc)

                # Get Recording Start in America/Detroit Timezone
                rec_start = utc_datetime.astimezone(tz.gettz(to_zone))
                return rec_start

        except Exception as e:
            print(f"Error parsing sync file: {e}")
            return None


def get_dat_timestamps(
    basepath: str or Path,
    sync: bool = False,
    start_end_only=False,
    local_time="America/Detroit",
    print_start_time_to_screen=True,
):
    """
    Gets timestamps for each frame in your dat file(s) in a given directory.

    IMPORTANT: in the event your .dat file has less frames than you timestamps.npy file,
    you MUST create a "dropped_end_frames.txt" file with the # of missing frames in the same folder
    to properly account for this offset.
    :param basepath: str, path to parent directory, holding your 'experiment' folder(s).
    :param sync: True = use 'synchronized_timestamps.npy' file, default = False
    :param start_end_only: True = only grab start and end timestamps
    :return:
    """
    basepath = Path(basepath)

    timestamp_files = get_timestamp_files(basepath, type="continuous", sync=sync)

    timestamps = []
    nrec, start_end_str, nframe_dat = [], [], []  # for start_end=True only
    nframe_end = -1

    # Backwards/forwards compatibility code
    # set_file = get_settings_filename(timestamp_files[0])  # get settings file name
    # set_folder = get_set_folder(timestamp_files[0])
    # oe_version = get_version_number(set_folder / set_file)
    # ts_units = "frames"
    # if oe_version >= "0.6":
    #     print("OE version >= 0.6 detected, check timestamps. Could be off by factor = Sampling Rate")
    #     ts_units = "seconds"  # timestamps.npy in 0.6.7 (and presumably all 0.6) is in seconds, NOT sample # (that's in sample_numbers.npy)

    for idf, file in enumerate(timestamp_files):

        rec_folder = file.parents[2]  # Get recording file
        start_time_rec, _ = get_recording_start(rec_folder, to_zone=local_time)
        # Get SR and sync frame info - for syncing to time of day.
        SR, sync_frame_rec = parse_sync_file(file.parents[2] / "sync_messages.txt")
        # # sync_frame info between recordings within the same experiment folder
        # _, sync_frame_rec = parse_sync_file(file.parents[2] / "sync_messages.txt")
        #
        # start_time_rec = start_time + pd.Timedelta((sync_frame_rec - sync_frame_exp) / SR, unit="sec")

        if print_start_time_to_screen:
            print("start time = " + str(start_time_rec))
        stamps = np.load(file)  # load in timestamps

        # Remove any dropped end frames.
        if (file.parent / "dropped_end_frames.txt").exists():
            with open((file.parent / "dropped_end_frames.txt"), "rb") as fp:
                nfile = fp.readlines(0)
                pattern = re.compile("[0-9]+")
                ndropped = int(pattern.search(str(nfile[0])).group(0))

            print(f"Dropping last {ndropped} frames per dropped_end_frames.txt file")
            stamps = stamps[0:-ndropped]
        if start_end_only:  # grab start and end timestamps only
            # Add up start and end frame numbers
            nframe_dat.extend([nframe_end + 1, nframe_end + len(stamps)])
            nframe_end = nframe_dat[-1]
            stamps = stamps[[0, -1]]
        timestamps.append(
            (
                start_time_rec + pd.to_timedelta((stamps - sync_frame_rec) / SR, unit="sec")
            ).to_frame(index=False)
        )  # Add in absolute timestamps, keep index of acquisition system

        if start_end_only:
            nrec.extend([idf, idf])
            start_end_str.extend(["start", "stop"])

    if not start_end_only:
        return pd.concat(timestamps)
    else:
        return pd.DataFrame(
            {
                "Recording": nrec,
                "Datetime": np.concatenate(timestamps).reshape(-1),
                "Condition": start_end_str,
                "nframe_dat": nframe_dat,
            }
        )


def create_sync_df(
    basepath: str or Path, sync: bool = False, sr_dat=30000, sr_eeg=1250
):
    """Create a dataframe with start and stop times of all recordings in a parent folder.
    nframe_dat and nframe_eeg assumes the files have been concatenated into one file.
    """

    start_stop_df = get_dat_timestamps(basepath, sync, start_end_only=True)

    # Calculate eeg frames
    nframe_eeg = (start_stop_df["nframe_dat"] / (sr_dat / sr_eeg)).values.astype(int)

    # Augment repeated frame numbers by one
    repeat_ind = np.where(np.diff(nframe_eeg) == 0)[0] + 1
    nframe_eeg[repeat_ind] += 1

    # Combine into one dataframe after calculating combined dat/eeg file times
    sync_df = pd.concat(
        (
            start_stop_df,
            pd.DataFrame(
                {
                    "dat_time": start_stop_df["nframe_dat"] / sr_dat,
                    "nframe_eeg": nframe_eeg,
                    "eeg_time": nframe_eeg / sr_eeg,
                }
            ),
        ),
        axis=1,
    )

    return sync_df


def get_timestamp_files(
    basepath: str or Path, type: str in ["continuous", "TTL"], sync: bool = False
):
    """
    Identify all timestamp files of a certain type.
    IMPORTANT NOTE: Due to the original OE file naming scheme, this function grabs files with 'timestamps' in units of
    frames (sample_number), NOT actual time.
    :param basepath: str of Path object of folder containing timestamp file(s)
    :param type: 'continuous' or 'TTL'
    :param sync: False(default) for 'timestamps.npy', True for 'synchronized_timestamps.npy'
    :return: list of all files in that directory matching the inputs
    """
    basepath = Path(basepath)

    # Backwards compatibility fix
    settings_file = sorted(Path(str(basepath).split("/experiment")[0]).glob("**/*settings.xml"))[0]
    oe_version = get_version_number(settings_file)
    ts_file_str = "**/*timestamps.npy"
    if Version(oe_version) >= Version("0.6"):
        # print("OE version >= 0.6 detected, check timestamps. Could be off by factor = Sampling Rate")
        # ts_units = "seconds"  # timestamps.npy in 0.6.7 (and presumably all 0.6) is in seconds, NOT sample # (tha
        ts_file_str = "**/*sample_numbers.npy"
    timestamps_list = sorted(basepath.glob(ts_file_str))

    assert len(timestamps_list) > 0, "No timestamps.npy files found, check if files exist and if appropriate inputs are being used"
    continuous_bool = ["continuous" in str(file_name) for file_name in timestamps_list]
    TTL_bool = ["TTL" in str(file_name) for file_name in timestamps_list]
    sync_bool = ["synchronized" in str(file_name) for file_name in timestamps_list]
    no_sync_bool = np.bitwise_not(sync_bool)

    if type == "continuous" and not sync:
        file_inds = np.where(np.bitwise_and(continuous_bool, no_sync_bool))[0]
    elif type == "continuous" and sync:
        file_inds = np.where(np.bitwise_and(continuous_bool, sync_bool))[0]
    elif type == "TTL" and not sync:
        file_inds = np.where(np.bitwise_and(TTL_bool, no_sync_bool))[0]
    elif type == "TTL" and sync:
        file_inds = np.where(np.bitwise_and(TTL_bool, sync_bool))[0]

    return [timestamps_list[ind] for ind in file_inds]


def get_lfp_timestamps(dat_times_or_folder, SRdat=30000, SRlfp=1250, **kwargs):
    """
    Gets all timestamps corresponding to a downsampled lfp or eeg file
    :param dat_times_or_folder: str, path to parent directory, holding your 'experiment' folder(s).
    OR pandas dataframe of timestamps from .dat file.
    :param SRdat: sample rate for .dat file
    :param SRlfp: sample rate for .lfp file
    :param **kwargs: inputs to get_dat_timestamps
    :return:
    """

    if isinstance(dat_times_or_folder, (str, Path)):
        dat_times = get_dat_timestamps(dat_times_or_folder, **kwargs)
    elif isinstance(dat_times_or_folder, (pd.DataFrame, pd.Series)):
        dat_times = dat_times_or_folder

    assert (
        np.round(SRdat / SRlfp) == SRdat / SRlfp
    ), "SRdat file must be an integer multiple of SRlfp "
    return dat_times.iloc[slice(0, None, int(SRdat / SRlfp))]


def load_all_ttl_events(
    basepath: str or Path, sanity_check_channel: int or None = None, **kwargs
):
    """Loads TTL events from digital input port on an OpenEphys box or Intan Recording Controller in BINARY format.
    Assumes you have left the directory structure intact! Flexible - can load from just one recording or all recordings.
    Combines all events into one dataframe with datetimes.

    :param TTLpath: folder where TTL files live
    :param sanity_check_channel: int or None (default), if not None specifies TTL channel to plot timestamps of
    :param kwargs: accepts all kwargs to load_ttl_events
    """
    basepath = Path(basepath)
    TTLpaths = sorted(basepath.glob("**/TTL*"))  # get all TTL folders
    # Grab corresponding continuous data folders
    exppaths = [file.parents[3] for file in TTLpaths]

    # Get sync data
    start_end_df = get_dat_timestamps(basepath, start_end_only=True, print_start_time_to_screen=False)

    # Concatenate everything together into one list
    events_all, nframes_dat = [], []
    for idr, (TTLfolder, expfolder) in enumerate(zip(TTLpaths, exppaths)):
        rec_start_df = start_end_df[(start_end_df.Recording == idr) & (start_end_df.Condition == 'start')]
        nframes_dat.append(rec_start_df.nframe_dat.values[0])
        events = load_ttl_events(TTLfolder, **kwargs)
        events["Recording"] = idr
        events_all.append(events)

        # This shouldn't be necessary for grabbing event timestamps.
        # nframes_dat.append(get_dat_timestamps(expfolder, sync=sync))

    # Now loop through and make everything into a datetime in case you are forced to use system times to synchronize everything later
    times_list = []
    for ide, (events, start_frame_dat) in enumerate(zip(events_all, nframes_dat)):
        event_dt_df = events_to_datetime(events, start_frame_dat)
        event_dt_df["Recording"] = ide
        times_list.append(event_dt_df)

    ttl_df = pd.concat(times_list)

    # Plot sanity check
    if sanity_check_channel is not None:
        assert isinstance(sanity_check_channel, int)
        ts_plot = ttl_df[ttl_df["channel_states"].abs() == sanity_check_channel]
        ttt = (ts_plot["datetimes"] - ts_plot["datetimes"].iloc[0]).values
        _, ax = plt.subplots()
        ax.plot(ttt)
        ax.set_xlabel(f"TTL{sanity_check_channel} #")
        ax.set_ylabel("Time elapsed from first TTL event (sec)")
        ax.set_title("Sanity check - should be monotonically increasing!")
    return ttl_df


def load_ttl_events(TTLfolder, zero_timestamps=True, event_names="", sync_info=True):
    """Loads TTL events for one recording folder and spits out a dictionary.
    IMPORTANT NOTE: Due to the original OE file naming scheme, this function grabs files with 'timestamps' in units of
    frames (sample_number), NOT actual time.

    :param TTLfolder: folder where your TTLevents live, recorded in BINARY format.
    :param zero_timestamps: True (default) = subtract start time in sync_messages.txt. This will align everything
    to the first frame in your .dat file.
    :param event_names: can pass a dictionary to keep track of what each event means, e.g.
    event_names = {1: 'optitrack_start', 2: 'lick'} would tell you channel1 = input from optitrack and
    channel2 = animal lick port activations
    :param sync_info: True (default), grabs sync related info if TTlfolder is in openephys directory structure
    :return: channel_states, channels, full_words, and timestamps in ndarrays
    """
    TTLfolder = Path(TTLfolder)

    # Backwards compatibility fix
    settings_file = sorted(TTLfolder.parents[4].glob("**/*settings.xml"))[0]
    oe_version = get_version_number(settings_file)
    varnames_load = ["channel_states", "channels", "full_words", "timestamps"]
    keynames_use = ["channel_states", "channels", "full_words", "timestamps"]
    if Version(oe_version) >= Version("0.6"):
        varnames_load = ["states", "timestamps", "full_words", "sample_numbers"]
        keynames_use = ["channel_states", "time_in_sec", "full_words", "timestamps"]

    # Load event times and states into a dict
    events = dict()
    for keyname, varname in zip(keynames_use, varnames_load):
        events[keyname] = np.load(TTLfolder / (varname + ".npy"))

    # Get sync info
    if sync_info:

        # Get SR and sync frame info - this gets you absolute time from start of recording! Use for syncing to time of day.
        sync_file_exp = (
            TTLfolder.parents[3] / "recording1/sync_messages.txt"
        )

        # Gets you time from start of dat file - use for syncing to frame number in a combined .dat file.
        sync_file_rec = TTLfolder.parents[2] / "sync_messages.txt"

        SR, exp_start = parse_sync_file(sync_file_exp)
        _, record_start = parse_sync_file(sync_file_rec)
        events["SR"] = SR
        events["recording_start_frame"] = record_start
        events["experiment_start_frame"] = exp_start

        # Zero timestamps
        if zero_timestamps:
            events["timestamps"] = events["timestamps"] - record_start
            # events["timestamps_from_exp_start"] =

        # # Grab start time from .xml file and keep it with events just in case
        # settings_file = TTLfolder.parents[4] / get_settings_filename(TTLfolder)
        # missing_set_file = False
        # try:
        #     exp_start_time = pd.to_datetime(
        #         XML2Dict(settings_file)["INFO"]["DATE"]
        #     )
        #     # events["start_time"] = pd.to_datetime(
        #     #     XML2Dict(settings_file)["INFO"]["DATE"]
        #     # )
        #
        # except FileNotFoundError:
        #     print("Settings file: " + str(settings_file) + " NOT FOUND")
        #
        #     # Find start time using filename
        #     p = re.compile(
        #         "[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-2]+[0-9]+-[0-6]+[0-9]+-[0-6]+[0-9]+"
        #     )
        #
        #     exp_start_time = pd.to_datetime(
        #         p.search(str(settings_file)).group(0), format="%Y-%m-%d_%H-%M-%S"
        #     )
        #     missing_set_file = True
        #     # events["start_time"] = pd.to_datetime(
        #     #     p.search(str(settings_file)).group(0), format="%Y-%m-%d_%H-%M-%S"
        #     # )

        # events["start_time"] = (exp_start_time +
        #                         pd.Timedelta((record_start - exp_start) / SR, unit="sec"))

        # # Print to screen to double-check!
        # if missing_set_file:
        #     print(
        #         str(events["start_time"])
        #         + " loaded from folder structure, be sure to double check!"
        #     )

        events["start_time"], _ = get_recording_start(TTLfolder.parents[2])

    # Last add in event_names dict
    events["event_names"] = event_names

    return events


def events_to_datetime(events, start_frame_dat=None):
    """Parses out channel_states and timestamps and calculates absolute datetimes for all events
    If start_frame_dat is specified, also outputs the sample number from each + start_frame_dat, useful
    for getting the correct frame number of an event in a combined dat file"""

    # First grab relevant keys from full events dictionary
    sub_dict = {key: events[key] for key in ["channel_states", "timestamps"]}
    sub_dict["datetimes"] = events["start_time"] + pd.to_timedelta(
        events["timestamps"] / [events["SR"]], unit="sec"
    )
    if start_frame_dat is not None:
        sub_dict["sample_number"] = sub_dict["timestamps"] + start_frame_dat

    # Now dump any event names into the appropriate rows
    if events["event_names"] is not None:
        event_names = np.empty_like(events["timestamps"], dtype=np.dtype(("U", 10)))
        for key in events["event_names"]:
            # Allocate appropriate name to all timestamps for a channel
            event_names[events["channel_states"] == key] = events["event_names"][key]
        sub_dict["event_name"] = event_names

    return pd.DataFrame.from_dict(sub_dict)


def recording_events_to_combined_time(
    event_df: pd.DataFrame,
    sync_df: pd.DataFrame,
    time_out="eeg_time",
    event_ts_key: str = "datetimes",
    sync_ts_key: str = "Datetime",
    event_align_key: str = "datetimes",
    sync_align_key: str = "Datetime",
    sr: int = 30000,
):

    """Infers appropriate start frame/time of each event if multiple recordings/experiments in the same folder
    have been combined
    :param event_df: dataframe with datetimes of each event
    :param sync_df: output of function create_sync_df
    :param time_out: eeg_time or dat_time
    :param event_ts_key: key to use to access datetimes in event_df
    :param sync_ts_key: key to use to access datetimes in sync_df
    :param event_align_key: key in event_df to align by
    :param sync_align_key: key in sync_df to align by"""
    # Calc and check that each event occurs in the same recording.
    try:
        nrec_start = [
            sync_df["Recording"][np.max(np.nonzero((start > sync_df[sync_ts_key]).values))]
            for start in event_df[event_ts_key]
        ]
        nrec_stop = [
            sync_df["Recording"][np.min(np.nonzero((start < sync_df[sync_ts_key]).values))]
            for start in event_df[event_ts_key]
        ]
    except ValueError:
        print("Event time falls either in different recordings or outside of all recording times. Check!")

    # Loop through each recording and calculate CS time in combined dat/eeg file
    if nrec_start == nrec_stop:
        event_time_comb = []
        for nrec, event_time in zip(nrec_start, event_df[event_align_key]):
            # Get correct start time of recording in the desired output time (eeg/dat) and timestamp
            rec_start_time = sync_df[
                (sync_df["Recording"] == nrec) & (sync_df["Condition"] == "start")
            ][time_out].values[0]
            rec_start_timestamp = sync_df[
                (sync_df["Recording"] == nrec) & (sync_df["Condition"] == "start")
            ][sync_align_key].iloc[0]

            if event_align_key == 'datetimes': #default
                event_dt = (event_time - rec_start_timestamp).total_seconds()

            elif event_align_key == 'sample_number': 
            #align using dataframes instead of datetime. still spits out relative seconds from start, 
            # but the time dif is calculated using df. Use if you have low datetime resolution in 
            # your ephys start time.
                df_dt = event_time  - rec_start_timestamp #in dataframes
                event_dt = df_dt/sr

            event_time_comb.append(event_dt + rec_start_time)
        event_time_comb = np.array(event_time_comb)

    else:
        good_bool = [start == stop for start, stop in zip(nrec_start, nrec_stop)]
        good_events = np.where(good_bool)[0]
        bad_events = np.where(~np.array(good_bool))[0]

        print(
            f"Event(s) # {bad_events + 1} occurs in between recordings and has(have) been left out"
        )
        # print(
        #     f"Recording start and end numbers do not all match. starts = {nrec_start}, ends = {nrec_stop}."
        # )
        # event_time_comb = np.nan

        event_time_comb = recording_events_to_combined_time(
            event_df.iloc[good_events], sync_df, time_out, event_ts_key, sync_ts_key
        )

    return event_time_comb


def get_version_number(settings_path):
    """Get OE version number"""
    settings_path = Path(settings_path)
    assert re.search("settings_?[0-9]?[0-9]?.xml",
                     settings_path.name) is not None, \
        "settings file is not of the format settings.xml or settings_#.xml"
    # assert settings_path.name == "settings.xml"

    settings_dict = XML2Dict(settings_path)

    return settings_dict["INFO"]["VERSION"]


def parse_sync_file(sync_file):
    """Grab synchronization info for a given session
    :param sync_file: path to 'sync_messages.txt' file in recording folder tree for that recording.
    :return: sync_frame: int, sync frame # when you hit the record button relative to hitting the play button
             SR: int, sampling rate in Hz. Subtract from TTL timestamps to get dat frame #.
    """

    # Read in file
    sync_lines = open(sync_file).readlines()

    oe_version = get_version_number(Path(sync_file).parents[2] / "settings.xml")

    try:
        # Grab sampling rate and sync time based on file structure
        if oe_version < "0.6":
            SR = int(
                sync_lines[1][
                    re.search("@", sync_lines[1])
                    .span()[1] : re.search("Hz", sync_lines[1])
                    .span()[0]
                ]
            )
            sync_frame = int(
                sync_lines[1][
                    re.search("start time: ", sync_lines[1])
                    .span()[1] : re.search("@[0-9]*Hz", sync_lines[1])
                    .span()[0]
                ]
            )
        else:
            SR = int(
                sync_lines[1][
                re.search("@ ", sync_lines[1])
                .span()[1]: re.search(" Hz", sync_lines[1])
                .span()[0]
                ]
            )
            sync_frame = int(
                sync_lines[1][
                re.search(": ", sync_lines[1])
                .span()[1]: re.search("\n", sync_lines[1])
                .span()[0]
                ]
            )

    except IndexError:  # Fill in from elsewhere if sync_messages missing info
        parent_dir = Path(sync_file).parent
        timestamp_files = sorted(parent_dir.glob("**/continuous/**/timestamps.npy"))
        assert len(timestamp_files) == 1, "Too many timestamps.npy files"
        sync_frame = np.load(timestamp_files[0])[0]

        structure_file = parent_dir / "structure.oebin"
        with open(structure_file) as f:
            data = f.read()
        structure = ast.literal_eval(data)
        SR = structure["continuous"][0]["sample_rate"]

    return SR, sync_frame


def get_set_folder(child_dir):
    """Gets the folder where your settings.xml file and experiment folders should live."""
    child_dir = Path(child_dir)
    expfolder_id = np.where(
        [
            str(child_dir.parents[id]).find("experiment") > -1
            for id in range(len(child_dir.parts) - 1)
        ]
    )[0].max()

    return child_dir.parents[expfolder_id + 1]


def get_settings_filename(child_dir):
    """Infers the settings file name from a child directory, e.g. the continuous or event recording folder

    :param child_dir: any directory below the top-level directory. Must include the "experiment#' directory!
    :return:
    """

    child_dir = Path(child_dir)
    expfolder = child_dir.parts[
        np.where(["experiment" in folder for folder in child_dir.parts])[0][0]
    ]

    if expfolder[-1] == "1":
        return "settings.xml"
    else:
        return "settings_" + expfolder[10:] + ".xml"


def LoadTTLEvents_full(
    Folder, Processor=None, Experiment=None, Recording=None, TTLport=1, mode="r+"
):
    """Load TTL in events recorded in binary format. Copied from https://github.com/open-ephys/analysis-tools
    Python3.Binary module for loading binary files and adjusted for TTL events. Keeps track of processor and other
    metadata, but probably too onerous for daily use"""
    Files = sorted(glob(Folder + "/**/TTL_" + str(TTLport) + "/*.npy", recursive=True))
    # InfoFiles = sorted(glob(Folder + '/*/*/structure.oebin'))

    event_data, timing_data = {}, {}
    print("Loading events from recording on TTL", TTLport, "...")
    for F, File in enumerate(Files):
        try:
            Exp, Rec, _, Proc, _, npy_file = File.split("/")[-6:]
            sync_file = open(
                os.path.join(
                    File.split("/" + Exp + "/" + Rec)[0] + "/" + Exp + "/" + Rec,
                    "sync_messages.txt",
                )
            ).readlines()
        except ValueError:  # for windows machines use the correct delimiter
            Exp, Rec, _, Proc, _, npy_file = File.split("\\")[-6:]
            sync_file = open(
                os.path.join(
                    File.split("\\" + Exp + "\\" + Rec)[0] + "\\" + Exp + "\\" + Rec,
                    "sync_messages.txt",
                )
            ).readlines()

        Exp = str(int(Exp[10:]) - 1)
        Rec = str(int(Rec[9:]) - 1)
        Proc = Proc.split(".")[-2].split("-")[-1]
        if "_" in Proc:
            Proc = Proc.split("_")[0]

        # Info = literal_eval(open(InfoFiles[F]).read())
        # ProcIndex = [Info['continuous'].index(_) for _ in Info['continuous']
        #              if str(_['recorded_processor_id']) == Proc][0]

        if Proc not in event_data.keys():
            event_data[Proc], timing_data[Proc] = {}, {}
        if Exp not in event_data[Proc]:
            event_data[Proc][Exp], timing_data[Proc][Exp] = {}, {}
        if Rec not in event_data[Proc][Exp]:
            event_data[Proc][Exp][Rec], timing_data[Proc][Exp][Rec] = {}, {}

        timing_data[Proc][Exp][Rec]["Rate"] = sync_file[1][
            re.search("@", sync_file[1])
            .span()[1] : re.search("Hz", sync_file[1])
            .span()[0]
        ]
        timing_data[Proc][Exp][Rec]["start_time"] = sync_file[1][
            re.search("start time: ", sync_file[1])
            .span()[1] : re.search("@[0-9]*Hz", sync_file[1])
            .span()[0]
        ]

        if Experiment:
            if int(Exp) != Experiment - 1:
                continue

        if Recording:
            if int(Rec) != Recording - 1:
                continue

        if Processor:
            if Proc != Processor:
                continue

        event_data[Proc][Exp][Rec][npy_file[:-4]] = np.load(File)

    return event_data, timing_data


"""
Created on 20170704 21:15:19

@author: Thawann Malfatti

Loads info from the settings.xml file.

Examples:
    File = '/Path/To/Experiment/settings.xml

    # To get all info the xml file can provide:
    AllInfo = SettingsXML.XML2Dict(File)

    # AllInfo will be a dictionary following the same structure of the XML file.

    # To get the sampling rate used in recording:
    Rate = SettingsXML.GetSamplingRate(File)

    # To get info only about channels recorded:
    RecChs = SettingsXML.GetRecChs(File)[0]

    # To get also the processor names:
    RecChs, PluginNames = SettingsXML.GetRecChs(File)

    # RecChs will be a dictionary:
    #
    # RecChs
    #     ProcessorNodeId
    #         ChIndex
    #             'name'
    #             'number'
    #             'gain'
    #         'PluginName'

"""


def FindRecProcs(Ch, Proc, RecChs):
    ChNo = Ch["number"]
    Rec = Proc["CHANNEL"][ChNo]["SELECTIONSTATE"]["record"]

    if Rec == "1":
        if Proc["NodeId"] not in RecChs:
            RecChs[Proc["NodeId"]] = {}
        RecChs[Proc["NodeId"]][ChNo] = Ch

    return RecChs


def Root2Dict(El):
    Dict = {}
    if list(El):
        for SubEl in El:
            if SubEl.keys():
                if SubEl.get("name"):
                    if SubEl.tag not in Dict:
                        Dict[SubEl.tag] = {}
                    Dict[SubEl.tag][SubEl.get("name")] = Root2Dict(SubEl)

                    Dict[SubEl.tag][SubEl.get("name")].update(
                        {K: SubEl.get(K) for K in SubEl.keys() if K != "name"}
                    )

                else:
                    Dict[SubEl.tag] = Root2Dict(SubEl)
                    Dict[SubEl.tag].update(
                        {K: SubEl.get(K) for K in SubEl.keys() if K != "name"}
                    )

            else:
                if SubEl.tag not in Dict:
                    Dict[SubEl.tag] = Root2Dict(SubEl)
                else:
                    No = len([k for k in Dict if SubEl.tag in k])
                    Dict[SubEl.tag + "_" + str(No + 1)] = Root2Dict(SubEl)
    else:
        if El.items():
            return dict(El.items())
        else:
            return El.text

    return Dict


def XML2Dict(File):
    Tree = ElementTree.parse(File)
    Root = Tree.getroot()
    Info = Root2Dict(Root)

    return Info


def GetSamplingRate(File):
    Info = XML2Dict(File)
    Error = "Cannot parse sample rate. Check your settings.xml file at SIGNALCHAIN>PROCESSOR>Sources/Rhythm FPGA."
    SignalChains = [_ for _ in Info.keys() if "SIGNALCHAIN" in _]

    try:
        for SignalChain in SignalChains:
            if "Sources/Rhythm FPGA" in Info[SignalChain]["PROCESSOR"].keys():
                if (
                    "SampleRateString"
                    in Info[SignalChain]["PROCESSOR"]["Sources/Rhythm FPGA"]["EDITOR"]
                ):
                    Rate = Info[SignalChain]["PROCESSOR"]["Sources/Rhythm FPGA"][
                        "EDITOR"
                    ]["SampleRateString"]
                    Rate = float(Rate.split(" ")[0]) * 1000
                elif (
                    Info[SignalChain]["PROCESSOR"]["Sources/Rhythm FPGA"]["EDITOR"][
                        "SampleRate"
                    ]
                    == "17"
                ):
                    Rate = 30000
                elif (
                    Info[SignalChain]["PROCESSOR"]["Sources/Rhythm FPGA"]["EDITOR"][
                        "SampleRate"
                    ]
                    == "16"
                ):
                    Rate = 25000
                else:
                    Rate = None
            else:
                Rate = None

        if not Rate:
            print(Error)
            return None
        else:
            return Rate

    except Exception as Ex:
        print(Ex)
        print(Error)
        return None


def GetRecChs(File):
    Info = XML2Dict(File)
    RecChs = {}
    ProcNames = {}

    if len([k for k in Info if "SIGNALCHAIN" in k]) > 1:
        for S in [k for k in Info if "SIGNALCHAIN_" in k]:
            for P, Proc in Info[S]["PROCESSOR"].items():
                Info["SIGNALCHAIN"]["PROCESSOR"][P + "_" + S[-1]] = Proc
            del Info[S]
    #     print('There are more than one signal chain in file. )
    #     Ind = input(')

    for P, Proc in Info["SIGNALCHAIN"]["PROCESSOR"].items():
        if "isSource" in Proc:
            if Proc["isSource"] == "1":
                SourceProc = P[:]
        else:
            if Proc["name"].split("/")[0] == "Sources":
                SourceProc = P[:]

        if "CHANNEL_INFO" in Proc and Proc["CHANNEL_INFO"]:
            for Ch in Proc["CHANNEL_INFO"]["CHANNEL"].values():
                RecChs = FindRecProcs(Ch, Proc, RecChs)

        elif "CHANNEL" in Proc:
            for Ch in Proc["CHANNEL"].values():
                RecChs = FindRecProcs(Ch, Proc, RecChs)

        else:
            continue

        if "pluginName" in Proc:
            ProcNames[Proc["NodeId"]] = Proc["pluginName"]
        else:
            ProcNames[Proc["NodeId"]] = Proc["name"]

    if Info["SIGNALCHAIN"]["PROCESSOR"][SourceProc]["CHANNEL_INFO"]:
        SourceProc = Info["SIGNALCHAIN"]["PROCESSOR"][SourceProc]["CHANNEL_INFO"][
            "CHANNEL"
        ]
    else:
        SourceProc = Info["SIGNALCHAIN"]["PROCESSOR"][SourceProc]["CHANNEL"]

    for P, Proc in RecChs.items():
        for C, Ch in Proc.items():
            if "gain" not in Ch:
                RecChs[P][C].update(
                    [c for c in SourceProc.values() if c["number"] == C][0]
                )

    return (RecChs, ProcNames)


if __name__ == "__main__":

    # Code to check `get_recording_start` function for all possible recording types
    # OE 0.6.7
    # start_time, prec = get_recording_start(Path('/data3/Anisomycin/Recording_Rats/Creampuff/2024_07_17_Anisomycin/2_POST/2024-07-17_10-42-12/Record Node 104/experiment1/recording1'))

    # OE 0.5.3 with PhoTimestamp
    # start_time, prec = get_recording_start(Path('/data2/Trace_FC/Recording_Rats/Finn/2022_01_20_training/1_tone_habituation/2022-01-20_12-28-41/Record Node 104/experiment1/recording1'))

    start_time, prec = get_recording_start(Path("/data2/Trace_FC/Recording_Rats/Django/2023_03_08_training/2_training/2023-03-08_12-15-15/Record Node 103/experiment1/recording1"))
    print(f"start_time={start_time}, precision={prec}")
