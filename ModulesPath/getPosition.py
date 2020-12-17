import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import linecache
from datetime import datetime, timedelta
from pathlib import Path
from parsePath import Recinfo
from mathutil import contiguous_regions
import xml.etree.ElementTree as ET
import re
from glob import glob
from dataclasses import dataclass


def getSampleRate(fileName):
    """Get Sample rate from csv header file - not set at 120Hz"""
    toprow = pd.read_csv(fileName, nrows=1, header=None)
    capture_FR = np.asarray(
        toprow[np.where(toprow == "Capture Frame Rate")[1][0] + 1][0], dtype=float
    )
    export_FR = np.asarray(
        toprow[np.where(toprow == "Export Frame Rate")[1][0] + 1][0], dtype=float
    )

    if capture_FR != export_FR:
        print("Careful! capture FR does NOT match export FR. Using export only.")

    return export_FR


def getnframes(fileName):
    """Get nframes from csv header file"""
    toprow = pd.read_csv(fileName, nrows=1, header=None)
    nframes_take = np.asarray(
        toprow[np.where(toprow == "Total Frames in Take")[1][0] + 1][0], dtype=float
    )
    nframes_export = np.asarray(
        toprow[np.where(toprow == "Total Exported Frames")[1][0] + 1][0], dtype=float
    )

    if nframes_take != nframes_export:
        print(
            "CAREFUL! # frames in take does not match # frames exported. Using # frames exported for analysis!"
        )

    return int(nframes_export)


def getunits(fileName):
    """determine if position data is in centimeters or meters"""
    toprow = pd.read_csv(fileName, nrows=1, header=None)
    units = toprow[np.where(toprow == "Length Units")[1][0] + 1][0]

    return units


def posfromCSV(fileName):
    """Import position data from OptiTrack CSV file"""
    posdata = pd.read_csv(fileName, header=[2, 4, 5])
    x0 = np.asarray(posdata["RigidBody", "Position", "X"])
    y0 = np.asarray(posdata["RigidBody", "Position", "Y"])
    z0 = np.asarray(posdata["RigidBody", "Position", "Z"])
    t = np.asarray(
        posdata.loc[:, ["Time (Seconds)" in _ for _ in posdata.keys()]]
    ).reshape(-1)

    xfill, yfill, zfill = interp_missing_pos(x0, y0, z0, t)

    # Now convert to centimeters
    units = getunits(fileName)
    if units.lower() == "centimeters":
        x, y, z = xfill, yfill, zfill
    elif units.lower() == "meters":
        x, y, z = xfill * 100, yfill * 100, zfill * 100
    else:
        raise Exception(
            "position data needs to be exported in either centimeters or meters"
        )

    return x, y, z, t


def interp_missing_pos(x, y, z, t):
    """Interpolate missing data points"""
    xgood, ygood, zgood = x, y, z
    idnan = contiguous_regions(np.isnan(x))  # identify missing data points

    for ids in idnan:
        missing_ids = range(ids[0], ids[-1])
        bracket_ids = ids + [-1, 0]
        xgood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], x[bracket_ids])
        ygood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], y[bracket_ids])
        zgood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], z[bracket_ids])

    return xgood, ygood, zgood


def posfromFBX(fileName):
    fileName = str(fileName)

    xpos, ypos, zpos = [], [], []
    with open(fileName) as f:
        next(f)
        for i, line in enumerate(f):

            m = "".join(line)

            if "KeyCount" in m:
                # print("break at i = " + str(i))
                track_begin = i + 2
                line_frame = linecache.getline(fileName, i + 2).strip().split(" ")
                total_frames = int(line_frame[1]) - 1
                break

    with open(fileName) as f:
        for _ in range(track_begin):
            next(f)

        for i, line in enumerate(f):
            # print(line)
            if len(xpos) > total_frames:
                break

            elif i < 1:
                # print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                # print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            xpos.extend(pos1)

        for line in f:

            if "KeyCount" in line:
                # print("break at i = " + str(i))
                break
            # else:  # NRK note: this errors out unpredictably on my computer. Is it necessary? doesn't the for break
            # automatically move onto the next line of code once you reach a "KeyCount" in the line being read?
            #     next(f)

        pos1 = []
        for i, line in enumerate(f):
            # print(i)
            if len(ypos) > total_frames:
                break

            elif i < 1:
                # print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                # print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            ypos.extend(pos1)

        for line in f:

            if "KeyCount" in line:
                # print("break at i = " + str(i))
                break
            # else:
            #     next(f)
        pos1 = []

        for i, line in enumerate(f):
            # print(line)

            if len(zpos) > total_frames:
                break

            elif i < 1:
                # print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                # print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            # line = next(f)
            zpos.extend(pos1)

    xpos = [float(_) for _ in xpos]
    ypos = [float(_) for _ in ypos]
    zpos = [float(_) for _ in zpos]

    return np.asarray(xpos), np.asarray(ypos), np.asarray(zpos)


def getStartTime(fileName):
    fileName = str(fileName)

    with open(fileName, newline="") as f:
        reader = csv.reader(f)
        row1 = next(reader)
        StartTime = [
            row1[i + 1] for i in range(len(row1)) if row1[i] == "Capture Start Time"
        ]
    # print(StartTime)
    tbegin = datetime.strptime(StartTime[0], "%Y-%m-%d %H.%M.%S.%f %p")
    return tbegin


class ExtractPosition:
    def __init__(self, basepath):
        """initiates position class

        Arguments:
            obj {class instance} -- should have the following attributes
                :param: obj.sessinfo.files.position --> filename for storing the positions
                :param: tracking_sf: accounts for any mismatch between calibration wand size (125mm) and the size used in OptiTrack.
                Prior to late 2020, we used a 500mm wand length in Motive (Calibration Box: Wanding->OptiWand->Wand Length)
                by accident, so the default is 4.
        """
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ------- defining file names ---------
        filePrefix = self._obj.files.filePrefix

        @dataclass
        class files:
            position: str = Path(str(filePrefix) + "_position.npy")

        self.files = files()
        self._load()

    def _load(self):
        if (f := self.files.position).is_file():
            posInfo = np.load(f, allow_pickle=True).item()
            self.x = posInfo["x"]
            self.y = posInfo["y"]
            self.z = posInfo["z"]
            self.t = posInfo["time"]  # in seconds
            self.datetime = posInfo["datetime"]  # in seconds
            self.tracking_sRate = getSampleRate(
                sorted((self._obj.basePath / "position").glob("*.csv"))[0]
            )
            self.speed = np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2) / (
                1 / self.tracking_sRate
            )
            self.data = pd.DataFrame(
                {
                    "time": self.t[1:],
                    "x": self.x[1:],
                    "y": self.y[1:],
                    "z": self.z[1:],
                    "speed": self.speed,
                    "datetime": self.datetime[1:],
                }
            )

    def __getitem__(self, epochs):
        pass

    def getPosition(self, method="from_metadata", scale=1.0):
        """get position data from files. All position related files should be in 'position' folder within basepath

        Parameters
        ----------
        method : str, optional
            [description], by default "from_metadata"
        scale : float, optional
            scale the extracted coordinates, by default 1.0
        """
        sRate = self._obj.sampfreq  # .dat file sampling frequency
        lfpsRate = self._obj.lfpsRate  # .eeg file sampling frequency
        basePath = Path(self._obj.basePath)
        metadata = self._obj.loadmetadata()

        nfiles = metadata.count()["StartTime"]

        # ------- collecting timepoints related to .dat file  --------
        data_time = []
        # transfer start times from the settings*.xml file and nframes in .dat file to each row of the metadata file
        durations = []
        if method == "from_metadata":
            for i, file_time in enumerate(metadata["StartTime"][:nfiles]):
                tbegin = datetime.strptime(file_time, "%Y-%m-%d_%H-%M-%S")
                nframes = metadata["nFrames"][i]
                duration = pd.Timedelta(nframes / sRate, unit="sec")
                tend = tbegin + duration
                trange = pd.date_range(
                    start=tbegin,
                    end=tend,
                    periods=int(duration.total_seconds() * self.tracking_sRate),
                )
                data_time.extend(trange)

        # grab timestamps directly from timestamps.npy files. Assumes you have preserved the OE file structure.
        elif method == "from_files":
            times_all = timestamps_from_oe(basePath, data_type="continuous")
            for i, times in enumerate(times_all):
                tbegin, tend = times[0], times[-1]
                duration = tend - tbegin
                durations.append(duration)
                trange = pd.date_range(
                    start=tbegin,
                    end=tend,
                    periods=int(duration.total_seconds() * self.tracking_sRate),
                )
                data_time.extend(trange)
        data_time = pd.to_datetime(data_time)

        # ------- deleting intervals that were deleted from .dat file after concatenating
        ndeletedintervals = metadata.count()["deletedStart (minutes)"]
        for i in range(ndeletedintervals):
            tnoisy_begin = data_time[0] + pd.Timedelta(
                metadata["deletedStart (minutes)"][i], unit="m"
            )
            tnoisy_end = data_time[0] + pd.Timedelta(
                metadata["deletedEnd (minutes)"][i], unit="m"
            )

            del_index = np.where((data_time > tnoisy_begin) & (data_time < tnoisy_end))[
                0
            ]

            data_time = np.delete(data_time, del_index)
            # data_time = data_time.indexer_between_time(
            #     pd.Timestamp(tnoisy_end), pd.Timestamp(tnoisy_begin)
            # )

        # ------- collecting timepoints related to position tracking ------
        posFolder = basePath / "position"
        posfiles = np.asarray(sorted(posFolder.glob("*.csv")))
        posfilestimes = np.asarray(
            [
                datetime.strptime(file.stem, "Take %Y-%m-%d %I.%M.%S %p")
                for file in posfiles
            ]
        )
        filesort_ind = np.argsort(posfilestimes).astype(int)
        posfiles = posfiles[filesort_ind]

        postime, posx, posy, posz = [], [], [], []

        for file in posfiles:
            print(file)
            tbegin = getStartTime(file)
            try:  # First try to load everything from CSV directly
                x, y, z, trelative = posfromCSV(file)
                assert (
                    len(x) > 0
                )  # Make sure you aren't just importing the header, if so engage except below
                trange = tbegin + pd.to_timedelta(trelative, unit="s")
                postime.extend(trange)

            except (
                FileNotFoundError,
                KeyError,
                pd.errors.ParserError,
            ):  # Get data from FBX file if not in CSV

                # Get time ranges for position files
                nframes_pos = getnframes(file)
                duration = pd.Timedelta(nframes_pos / self.tracking_sRate, unit="sec")
                tend = tbegin + duration
                trange = pd.date_range(start=tbegin, end=tend, periods=nframes_pos)

                x, y, z = posfromFBX(file.with_suffix(".fbx"))

                postime.extend(trange)
            posx.extend(x)
            posy.extend(y)
            posz.extend(z)

        postime = pd.to_datetime(postime[: len(posx)])
        posx = np.asarray(posx)
        posy = np.asarray(posy)
        posz = np.asarray(posz)

        # -------- interpolating positions for recorded data ------------
        xdata = np.interp(data_time, postime, posx) / scale
        ydata = np.interp(data_time, postime, posy) / scale
        zdata = np.interp(data_time, postime, posz) / scale
        time = np.linspace(0, len(xdata) / self.tracking_sRate, len(xdata))
        posVar = {
            "x": xdata,
            "y": zdata,
            "z": ydata,  # keep this data in case you are interested in rearing activity
            "time": time,
            "datetime": data_time,
            "trackingsRate": self.tracking_sRate,
        }

        np.save(self._obj.files.position, posVar)

        # now load this immediately into existence
        self._load()

    def plot(self):

        plt.clf()
        plt.plot(self.x, self.y)

    def export2Neuroscope(self):

        # neuroscope only displays positive values so translating the coordinates
        x = self.x + abs(min(self.x))
        y = self.y + abs(min(self.y))
        print(max(x))
        print(max(y))

        filename = self._obj.files.filePrefix.with_suffix(".pos")
        with filename.open("w") as f:
            for xpos, ypos in zip(x, y):
                f.write(f"{xpos} {ypos}\n")


def timestamps_from_oe(rec_folder, data_type="continuous"):
    """Gets timestamps for all recordings/experiments in a given recording folder. Assumes you have recorded
    in flat binary format in OpenEphys and left the directory structure intact. continuous data by default,
    set data_type='events' for TTL timestamps"""
    if isinstance(rec_folder, Path):
        oefolder = rec_folder
    else:
        oefolder = Path(rec_folder)

    # Identify and sort timestamp and settings files in ascending order
    if data_type in ["continuous"]:
        time_files = np.asarray(
            sorted(
                oefolder.glob("**/experiment*/**/" + data_type + "/**/timestamps.npy")
            )
        )
    else:
        raise ValueError("data_type must be " "continuous" "")
    set_files = np.asarray(sorted(oefolder.glob("**/settings*.xml")))
    sync_files = np.asarray(sorted(oefolder.glob("**/sync_messages.txt")))

    # Loop through and establish timeframes for each file
    times_abs = []
    for time, set_, sync_file in zip(time_files, set_files, sync_files):
        # load data
        timedata = np.load(time)
        myroot = ET.parse(set_).getroot()
        setdict = {}
        for elem in myroot[0]:
            setdict[elem.tag] = elem.text
        # setdict = XML2Dict(set_)
        SRuse, sync_start = get_sync_info(sync_file)

        # Identify absolute start times of each file...
        tbegin = datetime.strptime(setdict["DATE"], "%d %b %Y %H:%M:%S")
        tstamps = tbegin + pd.to_timedelta((timedata - sync_start) / SRuse, unit="sec")
        if len(times_abs) > 0 and tstamps[0] < times_abs[-1][-1]:
            raise Exception("Timestamps out of order - check directory structure!")
        times_abs.append(tstamps)

    return times_abs


def get_sync_info(_sync_file):
    sync_file_read = open(_sync_file).readlines()
    SR = int(
        sync_file_read[1][
            re.search("@", sync_file_read[1])
            .span()[1] : re.search("Hz", sync_file_read[1])
            .span()[0]
        ]
    )
    sync_start = int(
        sync_file_read[1][
            re.search("start time: ", sync_file_read[1])
            .span()[1] : re.search("@[0-9]*Hz", sync_file_read[1])
            .span()[0]
        ]
    )
    return SR, sync_start


if __name__ == "__main__":
    position = ExtractPosition(
        r"C:\Users\Nat\Documents\UM\Working\Opto\Jackie671\Jackie_3well_day4\Jackie_UTRACK_combined"
    )
    position.getPosition(method="from_files")
pass
