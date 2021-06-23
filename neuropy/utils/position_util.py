import csv
import linecache
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from ..utils.mathutil import contiguous_regions


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

    return int(export_FR)


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

    # ---- auto select which columns have rigid body position -------
    pos_columns = rigidbody_columns = None
    with open(fileName, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        line_count = 0
        for row in reader:
            if "Rigid Body" in row or "RigidBody" in row:
                rigidbody_columns = np.where(
                    np.bitwise_or(
                        np.array(row) == "Rigid Body", np.array(row) == "RigidBody"
                    )
                )[0]

            if "Position" in row:
                pos_columns = np.where(np.array(row) == "Position")[0]
                break
            line_count += 1

    rigidbody_pos_columns = np.intersect1d(pos_columns, rigidbody_columns)

    # second column is time so append that
    read_columns = np.append(1, rigidbody_pos_columns)

    posdata = pd.read_csv(
        fileName,
        skiprows=line_count + 1,
        skip_blank_lines=False,
        usecols=read_columns,
    )

    t = np.asarray(posdata.iloc[:, 0])
    x0 = np.asarray(posdata.iloc[:, 1])
    y0 = np.asarray(posdata.iloc[:, 2])
    z0 = np.asarray(posdata.iloc[:, 3])

    # if end frames are nan drop those
    # last_nan_region = contiguous_regions(np.isnan(x0))[-1]
    if np.isnan(x0[-1]):
        t, x0, y0, z0 = t[:-1], x0[:-1], y0[:-1], z0[:-1]

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
    """Get optitrack start time"""
    fileName = str(fileName)

    with open(fileName, newline="") as f:
        reader = csv.reader(f)
        row1 = next(reader)
        StartTime = [
            row1[i + 1] for i in range(len(row1)) if row1[i] == "Capture Start Time"
        ]
    # print(StartTime)
    tbegin = datetime.strptime(StartTime[0], "%Y-%m-%d %I.%M.%S.%f %p")
    return tbegin


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
