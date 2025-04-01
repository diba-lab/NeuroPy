import csv
import linecache
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import numpy as np
from datetime import datetime
import pandas as pd
from ..utils import position_util, mathutil
from pathlib import Path
from ..core import Position
from copy import deepcopy


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


def getnframes_fbx(fileName):
    fileName = str(fileName)

    with open(fileName) as f:
        next(f)
        for i, line in enumerate(f):

            m = "".join(line)

            if "KeyCount" in m:
                # print("break at i = " + str(i))
                # line_frame = linecache.getline(fileName, i + 2).strip().split(" ")

                break

    return int(m.strip().split(":")[1].strip())


def getunits(fileName):
    """determine if position data is in centimeters or meters"""
    toprow = pd.read_csv(fileName, nrows=1, header=None)
    units = toprow[np.where(toprow == "Length Units")[1][0] + 1][0]

    return units


def posfromCSV(fileName,get_rotation=False):
    """Import position data from OptiTrack CSV file
    get_rotation: bool, default False, to also pull rigid body rotation cols from csv.
    """

    # ---- auto select which columns have rigid body position -------
    pos_columns = rot_columns = rigidbody_columns = None
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

            if get_rotation:
                if "Rotation" in row:
                    rot_columns = np.where(np.array(row) == "Rotation")[0]

            if "Position" in row and (not get_rotation or "Rotation" in row):
                break  # Break only when all needed values are found

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
    # todo: potential bug here where csv file has missing timestamps at end that go to NaN and then don't get accounted for.
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
    
    if get_rotation:
        rigidbody_rot_columns = np.intersect1d(rot_columns, rigidbody_columns)

        rotdata = pd.read_csv(
            fileName,
            skiprows=line_count + 1,
            skip_blank_lines=False,
            usecols=rigidbody_rot_columns,
        )

        x0_r = np.asarray(rotdata.iloc[:, 1])
        y0_r = np.asarray(rotdata.iloc[:, 2])
        z0_r = np.asarray(rotdata.iloc[:, 3])

        if np.isnan(x0_r[-1]):
            x0_r, y0_r, z0_r = x0_r[:-1], y0_r[:-1], z0_r[:-1]

        xfill_r, yfill_r, zfill_r = interp_missing_pos(x0_r, y0_r, z0_r, t)

        # Now convert to centimeters
        units = getunits(fileName)
        if units.lower() == "centimeters":
            x_r, y_r, z_r = xfill_r, yfill_r, zfill_r
        elif units.lower() == "meters":
            x_r, y_r, z_r = xfill_r * 100, yfill_r * 100, zfill_r * 100

    if get_rotation:
        return x, y, z, x_r,y_r,z_r,t
    else:
        return x, y, z, t


def interp_missing_pos(x, y, z, t):
    """Interpolate missing data points"""
    xgood, ygood, zgood = x, y, z
    idnan = mathutil.contiguous_regions(np.isnan(x))  # identify missing data points

    for ids in idnan:
        missing_ids = range(max(0, ids[0]), min(len(xgood), ids[-1]))
        bracket_ids = [max(0, ids[0] - 1), min(len(t) - 1, ids[-1] + 1)]

        #missing_ids = range(ids[0], ids[-1])
        #bracket_ids = ids + [-1, 0]
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


class OptitrackIO:
    def __init__(self, dirname, scale_factor=1.0,sampling_rate = None,get_rotation = False) -> None:
        self.dirname = dirname
        self.scale_factor = scale_factor
        self.datetime = None
        self.time = None
        self.sampling_rate = sampling_rate
        self._parse_folder(get_rotation)

    def _parse_folder(self,get_rotation):
        """get position data from files. All position related files should be in 'position' folder within basepath

        Parameters
        ----------
        method : str, optional
            method to grab file start times: "from_metadata" (default) grabs from metadata.csv file,
                                             "from_files" grabs from timestamps.npy files in open-ephys folders
        scale : float, optional
            scale the extracted coordinates, by default 1.0
        """
        
        # ----- grab only files that are position files -------
        all_files = sorted(file for file in self.dirname.rglob("Take*") if file.suffix in {".csv", ".fbx"})

        if self.sampling_rate == None:
            sampling_rate = getSampleRate(all_files[0])
        else:
            sampling_rate = self.sampling_rate


        #sampling_rate = getSampleRate(sorted((self.dirname).glob("*.csv"))[0])

        # ------- collecting timepoints related to position tracking ------
        #currently the next few lines will break with fbx. fix if you're gonna use
        posfiles = np.asarray(all_files)
        posfilestimes = np.asarray([getStartTime(file) for file in posfiles])
        filesort_ind = np.argsort(posfilestimes).astype(int)
        posfiles = posfiles[filesort_ind]

        postime, posx, posy, posz = [], [], [], []
        datetime_starts, datetime_stops, datetime_nframes = [], [], []
        x_rot,y_rot,z_rot = [], [], []

        for file in posfiles:
            print(file)
            tbegin = getStartTime(file)

            if file.with_suffix(".fbx").is_file():
                # Get time ranges for position files
                nframes_pos = getnframes_fbx(file.with_suffix(".fbx"))
                duration = pd.Timedelta(nframes_pos / sampling_rate, unit="sec")
                tend = tbegin + duration
                trange = pd.date_range(start=tbegin, end=tend, periods=nframes_pos)

                x, y, z = posfromFBX(file.with_suffix(".fbx"))
                assert len(x) == nframes_pos
                postime.extend(trange)

            else:
                output = posfromCSV(file,get_rotation)

                if get_rotation:
                    x,y,z,rx,ry,rz,trelative = output
                    x_rot.extend(rx)
                    y_rot.extend(ry)
                    z_rot.extend(rz)

                else:
                    x,y,z,trelative = output

                # Make sure you aren't just importing the header, if so engage except
                assert len(x) > 0
                nframes_pos = len(x)
                trange = tbegin + pd.to_timedelta(trelative, unit="s")
                postime.extend(trange)
                tend = trange[-1]

            datetime_starts.append(tbegin)
            datetime_stops.append(tend.to_pydatetime())  # Explicitly returns a datetime and not a timestamp
            datetime_nframes.append(nframes_pos)
            posx.extend(x)
            posy.extend(y)
            posz.extend(z)

        postime = pd.to_datetime(postime)
        posx = np.asarray(posx)
        posy = np.asarray(posy)
        posz = np.asarray(posz)

        assert len(postime) == len(posx)

        self.x = posx * self.scale_factor
        self.y = posy * self.scale_factor
        self.z = posz * self.scale_factor
        self.datetime_array = postime
        self.datetime_starts = datetime_starts
        self.datetime_stops = datetime_stops
        self.datetime_nframes = datetime_nframes
        self.sampling_rate = sampling_rate

        if get_rotation:

            x_rot = np.asarray(x_rot)
            y_rot = np.asarray(y_rot)
            z_rot = np.asarray(z_rot)
            self.x_rot = x_rot * self.scale_factor
            self.y_rot = y_rot * self.scale_factor
            self.z_rot = z_rot * self.scale_factor

    def get_position_at_datetimes(self, dt):

        x = np.interp(dt, self.datetime_array, self.x)
        y = np.interp(dt, self.datetime_array, self.y)
        z = np.interp(dt, self.datetime_array, self.z)

        return x, y, z

    def to_position(self, t_start=0):
        return Position(np.array([self.x, self.y, self.z]), t_start=t_start, sampling_rate=self.sampling_rate)

    def to_position_with_datetime(self, t_start=0):
        position_df = pd.DataFrame({
            "datetime": self.datetime_array,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        })
        position_df.attrs['metadata'] = {
            't_start': t_start,
            'sampling_rate': self.sampling_rate
        }
        return position_df

    def remove_negatives(self, ref_time):
        """
        Remove position data that is before the referenced time. Used when motive is started during a recording that
        precedes the relevant recording

        Parameters
        ----------
        ref_time : float, optional
            The reference time to subtract from the datetime_array, by default 0.
        """
        if not isinstance(ref_time, pd.Timestamp):
            ref_time = pd.Timestamp(ref_time)  # Ensure ref_time is a pandas Timestamp
            print("Reference time was not a timestamp, converting to timestamp")

        relative_times = (self.datetime_array - ref_time).total_seconds()
        mask = relative_times >= 0

        # Filter only per-frame data
        self.datetime_array = self.datetime_array[mask]
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.z = self.z[mask]

        self.datetime_starts = self.datetime_array[0]
        self.datetime_stops = self.datetime_array[-1]
        self.datetime_nframes = len(self.datetime_array)

    def old_stuff(self):
        """get position data from files. All position related files should be in 'position' folder within basepath

        Parameters
        ----------
        method : str, optional
            method to grab file start times: "from_metadata" (default) grabs from metadata.csv file,
                                             "from_files" grabs from timestamps.npy files in open-ephys folders
        scale : float, optional
            scale the extracted coordinates, by default 1.0
        """
        sRate = self._obj.sampfreq  # .dat file sampling frequency
        basePath = Path(self._obj.basePath)
        metadata = self._obj.loadmetadata

        # ------- collecting timepoints related to .dat file  --------
        data_time = []
        # transfer start times from the settings*.xml file and nframes in .dat file to each row of the metadata file
        tracking_sRate = position_util.getSampleRate(
            sorted((self._obj.basePath / "position").glob("*.csv"))[0]
        )
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
                    periods=int(duration.total_seconds() * tracking_sRate),
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
                    periods=int(duration.total_seconds() * tracking_sRate),
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
            tbegin = position_util.getStartTime(file)

            if file.with_suffix(".fbx").is_file():
                # Get time ranges for position files
                nframes_pos = position_util.getnframes(file)
                duration = pd.Timedelta(nframes_pos / tracking_sRate, unit="sec")
                tend = tbegin + duration
                trange = pd.date_range(start=tbegin, end=tend, periods=nframes_pos)

                x, y, z = position_util.posfromFBX(file.with_suffix(".fbx"))

                postime.extend(trange)

            else:  # First try to load everything from CSV directly
                x, y, z, trelative = position_util.posfromCSV(file)
                # Make sure you arent't just importing the header, if so engage except
                assert len(x) > 0
                trange = tbegin + pd.to_timedelta(trelative, unit="s")
                postime.extend(trange)

            posx.extend(x)
            posy.extend(y)
            posz.extend(z)
        postime = pd.to_datetime(postime[: len(posx)])
        posx = np.asarray(posx)
        posy = np.asarray(posy)
        posz = np.asarray(posz)

        # -------- interpolating positions for recorded data ------------
        xdata = np.interp(data_time, postime, posx) * scale
        ydata = np.interp(data_time, postime, posy) * scale
        zdata = np.interp(data_time, postime, posz) * scale

        time = np.linspace(0, len(xdata) / tracking_sRate, len(xdata))
        posVar = {
            "x": xdata,
            "y": zdata,
            "z": ydata,  # keep this data in case you are interested in rearing activity
            "time": time,
            "datetime": data_time,
            "trackingsRate": tracking_sRate,
        }

        self.x = xdata
        self.y = zdata
        self.z = ydata
        self.time = time
        self.tracking_srate = tracking_sRate

        self.save()
