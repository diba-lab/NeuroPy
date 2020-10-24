import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import linecache
from datetime import datetime, timedelta
from pathlib import Path
from parsePath import Recinfo


def posfromFBX(fileName):
    fileName = str(fileName)

    xpos, ypos, zpos = [], [], []
    with open(fileName) as f:
        next(f)
        for i, line in enumerate(f):

            m = "".join(line)

            if "KeyCount" in m:
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
                break
            else:
                next(f)

        pos1 = []
        for i, line in enumerate(f):
            # print(line)
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
                break
            else:
                next(f)

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

    tracking_sRate = 120  # position sample rate

    def __init__(self, basepath):
        """initiates position class

        Arguments:
            obj {class instance} -- should have the following attributes
                obj.sessinfo.files.position --> filename for storing the positions
        """
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        posfile = self._obj.files.position
        if os.path.exists(posfile):
            posInfo = self._load(posfile).item()
            self.x = posInfo["x"] / 4  # in seconds
            self.y = posInfo["y"] / 4  # in seconds
            self.t = posInfo["time"]  # in seconds
            self.datetime = posInfo["datetime"]  # in seconds
            self.speed = np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2) / (
                1 / self.tracking_sRate
            )

        else:
            "Position file does not exist....did not load _position.npy"

    def _load(self, posfile):
        return np.load(posfile, allow_pickle=True)

    def getPosition(self):
        sRate = self._obj.sampfreq  # .dat file sampling frequency
        basePath = Path(self._obj.basePath)
        metadata = self._obj.metadata

        nfiles = metadata.count()["StartTime"]

        # ------- collecting timepoints related to .dat file  --------
        data_time = []
        for i, file_time in enumerate(metadata["StartTime"][:nfiles]):
            tbegin = datetime.strptime(file_time, "%Y-%m-%d_%H-%M-%S")
            nframes = metadata["nFrames"][i]
            duration = pd.Timedelta(nframes / sRate, unit="sec")
            tend = tbegin + duration
            trange = pd.date_range(
                start=tbegin,
                end=tend,
                periods=int(duration.seconds * self.tracking_sRate),
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

            fileinfo = pd.read_csv(file, header=None, nrows=1)
            # required values are in column 11 and 13 of .csv file
            tbegin = datetime.strptime(fileinfo.iloc[0][11], "%Y-%m-%d %I.%M.%S.%f %p")
            nframes = fileinfo.iloc[0][13]
            duration = pd.Timedelta(nframes / self.tracking_sRate, unit="sec")
            tend = tbegin + duration
            trange = pd.date_range(start=tbegin, end=tend, periods=nframes)

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
        xdata = np.interp(data_time, postime, posx)
        ydata = np.interp(data_time, postime, posy)
        zdata = np.interp(data_time, postime, posz)
        time = np.linspace(0, len(xdata) / 120, len(xdata))

        posVar = {
            "x": xdata,
            "y": zdata,  # as in optitrack the z coordinates gives the y information
            "time": time,
            "datetime": data_time,
            "trackingsRate": self.tracking_sRate,
        }

        np.save(self._obj.files.position, posVar)

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

