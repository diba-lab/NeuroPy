from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from ..parsePath import Recinfo
from ..core import Position, Epoch
from ..utils import position_util
from .. import plotting


class SessPosition(Position):
    def __init__(self, basepath: Recinfo) -> None:

        self._obj = basepath

        filePrefix = self._obj.files.filePrefix
        filename = filePrefix.with_suffix(".position.npy")
        super().__init__(filename=filename)
        self.load()

    @property
    def video_start_time(self):
        posFolder = Path(self._obj.basePath) / "position"
        posfiles = np.asarray(sorted(posFolder.glob("*.csv")))
        return position_util.getStartTime(posfiles[0])

    def from_optitrack(self, method="from_metadata", scale=1.0):
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
        metadata = self._obj.loadmetadata()

        nfiles = metadata.count()["StartTime"]

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
            times_all = position_util.timestamps_from_oe(
                basePath, data_type="continuous"
            )
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
            # try:  # First try to load everything from CSV directly
            #     x, y, z, trelative = posfromCSV(file)
            #     # Make sure you arent't just importing the header, if so engage except
            #     assert len(x) > 0
            #     trange = tbegin + pd.to_timedelta(trelative, unit="s")
            #     postime.extend(trange)

            # except (
            #     FileNotFoundError,
            #     KeyError,
            #     pd.errors.ParserError,
            # ):  # Get data from FBX file if not in CSV

            #     # Get time ranges for position files
            #     nframes_pos = getnframes(file)
            #     duration = pd.Timedelta(nframes_pos / tracking_sRate, unit="sec")
            #     tend = tbegin + duration
            #     trange = pd.date_range(start=tbegin, end=tend, periods=nframes_pos)

            #     x, y, z = posfromFBX(file.with_suffix(".fbx"))

            #     postime.extend(trange)

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

    def to_neuroscope(self):

        # neuroscope only displays positive values so translating the coordinates
        x = self.x + abs(min(self.x))
        y = self.y + abs(min(self.y))
        print(max(x))
        print(max(y))

        filename = self._obj.files.filePrefix.with_suffix(".pos")
        with filename.open("w") as f:
            for xpos, ypos in zip(x, y):
                f.write(f"{xpos} {ypos}\n")

    def plot(self):
        plotting.plot_position(self.x, self.y)
