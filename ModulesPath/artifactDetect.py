from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
from parsePath import Recinfo


class findartifact:
    """Detects noisy periods uding downsampled data

    Attributes
    ------------
    time: array,
        time periods which are noisy

    Methods
    ------------
    removefrom:
        removes noisy timestamps
    """

    def __init__(self, obj):

        if isinstance(obj, Recinfo):
            self._obj = obj
        else:
            self._obj = Recinfo(obj)

        self.time = None

        # ----- defining file names ---------
        filePrefix = self._obj.files.filePrefix

        @dataclass
        class files:
            dead: str = filePrefix.with_suffix(".dead")
            artifact: str = filePrefix.with_suffix(".artifact.npy")

        self.files = files()

        # ----- loading files --------
        if self.files.artifact.is_file():
            self._load()
        elif Path(self.files.dead).is_file():
            with self.files.dead.open("r") as f:
                noisy = []
                for line in f:
                    epc = line.split(" ")
                    epc = [float(_) for _ in epc]
                    noisy.append(epc)
                noisy = np.asarray(noisy) / 1000
                self.time = noisy  # in seconds

    def _load(self):
        data = np.load(self.files.artifact, allow_pickle=True).item()
        self.threshold = data["threshold"]
        self.chan = data["channel"]
        self.time = data["time"]

    def removefrom(self, lfp, timepoints):
        """Deletes detected artifacts from the 'lfp'

        Args:
            lfp ([array]): lfp signal
            timepoints ([array]): seconds, corresponding time stamps of the lfp

        Returns:
            [array]: artifact deleted lfp
        """
        # --- if a period is given, then convert it to timepoints------
        if len(timepoints) == 2:
            timepoints = np.linspace(timepoints[0], timepoints[1], len(lfp))

        if self.time is not None:
            dead_indx = np.concatenate(
                [
                    np.where((timepoints > start) & (timepoints < end))[0]
                    for (start, end) in self.time
                ]
            )
            lfp = np.delete(lfp, dead_indx, axis=-1)
        return lfp

    def usingZscore(self, chans=None, thresh=5):
        """
        calculating periods to exclude for analysis using simple z-score measure
        """
        if chans is None:
            chans = np.random.choice(self._obj.recinfo.goodchans, 4)

        eegSrate = self._obj.lfpSrate
        lfp = self._obj.recinfo.geteeg(chans=chans)
        lfp = np.median(lfp, axis=0)
        zsc = np.abs(stat.zscore(lfp))

        artifact_binary = np.where(zsc > thresh, 1, 0)
        artifact_binary = np.concatenate(([0], artifact_binary, [0]))
        artifact_diff = np.diff(artifact_binary)
        artifact_start = np.where(artifact_diff == 1)[0]
        artifact_end = np.where(artifact_diff == -1)[0]

        firstPass = np.vstack((artifact_start - 10, artifact_end + 2)).T

        minInterArtifactDist = 5 * eegSrate
        secondPass = []
        artifact = firstPass[0]
        for i in range(1, len(artifact_start)):
            if firstPass[i, 0] - artifact[1] < minInterArtifactDist:
                # Merging artifacts
                artifact = [artifact[0], firstPass[i, 1]]
            else:
                secondPass.append(artifact)
                artifact = firstPass[i]

        secondPass.append(artifact)

        # --- converting to required time units for various puposes ------
        artifact_ms = np.asarray(secondPass) / (eegSrate / 1000)  # ms
        artifact_s = np.asarray(secondPass) / eegSrate  # seconds

        # --- writing to file for visualizing in neuroscope and spyking circus ----
        file_neuroscope = self._obj.recinfo.files.filePrefix.with_suffix(".evt.art")
        circus_file = self._obj.recinfo.files.filePrefix.with_suffix(".dead")
        with file_neuroscope.open("w") as a, circus_file.open("w") as c:
            for beg, stop in artifact_ms:
                a.write(f"{beg} start\n{stop} end\n")
                c.write(f"{beg} {stop}\n")

        data = {"channel": chans, "time": artifact_s, "threshold": thresh}
        np.save(self.files.artifact, data)

        self._load()
        return zsc

    def plot(self, chan=None):

        if chan is None:
            chan = np.random.choice(self._obj.recinfo.goodchans)

        lfp = self._obj.utils.geteeg(chans=chan)
        zsc = np.abs(stat.zscore(lfp))
        artifact = self.time * self._obj.recinfo.lfpSrate

        _, ax = plt.subplots(1, 1)
        ax.plot(zsc, "gray")
        ax.axhline(self.threshold, color="#37474F", ls="--")
        ax.plot(
            artifact[:, 0], self.threshold * np.ones(artifact.shape[0]), "r|", ms="10"
        )
        ax.plot(
            artifact[:, 1], self.threshold * np.ones(artifact.shape[0]), "k|", ms="10"
        )
        ax.set_xlabel("frames")
        ax.set_ylabel("Absolute zscore")

        ax.legend(["zsc-lfp", "threshold", "art-start", "art-end"])

    def createCleanDat(self):

        # for shankID in range(3, 9):
        #     print(shankID)

        #     DatFileOG = (
        #         folderPath
        #         + "Shank"
        #         + str(shankID)
        #         + "/RatJDay2_Shank"
        #         + str(shankID)
        #         + ".dat"
        #     )
        #     DestFolder = (
        #         folderPath
        #         + "Shank"
        #         + str(shankID)
        #         + "/RatJDay2_Shank"
        #         + str(shankID)
        #         + "_denoised.dat"
        #     )

        #     nChans = 8
        #     SampFreq = 30000

        #     b = []
        #     for i in range(len(Data_start)):

        #         start_time = Data_start[i]
        #         end_time = Data_end[i]

        #         duration = end_time - start_time  # in seconds
        #         b.append(
        #             np.memmap(
        #                 DatFileOG,
        #                 dtype="int16",
        #                 mode="r",
        #                 offset=2 * nChans * int(SampFreq * start_time),
        #                 shape=(nChans * int(SampFreq * duration)),
        #             )
        #         )

        #     c = np.memmap(
        #         DestFolder, dtype="int16", mode="w+", shape=sum([len(x) for x in b])
        #     )

        #     del c
        #     d = np.memmap(
        #         DestFolder, dtype="int16", mode="r+", shape=sum([len(x) for x in b])
        #     )

        #     sizeb = [0]
        #     sizeb.extend([len(x) for x in b])
        #     sizeb = np.cumsum(sizeb)

        #     for i in range(len(b)):

        #         d[sizeb[i] : sizeb[i + 1]] = b[i]
        #         # d[len(b[i]) : len(b1) + len(b2)] = b2
        #     del d
        #     del b
        pass
