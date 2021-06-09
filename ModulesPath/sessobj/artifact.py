from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
from ..parsePath import Recinfo
from ..core import Epoch


class Artifact(Epoch):
    """Detects noisy periods using downsampled data

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

        # ----- defining file names ---------
        filePrefix = self._obj.files.filePrefix
        filename = filePrefix.with_suffix(".artifact.npy")
        super().__init__(filename=filename)

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

    def getframes(self):
        eegSrate = self._obj.lfpSrate
        noisy_intervals = (self.time * eegSrate).astype(int) - 1  # zero indexing
        noisy_frames = np.concatenate(
            [np.arange(beg, end) for (beg, end) in noisy_intervals]
        )
        # correcting for any rounding error mostly an issue when artifacts are at end
        noisy_frames = noisy_frames[noisy_frames < self._obj.getNframesEEG]
        return noisy_frames

    def detect(self, chans=None, thresh=5, method="zscore"):
        """
        calculating periods to exclude for analysis using simple z-score measure
        """
        if chans is None:
            chans = np.random.choice(self._obj.goodchans, 4)

        eegSrate = self._obj.lfpSrate
        lfp = self._obj.geteeg(chans=chans)
        if isinstance(chans, list):
            lfp = np.asarray(lfp)
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

        artifact_s = np.asarray(secondPass) / eegSrate  # seconds

        epochs = pd.DataFrame(
            {"start": artifact_s[:, 0], "stop": artifact_s[:, 1], "label": ""}
        )
        metadata = {"channels": chans, "thresh": thresh}
        self.epochs, self.metadata = epochs, metadata
        self.save()

    # def export2neuroscope(self):
    #     # --- converting to required time units for export ------
    #     artifact_ms = self.time * 1000  # ms

    #     # --- writing to file for neuroscope and spyking circus ----
    #     file_neuroscope = self.files.neuroscope
    #     with file_neuroscope.open("w") as file:
    #         for beg, stop in artifact_ms:
    #             file.write(f"{beg} start\n{stop} end\n")

    def to_spyking_circus(self, ext=".dead"):
        # --- writing to file for neuroscope and spyking circus ----
        circus_file = self._obj.files.filePrefix.with_suffix(ext)
        with circus_file.open("w") as file:
            for epoch in self.epochs.itertuples():
                file.write(f"{epoch.start*1000} {epoch.stop*1000}\n")  # unit: ms

    def plot(self):

        chans = self.metadata["channels"]
        threshold = self.metadata["thresh"]
        lfp = self._obj.geteeg(chans=chans)
        if not isinstance(chans, int):
            lfp = np.asarray(lfp)
            lfp = np.median(lfp, axis=0)

        zsc = np.abs(stat.zscore(lfp))
        factor = 5
        downsample_srate = int(self._obj.lfpSrate / factor)
        zsc_downsampled = zsc[::factor]
        epochs = np.asarray(self.epochs[["start", "stop"]])
        artifact = epochs * downsample_srate

        _, ax = plt.subplots(1, 1)
        ax.plot(zsc_downsampled, "gray")
        ax.axhline(threshold, color="#37474F", ls="--")
        ax.plot(artifact[:, 0], threshold * np.ones(artifact.shape[0]), "r|", ms="10")
        ax.plot(artifact[:, 1], threshold * np.ones(artifact.shape[0]), "k|", ms="10")
        ax.set_xlabel("frames")
        ax.set_ylabel("Absolute zscore")

        ax.legend(["zsc-lfp", "threshold", "art-start", "art-end"])
