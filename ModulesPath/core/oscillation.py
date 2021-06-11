import numpy as np
import pandas as pd
from scipy import stats
from ..utils import signal_process
from .epoch import Epoch


class Oscillation:
    def __init__(self, freq_band: tuple, fs=1250, **kwargs) -> None:
        super().__init__(**kwargs)
        self.freq_band = freq_band
        self.fs = fs

    def get_best_channels(self, lfps):
        """Channel which represent high spindle power during nrem across all channels"""
        avg_bandpower = np.zeros(len(lfps))
        for i, lfp in enumerate(lfps):
            filtered = signal_process.filter_sig.bandpass(
                lfp, lf=self.freq_band[0], hf=self.freq_band[1]
            )
            amplitude_envelope = np.abs(signal_process.hilbertfast(filtered))
            avg_bandpower[i] = np.mean(amplitude_envelope)

        descending_order = np.argsort(avg_bandpower)[::-1]
        return descending_order

    def detect(self, lfps, thresh, mindur, maxdur, mergedist, ignore_times=None):
        """[summary]

        Parameters
        ----------
        thresh : tuple, optional
            low and high threshold for detection
        mindur : float, optional
            minimum duration of epoch
        maxdur : float, optiona
        chans : list
            channels used for epoch detection, if None then chooses best chans
        """

        zscsignal = []
        lf, hf = self.freq_band
        lowthresh, highthresh = thresh
        for lfp in lfps:
            yf = signal_process.filter_sig.bandpass(lfp, lf=lf, hf=hf)
            zsc_chan = stats.zscore(np.abs(signal_process.hilbertfast(yf)))
            zscsignal.append(zsc_chan)

        zscsignal = np.asarray(zscsignal)

        # ---------setting noisy periods zero --------
        if ignore_times is not None:
            assert ignore_times.ndim == 2, "ignore_times should be 2 dimensional array"
            noisy_frames = np.concatenate(
                [
                    (np.arange(start, stop) * self.fs).astype(int)
                    for (start, stop) in ignore_times
                ]
            )

            zscsignal[:, noisy_frames] = 0

        # ------hilbert transform --> binarize by > than lowthreshold
        maxPower = np.max(zscsignal, axis=0)
        ThreshSignal = np.where(zscsignal > lowthresh, 1, 0).sum(axis=0)
        ThreshSignal = np.diff(np.where(ThreshSignal > 0, 1, 0))
        start = np.where(ThreshSignal == 1)[0]
        stop = np.where(ThreshSignal == -1)[0]

        # --- getting rid of incomplete epochs at begining or end ---------
        if start[0] > stop[0]:
            stop = stop[1:]
        if start[-1] > stop[-1]:
            start = start[:-1]

        firstPass = np.vstack((start, stop)).T
        print(f"{len(firstPass)} epochs detected initially")

        # --------merging close epochs------------
        min_inter_epoch_samples = mergedist * self.fs
        secondPass = []
        epoch = firstPass[0]
        for i in range(1, len(firstPass)):
            if firstPass[i, 0] - epoch[1] < min_inter_epoch_samples:
                epoch = [epoch[0], firstPass[i, 1]]
            else:
                secondPass.append(epoch)
                epoch = firstPass[i]
        secondPass.append(epoch)
        secondPass = np.asarray(secondPass)
        print(f"{len(secondPass)} epochs reamining after merging close ones")

        # ------delete epochs with less than threshold power--------
        thirdPass = []
        peakpower, peaktime = [], []

        for i in range(0, len(secondPass)):
            maxValue = max(maxPower[secondPass[i, 0] : secondPass[i, 1]])
            if maxValue > highthresh:
                thirdPass.append(secondPass[i])
                peakpower.append(maxValue)
                peaktime.append(
                    secondPass[i, 0]
                    + np.argmax(maxPower[secondPass[i, 0] : secondPass[i, 1]])
                )
        thirdPass = np.asarray(thirdPass)
        print(
            f"{len(thirdPass)} epochs reamining after deleting epochs with weaker power"
        )

        ripple_duration = np.diff(thirdPass, axis=1) / self.fs
        epochs = pd.DataFrame(
            {
                "start": thirdPass[:, 0],
                "stop": thirdPass[:, 1],
                "peakpower": peakpower,
                "peaktime": np.asarray(peaktime),
                "duration": ripple_duration.squeeze(),
            }
        )

        # ---------delete very short epochs--------
        epochs = epochs[epochs.duration >= mindur]
        print(f"{len(epochs)} epochs reamining after deleting short epochs")

        # ----- delete epochs with unrealistic high power
        # artifactRipples = np.where(peakpower > maxPeakPower)[0]
        # fourthPass = np.delete(thirdPass, artifactRipples, 0)
        # peakpower = np.delete(peakpower, artifactRipples)

        # ---------delete very long epochs---------
        epochs = epochs[epochs.duration <= maxdur]
        print(f"{len(epochs)} epochs reamining after deleting very long epochs")

        # ----- converting to all time stamps to seconds --------
        epochs[["start", "stop", "peakpower", "peaktime"]] /= self.fs  # seconds

        epochs = epochs.reset_index(drop=True)
        epochs["label"] = ""
        metadata = {
            "params": {
                "lowThres": lowthresh,
                "highThresh": highthresh,
                "freq_band": self.freq_band,
                "mindur": mindur,
                "maxdur": maxdur,
                "mergedist": mergedist,
            },
        }

        return epochs, metadata
