import numpy as np
import pandas as pd
from scipy import stats
from .. import signal_process


class Oscillation:
    def __init__(self) -> None:
        pass

    def detect(
        self,
        eeg,
        lowFreq,
        highFreq,
        lowthresh=1,
        highthresh=5,
        mindur=0.05,
        maxdur=0.450,
        mergedist=0.05,
        fs=1250,
        ignore_times=None,
    ):
        """[summary]

        Parameters
        ----------
        lowFreq : int, optional
            [description], by default 150
        highFreq : int, optional
            [description], by default 240
        chans : list
            channels used for ripple detection, if None then chooses best chans
        """

        zscsignal = []
        sharpWv_sig = np.zeros(eeg[0].shape[-1])
        for lfp in eeg:
            yf = signal_process.filter_sig.bandpass(lfp, lf=lowFreq, hf=highFreq)
            zsc_chan = stats.zscore(np.abs(signal_process.hilbertfast(yf)))
            zscsignal.append(zsc_chan)

            broadband = signal_process.filter_sig.bandpass(lfp, lf=2, hf=50)
            sharpWv_sig += stats.zscore(np.abs(signal_process.hilbertfast(broadband)))
        zscsignal = np.asarray(zscsignal)

        # ---------setting noisy period zero --------
        artifact = findartifact(self._obj)
        if artifact.time is not None:
            noisy_frames = artifact.getframes()
            zscsignal[:, noisy_frames] = 0

        # ------hilbert transform --> binarize by > than lowthreshold
        maxPower = np.max(zscsignal, axis=0)
        ThreshSignal = np.where(zscsignal > lowthresh, 1, 0).sum(axis=0)
        ThreshSignal = np.diff(np.where(ThreshSignal > 0, 1, 0))
        start = np.where(ThreshSignal == 1)[0]
        stop = np.where(ThreshSignal == -1)[0]

        # --- getting rid of incomplete ripples at begining or end ---------
        if start[0] > stop[0]:
            stop = stop[1:]
        if start[-1] > stop[-1]:
            start = start[:-1]

        firstPass = np.vstack((start, stop)).T
        print(f"{len(firstPass)} ripples detected initially")

        # --------merging close ripples------------
        minInterRippleSamples = mergedist * fs
        secondPass = []
        ripple = firstPass[0]
        for i in range(1, len(firstPass)):
            if firstPass[i, 0] - ripple[1] < minInterRippleSamples:
                ripple = [ripple[0], firstPass[i, 1]]
            else:
                secondPass.append(ripple)
                ripple = firstPass[i]

        secondPass.append(ripple)
        secondPass = np.asarray(secondPass)
        print(f"{len(secondPass)} ripples reamining after merging")

        # ------delete ripples with less than threshold power--------
        thirdPass = []
        peakNormalizedPower, peaktime, peakSharpWave = [], [], []

        for i in range(0, len(secondPass)):
            maxValue = max(maxPower[secondPass[i, 0] : secondPass[i, 1]])
            if maxValue > highthresh:
                thirdPass.append(secondPass[i])
                peakNormalizedPower.append(maxValue)
                peaktime.append(
                    secondPass[i, 0]
                    + np.argmax(maxPower[secondPass[i, 0] : secondPass[i, 1]])
                )
                peakSharpWave.append(
                    secondPass[i, 0]
                    + np.argmax(sharpWv_sig[secondPass[i, 0] : secondPass[i, 1]])
                )
        thirdPass = np.asarray(thirdPass)
        print(f"{len(thirdPass)} ripples reamining after deleting weak ripples")
        print(thirdPass.shape)

        ripple_duration = np.diff(thirdPass, axis=1) / fs
        ripples = pd.DataFrame(
            {
                "start": thirdPass[:, 0],
                "stop": thirdPass[:, 1],
                "peakNormalizedPower": peakNormalizedPower,
                "peakSharpWave": np.asarray(peakSharpWave),
                "peaktime": np.asarray(peaktime),
                "duration": ripple_duration.squeeze(),
            }
        )

        # ---------delete very short ripples--------
        ripples = ripples[ripples.duration >= mindur]
        print(f"{len(ripples)} ripples reamining after deleting short ripples")

        # ----- delete ripples with unrealistic high power
        # artifactRipples = np.where(peakNormalizedPower > maxPeakPower)[0]
        # fourthPass = np.delete(thirdPass, artifactRipples, 0)
        # peakNormalizedPower = np.delete(peakNormalizedPower, artifactRipples)

        # ---------delete very long ripples---------
        ripples = ripples[ripples.duration <= maxdur]
        print(f"{len(ripples)} ripples reamining after deleting very long ripples")

        # ----- converting to all time stamps to seconds --------
        ripples[["start", "stop", "peakSharpWave", "peaktime"]] /= fs  # seconds

        epochs = ripples.reset_index(drop=True)
        epochs["label"] = ""
        metadata = {
            "DetectionParams": {
                "lowThres": lowthresh,
                "highThresh": highthresh,
                "lowFreq": lowFreq,
                "highFreq": highFreq,
                "mindur": mindur,
                "maxdur": maxdur,
                "mergedist": mergedist,
            },
        }

        return epochs, metadata
