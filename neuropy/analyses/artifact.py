import pandas as pd
import numpy as np
from scipy import stats
from ..core import Epoch
from ..core import Signal
from ..utils import signal_process


def detect_artifact_epochs(signal: Signal, thresh=4, filt: list or np.ndarray = None):
    """
    calculating periods to exclude for analysis using simple z-score measure
    :param filt = list of lower and upper limits with which to filter signal, e.g.
    [3, 3000] -> bandpass between and 3000 Hz while [45, None] -> high-pass above 45.
    """

    sampling_rate = signal.sampling_rate

    if signal.n_channels > 1:
        sig_raw = np.median(signal.traces, axis=0)
    else:
        sig_raw = signal.traces.reshape((-1))

    # NRK todo: does this need to go BEFORE taking the median of the signal above?
    # After condensing into one trace, filter things
    if filt is not None:
        assert len(filt) == 2, "Inputs for filtering signal must be length = 2"
        if filt[0] is not None:  # highpass
            sig = signal_process.filter_sig.highpass(sig_raw, filt[0], fs=sampling_rate)
        elif filt[1] is not None:  # lowpass
            sig = signal_process.filter_sig.lowpass(sig_raw, filt[1], fs=sampling_rate)
        elif filt[0] is not None and filt[1] is not None:  # bandpass
            sig = signal_process.filter_sig.bandpass(
                sig_raw, filt[0], filt[1], fs=sampling_rate
            )
    elif filt is None:
        sig = sig_raw

    zsc = np.abs(stats.zscore(sig, axis=-1))
    artifact_binary = np.where(zsc > thresh, 1, 0)
    artifact_binary = np.concatenate(([0], artifact_binary, [0]))
    artifact_diff = np.diff(artifact_binary)
    artifact_start = np.where(artifact_diff == 1)[0]
    artifact_end = np.where(artifact_diff == -1)[0]

    firstPass = np.vstack((artifact_start - 10, artifact_end + 10)).T

    minInterArtifactDist = 5 * sampling_rate
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

    artifact_s = np.asarray(secondPass) / sampling_rate  # seconds

    epochs = pd.DataFrame(
        {"start": artifact_s[:, 0], "stop": artifact_s[:, 1], "label": ""}
    )
    metadata = {"threshold": thresh}

    return Epoch(epochs, metadata)
