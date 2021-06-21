import pandas as pd
import numpy as np
from scipy import stats
from ..core import Epoch
from ..core import Signal


def detect_artifact_epochs(signal: Signal, thresh=4, method="zscore"):
    """
    calculating periods to exclude for analysis using simple z-score measure
    """

    sampling_rate = signal.sampling_rate

    if signal.n_channels > 1:
        sig = np.median(signal.traces, axis=0)
    else:
        sig = signal.traces.reshape((-1))

    zsc = np.abs(stats.zscore(sig, axis=-1))
    artifact_binary = np.where(zsc > thresh, 1, 0)
    artifact_binary = np.concatenate(([0], artifact_binary, [0]))
    artifact_diff = np.diff(artifact_binary)
    artifact_start = np.where(artifact_diff == 1)[0]
    artifact_end = np.where(artifact_diff == -1)[0]

    firstPass = np.vstack((artifact_start - 10, artifact_end + 2)).T

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
