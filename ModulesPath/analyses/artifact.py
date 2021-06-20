import pandas as pd
import numpy as np
import scipy.stats as stat
from ..core import Epoch
from ..core import Signal


def detect_artifact_epochs(signal: Signal, thresh=4, method="zscore"):
    """
    calculating periods to exclude for analysis using simple z-score measure
    """

    sampling_rate = signal.sampling_rate

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

    return Epoch(epochs, metadata)
