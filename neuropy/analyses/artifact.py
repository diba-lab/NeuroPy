import pandas as pd
import numpy as np
from scipy import stats
from ..core import Epoch
from ..core import Signal


def detect_artifact_epochs(signal: Signal, thresh=4, edge_cutoff=2, merge=5):
    """
    calculating artifact periods using z-score measure

    Parameters
    ----------
    signal : core.Signal
        neuropy.signal object
    thresh : int, optional
        zscore value above which it is considered noisy, by default 4
    edge_cutoff : int, optional
        zscore value, boundries are extended to this value, by default 2
    merge : int,
        artifacts less than this seconds apart are merged, default 5 seconds
    method : str, optional
        [description], by default "zscore"

    Returns
    -------
    core.Epoch
    """

    assert edge_cutoff <= thresh, "edge_cutoff can not be bigger than thresh"
    sampling_rate = signal.sampling_rate

    if signal.n_channels > 1:
        sig = np.median(signal.traces, axis=0)
    else:
        sig = signal.traces.reshape((-1))

    # ---- zscoring and identifying start and stops---------
    zsc = np.abs(stats.zscore(sig, axis=-1))
    artifact_binary = np.where(zsc > thresh, 1, 0)
    artifact_binary = np.concatenate(([0], artifact_binary, [0]))
    artifact_diff = np.diff(artifact_binary)
    artifact_start = np.where(artifact_diff == 1)[0]
    artifact_end = np.where(artifact_diff == -1)[0]
    firstPass = np.vstack((artifact_start, artifact_end)).T

    # --- extending the edges of artifact region --------
    edge_indx = np.where(zsc < edge_cutoff)[0]
    for i in range(firstPass.shape[0]):
        a, b = firstPass[i]
        left = edge_indx[np.max(np.where((a - edge_indx) > 0)[0])]
        right = edge_indx[np.min(np.where((edge_indx - b) > 0)[0])]
        firstPass[i] = [left, right]

    # --- merging neighbours -------
    minInterArtifactDist = merge * sampling_rate
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
