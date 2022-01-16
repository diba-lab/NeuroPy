import pandas as pd
import numpy as np
from scipy import stats
from ..core import Epoch
from ..core import Signal
from ..utils import signal_process


def detect_artifact_epochs(
    signal: Signal, thresh=4, edge_cutoff=2, merge=5, filt: list or np.ndarray = None
):
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
    filt : list, optional
        lower and upper limits with which to filter signal, e.g. 3, 3000] ->
        bandpass between and 3000 Hz while [45, None] -> high-pass above 45.

    Returns
    -------
    core.Epoch
    """

    assert edge_cutoff <= thresh, "edge_cutoff can not be bigger than thresh"
    sampling_rate = signal.sampling_rate

    if signal.n_channels > 1:
        sig_raw = np.mean(signal.traces, axis=0)

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

    # ---- zscoring and identifying start and stops---------
    zsc = np.abs(stats.zscore(sig, axis=-1))
    artifact_binary = np.where(zsc > thresh, 1, 0)
    artifact_binary = np.concatenate(([0], artifact_binary, [0]))
    artifact_diff = np.diff(artifact_binary)
    artifact_start = np.where(artifact_diff == 1)[0]
    artifact_end = np.where(artifact_diff == -1)[0]
    firstPass = np.vstack((artifact_start, artifact_end)).T

    # --- extending the edges of artifact region --------
    edge_binary = np.where(zsc > edge_cutoff, 1, 0)
    edge_binary = np.concatenate(([0], edge_binary, [0]))
    edge_diff = np.diff(edge_binary)
    edge_start = np.where(edge_diff == 1)[0]
    edge_end = np.where(edge_diff == -1)[0]

    edge_start = edge_start[np.digitize(firstPass[:, 0], edge_start) - 1]
    edge_end = edge_end[np.digitize(firstPass[:, 1], edge_end)]
    firstPass[:, 0], firstPass[:, 1] = edge_start, edge_end

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
