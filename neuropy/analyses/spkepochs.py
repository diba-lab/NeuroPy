import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import find_peaks
from ..utils import mathutil
from .. import core


def detect_off_epochs(mua: core.Mua, ignore_epochs: core.Epoch = None):
    """Detects OFF periods using multiunit activity. During these epochs neurons stop almost stop firing. These off periods were reported by Vyazovskiy et al. 2011 in cortex for sleep deprived animals.

    Parameters
    ----------
    mua : core.Mua object
        mua object holds total number of spikes in each bin
    ignore_epochs: core.Epoch
        ignore these epochs from getting detected

    References
    ----------
    1) Vyazovskiy, V. V., Olcese, U., Hanlon, E. C., Nir, Y., Cirelli, C., & Tononi, G. (2011). Local sleep in awake rats. Nature, 472(7344), 443–447. https://doi.org/10.1038/nature10009


    """

    time = mua.time
    frate = mua.firing_rate

    # off periods
    off = np.diff(np.where(frate < np.median(frate), 1, 0))
    start_off = np.where(off == 1)[0]
    end_off = np.where(off == -1)[0]

    if start_off[0] > end_off[0]:
        end_off = end_off[1:]
    if start_off[-1] > end_off[-1]:
        start_off = start_off[:-1]

    offperiods = np.vstack((start_off, end_off)).T
    duration = np.diff(offperiods, axis=1).squeeze()

    # ---- calculate minimum instantenous frate within intervals ------
    minValue = np.zeros(len(offperiods))
    for i in range(0, len(offperiods)):
        minValue[i] = min(frate[offperiods[i, 0] : offperiods[i, 1]])

    # --- selecting only top 10 percent of lowest peak instfiring -----
    quantiles = pd.qcut(minValue, 10, labels=False)
    top10percent = np.where(quantiles == 0)[0]
    offperiods = offperiods[top10percent, :]
    duration = duration[top10percent]

    events = pd.DataFrame(
        {
            "start": time[offperiods[:, 0]],
            "stop": time[offperiods[:, 1]],
            "duration": duration * mua.bin_size,
            "label": "",
        }
    )

    return core.Epoch(events)


def detect_pbe_epochs(
    mua: core.Mua,
    thresh=(3, None),
    edge_cutoff=0.5,
    duration=(0.1, None),
    distance=None,
):
    """Detects putative population burst events

    Parameters
    ----------
    thresh : tuple, optional
        values based on zscore i.e, events with firing rate above thresh[0] and peak exceeding thresh[1], by default (0, 3) --> above mean and greater than 3 SD
    duration : float, optional
        minimum and maximum duration of pbe, in seconds, default = (0.1,None) seconds
    distance : float, optioal
        if two events are less than this time apart, they are merged, in seconds
    """

    assert len(thresh) == 2, "thresh can only have two elements"
    if distance is None:
        distance = 1e-6
    else:
        distance = distance / mua.bin_size

    min_dur, max_dur = duration
    params = {
        "thresh": thresh,
        "duration": duration,
        "distance": distance,
    }

    lowthresh, highthresh = thresh
    n_spikes = stats.zscore(mua.spike_counts)
    n_spikes_thresh = np.where(n_spikes >= edge_cutoff, n_spikes, 0)
    peaks, props = find_peaks(
        n_spikes_thresh, height=[lowthresh, highthresh], prominence=0.5
    )
    starts, stops = props["left_bases"], props["right_bases"]
    peaks_n_spikes = n_spikes_thresh[peaks]

    # ----- merge overlapping epochs ------
    n_epochs = len(starts)
    ind_delete = []
    for i in range(n_epochs - 1):
        if starts[i + 1] - stops[i] < distance:

            # stretch the second epoch to cover the range of both epochs
            starts[i + 1] = min(starts[i], starts[i + 1])
            stops[i + 1] = max(stops[i], stops[i + 1])

            peaks_n_spikes[i + 1] = max(peaks_n_spikes[i], peaks_n_spikes[i + 1])
            peaks[i + 1] = [peaks[i], peaks[i + 1]][
                np.argmax([peaks_n_spikes[i], peaks_n_spikes[i + 1]])
            ]

            ind_delete.append(i)

    epochs_arr = np.vstack((starts, stops, peaks, peaks_n_spikes)).T
    starts, stops, peaks, peaks_n_spikes = np.delete(epochs_arr, ind_delete, axis=0).T

    time = np.asarray(mua.time)
    epochs_df = pd.DataFrame(
        {
            "start": time[starts.astype("int")],
            "stop": time[stops.astype("int")],
            "peak_time": time[peaks.astype("int")],
            "peak_counts": peaks_n_spikes,
            "label": "pbe",
        }
    )
    epochs = core.Epoch(epochs=epochs_df)
    # ------duration thresh---------
    epochs = epochs.duration_slice(min_dur=min_dur, max_dur=max_dur)
    print(f"{len(epochs)} epochs reamining with durations within ({min_dur},{max_dur})")

    epochs.metadata = params
    return epochs


def detect_lowstates_epochs():
    pass
