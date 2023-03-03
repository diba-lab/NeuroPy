import numpy as np
try:
    import modin.pandas as pd # modin is a drop-in replacement for pandas that uses multiple cores
except ImportError:
    import pandas as pd # fallback to pandas when modin isn't available
import scipy.stats as stats
from neuropy.utils import mathutil
from .. import core


def detect_local_sleep_epochs(mua: core.Mua, ignore_epochs: core.Epoch = None):
    """Detects local OFF events in within period

    Parameters
    ----------
    period : list,array-like
        period in seconds

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
            "duration": duration / 1000,
            "label": "",
        }
    )

    return core.Epoch(events)


def detect_pbe_epochs(
    mua: core.Mua, thresh=(0, 3), min_dur=0.1, merge_dur=0.01, max_dur=1.0
):
    """Detects putative population burst events

    Parameters
    ----------
    thresh : tuple, optional
        values based on zscore i.e, events with firing rate above thresh[0] and peak exceeding thresh[1], by default (0, 3) --> above mean and greater than 3 SD
    min_dur : float, optional
        minimum duration of a pop burst event, in seconds, default = 0.1 seconds
    merge_dur : float, optioal
        if two events are less than this time apart, they are merged, in seconds
    max_dur : float, optional
        events only lasting below this duration
    """
    assert len(thresh) == 2, "thresh can only have two elements"
    params = {
        "thresh": thresh,
        "min_dur": min_dur,
        "merge_dur": merge_dur,
        "max_dur": max_dur,
    }

    sampling_rate = 1 / mua.bin_size  # sampling rate of mua
    min_dur = min_dur * sampling_rate
    merge_dur = merge_dur * sampling_rate
    events = mathutil.threshPeriods(
        stats.zscore(mua.firing_rate),
        lowthresh=thresh[0],
        highthresh=thresh[1],
        minDuration=min_dur,
        minDistance=merge_dur,
    )

    time = np.asarray(mua.time)
    pbe_times = time[events]

    events = pd.DataFrame(
        {
            "start": pbe_times[:, 0],
            "stop": pbe_times[:, 1],
            "duration": np.diff(pbe_times, axis=1).squeeze(),
            "label": "",
        }
    )

    events = events[events.duration < max_dur].reset_index(drop=True)

    return core.Epoch(events, params)


def detect_lowstates_epochs():
    pass
