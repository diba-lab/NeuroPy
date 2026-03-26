from dataclasses import dataclass
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as filtSig
import scipy.signal as sg
from joblib import Parallel, delayed
from scipy import fftpack, stats
from scipy.fftpack import next_fast_len
from scipy.ndimage import gaussian_filter
import seaborn as sns
from scipy.interpolate import interp2d

from neuropy.core.signal import Signal

class PSTH:
    def __init__(
        self,
        signal: Signal,
        event_start_times: pd.Series,
        event_end_times: pd.Series,
        cell_id: int or None,
        start_buffer_sec: float = 10.0,
        end_buffer_sec: float = 10.0,
        auto_generate=True,
    ):
        pass

def get_peth(traces: np.array, t_start: np.array, sample_rate: int, event_times: np.ndarray or pd.Series, buffer_sec: tuple or list or np.array):
    """Get time histogram of signal traces centered on events with buffer_sec before / after
    :param: signal: nchannels x ntimes array of signal of interest, assumes constant sample_rate below
    :param: t_start: start time of signal in seconds
    :param: sample_rate: sample rate of signal
    :param: event_times: events around which to gather signal
    :buffer_sec: (start_buffer, end_buffer): times before / after event_times to keep"""

    assert len(buffer_sec) == 2
    buffer_sec = np.abs(buffer_sec) # Make everything positive

    event_frames = ((event_times - t_start) * sample_rate).astype(int)
    buffer_frames = np.array(buffer_sec * sample_rate).astype(int)
    last_frame = traces.shape[1]

    peth = []
    for event_frame in event_frames:
        start_frame = event_frame - buffer_frames[0]
        stop_frame = event_frame + buffer_frames[1]
        if (start_frame < 0) or (stop_frame > last_frame):  # skip if event is too close to signal start or end
            continue
        else:
            peth.append(traces[:, start_frame:stop_frame])

    peth = np.array(peth)
    times = np.linspace(-buffer_sec[0], buffer_sec[1], peth.shape[1])

    return times, peth


