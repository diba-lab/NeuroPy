import numpy as np
from .. import core
import pandas as pd


def corr_across_time_window(neurons: core.Neurons, window=300, bin_size=0.25):
    """Correlation of pairwise correlations across time

    Parameters
    ----------
    neurons: core.Neruons
        neurons used for calculation
    window : int, optional
        dividing the period into this size window, by default 300 seconds
    bin_size : float, optional
        [description], by default 0.25
    """

    n_windows = int((neurons.t_stop - neurons.t_start) / window)
    window_starts = np.arange(n_windows) * window + neurons.t_start

    pair_corr = []
    for start in window_starts:
        pair_corr.append(
            neurons.time_slice(start, start + window)
            .get_binned_spiketrains(bin_size=bin_size)
            .get_pairwise_corr()
        )
    pair_corr = np.asarray(pair_corr).T
    corr_across_time = pd.DataFrame(pair_corr).corr()
    return corr_across_time.values
