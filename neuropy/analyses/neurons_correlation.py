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


def corr_pairwise(
    neurons: core.Neurons,
    ids,
    period,
    cross_shanks=False,
    binsize=0.25,
    window=None,
    slideby=None,
):
    """Calculates pairwise correlation between given spikes within given period

    Parameters
    ----------
    spikes : list
        list of spike times
    period : list
        time period within which it is calculated , in seconds
    cross_shanks: bool,
        whether restrict pairs to only across shanks, by default False (all pairs)
    binsize : float, optional
        binning of the time period, by default 0.25 seconds

    Returns
    -------
    N-pairs
        pairwise correlations
    """

    spikes = self.get_spiketrains(ids)
    bins = np.arange(period[0], period[1], binsize)
    spk_cnts = np.asarray([np.histogram(cell, bins=bins)[0] for cell in spikes])

    time_points = None
    if window is not None:
        nbins_window = int(window / binsize)
        if slideby is None:
            slideby = window
        nbins_slide = int(slideby / binsize)
        spk_cnts = np.lib.stride_tricks.sliding_window_view(
            spk_cnts, nbins_window, axis=1
        )[:, ::nbins_slide, :]
        time_points = np.lib.stride_tricks.sliding_window_view(
            bins, nbins_window, axis=-1
        )[::nbins_slide, :].mean(axis=-1)

    # ----- indices for cross shanks correlation -------
    shnkId = self.get_shankID(ids)
    selected_pairs = np.tril_indices(len(ids), k=-1)
    if cross_shanks:
        selected_pairs = np.nonzero(
            np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1))
        )

    corr = []
    if spk_cnts.ndim == 2:
        corr = np.corrcoef(spk_cnts)[selected_pairs]
    else:
        for w in range(spk_cnts.shape[1]):
            corr.append(np.corrcoef(spk_cnts[:, w, :])[selected_pairs])
        corr = np.asarray(corr).T

    return corr, time_points
