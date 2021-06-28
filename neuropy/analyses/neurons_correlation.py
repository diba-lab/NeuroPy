import numpy as np
from .. import core


def corr_across_time_window(
    neurons: core.Neurons, cell_ids, period, window=300, binsize=0.25
):
    """Correlation of pairwise correlation across a period by dividing into window size epochs

    Parameters
    ----------
    period: array like
        time period where the pairwise correlations are calculated, in seconds
    window : int, optional
        dividing the period into this size window, by default 900
    binsize : float, optional
        [description], by default 0.25
    """

    spikes = self.get_spiketrains(cell_ids)
    epochs = np.arange(period[0], period[1], window)

    pair_corr_epoch = []
    for i in range(len(epochs) - 1):
        epoch_bins = np.arange(epochs[i], epochs[i + 1], binsize)
        spkcnt = np.asarray([np.histogram(x, bins=epoch_bins)[0] for x in spikes])
        epoch_corr = np.corrcoef(spkcnt)
        pair_corr_epoch.append(epoch_corr[np.tril_indices_from(epoch_corr, k=-1)])
    pair_corr_epoch = np.asarray(pair_corr_epoch)

    # masking nan values in the array
    pair_corr_epoch = np.ma.array(pair_corr_epoch, mask=np.isnan(pair_corr_epoch))
    corr = np.ma.corrcoef(pair_corr_epoch)  # correlation across windows
    time = epochs[:-1] + window / 2
    self.corr, self.time = corr, time
    return np.asarray(corr), time


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
