import numpy as np
from .. import core


def firing_rate_stability(neurons: core.Neurons, periods, thresh=0.3):

    spikes = neurons.spiketrains
    spks = spikes.times
    nCells = len(spks)

    # --- number of spikes in each bin ------
    bin_dur = np.asarray([np.diff(window) for window in periods]).squeeze()
    total_dur = np.sum(bin_dur)
    nspks_period = np.asarray(
        [np.histogram(cell, bins=np.concatenate(periods))[0][::2] for cell in spks]
    )
    assert nspks_period.shape[0] == nCells

    total_spks = np.sum(nspks_period, axis=1)

    nperiods = len(periods)
    mean_frate = total_spks / total_dur

    # --- calculate meanfr in each bin and the fraction of meanfr over all bins
    frate_period = nspks_period / np.tile(bin_dur, (nCells, 1))
    fraction = frate_period / mean_frate.reshape(-1, 1)
    assert frate_period.shape == fraction.shape

    isStable = np.where(fraction >= thresh, 1, 0)
    spkinfo = spikes.info[["q", "shank"]].copy()
    spkinfo["stable"] = isStable.all(axis=1).astype(int)

    stbl = {
        "stableinfo": spkinfo,
        "isStable": isStable,
        "bins": periods,
        "thresh": thresh,
    }

    return stbl


def isi_stability(neurons: core.Neurons, window=3600, isi_thresh=4):
    """stability using ISI contamination

    References
    ----------
    1. Pacheco, A. T., Bottorff, J., Gao, Y., & Turrigiano, G. G. (2021). Sleep promotes downward firing rate homeostasis. Neuron, 109(3), 530-544.
        "Some neurons were lost during the recording, presumably due to electrode drift or gliosis. To establish ‘‘on’’ and ‘‘off’’ times for neurons, we used ISI contamination: when hourly % of ISIs < 3 msec was above 4%, unit was considered to be offline. Based on these 'on' and 'off' times, only units that were online for 80% of the experiment were used for analysis."

    Parameters
    ----------
    neurons : core.Neurons
        neurons for calculating stability
    window : int, optional
        window size over which isi contamination to be calculated, by default 3600
    isi_thresh : int, optional
        isi percentage threshold above which neurons are considered off in that window, by default 4 percent

    Returns
    -------
    boolean array, 1 ==> on, 0 ==> off
        [description]
    """
    t_start, t_stop = neurons.t_start, neurons.t_stop

    t_windows = np.arange(t_start, t_stop + window, window)

    isi_contamination = []
    for w in t_windows[:-1]:
        neurons_window = neurons.time_slice(w, w + window)
        isi_3ms = np.sum(neurons_window.get_isi(n_bins=3), axis=1)
        n_spikes = neurons_window.get_n_spikes()
        isi_contamination.append(isi_3ms / n_spikes)

    isi_contamination = np.asarray(isi_contamination) * 100
    isi_bool = np.where(isi_contamination > isi_thresh, 0, 1)

    return isi_bool.T


def waveform_similarity(self):
    pass


def refPeriodViolation(self):

    spks = self._obj.spikes.times

    fp = 0.05  # accepted contamination level
    T = self._obj.epochs.totalduration
    taur = 2e-3
    tauc = 1e-3
    nbadspikes = lambda N: 2 * (taur - tauc) * (N ** 2) * (1 - fp) * fp / T

    nSpks = [len(_) for _ in spks]
    expected_violations = [nbadspikes(_) for _ in nSpks]

    self.expected_violations = np.asarray(expected_violations)

    isi = [np.diff(_) for _ in spks]
    ref = np.array([0, 0.002])
    zerolag_spks = [np.histogram(_, bins=ref)[0] for _ in isi]

    self.violations = np.asarray(zerolag_spks)


def isolationDistance(self):
    pass
