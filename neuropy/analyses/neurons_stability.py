import numpy as np

from .. import core


def firing_rate_stability(neurons: core.Neurons, window=3600, thresh=0.3):
    """Stability using firing rate

    Parameters
    ----------
    neurons : core.Neurons
        [description]
    periods : [type]
        [description]
    thresh : float, optional
        [description], by default 0.3

    Returns
    -------
    [type]
        [description]
    """

    frate = (neurons.firing_rate).reshape(-1, 1)
    binned_spk = neurons.get_binned_spiketrains(bin_size=window)
    window_frate = binned_spk.spike_counts / binned_spk.bin_size
    frate_ratio = window_frate / frate
    stability_bool = np.where(frate_ratio > thresh, 1, 0)
    good_windows = np.sum(stability_bool, axis=1)
    neuron_bool = np.where(good_windows == binned_spk.n_bins, 1, 0)

    return neuron_bool


def isi_stability(neurons: core.Neurons, window=3600, isi_thresh=0.04, on_thresh=0.8):
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
    boolean array
        [description]
    """
    t_start, t_stop = neurons.t_start, neurons.t_stop

    t_windows = np.arange(t_start, t_stop + window, window)

    isi_contamination = []
    for w in t_windows[:-1]:
        neurons_window = neurons.time_slice(w, w + window)
        isi_3ms = np.sum(neurons_window.get_isi(n_bins=3), axis=1)
        n_spikes = neurons_window.n_spikes
        isi_contamination.append(isi_3ms / n_spikes)

    isi_contamination = np.asarray(isi_contamination).T
    isi_bool = np.where(isi_contamination > isi_thresh, 0, 1)
    on_window = np.sum(isi_bool, axis=1) / isi_bool.shape[1]
    stability_bool = on_window > on_thresh
    return stability_bool


def waveform_stability(neurons: core.Neurons):
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
