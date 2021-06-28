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
