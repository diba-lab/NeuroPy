"""Calculate and test millisecond-scale connectivity between neurons a la Diba et al. (2014) and English/McKenzie
 et al. (2017)"""

import numpy as np

try:
    import ccg_gpu as ccg
    import cupy as cp

    cuda = True
except ModuleNotFoundError:
    import ccg

    cuda = False


def eran_conv(ccg, window_width=5, wintype="gauss", hollow_frac=None):
    """Estimate chance-level correlations using convolution method from Stark and Abeles (2009, J. Neuro Methods).

    :param ccg:
    :param window_width:
    :param wintype:
    :param hollow_frac:
    :return: pvals:
             pred:
             qvals:
    """
    pvals, pred, qvals = None
    assert wintype in ["gauss", "rect", "triang"]

    # Auto-assign appropriate hollow fraction if not specified
    if hollow_frac is None:
        if wintype == "gauss":
            hollow_frac = 0.6
        elif wintype == "rect":
            hollow_frac = 0.42
        elif wintype == "triang":
            hollow_frac = 0.63

    return pvals, pred, qvals


def ccg_jitter(
    spike_trains,
    clusters,
    SampleRate=30000,
    binsize=0.0005,
    duration=0.02,
    jscale=5,
    njitter=100,
    alpha=0.05,
):

    # TODO: make this take the same inputs as correlograms? e.g. spikes from both clusters sorted by time with corresponding cluster ids?
    # # Make spike trains into 1d numpy array
    # spikes0 = spike_trains[0]
    # spikes1 = spike_trains[1]

    # set up variables
    halfbins = (
        cp.round(duration / binsize / 2) if cuda else np.round(duration / binsize / 2)
    )
    one_ms = 0.001
    spikes_sorted, clus_sorted = ccg_spike_assemble(spike_trains)
    spikes1 = spikes_sorted[
        clus_sorted == 1
    ]  # keep all spike times from cluster 1 for easy manipulation during jitter step

    # run ccg on actual data
    correlograms = ccg.correlograms(
        spikes_sorted,
        clus_sorted,
        bin_size=binsize,
        window_size=duration,
        sample_rate=SampleRate,
    )

    # Now run on jittered spike-trains!
    # TODO: implement this in ALL cupy and compare times...does it matter if the spike jitter code is in numpy? Answer: it does 16ms with numpy vs 1 with cupy.
    nspikes1 = len(spikes1)
    ccgj = []
    for n in range(njitter):

        # Jitter spikes in second cluster
        if cuda:
            spike_trains[1] = (
                cp.round(
                    (
                        spikes1
                        + 2 * (one_ms * jscale) * cp.random.rand(nspikes1)
                        - 1 * one_ms * jscale
                    )
                    * SampleRate
                )
                / SampleRate
            )
        else:
            spike_trains[1] = (
                np.round(
                    (
                        spikes1
                        + 2 * (one_ms * jscale) * np.random.rand(nspikes1)
                        - 1 * one_ms * jscale
                    )
                    * SampleRate
                )
                / SampleRate
            )

        # NRK TODO: start debugging here!
        spikes_sorted, clus_sorted = ccg_spike_assemble(spike_trains)

        # re-run ccg
        ccgj.append(
            ccg.correlograms(
                spikes_sorted,
                clus_sorted,
                bin_size=binsize,
                window_size=duration,
                sample_rate=SampleRate,
            )
        )


def ccg_spike_assemble(spike_trains):
    """Assemble an array of sorted spike times and cluIDs for the input cluster ids the list clus_use """
    spikes_all, clus_all = [], []
    for ids, spike_train in enumerate(spike_trains):
        spikes_all.append(spike_train),
        clus_all.append(np.ones_like(spike_train) * ids)
    if cuda:
        spikes_all, clus_all = cp.concatenate(spikes_all), cp.concatenate(clus_all)
    else:
        spikes_all, clus_all = np.concatenate(spikes_all), np.concatenate(clus_all)
    spikes_sorted = spikes_all[spikes_all.argsort()]
    clus_sorted = clus_all[spikes_all.argsort()]

    return spikes_sorted, clus_sorted.astype("int")
