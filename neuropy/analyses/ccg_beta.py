import numpy as np
import cupy as cp


# Define acceptable dtypes
_ACCEPTED_ARRAY_DTYPES = (
    float,
    int,
    bool,
)


def _as_array(arr, dtype=None, use_cupy=False):
    """Convert an object to a numerical array using NumPy or CuPy."""
    lib = cp if use_cupy else np
    if arr is None:
        return None
    if isinstance(arr, lib.ndarray) and dtype is None:
        return arr
    if isinstance(arr, (int, float)):
        arr = [arr]
    out = lib.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    if out.dtype not in _ACCEPTED_ARRAY_DTYPES:
        raise ValueError(
            f"'arr' seems to have an invalid dtype: {out.dtype}"
        )
    return out


def _index_of(arr, lookup, use_cupy=False):
    """Replace scalars in an array by their indices in a lookup table."""
    lib = cp if use_cupy else np
    lookup = lib.asarray(lookup, dtype=lib.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = lib.zeros(m + 1, dtype=int)

    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = lib.arange(len(lookup))
    return tmp[arr]


def _unique(x, use_cupy=False):
    """
    Faster version of np.unique().
    This version is restricted to 1D arrays of non-negative integers.
    It is only faster if len(x) >> len(unique(x)).
    """
    lib = cp if use_cupy else np
    if x is None or len(x) == 0:
        return lib.array([], dtype=int)
    x = _as_array(x, use_cupy=use_cupy)
    x = x[x >= 0]
    bc = lib.bincount(x)
    return lib.nonzero(bc)[0]


def _increment(arr, indices, use_cupy=False):
    """
    Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account.
    """
    lib = cp if use_cupy else np
    arr = _as_array(arr, use_cupy=use_cupy)
    indices = _as_array(indices, use_cupy=use_cupy)
    bbins = lib.bincount(indices)
    arr[: len(bbins)] += bbins
    return arr


def _diff_shifted(arr, steps=1, use_cupy=False):
    """Calculate the difference between elements shifted by steps."""
    arr = _as_array(arr, use_cupy=use_cupy)
    return arr[steps:] - arr[: len(arr) - steps]


def _create_correlograms_array(n_clusters, winsize_bins, use_cupy=False):
    """Create an array to hold correlograms."""
    lib = cp if use_cupy else np
    return lib.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1), dtype=lib.int32)


def _symmetrize_correlograms(correlograms, use_cupy=False):
    """Return the symmetrized version of the CCG arrays."""
    lib = cp if use_cupy else np
    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # We symmetrize c[i, j, 0].
    # This is necessary because the algorithm in correlograms() is sensitive to the order of identical spikes.
    correlograms[..., 0] = lib.maximum(correlograms[..., 0], correlograms[..., 0].T)

    sym = correlograms[..., 1:][..., ::-1]
    sym = lib.transpose(sym, (1, 0, 2))

    return lib.dstack((sym, correlograms))


def firing_rate(spike_clusters, cluster_ids=None, bin_size=None, duration=None):
    """Compute the average number of spikes per cluster per bin."""
    # Take the cluster order into account.
    if cluster_ids is None:
        cluster_ids = _unique(spike_clusters)
    else:
        cluster_ids = _as_array(cluster_ids)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters_i = _index_of(spike_clusters, cluster_ids)

    assert bin_size > 0
    bc = np.bincount(spike_clusters_i)
    # Handle the case where the last cluster(s) are empty.
    if len(bc) < len(cluster_ids):
        n = len(cluster_ids) - len(bc)
        bc = np.concatenate((bc, np.zeros(n, dtype=bc.dtype)))
    assert bc.shape == (len(cluster_ids),)
    return bc * np.c_[bc] * (bin_size / (duration or 1.0))


def correlograms(
    neurons,
    sample_rate=1.0,
    bin_size=None,
    window_size=None,
    symmetrize=True,
    use_cupy=False
):
    """Compute cross-correlograms."""
    lib = cp if use_cupy else np

    spike_times = lib.concatenate(neurons.spiketrains)
    spike_clusters = lib.concatenate([
        lib.full(len(spiketrain), cluster_id)
        for spiketrain, cluster_id in zip(neurons.spiketrains, neurons.neuron_ids)
    ])

    sort_ind = lib.argsort(spike_times)
    spike_times = spike_times[sort_ind]
    spike_clusters = spike_clusters[sort_ind]
    spike_samples = (spike_times * sample_rate).astype(int)

    bin_size = lib.clip(bin_size, 1e-5, 1e5)
    binsize = int(sample_rate * bin_size)

    window_size = lib.clip(window_size, 1e-5, 1e5)
    winsize_bins = 2 * int(0.5 * window_size / bin_size) + 1

    clusters = _unique(spike_clusters, use_cupy=use_cupy)
    n_clusters = len(clusters)
    spike_clusters_i = _index_of(spike_clusters, clusters, use_cupy=use_cupy)

    shift = 1
    mask = lib.ones_like(spike_samples, dtype=bool)
    correlograms = _create_correlograms_array(n_clusters, winsize_bins, use_cupy=use_cupy)

    while mask[:-shift].any():
        spike_diff = _diff_shifted(spike_samples, shift, use_cupy=use_cupy)
        spike_diff_b = spike_diff // binsize
        mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False
        m = mask[:-shift].copy()
        d = spike_diff_b[m]

        indices = lib.ravel_multi_index(
            (
                spike_clusters_i[:-shift][m],
                spike_clusters_i[+shift:][m],
                d
            ),
            correlograms.shape
        )
        _increment(correlograms.ravel(), indices, use_cupy=use_cupy)
        shift += 1

    if symmetrize:
        return _symmetrize_correlograms(correlograms, use_cupy=use_cupy)
    else:
        return correlograms
