import numpy as np
try:
    import cupy as cp
except ImportError:
    print("Error importing CuPy. Install CuPy to speed up correlogram calculations.")
    cp = None


# Define acceptable dtypes
_ACCEPTED_ARRAY_DTYPES = (
    float,
    int,
    bool,
)


# Assemble Spike Arrays
def _np_assemble_spike_arrays(neurons, sample_rate):
    """
    Assemble spike arrays for neurons from neurons object using NumPy.
    """
    # Get spike times from neurons
    spike_times = np.concatenate(neurons.spiketrains)

    # Get neuron clusters
    spike_clusters = np.concatenate([
        np.full(len(spiketrain), cluster_id)
        for spiketrain, cluster_id in zip(neurons.spiketrains, neurons.neuron_ids)
    ])

    # Sort spike times and neuron clusters
    sort_ind = np.argsort(spike_times)

    # Get all sorted arrays
    spike_times = spike_times[sort_ind]
    spike_clusters = spike_clusters[sort_ind]
    spike_samples = (spike_times * sample_rate).astype(int)

    return spike_times, spike_clusters, spike_samples


def _cp_assemble_spike_arrays(neurons, sample_rate):
    """
    Assemble spike arrays for neurons from neurons object using CuPy.
    """
    spike_times = cp.concatenate([cp.asarray(spiketrain) for spiketrain in neurons.spiketrains])

    # Get neuron clusters
    spike_clusters = cp.concatenate([
        cp.full(len(spiketrain), cluster_id, dtype=cp.int32)
        for spiketrain, cluster_id in zip(neurons.spiketrains, neurons.neuron_ids)
    ])

    # Sort spike times and neuron clusters
    sort_ind = cp.argsort(spike_times)

    # Get all sorted arrays
    spike_times = spike_times[sort_ind]
    spike_clusters = spike_clusters[sort_ind]
    spike_samples = (spike_times * sample_rate).astype(cp.int32)

    return spike_times, spike_clusters, spike_samples


# Create Arrays
def _np_as_array(arr, dtype=None):
    """
    Convert an object to a numerical NumPy array.
    Avoid a copy if possible.
    """
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and dtype is None:
        return arr
    if isinstance(arr, (int, float)):
        arr = [arr]
    out = np.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    if out.dtype not in _ACCEPTED_ARRAY_DTYPES:
        raise ValueError(
            "'arr' seems to have an invalid dtype: " "{0:s}".format(str(out.dtype))
        )
    return out


def _cp_as_array(arr, dtype=None):
    """
    Convert an object to a numerical CuPy array.
    """
    if arr is None:
        return None
    if isinstance(arr, cp.ndarray) and dtype is None:
        return arr
    if isinstance(arr, (int, float)):
        arr = [arr]
    out = cp.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    # Check for accepted CuPy dtypes
    accepted_dtypes = (cp.float32, cp.float64, cp.int32, cp.int64, cp.bool_)
    if out.dtype not in accepted_dtypes:
        raise ValueError(
            f"'arr' seems to have an invalid dtype: {out.dtype}"
        )
    return out


# Create Index array via lookup
def _np_index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.
    Implicitely assume that:
    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.
    This is not checked for performance reasons.
    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    # TODO: assertions to disable in production for performance reasons.
    # TODO: np.searchsorted(lookup, arr) is faster on small arrays with large
    # values
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


def _cp_index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.
    Implicitly assume that:
    * All elements of arr and lookup are non-negative integers.
    * All elements of arr belong to lookup.
    This is not checked for performance reasons.
    """
    # Convert lookup to a CuPy array of int32
    lookup = cp.asarray(lookup, dtype=cp.int32)

    # Determine the size of the temporary array
    m = (lookup.max().item() if len(lookup) else 0) + 1  # Convert to Python int

    # Create the temporary array on the GPU
    tmp = cp.zeros(int(m + 1), dtype=cp.int32)  # Ensure size is an integer

    # Ensure that -1 values are kept
    tmp[-1] = -1

    # Map lookup values to their indices
    if len(lookup):
        tmp[lookup] = cp.arange(len(lookup), dtype=cp.int32)

    # Convert arr to CuPy array and return mapped indices
    arr = cp.asarray(arr, dtype=cp.int32)
    return tmp[arr]


# Get unique values
def _np_unique(x):
    """Faster version of np.unique().
    This version is restricted to 1D arrays of non-negative integers.
    It is only faster if len(x) >> len(unique(x)).
    """
    if x is None or len(x) == 0:
        return np.array([], dtype=int)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = _np_as_array(x)
    x = x[x >= 0]
    bc = np.bincount(x)
    return np.nonzero(bc)[0]


def _unique_cupy(x):
    """
    CuPy implementation of _np_unique
    """
    if x is None or len(x) == 0:
        return cp.array([], dtype=cp.int32)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = _cp_as_array(x)
    x = x[x >= 0]
    bc = np.bincount(x)
    return np.nonzero(bc)[0]



def _np_increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    arr = _np_as_array(arr)
    indices = _np_as_array(indices)
    bbins = np.bincount(indices)
    arr[: len(bbins)] += bbins
    return arr


def _cp_increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    arr = _cp_as_array(arr)
    indices = _cp_as_array(indices)
    bbins = cp.asarray(
        np.bincount(cp.asnumpy(indices))
    )  # NRK can you make this cupy? Maybe add in try/except statement?
    arr[: len(bbins)] += bbins
    return arr


def _diff_shifted(arr, steps=1):
    arr = _np_as_array(arr)
    return arr[steps:] - arr[: len(arr) - steps]


def _cp_diff_shifted(arr, steps=1):
    arr = _cp_as_array(arr)
    return arr[steps:] - arr[: len(arr) - steps]


def _np_create_correlograms_array(n_clusters, winsize_bins):
    return np.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1), dtype=np.int32)


def _cp_create_correlograms_array(n_clusters, winsize_bins):
    """Create an empty correlograms array using CuPy."""
    return cp.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1), dtype=cp.int32)


def _np_symmetrize_correlograms(correlograms):
    """Return the symmetrized version of the CCG arrays."""

    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # We symmetrize c[i, j, 0].
    # This is necessary because the algorithm in correlograms()
    # is sensitive to the order of identical spikes.
    correlograms[..., 0] = np.maximum(correlograms[..., 0], correlograms[..., 0].T)

    sym = correlograms[..., 1:][..., ::-1]
    sym = np.transpose(sym, (1, 0, 2))

    return np.dstack((sym, correlograms))


def _cp_symmetrize_correlograms(correlograms):
    """Return the symmetrized version of the CCG arrays using CuPy."""
    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # Symmetrize correlograms[..., 0]
    correlograms[..., 0] = cp.maximum(correlograms[..., 0], correlograms[..., 0].T)

    # Symmetrize the remaining bins
    sym = correlograms[..., 1:][..., ::-1]
    sym = cp.transpose(sym, (1, 0, 2))

    return cp.dstack((sym, correlograms))


def firing_rate(spike_clusters, cluster_ids=None, bin_size=None, duration=None):
    """Compute the average number of spikes per cluster per bin."""

    # Take the cluster order into account.
    if cluster_ids is None:
        cluster_ids = _np_unique(spike_clusters)
    else:
        cluster_ids = _np_as_array(cluster_ids)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters_i = _np_index_of(spike_clusters, cluster_ids)

    assert bin_size > 0
    bc = np.bincount(spike_clusters_i)
    # Handle the case where the last cluster(s) are empty.
    if len(bc) < len(cluster_ids):
        n = len(cluster_ids) - len(bc)
        bc = np.concatenate((bc, np.zeros(n, dtype=bc.dtype)))
    assert bc.shape == (len(cluster_ids),)
    return bc * np.c_[bc] * (bin_size / (duration or 1.0))


def np_spike_correlations(
        neurons,
        neuron_inds,
        sample_rate=1.0,
        bin_size=None,
        window_size=None,
        symmetrize=True,
):
    """
    Compute all pairwise cross-correlations among neurons(clusters) given in neurons class.

    Parameters
    ----------
    neurons : core.neurons
        neurons obj containing spiketrains and related info
    sample_rate : float
        Sampling rate.
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.

    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise CCGs.
    """

    # Convert to array if int
    if isinstance(neuron_inds, int):
        neuron_inds = [neuron_inds]

    neurons = neurons.neuron_slice(neuron_inds=neuron_inds)

    # Get spike times from neurons
    spike_times, spike_clusters, spike_samples = _np_assemble_spike_arrays(neurons, sample_rate)

    # Get binsize
    bin_size = np.clip(bin_size, 1e-5, 1e5)
    binsize = int(sample_rate * bin_size)
    assert binsize >= 1

    # Get window-size dependent bins
    window_size = np.clip(window_size, 1e-5, 1e5)
    winsize_bins = 2 * int(0.5 * window_size / bin_size) + 1

    assert winsize_bins >= 1
    assert winsize_bins % 2 == 1

    # Get unique neuron clusters
    clusters = _np_unique(spike_clusters)
    n_clusters = len(clusters)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters_i = _np_index_of(spike_clusters, clusters)

    # Shift between the two copies of the spike trains.
    shift = 1

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_samples, dtype=bool)
    correlograms = _np_create_correlograms_array(n_clusters, winsize_bins)

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        spike_diff_b = spike_diff // binsize

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False

        # Cache the masked spike delays.
        m = mask[:-shift].copy()

        # Update the masks given the clusters to update.
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        indices = np.ravel_multi_index(
            (
                spike_clusters_i[:-shift][m],
                spike_clusters_i[+shift:][m],
                d
            ),
            correlograms.shape
        )

        # Increment the matching spikes in the correlograms array.
        _np_increment(correlograms.ravel(), indices)
        shift += 1

    if symmetrize:
        return _np_symmetrize_correlograms(correlograms)
    else:
        return correlograms


def cp_spike_correlations(
        neurons,
        neuron_inds,
        sample_rate=1.0,
        bin_size=None,
        window_size=None,
        symmetrize=True,
):
    """
    Compute all pairwise cross-correlations among neurons(clusters) given in neurons class.

    Parameters
    ----------
    neurons : core.neurons
        neurons obj containing spiketrains and related info
    sample_rate : float
        Sampling rate.
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.

    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise CCGs.
    """

    # Convert to array if int
    if isinstance(neuron_inds, int):
        neuron_inds = [neuron_inds]

    neurons = neurons.neuron_slice(neuron_inds=neuron_inds)

    # Get spike times from neurons
    spike_times, spike_clusters, spike_samples = _cp_assemble_spike_arrays(neurons, sample_rate)

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    binsize = int(sample_rate * bin_size)  # in samples

    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    winsize_bins = 2 * int(0.5 * window_size / bin_size) + 1

    # Get unique neuron clusters
    clusters = _unique_cupy(spike_clusters)
    n_clusters = len(clusters)

    spike_clusters_i = _cp_index_of(spike_clusters, clusters)

    shift = 1

    mask = cp.ones_like(spike_samples, dtype=cp.bool_)
    correlograms = _cp_create_correlograms_array(n_clusters, winsize_bins)

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _cp_diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        spike_diff_b = spike_diff // binsize

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m]

        # # Update the masks given the clusters to update.
        # m0 = cp.in1d(spike_clusters[:-shift], clusters)
        # m = m & m0
        # d = spike_diff_b[m]
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        indices = cp.ravel_multi_index(
            (spike_clusters_i[:-shift][m], spike_clusters_i[+shift:][m], d),
            correlograms.shape,
        )

        # Increment the matching spikes in the correlograms array.
        _cp_increment(correlograms.ravel(), indices)

        shift += 1

    if symmetrize:
        return _cp_symmetrize_correlograms(correlograms)
    else:
        return correlograms


def spike_correlations(
        neurons,
        neuron_inds,
        sample_rate=1.0,
        bin_size=None,
        window_size=None,
        symmetrize=True,
        use_cupy=False
):
    if use_cupy:
        correlograms = cp_spike_correlations(neurons, neuron_inds, sample_rate=sample_rate, bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
    else:
        correlograms = np_spike_correlations(neurons, neuron_inds, sample_rate=sample_rate, bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
    return correlograms