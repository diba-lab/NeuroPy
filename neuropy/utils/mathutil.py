import numpy as np
import math
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from scipy import stats
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from .ccg import correlograms
import scipy.signal as sg
import typing


def choose_elementwise(x, y, condition):
    assert type(x) == type(y), "Both inputs should have same type"
    assert type(condition) is np.ndarray, "condition should be a boolean array"

    if type(x) is np.ndarray:
        assert x.shape == y.shape, "Input arrays should have same shape"
        out = np.zeros_like(x)
        out[..., condition] = x[..., condition]
        out[..., ~condition] = y[..., ~condition]
    else:
        try:
            out = [x if cond else y for (x, y, cond) in zip(x, y, condition)]
        except:
            raise TypeError("Inpvalid inputs")

    return out


def gaussian_kernel1D(sigma, bin_size, truncate=4.0):
    """Get a gaussian kernel

    Parameters
    ----------
    sigma : float
        standard deviation of the kernel
    bin_size : float
        bin size of the kernel
    truncate: float
        limit kernel to this standard deviation,default = 4.0

    Returns
    -------
    np.array
        gaussian kernel
    """
    t_gauss = np.arange(-truncate * sigma, truncate * sigma, bin_size)
    gaussian = np.exp(-(t_gauss**2) / (2 * sigma**2))
    gaussian /= np.sum(gaussian)
    return gaussian


def min_max_scaler(x, axis=-1):
    """Scales the values x to lie between 0 and 1 along the specified axis

    Parameters
    ----------
    x : np.array
        numpy ndarray

    Returns
    -------
    np.array
        scaled array


    ERRORS:
        2023-03-02 - ValueError: zero-size array to reduction operation minimum which has no identity
            Occurs when np.shape(x) is (0,)


    """
    min_val = np.min(x, axis=axis, keepdims=True)
    range_val = np.ptp(x, axis=axis, keepdims=True)
    return (x - min_val)  / np.where(range_val == 0,1,range_val)


def min_max_external_scaler(x, xmin, xptp):
    """Scales the values of x according to a specified min value and peak-to-peak values.
    Cousin to min_max_scaler, useful for comparing firing rates across different conditions

    Parameters
    ----------
    x: np.array
    xmin: np.array which matches the shape of x in one direction and is shape 1 in the other direction
    xptp: same as xmin but specifying the range of the data

    Returns
    -------
    scaled np.array
    """

    # Check to make sure xmin and xptp are broadcastable to x.
    req_shape = (x.shape[0],1)
    if xmin.ndim == 1 and xmin.shape[0] == x.shape[0]:
        xmin = xmin[:,np.newaxis]

    if xptp.ndim == 1 and xptp.shape[0] == x.shape[0]:
        xptp = xptp[:,np.newaxis]

    if xmin.shape != req_shape:
        raise ValueError(f"Array with shape {xmin.shape} cannot be made broadcastable to {req_shape}.")

    if xptp.shape != req_shape:
        raise ValueError(f"Array with shape {xptp.shape} cannot be made broadcastable to {req_shape}.")

    return (x - xmin) / xptp


def bounded(v: float, vmin: float = -np.inf, vmax: float = np.inf) -> float:
    """Returns the value bounded between two optional lower and upper bounds.
    It clips to the bounds.

    Usage:
        from neuropy.utils.mathutil import bounded

        v_bounded = bounded(value, vmin=0.0, vmax=1.0)
    
    Usage 2:
        value = [-150, -65, -0.9, 0, 0.9, 65, 150]

        bounded(value, vmin=0.0, vmax=1.0) # array([0. , 0. , 0. , 0. , 0.9, 1. , 1. ])
        bounded(value, vmin=-1.0, vmax=1.0) # array([-1. , -1. , -0.9,  0. ,  0.9,  1. ,  1. ])

        
    """
    if np.isscalar(v):
        if np.isnan(v):
            return np.nan  # Return NaN directly if v is NaN
        return min(max(v, vmin), vmax)  # No need for numpy functions for scalar
    
    # If v is array-like
    v = np.array(v)
    v = np.clip(v, vmin, vmax)
    return v

    
def map_to_fixed_range(lin_pos, x_min:float=0.0, x_max:float=1.0):
    # Normalize lin_pos to the range [0, 1]
    lin_pos_normalized = (lin_pos - np.nanmin(lin_pos)) / (np.nanmax(lin_pos) - np.nanmin(lin_pos))
    # Scale the normalized lin_pos to the range [x_min, x_max]
    mapped_pos = lin_pos_normalized * (x_max - x_min) + x_min
    return mapped_pos

# @function_attributes(short_name=None, tags=['map','values','transform'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-31 16:53', related_items=[])
def map_value(value, from_range: tuple[float, float], to_range: tuple[float, float]):
    """ Maps values from a range `from_low_high_tuple`: (a, b) to a new range `to_low_high_tuple`: (A, B). Similar to arduino's `map(value, fromLow, fromHigh, toLow, toHigh)` function
    
    Usage:
        from neuropy.utils.mathutil import map_value
        track_change_mapped_idx = map_value(track_change_time, (Flat_epoch_time_bins_mean[0], Flat_epoch_time_bins_mean[-1]), (0, (num_epochs-1)))
        track_change_mapped_idx
        
    Example 2: Defining Shortcut mapping
        map_value_time_to_epoch_idx_space = lambda v: map_value(v, (Flat_epoch_time_bins_mean[0], Flat_epoch_time_bins_mean[-1]), (0, (num_epochs-1))) # same map

    """
    # Calculate the ratio of the input value relative to the input range
    from_low, from_high = from_range
    to_low, to_high = to_range
    ratio = (value - from_low) / (from_high - from_low)
    # Map the ratio to the output range
    mapped_value = to_low + (to_high - to_low) * ratio
    # Return the mapped value
    return mapped_value


def compute_grid_bin_bounds(*args):
    """ computes the (min, max) bound for each passed array and returns a tuple of these (min, max) tuples. 
    from neuropy.utils.mathutil import compute_grid_bin_bounds
    
    grid_bin_bounds: `((x_min, x_max), (y_min, y_max), ...)
    
    """
    grid_bin_bounds = []
    for data in args:
        if data is not None:
            bounds = (np.nanmin(data), np.nanmax(data))
        else:
            bounds = None # append None to the array
        grid_bin_bounds.append(bounds)
    return tuple(grid_bin_bounds)



def cdf(x, bins):
    """Returns cummulative distribution for x at bins"""
    return np.cumsum(np.histogram(x, bins, density=True)[0])


def getICA_Assembly(x):
    """extracting statisticaly independent components from significant eigenvectors as detected using Marcenko-Pasteur distributionvinput = Matrix  (m x n) where 'm' are the number of cells and 'n' time bins ICA weights thus extracted have highiest weight positive (as done in Gido M. van de Ven et al. 2016) V = ICA weights for each neuron in the coactivation (weight having the highiest value is kept positive) M1 =  originally extracted neuron weights

    Arguments:
        x {[ndarray]} -- [an array of size n * m]

    Returns:
        [type] -- [Independent assemblies]
    """

    zsc_x = stats.zscore(x, axis=1)

    corrmat = (zsc_x @ zsc_x.T) / x.shape[1]

    lambda_max = (1 + np.sqrt(1 / (x.shape[1] / x.shape[0]))) ** 2
    eig_val, eig_mat = np.linalg.eigh(corrmat)
    get_sigeigval = np.where(eig_val > lambda_max)[0]
    n_sigComp = len(get_sigeigval)
    pca_fit = PCA(n_components=n_sigComp, whiten=False).fit_transform(x)

    ica_decomp = FastICA(n_components=None, whiten=False).fit(pca_fit)
    W = ica_decomp.components_
    V = eig_mat[:, get_sigeigval] @ W.T

    return V


def threshPeriods(sig, lowthresh=1, highthresh=2, minDistance=30, minDuration=50):

    ThreshSignal = np.diff(np.where(sig > lowthresh, 1, 0))
    start = np.where(ThreshSignal == 1)[0]
    stop = np.where(ThreshSignal == -1)[0]

    if start[0] > stop[0]:
        stop = stop[1:]
    if start[-1] > stop[-1]:
        start = start[:-1]

    firstPass = np.vstack((start, stop)).T

    # ===== merging close events
    secondPass = []
    event = firstPass[0]
    for i in range(1, len(firstPass)):
        if firstPass[i, 0] - event[1] < minDistance:
            # Merging states
            event = [event[0], firstPass[i, 1]]
        else:
            secondPass.append(event)
            event = firstPass[i]

    secondPass.append(event)
    secondPass = np.asarray(secondPass)
    event_duration = np.diff(secondPass, axis=1).squeeze()

    # delete very short events
    shortevents = np.where(event_duration < minDuration)[0]
    thirdPass = np.delete(secondPass, shortevents, 0)
    event_duration = np.delete(event_duration, shortevents)

    # keep only events with peak above highthresh
    fourthPass = []
    # peakNormalizedPower, peaktime = [], []
    for i in range(len(thirdPass)):
        maxValue = max(sig[thirdPass[i, 0] : thirdPass[i, 1]])
        if maxValue >= highthresh:
            fourthPass.append(thirdPass[i])
            # peakNormalizedPower.append(maxValue)
            # peaktime.append(
            #     [
            #         secondPass[i, 0]
            #         + np.argmax(zscsignal[secondPass[i, 0] : secondPass[i, 1]])
            #     ]
            # )

    return np.asarray(fourthPass)


def _unpack_args(values, fs=1):
    """Parsing argument for thresh_epochs"""
    try:
        val_min, val_max = values
    except (TypeError, ValueError):
        val_min, val_max = (values, None)

    val_min = val_min * fs
    val_max = val_max * fs if val_max is not None else None

    return val_min, val_max


def thresh_epochs(arr: np.ndarray, thresh, length, sep=0, boundary=0, fs=1):
    hmin, hmax = _unpack_args(thresh)  # does not need fs
    lmin, lmax = _unpack_args(length, fs=fs)
    sep = sep * fs + 1e-6

    assert hmin >= boundary, "boundary must be smaller than min thresh"

    arr_thresh = np.where(arr >= boundary, arr, 0)
    peaks, props = sg.find_peaks(arr_thresh, height=[hmin, hmax], prominence=0)

    starts, stops = props["left_bases"], props["right_bases"]
    peaks_values = arr_thresh[peaks]

    # ----- merge overlapping epochs ------
    n_epochs = len(starts)
    ind_delete = []
    for i in range(n_epochs - 1):
        if (starts[i + 1] - stops[i]) < sep:
            # stretch the second epoch to cover the range of both epochs
            starts[i + 1] = min(starts[i], starts[i + 1])
            stops[i + 1] = max(stops[i], stops[i + 1])

            peaks_values[i + 1] = max(peaks_values[i], peaks_values[i + 1])
            peaks[i + 1] = [peaks[i], peaks[i + 1]][
                np.argmax([peaks_values[i], peaks_values[i + 1]])
            ]
            ind_delete.append(i)

    epochs_arr = np.vstack((starts, stops, peaks, peaks_values)).T
    epochs_arr = np.delete(epochs_arr, ind_delete, axis=0)

    # ----- duration thresholds ------
    epochs_length = epochs_arr[:, 1] - epochs_arr[:, 0]
    if lmax is None:
        lmax = epochs_length.max()
    ind_keep = (epochs_length >= lmin) & (epochs_length <= lmax)

    starts, stops, peaks, peaks_values = epochs_arr[ind_keep, :].T

    return starts / fs, stops / fs, peaks / fs, peaks_values


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index. Taken directly from stackoverflow:
    https://stackoverflow.com/questions/4494404/find-large-number-of-
    consecutive-values-fulfilling-condition-in-a-numpy-array
        
    Usage:
        from neuropy.utils.mathutil import contiguous_regions
        idnan = mathutil.contiguous_regions(np.isnan(x))  # identify missing data points

            for ids in idnan:
                missing_ids = range(ids[0], ids[-1])
                bracket_ids = ids + [-1, 0]
                xgood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], x[bracket_ids])
                ygood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], y[bracket_ids])
                zgood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], z[bracket_ids])

    """

    # Find the indices of changes in "condition"
    d = np.diff(condition)
    (idx,) = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def schmitt_threshold(arr: np.array, low_thresh: float, high_thresh: float):
    """Detect high and low states in an array using two thresholds (Schmitt trigger). Works best for bimodal data.

    Parameters
    ----------
    arr : np.array
        array for threshold detection
    low_thresh : float
        low threshold
    high_thresh : float
        high threshold

    Returns
    -------
    logical array
        array of 0(low state) and 1(high state)
    """
    states = np.zeros_like(arr)
    first_low = np.where(arr <= low_thresh)[0][0]
    first_high = np.where(arr >= high_thresh)[0][0]
    first_ind = np.min([first_low, first_high])
    states[:first_ind] = np.argmin([first_low, first_high])
    current_state = states[first_ind - 1]

    for i in range(first_ind, len(arr)):
        if arr[i] >= high_thresh:
            current_state = 1
        if arr[i] <= low_thresh:
            current_state = 0

        states[i] = current_state

    return states


def bimodal_classify(
    arr,
    ret_params=False,
    threshold_type: typing.Literal["default", "schmitt"] = "default",
):
    """Gaussian classification of features into two components.

    Parameters
    ----------
    arr : 1D or 2D array
        features to be classified.
    ret_params : bool, optional
        Gaussian fit parameters are retured if true , by default False
    plot : bool, optional
        _description_, by default False
    ax : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    assert arr.ndim < 2, "Only 1D data accepted"

    clus = GaussianMixture(
        n_components=2, init_params="k-means++", max_iter=200, n_init=10
    ).fit(arr[:, None])
    labels = clus.predict(arr[:, None])
    clus_means = clus.means_.squeeze()

    # --- order cluster labels by increasing mean (low=0, high=1) ------
    sort_idx = np.argsort(clus_means)
    label_map = np.zeros_like(sort_idx)
    label_map[sort_idx] = np.arange(len(sort_idx))
    fixed_labels = label_map[labels.astype("int")]
    means = clus_means[sort_idx]
    covs = clus.covariances_.squeeze()[sort_idx]
    weights = clus.weights_[sort_idx]

    if threshold_type == "schmitt":
        bins = np.linspace(arr.min(), arr.max(), 200)
        full_fit = np.zeros_like(bins)
        for i in range(2):
            full_fit += stats.norm.pdf(bins, means[i], np.sqrt(covs[i])) * weights[i]

        bins_between_peaks = (bins > means[0]) & (bins < means[1])
        thresh_ind = np.argmin(full_fit[bins_between_peaks])
        thresh_val = bins[bins_between_peaks][thresh_ind]
        low_thresh = (means[0] + thresh_val) / 2
        high_thresh = (means[1] + thresh_val) / 2

        fixed_labels = schmitt_threshold(arr, low_thresh, high_thresh)

    if ret_params:
        params_dict = dict(
            weights=weights,
            means=means,
            covariances=covs,
        )
        return fixed_labels, params_dict
    else:
        return fixed_labels


def hmmfit1d(Data, ret_means=False, **kwargs):
    # hmm states on 1d data and returns labels with highest mean = highest label
    flag = None
    if np.isnan(Data).any():
        nan_indices = np.where(np.isnan(Data) == 1)[0]
        non_nan_indices = np.where(np.isnan(Data) == 0)[0]
        Data_og = Data
        Data = np.delete(Data, nan_indices)
        hmmlabels = np.nan * np.ones(len(Data_og))
        flag = 1

    Data = (np.asarray(Data)).reshape(-1, 1)
    models = []
    scores = []
    for i in range(10):
        model = GaussianHMM(n_components=2, n_iter=10, random_state=i, **kwargs)
        model.fit(Data)
        models.append(model)
        scores.append(model.score(Data))
    model = models[np.argmax(scores)]

    hidden_states = model.predict(Data)
    mus = np.squeeze(model.means_)
    sigmas = np.squeeze(np.sqrt(model.covars_))
    transmat = np.array(model.transmat_)

    idx = np.argsort(mus)
    mus = mus[idx]
    sigmas = sigmas[idx]
    transmat = transmat[idx, :][:, idx]

    state_dict = {}
    states = [i for i in range(4)]
    for i in idx:
        state_dict[idx[i]] = states[i]

    relabeled_states = np.asarray([state_dict[h] for h in hidden_states])
    relabeled_states[:2] = [0, 0]
    relabeled_states[-2:] = [0, 0]

    if flag:
        hmmlabels[non_nan_indices] = relabeled_states

    else:
        hmmlabels = relabeled_states

    if ret_means:
        return hmmlabels, mus
    else:
        return hmmlabels


def eventpsth(ref, event, fs, quantparam=None, binsize=0.01, window=1, nQuantiles=1):
    """psth of 'event' with respect to 'ref'

    Parameters
    ----------
    ref (array):
        1-D array of timings of reference event in seconds
    event (1D array):
        timings of events whose psth will be calculated
    fs:
        sampling rate
    quantparam (1D array):
        values used to divide 'ref' into quantiles
    binsize (float, optional):
        [description]. Defaults to 0.01.
    window (int, optional):
        [description]. Defaults to 1.
    nQuantiles (int, optional):
        [description]. Defaults to 10.

    Returns
    -------
        [type]: [description]
    """

    ref = np.asarray(ref)
    event = np.asarray(event)

    if quantparam is not None:
        assert len(event) == len(quantparam), print(
            "length of quantparam must be same as ref events"
        )
        quantiles = pd.qcut(quantparam, nQuantiles, labels=False)

        quants, eventid = [], []
        for category in range(nQuantiles):
            indx = np.where(quantiles == category)[0]
            quants.append(ref[indx])
            eventid.append(category * np.ones(len(indx)).astype(int))

        quants.append(event)
        eventid.append(((nQuantiles + 1) * np.ones(len(event))).astype(int))

        quants = np.concatenate(quants)
        eventid = np.concatenate(eventid)
    else:
        quants = np.concatenate((ref, event))
        eventid = np.concatenate([np.ones(len(ref)), 2 * np.ones(len(event))]).astype(
            int
        )

    sort_ind = np.argsort(quants)

    ccg = correlograms(
        quants[sort_ind],
        eventid[sort_ind],
        sample_rate=fs,
        bin_size=binsize,
        window_size=window,
    )

    return ccg[:-1, -1, :]


def gini(arr, eps=1e-8):
    """
    Calculate the Gini coefficient of a numpy array.

    Source: PyGini
    https://github.com/mckib2/pygini/blob/master/pygini/gini.py

    -----
    Based on bottom eq on [2]_.
    References
    ----------
    .. [2]_ http://www.statsdirect.com/help/
            default.htm#nonparametric_methods/gini.htm
    """

    # All values are treated equally, arrays must be 1d and > 0:
    arr = np.abs(arr).flatten() + eps

    # Values must be sorted:
    arr = np.sort(arr)

    # Index per array element:
    index = np.arange(1, arr.shape[0] + 1)

    # Number of array elements:
    N = arr.shape[0]

    # Gini coefficient:
    return (np.sum((2 * index - N - 1) * arr)) / (N * np.sum(arr))
