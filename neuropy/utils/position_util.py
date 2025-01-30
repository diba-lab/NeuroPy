import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from .. import core
from neuropy.utils.mathutil import contiguous_regions, threshPeriods, compute_grid_bin_bounds, map_value
from neuropy.utils.mixins.binning_helpers import compute_spanning_bins

from enum import Enum


class RegularizationApproach(Enum):
    """Docstring for RegularizationApproach."""
    RAW_VALUES = "raw_values"
    SUBTRACT_MIN = "subtract_min"
    RESTORE_X_RANGE = "restore_x_range" # restores the original range of the x values after performing the linearization.
    

def linearize_position_df(pos_df: pd.DataFrame, sample_sec=3, method="isomap", sigma=2, override_position_sampling_rate_Hz=None, regularization_approach:RegularizationApproach=RegularizationApproach.RAW_VALUES):
    """linearize trajectory. Use method='PCA' for off-angle linear track, method='ISOMAP' for any non-linear track.
    ISOMAP is more versatile but also more computationally expensive.

    Parameters
    ----------
    track_names: list of track names, each must match an epoch in epochs class.
    sample_sec : int, optional
        sample a point every sample_sec seconds for training ISOMAP, by default 3. Lower it if inaccurate results
    method : str, optional
        by default 'ISOMAP' (for any continuous track, untested on t-maze as of 12/22/2020) or
        'PCA' (for straight tracks)
    override_position_sampling_rate_Hz: float, optional - ignored except for method="isomap". If provided, used for downsampling dataframe prior to computation. Otherwise sampling rate is approximated from pos_df['t'] column.
    
    Modifies:
        Adds the 'lin_pos' column to the provided position dataframe.
    """    
    xy_pos = pos_df[['x','y']].to_numpy()
    
    xlinear = None
    if method.lower() == "pca":
        pca = PCA(n_components=1)
        xlinear = pca.fit_transform(xy_pos).squeeze()
    elif method.lower() == "isomap":
        imap = Isomap(n_neighbors=5, n_components=2)
        # downsample points to reduce memory load and time
        if override_position_sampling_rate_Hz is not None:
            position_sampling_rate_Hz = override_position_sampling_rate_Hz
        else:
            # compute sampling rate from the 't' column:
            assert 't' in pos_df.columns
            position_sampling_rate_Hz = 1.0 / np.nanmean(np.diff(pos_df['t'].to_numpy())) # In Hz, returns 29.969777
        num_end_samples = np.round(int(position_sampling_rate_Hz) * sample_sec)
        pos_ds = xy_pos[0:-1:num_end_samples]
        imap.fit(pos_ds)
        iso_pos = imap.transform(xy_pos)
        # Keep iso_pos here in case we want to use 2nd dimension (transverse to track) in future...
        if iso_pos.std(axis=0)[0] < iso_pos.std(axis=0)[1]:
            iso_pos[:, [0, 1]] = iso_pos[:, [1, 0]]
        xlinear = iso_pos[:, 0]
    else:
        print('ERROR: invalid method name: {}'.format(method))
        
    if (sigma is not None) and (sigma > 0.0):
        xlinear = gaussian_filter1d(xlinear, sigma=sigma) # smooth
        
    if regularization_approach.name == RegularizationApproach.SUBTRACT_MIN.name:
        xlinear -= np.min(xlinear) # required to prevent mapping to negative values
    elif regularization_approach.name == RegularizationApproach.RESTORE_X_RANGE.name:
        xlinear = -1.0 * xlinear # flip over the y-axis first
        lin_pos_bounds = compute_grid_bin_bounds(xlinear)[0]
        x_bounds = compute_grid_bin_bounds(pos_df['x'].to_numpy())[0]
        # print(f'lin_pos_bounds: {lin_pos_bounds}, x_bounds: {x_bounds}')
        xlinear = map_value(xlinear, lin_pos_bounds, x_bounds) # map xlinear from its current bounds range to the xbounds range
    else:
        assert regularization_approach.name == RegularizationApproach.RAW_VALUES.name, f"Invalid regularization approach!"
    pos_df['lin_pos'] = xlinear # add the linearized position to the dataframe as the 'lin_pos' column
    return pos_df


def linearize_position(position: core.Position, sample_sec=3, method="isomap", sigma=2, **kwargs) -> core.Position:
    """linearize trajectory. Use method='PCA' for off-angle linear track, method='ISOMAP' for any non-linear track.
    ISOMAP is more versatile but also more computationally expensive.

    Parameters
    ----------
    position: core.Position
        Position object containing spatial information
    sample_sec: int, optional
        Sample a point every sample_sec seconds for training ISOMAP, by default 3. Lower it if inaccurate results.
    method: str, optional
        by default 'ISOMAP' (for any continuous track, untested on t-maze as of 12/22/2020) or
        'PCA' (for straight tracks)
    sigma: int, optional
        Gaussian filter smoothing parameter, by default 2.
    dimensions: list, optional
        List of spatial dimensions to use, by default ["x", "y"].

    Returns
    -------
    core.Position
        A new Position object with linearized traces.
    """
    if isinstance(position, pd.DataFrame):
        pos_df = linearize_position_df(position, sample_sec=sample_sec, method=method, sigma=sigma, override_position_sampling_rate_Hz=None, **kwargs)
        xlinear = pos_df['lin_pos'].to_numpy()
        return core.Position.from_separate_arrays(pos_df['t'].to_numpy(), xlinear, lin_pos=xlinear, metadata=None)
    else:
        pos_df = position.to_dataframe() # convert from a Position object
        pos_df = linearize_position_df(pos_df, sample_sec=sample_sec, method=method, sigma=sigma, override_position_sampling_rate_Hz=position.sampling_rate, **kwargs)
        xlinear = pos_df['lin_pos'].to_numpy()
        return core.Position.from_separate_arrays(position.time, xlinear, lin_pos=xlinear, metadata=position.metadata)


# Run Direction calculation from main
def run_direction(
    position: core.Position,
    speed_thresh=(20, None),
    boundary=8.0,
    duration=(0.5, None),
    sep=1,
    min_distance=10,
    sigma=0.1,
):
    """Divide running epochs into up (increasing values) and down (decreasing values).
    Currently only works for one dimensional position data

    Parameters
    ----------
    speed_thresh : tuple, optional
        low and high speed threshold for speed, by default (10, 20) in cm/s
    boundary: float
        boundaries of epochs are extended to this value, in cm/s
    duration : int, optional
        min and max duration of epochs, in seconds
    sep: int, optional
        epochs separated by less than this many seconds will be merged
    min_distance : int, optional
        the animal should cover this much distance in one direction within the lap to be included, by default 50 cm
    sigma : int, optional
        speed is smoothed, increase if epochs are fragmented, by default 10
    plot : bool, optional
        plots the epochs with position and speed data, by default True
    """

    metadata = locals()
    metadata.pop("position")
    assert position.ndim == 1, "Run direction only supports one dimensional position"

    sampling_rate = position.sampling_rate
    dt = 1 / sampling_rate
    x = position.x
    speed = gaussian_filter1d(position.speed, sigma=sigma / dt)

    starts, stops, peak_time, peak_speed = thresh_epochs(
        arr=speed,
        thresh=speed_thresh,
        length=duration,
        sep=sep,
        boundary=boundary,
        fs=sampling_rate,
    )

    high_speed = np.vstack((starts, stops)).T
    high_speed = high_speed * sampling_rate  # convert to index locations
    val = []
    for epoch in high_speed.astype("int"):
        displacement = x[epoch[1]] - x[epoch[0]]
        # distance = np.abs(np.diff(x[epoch[0] : epoch[1]])).sum()

        if np.abs(displacement) > min_distance:
            if displacement < 0:
                val.append(-1)
            elif displacement > 0:
                val.append(1)
        else:
            val.append(0)
    val = np.asarray(val)

    # ---- deleting epochs where animal ran a little distance------
    ind_keep = val != 0
    high_speed = high_speed[ind_keep, :]
    val = val[ind_keep]
    peak_time = peak_time[ind_keep]
    peak_speed = peak_speed[ind_keep]

    high_speed = np.around(high_speed / sampling_rate + position.t_start, 2)
    data = pd.DataFrame(high_speed, columns=["start", "stop"])
    data["label"] = np.where(val > 0, "up", "down")
    data["peak_time"] = peak_time + position.t_start
    data["peak_speed"] = peak_speed

    return core.Epoch(epochs=data, metadata=metadata)


def calculate_run_direction(
    position: core.Position,
    speedthresh=(10, 20),
    merge_dur=2,
    min_dur=2,
    smooth_speed=50,
    min_dist=50,
):
    """Divide running epochs into up (increasing values) and down (decreasing values).
    Currently only works for one dimensional position data

    Parameters
    ----------
    speedthresh : tuple, optional
        low and high speed threshold for speed, by default (10, 20)
    merge_dur : int, optional
        two epochs if less than merge_dur (seconds) apart they will be merged , by default 2 seconds
    min_dur : int, optional
        minimum duration of a run epoch, by default 2 seconds
    smooth_speed : int, optional
        speed is smoothed, increase if epochs are fragmented, by default 50
    min_dist : int, optional
        the animal should cover this much distance in one direction within the lap to be included, by default 50
    plot : bool, optional
        plots the epochs with position and speed data, by default True
    """

    assert position.ndim == 1, "Run direction only supports one dimensional position"

    trackingsampling_rate = position.time
    posdata = position.to_dataframe()

    posdata = posdata[(posdata.time > period[0]) & (posdata.time < period[1])]
    x = posdata.linear
    time = posdata.time
    speed = posdata.speed
    speed = gaussian_filter1d(posdata.speed, sigma=smooth_speed)

    high_speed = threshPeriods(
        speed,
        lowthresh=speedthresh[0],
        highthresh=speedthresh[1],
        minDistance=merge_dur * trackingsampling_rate,
        minDuration=min_dur * trackingsampling_rate,
    )

    high_speed = np.vstack((starts, stops)).T
    high_speed = high_speed * sampling_rate  # convert to index locations
    val = []
    for epoch in high_speed.astype("int"):
        displacement = x[epoch[1]] - x[epoch[0]]
        # distance = np.abs(np.diff(x[epoch[0] : epoch[1]])).sum()

        if np.abs(displacement) > min_dist:
            if displacement < 0:
                val.append(-1)
            elif displacement > 0:
                val.append(1)
        else:
            val.append(0)
    val = np.asarray(val)

    # ---- deleting epochs where animal ran a little distance------
    ind_keep = val != 0
    high_speed = high_speed[ind_keep, :]
    val = val[ind_keep]
    peak_time = peak_time[ind_keep]
    peak_speed = peak_speed[ind_keep]

    high_speed = np.around(high_speed / trackingsampling_rate + period[0], 2)
    data = pd.DataFrame(high_speed, columns=["start", "stop"])
    # data["duration"] = np.diff(high_speed, axis=1)
    data["direction"] = np.where(val > 0, "forward", "backward")

    self.epochs = run_epochs

    return run_epochs


# def calculate_run_epochs(
#     position: core.Position,
#     speedthresh=(10, 20),
#     merge_dur=2,
#     min_dur=2,
#     smooth_speed=50,
# ):
#     """Divide running epochs into forward and backward.
#     Currently only works for one dimensional position data
#
#     Parameters
#     ----------
#     speedthresh : tuple, optional
#         low and high speed threshold for speed, by default (10, 20)
#     merge_dur : int, optional
#         two epochs if less than merge_dur (seconds) apart they will be merged , by default 2 seconds
#     min_dur : int, optional
#         minimum duration of a run epoch, by default 2 seconds
#     smooth_speed : int, optional
#         speed is smoothed, increase if epochs are fragmented, by default 50
#     min_dist : int, optional
#         the animal should cover this much distance in one direction within the lap to be included, by default 50
#     plot : bool, optional
#         plots the epochs with position and speed data, by default True
#     """
#
#     sampling_rate = position.sampling_rate
#
#     x = position.x
#     time = position.time
#     speed = position.speed
#     speed = gaussian_filter1d(position.speed, sigma=smooth_speed)
#
#     high_speed = threshPeriods(
#         speed,
#         lowthresh=speedthresh[0],
#         highthresh=speedthresh[1],
#         minDistance=merge_dur * sampling_rate,
#         minDuration=min_dur * sampling_rate,
#     )
#     val = []
#     for epoch in high_speed:
#         displacement = x[epoch[1]] - x[epoch[0]]
#         # distance = np.abs(np.diff(x[epoch[0] : epoch[1]])).sum()
#
#         if np.abs(displacement) > min_dist:
#             if displacement < 0:
#                 val.append(-1)
#             elif displacement > 0:
#                 val.append(1)
#         else:
#             val.append(0)
#     val = np.asarray(val)
#
#     # ---- deleting epochs where animal ran a little distance------
#     high_speed = np.delete(high_speed, np.where(val == 0)[0], axis=0)
#     val = np.delete(val, np.where(val == 0)[0])
#
#     high_speed = np.around(high_speed / sampling_rate + period[0], 2)
#     data = pd.DataFrame(high_speed, columns=["start", "stop"])
#     # data["duration"] = np.diff(high_speed, axis=1)
#     data["direction"] = np.where(val > 0, "forward", "backward")
#
#     return run_epochs


def compute_position_grid_size(*any_1d_series, num_bins:tuple):
    """  Computes the required bin_sizes from the required num_bins (for each dimension independently)
    Usage:
    out_grid_bin_size, out_bins, out_bins_infos = compute_position_grid_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64))
    active_grid_bin = tuple(out_grid_bin_size)
    print(f'active_grid_bin: {active_grid_bin}') # (3.776841861770752, 1.043326930905373)
    
    History:
        Extracted from pyphocorehelpers.indexing_helpers import compute_position_grid_size for use in BaseDataSessionFormats
    
    """
    assert (len(any_1d_series)) == len(num_bins), f'(len(other_1d_series)) must be the same length as the num_bins tuple! But (len(other_1d_series)): {(len(any_1d_series))} and len(num_bins): {len(num_bins)}!'
    num_series = len(num_bins)
    out_bins = []
    out_bins_info = []
    out_bin_grid_step_size = np.zeros((num_series,))

    for i in np.arange(num_series):
        xbins, xbin_info = compute_spanning_bins(any_1d_series[i], num_bins=num_bins[i])
        out_bins.append(xbins)
        out_bins_info.append(xbin_info)
        out_bin_grid_step_size[i] = xbin_info.step

    return out_bin_grid_step_size, out_bins, out_bins_info

