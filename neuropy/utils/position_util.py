import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.ndimage import gaussian_filter1d

from .. import core
from ..utils.mathutil import contiguous_regions, threshPeriods


def linearize_position(position: core.Position, sample_sec=3, method="isomap", sigma=2):
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

    """
    xpos = position.x
    ypos = position.y

    xy_pos = np.vstack((xpos, ypos)).T
    xlinear = None
    if method == "pca":
        pca = PCA(n_components=1)
        xlinear = pca.fit_transform(xy_pos).squeeze()
    elif method == "isomap":
        imap = Isomap(n_neighbors=5, n_components=2)
        # downsample points to reduce memory load and time
        pos_ds = xy_pos[0 : -1 : np.round(int(position.sampling_rate) * sample_sec)]
        imap.fit(pos_ds)
        iso_pos = imap.transform(xy_pos)
        # Keep iso_pos here in case we want to use 2nd dimension (transverse to track) in future...
        if iso_pos.std(axis=0)[0] < iso_pos.std(axis=0)[1]:
            iso_pos[:, [0, 1]] = iso_pos[:, [1, 0]]
        xlinear = iso_pos[:, 0]

    xlinear = gaussian_filter1d(xlinear, sigma=sigma)
    return core.Position(
        traces=xlinear, t_start=position.t_start, sampling_rate=position.sampling_rate
    )


def calculate_run_direction(
    position: core.Position,
    speedthresh=(10, 20),
    merge_dur=2,
    min_dur=2,
    smooth_speed=50,
    min_dist=50,
):
    """Divide running epochs into forward and backward.
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

    trackingsampling_rate = self._position.time
    posdata = self._position.to_dataframe()

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
    val = []
    for epoch in high_speed:
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
    high_speed = np.delete(high_speed, np.where(val == 0)[0], axis=0)
    val = np.delete(val, np.where(val == 0)[0])

    high_speed = np.around(high_speed / trackingsampling_rate + period[0], 2)
    data = pd.DataFrame(high_speed, columns=["start", "stop"])
    # data["duration"] = np.diff(high_speed, axis=1)
    data["direction"] = np.where(val > 0, "forward", "backward")

    self.epochs = run_epochs

    return run_epochs


def calculate_run_epochs(
    position: core.Position,
    speedthresh=(10, 20),
    merge_dur=2,
    min_dur=2,
    smooth_speed=50,
):
    """Divide running epochs into forward and backward.
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

    sampling_rate = position.sampling_rate

    x = position.x
    time = position.time
    speed = position.speed
    speed = gaussian_filter1d(position.speed, sigma=smooth_speed)

    high_speed = threshPeriods(
        speed,
        lowthresh=speedthresh[0],
        highthresh=speedthresh[1],
        minDistance=merge_dur * sampling_rate,
        minDuration=min_dur * sampling_rate,
    )
    val = []
    for epoch in high_speed:
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
    high_speed = np.delete(high_speed, np.where(val == 0)[0], axis=0)
    val = np.delete(val, np.where(val == 0)[0])

    high_speed = np.around(high_speed / sampling_rate + period[0], 2)
    data = pd.DataFrame(high_speed, columns=["start", "stop"])
    # data["duration"] = np.diff(high_speed, axis=1)
    data["direction"] = np.where(val > 0, "forward", "backward")

    return run_epochs
