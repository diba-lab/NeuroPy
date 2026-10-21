import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.interpolate import interp1d

from .. import core
from ..utils.mathutil import contiguous_regions, thresh_epochs

def linearize_position(position: core.Position, sample_sec=3, method="isomap", sigma=2, dimensions=["x", "y"],circular=False):
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
    circular: bool, optional
        If True, assumes the track is circular and uses polar coordinates for linearization.

    Returns
    -------
    core.Position
        A new Position object with linearized traces.
    """
    # Extract the specified dimensions
    pos_components = []
    for dim in dimensions:
        if hasattr(position, dim):
            pos_components.append(getattr(position, dim))
        else:
            raise ValueError(f"Dimeinos '{dim}' not found in the position object.")

    # Combined dimensions
    pos_array = np.vstack(pos_components).T

    xlinear = None

    if circular: #will have issues if x/y have nans. interpolate before sending in.

        #need to get xlim,ylim, set 0 as the center of those
        x, y = pos_array[:, 0], pos_array[:, 1]
        xcenter = np.nanmean([np.nanmin(x),np.nanmax(x)])
        ycenter = np.nanmean([np.nanmin(y),np.nanmax(y)])

        x = x-xcenter
        y = y-ycenter

        theta = np.arctan2(y,x);
        theta[theta < 0] += 2*np.pi; #have all between 0 and 2pi

        #theta_valid = theta[~np.isnan(theta)]
        #theta_unwrapped = np.unwrap(theta_valid)
        #theta[~np.isnan(theta)] = theta_unwrapped
    
        xlinear = theta

    else:

        if method.lower() == "pca":
            pca = PCA(n_components=1)
            xlinear = pca.fit_transform(pos_array).squeeze()
        elif method.lower() == "isomap":
            imap = Isomap(n_neighbors=5, n_components=2)
            
            # Eliminate any NaNs present in data
            nan_bool = np.bitwise_or(np.isnan(pos_array[:, 0]), np.isnan(pos_array[:, 1]))
            pos_array_good = pos_array[~nan_bool, :]

            # Downsample points to reduce memory load and time
            pos_ds = pos_array[0 : -1 : np.round(int(position.sampling_rate) * sample_sec)]
            imap.fit(pos_ds)
            iso_pos = imap.transform(pos_array)

            # Keep iso_pos here in case we want to use 2nd dimension (transverse to track)
            if iso_pos.std(axis=0)[0] < iso_pos.std(axis=0)[1]:
                iso_pos[:, [0,1]] = iso_pos[:, [1,0]]
            xlinear = iso_pos[:,0]

        xlinear = gaussian_filter1d(xlinear, sigma=sigma)
        xlinear -= np.min(xlinear)
        xlinear_good = np.ones_like(nan_bool)*np.nan
        xlinear_good[~nan_bool] = xlinear       

        return core.Position(
            traces=xlinear, t_start=position.t_start, sampling_rate=position.sampling_rate
        )

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
    data["label"] = np.where(val > 0, "right", "left")
    data["peak_time"] = peak_time + position.t_start
    data["peak_speed"] = peak_speed

    return core.Epoch(epochs=data, metadata=metadata)

if __name__ == "__main__":
    pass




