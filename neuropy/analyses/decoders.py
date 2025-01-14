import warnings
from typing import Optional, Union
from pathlib import Path
from copy import deepcopy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from nptyping import NDArray
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.special import factorial

from neuropy.analyses.placefields import PfND

from neuropy import core
# from .. import core

from neuropy.utils.mixins.binning_helpers import BinningContainer, BinningInfo # for epochs_spkcount getting the correct time bins
from neuropy.utils.mixins.binning_helpers import build_spanning_grid_matrix # for Decode2d reverse transformations from flat points

from attrs import define, field, Factory

class RadonTransformComputation:
    """
    Helpers for radon transform:

    from neuropy.analyses.decoders import RadonTransformComputation

    """
    @classmethod
    def phi(cls, velocity, time_bin_size, pos_bin_size):
        return np.arctan(velocity * time_bin_size / pos_bin_size)
    
    @classmethod
    def rho(cls, icpt, t_mid, x_mid, velocity, time_bin_size, pos_bin_size):
        phi = cls.phi(velocity=velocity, time_bin_size=time_bin_size, pos_bin_size=pos_bin_size)
        return ((icpt + (velocity * t_mid) - x_mid)/pos_bin_size) * np.sin(phi)
    
    # Conversion functions: ______________________________________________________________________________________________ #
    @classmethod
    def convert_real_space_x_to_index_space_ri(cls, ri_mid, x_mid, pos_bin_size):
        convert_real_space_x_to_index_space_ri = lambda x: (((x - x_mid)/pos_bin_size) + ri_mid)
        return convert_real_space_x_to_index_space_ri

    @classmethod
    def convert_real_time_t_to_index_time_ci(cls, ci_mid, t_mid, time_bin_size):
        convert_real_time_t_to_index_time_ci = lambda t: (((t - t_mid)/time_bin_size) + ci_mid)
        return convert_real_time_t_to_index_time_ci

    # Used in `radon_transform` __________________________________________________________________________________________ #
    @classmethod
    def velocity(cls, phi, time_bin_size, pos_bin_size):
        """ Not working.
        
        """
        return pos_bin_size / (time_bin_size * np.tan(phi)) # 1/np.tan(x) == cot(x)
    
    @classmethod
    def intercept(cls, phi, rho, t_mid, x_mid, time_bin_size, pos_bin_size):
        """ Not working.
            t_mid, x_mid: the continuous-time versions
        """
        return (
            (pos_bin_size * t_mid) / (time_bin_size * np.tan(phi))
            + (rho / np.sin(phi)) * pos_bin_size
            + x_mid
        )

    @classmethod
    def y_line_idxs(cls, phi, rho, ci_mid, ri_mid, ci_mat=None):
        """ 
        returns a lambda function that takes `ci_mat`
        y_line_idxs_fn = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, ci_mid=ci_mid, ri_mid=ri_mid)
        y_line_idxs = y_line_idxs_fn(ci_mat=ci_mat)

        ---

        y_line_idxs = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, ci_mid=ci_mid, ri_mid=ri_mid, ci_mat=ci_mat)

        """
        if ci_mat is None:
            # return a lambda function:
            return lambda ci_mat: np.rint(((rho - (ci_mat - ci_mid) * np.cos(phi)) / np.sin(phi)) + ri_mid).astype("int")
        else:
            # return the literal
            return np.rint(((rho - (ci_mat - ci_mid) * np.cos(phi)) / np.sin(phi)) + ri_mid).astype("int") 
    
    @classmethod
    def y_line(cls, phi, rho, t_mid, x_mid, t_mat=None):
        """
        
        y_line = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, t_mid=t_mid, x_mid=x_mid, t_mat=t_mat)
        """
        if t_mat is None:
            # return a lambda function:
            return lambda t: np.rint(((rho - (t - t_mid) * np.cos(phi)) / np.sin(phi)) + x_mid).astype("int")
        else:
            # return the literal
            return np.rint(((rho - (t_mat - t_mid) * np.cos(phi)) / np.sin(phi)) + x_mid).astype("int") 


        # y_line = ((rho_mat - (ci_mat - ci_mid) * np.cos(phi_mat)) / np.sin(phi_mat)) + ri_mid # (t_mat - ci_mid): makes it not matter whether absolute time bins or time bin indicies were used here:
        y_line = ((rho - (t_mat - t_mid) * np.cos(phi)) / np.sin(phi)) + x_mid
        return np.rint(y_line).astype("int") # (5000, 6) - (nlines, n_t)


    @classmethod
    def compute_score(cls, arr: NDArray, y_line_idxs: NDArray, nlines: int, n_neighbours:int):
        assert np.ndim(arr) >= 2
        n_pos, n_t = np.shape(arr)

        # using convolution to sum neighbours
        arr = np.apply_along_axis(
            np.convolve, axis=0, arr=arr, v=np.ones(2 * n_neighbours + 1), mode="same"
        )

        posterior = np.zeros((nlines, n_t)) # allocate output posterior

        # n_pos = np.shape(arr)[0]
        y_line_idxs = np.rint(y_line_idxs).astype("int")
        # if line falls outside of array in a given bin, replace that with median posterior value of that bin across all positions
        t_out = np.where((y_line_idxs < 0) | (y_line_idxs > (n_pos - 1)))
        t_in = np.where((y_line_idxs >= 0) & (y_line_idxs <= (n_pos - 1)))
        posterior[t_out] = np.median(arr[:, t_out[1]], axis=0)
        posterior[t_in] = arr[y_line_idxs[t_in], t_in[1]]

        # old_settings = np.seterr(all="ignore")
        posterior_mean = np.nanmean(posterior, axis=1)

        best_line_idx: int = np.argmax(posterior_mean)
        score = posterior_mean[best_line_idx]

        # np.seterr(**old_settings)
        return score, best_line_idx, (posterior, posterior_mean, y_line_idxs, (t_in, t_out))



@define(slots=False)
class RadonTransformDebugValue:
    t: NDArray = field()
    n_t: int = field()
    ci_mid: float = field()

    pos: NDArray = field()
    n_pos: int = field()
    ri_mid: float = field()

    diag_len: float = field()

    y_line_idxs: NDArray = field()
    y_line: NDArray = field() # these come back with all elements the same for a given line index? like [73, 73, 73, 73, 73, 73]
    t_out: NDArray = field()
    t_in: NDArray = field()

    posterior: NDArray = field()
    posterior_mean: NDArray = field()
    best_line_idx: int = field()
    best_phi: float = field()
    best_rho: float = field()
    
    ## real world
    time_mid: float = field()
    pos_mid: float = field()

    # @property
    # def n_t(self) -> int:
    #     return len(self.t)
    @property
    def ci(self) -> NDArray:
        """ ci: time indicies """
        return np.arange(self.n_t)

    # @property
    # def n_pos(self) -> int:
    #     return len(self.pos)
    @property
    def ri(self) -> NDArray:
        return np.arange(self.n_pos) # pos: position bin indicies

    @property
    def best_y_line(self) -> NDArray:
        """The best_y_line property."""
        return np.squeeze(self.y_line[self.best_line_idx, :])

    @property
    def best_y_line_idxs(self) -> NDArray:
        """The best_y_line property."""
        return np.squeeze(self.y_line_idxs[self.best_line_idx, :])


def radon_transform(arr: NDArray, nlines:int=10000, dt:float=1, dx:float=1, n_neighbours:int=1, enable_return_neighbors_arr=False, t0: Optional[float]=None, x0: Optional[float]=None):
    """Line fitting algorithm primarily used in decoding algorithm, a variant of radon transform, algorithm based on Kloosterman et al. 2012

    from neuropy.analyses.decoders import radon_transform
    
    Parameters
    ----------
    arr : 2d array
        time axis is represented by columns, position axis is represented by rows
    dt : float
        time binsize in seconds, only used for velocity/intercept calculation
    dx : float
        position binsize in cm, only used for velocity/intercept calculation
    n_neighbours : int,
        probability in each bin is replaced by sum of itself and these many 'neighbours' column wise, default 1 neighbour

    NOTE: when returning velocity the sign is flipped to match with position going from bottom to up

    Returns
    -------
    score:
        sum of values (posterior) under the best fit line
    velocity:
        speed of replay in cm/s
    intercept:
        intercept of best fit line

    References
    ----------
    1) Kloosterman et al. 2012
    """
    if t0 is None:
        t0 = 0.0

    if x0 is None:
        x0 = 0.0
        
    # if time_bin_centers is None:
    #     time_bin_centers = np.arange(arr.shape[1]) # index from [0, ... (NT-1)]
    # else:
    #     assert len(time_bin_centers) == np.shape(arr)[1]
    
    ci: NDArray = np.arange(arr.shape[1]) # ci: time indicies
    t: NDArray = (ci*float(dt)) + t0 # t: time bins in real seconds. When t0 is provided, these appear to be good.
    n_t: int = len(t)
    # ci_mid = (n_t + 1) / 2 - 1 # index space
    ci_mid: float = (float(n_t) / 2.0) # index space
    # time_mid = ((float(n_t) * dt) / 2.0) # real space
    time_mid: float = (t[-1] + t[0]) / 2.0 # real space

    # pos = np.arange(arr.shape[0]) # pos: position bin indicies
    ri: NDArray = np.arange(arr.shape[0]) # pos: position bin indicies
    pos: NDArray = (ri*float(dx)) + x0 # pos: position bin centers. When x0 is provided these perfeclty match `xbin_centers`
    n_pos: int = len(pos)
    # ri_mid = (n_pos + 1) / 2 - 1 # index space
    ri_mid: float = (float(n_pos) / 2.0) # index space
    # pos_mid = ((float(n_pos) * dx) / 2.0) # real space
    pos_mid: float = ((float(pos[-1]) + float(pos[0])) / 2.0) # real space

    diag_len: float = np.sqrt((n_t - 1) ** 2 + (n_pos - 1) ** 2)


    # exclude stationary events by choosing phi little below 90 degree
    # NOTE: angle of line is given by (90-phi), refer Kloosterman 2012
    phi = np.random.uniform(low=(-np.pi / 2), high=(np.pi / 2), size=nlines) # (nlines, )
    rho = np.random.uniform(low=-diag_len / 2, high=diag_len / 2, size=nlines) # (nlines, )

    rho_mat = np.tile(rho, (n_t, 1)).T
    phi_mat = np.tile(phi, (n_t, 1)).T
    
    ci_mat = np.tile(ci, (nlines, 1))
    t_mat = np.tile(t, (nlines, 1))

    # y_line = ((rho_mat - (ci_mat - ci_mid) * np.cos(phi_mat)) / np.sin(phi_mat)) + ri_mid # (t_mat - ci_mid): makes it not matter whether absolute time bins or time bin indicies were used here:
    # y_line_idxs = ((rho_mat - (ci_mat - ci_mid) * np.cos(phi_mat)) / np.sin(phi_mat)) + ri_mid
    # y_line_idxs = np.rint(y_line_idxs).astype("int")

    y_line_idxs = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, ci_mid=ci_mid, ri_mid=ri_mid, ci_mat=ci_mat) # (5000, 6) - (nlines, n_t) - note that the indicies returned can be actually outside the matrix bounds - e.g. negative (not python-wrapping index negative) or larger than the number of position bins
    y_line = RadonTransformComputation.y_line(phi=phi_mat, rho=rho_mat, t_mid=time_mid, x_mid=pos_mid, t_mat=t_mat) # seemingly incorrect

    # y_line = ((rho_mat - (t_mat - time_mid) * np.cos(phi_mat)) / np.sin(phi_mat)) + ri_mid ## 2024-05-07 - This seemed to be working, but it shouldn't be.

    # y_line = ((rho_mat - (t_mat - time_mid) * np.cos(phi_mat)) / np.sin(phi_mat)) + pos_mid
    # y_line = np.rint(y_line).astype("int") # (5000, 6) - (nlines, n_t)

    # old_settings = np.seterr(all="ignore")
    with np.errstate(all="ignore"):
        # posterior_mean = np.nanmean(posterior, axis=1)

        # best_line_idx: int = np.argmax(posterior_mean)
        # score = posterior_mean[best_line_idx]

        # score, best_line_idx, (posterior, posterior_mean, y_line, (t_in, t_out)) = RadonTransformComputation.compute_score(arr=arr, y_line=y_line, nlines=nlines, n_neighbours=n_neighbours)

        score, best_line_idx, (posterior, posterior_mean, y_line_idxs, (t_in, t_out)) = RadonTransformComputation.compute_score(arr=arr, y_line_idxs=y_line_idxs, nlines=nlines, n_neighbours=n_neighbours)
        best_phi = phi[best_line_idx]
        best_rho = rho[best_line_idx]
        best_y_line_idxs = np.squeeze(y_line_idxs[best_line_idx, :]) # (n_t, ) - confirmed to be correct
        # best_y_line = np.squeeze(y_line[best_line_idx, :]) # (n_t, ) # incorrect
        # converts to real world values

        ## Pho 2024-02-15 - Validated that below matches the original manuscript
        # velocity = RadonTransformComputation.velocity(phi=best_phi, time_bin_size=dt, pos_bin_size=dx)
        # intercept = RadonTransformComputation.intercept(phi=best_phi, rho=best_rho, t_mid=time_mid, x_mid=pos_mid, time_bin_size=dt, pos_bin_size=dx)

        ## Compute the correct intercept and velocity/slope from the debug line which seems to be correct:
        is_inside_matrix = np.logical_and((best_y_line_idxs >= 0), (best_y_line_idxs < n_pos))
        inside_matrix_only_best_y_line_idxs = best_y_line_idxs[is_inside_matrix]
        inside_matrix_only_t = t[is_inside_matrix]
        best_inside_y_line = np.array([pos[an_idx] for an_idx in inside_matrix_only_best_y_line_idxs])    
        velocity = (best_inside_y_line[-1]-best_inside_y_line[0])/(inside_matrix_only_t[-1]-inside_matrix_only_t[0]) # IndexError: index -1 is out of bounds for axis 0 with size 0 -- occuring with a (58, 1) array of all NaNs
        intercept = best_inside_y_line[0]-(velocity * inside_matrix_only_t[0])
        
        # best_y_line = np.array([pos[an_idx] for an_idx in best_y_line_idxs]) # (n_t, )

        # y_line = np.interp(t, xp=np.squeeze(inside_matrix_only_t), fp=np.squeeze(inside_matrix_only_best_y_line_idxs))


        # inside_only: (-48.92679149792471, 4450.167735614992)
        # best_y_line_segment = np.array([float(x0), (float(x0) + float(dx))])
        # t_segment = np.array([float(t0), float(t0)+float(dt)])
        # velocity = (best_y_line_segment[-1]-best_y_line_segment[0])/(t_segment[-1]-t_segment[0])
        # intercept = best_y_line_segment[0]-(velocity * t_segment[0]) # (19.027085582525963, -1566.9703125223657)

    # np.seterr(**old_settings)

    if enable_return_neighbors_arr:
        ## compute the real y_line for the debug value:
        y_line = (velocity * t) + intercept

        debug_info = RadonTransformDebugValue(t=t, n_t=n_t, ci_mid=ci_mid, time_mid=time_mid, 
            pos=pos, n_pos=n_pos, ri_mid=ri_mid, pos_mid=pos_mid,
            diag_len=diag_len, y_line_idxs=y_line_idxs, y_line=y_line, t_out=t_out, t_in=t_in, posterior=posterior, posterior_mean=posterior_mean,
            best_line_idx=best_line_idx, best_phi=best_phi, best_rho=best_rho,
         )
        return score, -velocity, intercept, (n_neighbours, arr.copy(), debug_info)
    else:
        return score, -velocity, intercept


def old_radon_transform(arr, nlines=5000):
    """Older version of the radon_transform that only returns the score and the slope (no intercept). Line fitting algorithm primarily used in decoding algorithm, a variant of radon transform, algorithm based on Kloosterman et al. 2012

    Parameters
    ----------
    arr : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    References
    ----------
    1) Kloosterman et al. 2012
    
    Usage:
        from neuropy.analyses.decoders import old_radon_transform

    """
    t = np.arange(arr.shape[1])
    nt = len(t)
    tmid = (nt + 1) / 2
    pos = np.arange(arr.shape[0])
    npos = len(pos)
    pmid = (npos + 1) / 2
    arr = np.apply_along_axis(np.convolve, axis=0, arr=arr, v=np.ones(3))

    theta = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=nlines)
    diag_len = np.sqrt((nt - 1) ** 2 + (npos - 1) ** 2)
    intercept = np.random.uniform(low=-diag_len / 2, high=diag_len / 2, size=nlines)

    cmat = np.tile(intercept, (nt, 1)).T
    mmat = np.tile(theta, (nt, 1)).T
    tmat = np.tile(t, (nlines, 1))
    posterior = np.zeros((nlines, nt))

    y_line = (((cmat - (tmat - tmid) * np.cos(mmat)) / np.sin(mmat)) + pmid).astype(int)
    t_out = np.where((y_line < 0) | (y_line > npos - 1))
    t_in = np.where((y_line >= 0) & (y_line <= npos - 1))
    posterior[t_out] = np.median(arr[:, t_out[1]], axis=0)
    posterior[t_in] = arr[y_line[t_in], t_in[1]]

    posterior_sum = np.nanmean(posterior, axis=1)
    max_line = np.argmax(posterior_sum)
    slope = -(1 / np.tan(theta[max_line]))

    return posterior_sum[max_line], slope


def wcorr(arr: NDArray) -> float:
    """weighted correlation
    Encountering issue when nx == 1, as in there is only one time bin, in which the wcorr doesn't make any sense.
    """
    # with warnings.catch_warnings():
    #     warnings.simplefilter("error", RuntimeWarning)

    nx, ny = arr.shape[1], arr.shape[0]
    y_mat: NDArray = np.tile(np.arange(ny)[:, np.newaxis], (1, nx))
    x_mat: NDArray = np.tile(np.arange(nx), (ny, 1))
    arr_sum: float = np.nansum(arr)
    ey: float = np.nansum(arr * y_mat) / arr_sum  # RuntimeWarning: invalid value encountered in double_scalars
    ex: float = np.nansum(arr * x_mat) / arr_sum  # RuntimeWarning: invalid value encountered in double_scalars
    cov_xy: float = np.nansum(arr * (y_mat - ey) * (x_mat - ex)) / arr_sum # RuntimeWarning: invalid value encountered in double_scalars
    cov_yy: float = np.nansum(arr * (y_mat - ey) ** 2) / arr_sum # RuntimeWarning: invalid value encountered in double_scalars
    cov_xx: float = np.nansum(arr * (x_mat - ex) ** 2) / arr_sum # RuntimeWarning: invalid value encountered in double_scalars

    return cov_xy / np.sqrt(cov_xx * cov_yy)


def jump_distance(posteriors, jump_stat="mean", norm=True):
    """Calculate jump distance for posterior matrices"""

    if jump_stat == "mean":
        f = np.mean
    elif jump_stat == "median":
        f = np.median
    elif jump_stat == "max":
        f = np.max
    else:
        raise ValueError("Invalid jump_stat. Valid values: mean, median, max")

    dx = 1 / posteriors[0].shape[0] if norm else 1
    jd = np.array([f(np.abs(np.diff(np.argmax(_, axis=0)))) for _ in posteriors])

    return jd * dx


def column_shift(arr, shifts=None):
    """Circular shift columns independently by a given amount"""

    assert arr.ndim == 2, "only 2d arrays accepted"

    if shifts is None:
        rng = np.random.default_rng()
        shifts = rng.integers(-arr.shape[0], arr.shape[0], arr.shape[1])

    assert arr.shape[1] == len(shifts)

    shifts = shifts % arr.shape[0]
    rows_indx, columns_indx = np.ogrid[: arr.shape[0], : arr.shape[1]]

    rows_indx = rows_indx - shifts[np.newaxis, :]

    return arr[rows_indx, columns_indx]


def epochs_spkcount(neurons: Union[core.Neurons, pd.DataFrame], epochs: Union[core.Epoch, pd.DataFrame], bin_size=0.01, slideby=None, export_time_bins:bool=False, included_neuron_ids=None, debug_print:bool=False, use_single_time_bin_per_epoch: bool=False):
    """Binning events and calculating spike counts

    Args:
        neurons (Union[core.Neurons, pd.DataFrame]): _description_
        epochs (Union[core.Epoch, pd.DataFrame]): _description_
        bin_size (float, optional): _description_. Defaults to 0.01.
        slideby (_type_, optional): _description_. Defaults to None.
        export_time_bins (bool, optional): If True returns a list of the actual time bin centers for each epoch in time_bins. Defaults to False.
        included_neuron_ids (bool, optional): Only relevent if using a spikes_df for the neurons input. Ensures there is one spiketrain built for each neuron in included_neuron_ids, even if there are no spikes.
        debug_print (bool, optional): _description_. Defaults to False.
        use_single_time_bin_per_epoch (bool, optional): If True, a single time bin is used per epoch instead of using the provided `bin_size`. This means that each epoch will have exactly one bin, but it will be variablely-sized depending on the epoch's duration. Defaults to false.
        
    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        list: spkcount - one for each epoch in filter_epochs
        list: nbins - A count of the number of time bins that compose each decoding epoch e.g. nbins: [7 2 7 1 5 2 7 6 8 5 8 4 1 3 5 6 6 6 3 3 4 3 6 7 2 6 4 1 7 7 5 6 4 8 8 5 2 5 5 8]
        list: time_bin_containers_list - None unless export_time_bins is True. 
        
    Usage:
    
        spkcount, nbins, time_bin_containers_list = 
        
    
        
    Extra:
    
        If the epoch is shorter than the bin_size the time_bins returned should be the edges of the epoch
        
        
    """
    from neuropy.core.epoch import ensure_dataframe

    # Handle extracting the spiketrains, which are a list with one entry for each neuron and each list containing the timestamps of the spike event
    if isinstance(neurons, core.Neurons):
        spiketrains = neurons.spiketrains
    elif isinstance(neurons, pd.DataFrame):
        # a spikes_df is passed in, build the spiketrains
        spikes_df = neurons
        spiketrains = spikes_df.spikes.get_unit_spiketrains(included_neuron_ids=included_neuron_ids)
    else:
        raise NotImplementedError

    # Handle either core.Epoch or pd.DataFrame objects:
    epoch_df = ensure_dataframe(epochs)
    n_epochs = np.shape(epoch_df)[0] # there is one row per epoch

    spkcount = []
    if export_time_bins:
        time_bin_containers_list = []
    else:
        time_bin_containers_list = None

    nbins = np.zeros(n_epochs, dtype="int")

    if slideby is None:
        slideby = bin_size
    if debug_print:
        print(f'slideby: {slideby}')

    if not use_single_time_bin_per_epoch:
        window_shape  = int(bin_size * 1000) # Ah, forces integer binsizes!
        if debug_print:
            print(f'window_shape: {window_shape}')

    # ----- little faster but requires epochs to be non-overlapping ------
    # bins_epochs = []
    # for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
    #     bins = np.arange(epoch.start, epoch.stop, bin_size)
    #     nbins[i] = len(bins) - 1
    #     bins_epochs.extend(bins)
    # spkcount = np.asarray(
    #     [np.histogram(_, bins=bins_epochs)[0] for _ in spiketrains]
    # )

    # deleting unwanted columns that represent time between events
    # cumsum_nbins = np.cumsum(nbins)
    # del_columns = cumsum_nbins[:-1] + np.arange(len(cumsum_nbins) - 1)
    # spkcount = np.delete(spkcount, del_columns.astype(int), axis=1)

    for i, epoch in enumerate(epoch_df.itertuples()):
        #TODO 2024-01-25 16:52: - [ ] It seems that when the epoch duration is shorter than the bin size we should impose the same bins as the single-time-bin-per-epoch case, but idk what to do with the slideby.
        # Something like: if (use_single_time_bin_per_epoch or (window_shape > spkcount_.shape[1])): 
        if use_single_time_bin_per_epoch:
            bins = np.array([epoch.start, epoch.stop]) # NOTE: not subdivided
        else:
            # fixed time-bin duration -> variable num time bins per epoch depending on epoch length    
            # first dividing in 1ms
            bins = np.arange(epoch.start, epoch.stop, 0.001) # subdivided by 1ms, so way more bins here than we'd expect for either bin_centers or bin_edges
            
        spkcount_ = np.asarray(
            [np.histogram(_, bins=bins)[0] for _ in spiketrains]
        )
        if debug_print:
            print(f'i: {i}, epoch: [{epoch.start}, {epoch.stop}], bins: {np.shape(bins)}, np.shape(spkcount_): {np.shape(spkcount_)}')
        
        # the 2nd condition ((window_shape > spkcount_.shape[1])) prevents ValueError: window shape cannot be larger than input array shape spkcount_.shape: (80,60), window_shape: 75
        if (use_single_time_bin_per_epoch or (window_shape > spkcount_.shape[1])): 
            slide_view = spkcount_  # In this case, your spike count stays as it is
            nbins[i] = 1 # always 1 bin. #TODO 2024-01-19 04:45: - [ ] What is slide_view and do I need it?
        else:        
            slide_view = np.lib.stride_tricks.sliding_window_view(spkcount_, window_shape, axis=1)[:, :: int(slideby * 1000), :].sum(axis=2) # ValueError: window shape cannot be larger than input array shape spkcount_.shape: (80,60), window_shape: 75
            nbins[i] = slide_view.shape[1]
        
        if export_time_bins:
            if debug_print:
                print(f'nbins[i]: {nbins[i]}') # nbins: 20716
            
            # reduced_time_bins: only the FULL number of bin *edges*
            # reduced_time_bins # array([22.26, 22.36, 22.46, ..., 2093.66, 2093.76, 2093.86])
            if use_single_time_bin_per_epoch:
                # For single bin case, the bin edges are just the epoch start and stop times (which are importantly smaller than the time_bin_size)
                reduced_time_bin_edges = bins
                # And the bin center is just the middle of the epoch
                reduced_time_bin_centers = np.asarray([(epoch.start + epoch.stop) / 2])
                actual_window_size = float(epoch.stop - epoch.start) # the actual (variable) bin size
                assert len(reduced_time_bin_edges) >= 2, f"epochs_spkcount(...): epoch[{i}], nbins[{i}]: cannot build extents because len(reduced_time_bin_edges) < 2: reduced_time_bin_edges: {reduced_time_bin_edges}"
                manual_center_info = BinningInfo(variable_extents=(reduced_time_bin_edges[0], reduced_time_bin_edges[-1]), step=actual_window_size, num_bins=len(reduced_time_bin_centers)) # BinningInfo(variable_extents: tuple, step: float, num_bins: int)
                # center_info = BinningContainer.build_center_binning_info(reduced_time_bin_centers, reduced_time_bin_edges) # the second argument (edge_extents) is just the edges
                bin_container = BinningContainer(edges=reduced_time_bin_edges, centers=reduced_time_bin_centers, center_info=manual_center_info) # have to manually provide center_info because it doesn't work with two or less entries.
                
            else:
                reduced_slide_by_amount = int(slideby * 1000)
                reduced_time_bin_edges = bins[:: reduced_slide_by_amount] # WTH does this notation mean?
                
                # assert len(reduced_time_bin_edges) >= 2, f"epochs_spkcount(...): epoch[{i}], nbins[{i}]: cannot build extents because len(reduced_time_bin_edges) < 2: reduced_time_bin_edges: {reduced_time_bin_edges}"
                assert len(reduced_time_bin_edges) > 0, f"epochs_spkcount(...): epoch[{i}], nbins[{i}]: cannot build extents because reduced_time_bin_edges is empty (len(reduced_time_bin_edges) == 0): reduced_time_bin_edges: {reduced_time_bin_edges}"
                
                try:
                    n_reduced_edges: int = len(reduced_time_bin_edges)
                    if n_reduced_edges == 1:
                        # Built using `epoch` - have to manually build center_info from subsampled `bins` because it doesn't work with two or less entries.
                        print(f'ERROR: epochs_spkcount(...): epoch[{i}], nbins[{i}]: {nbins[i]} - TODO 2024-08-07 19:11: Building BinningContainer for epoch with fewer than 2 edges (occurs when epoch duration is shorter than the bin size). Using the epoch.start, epoch.stop as the two edges (giving a single bin) but this might be off and cause problems, as they are the edges of the epoch but maybe not "real" edges?')
                        reduced_time_bin_edges = np.array([epoch.start, epoch.stop])
                        # reduced_time_bin_edges = deepcopy(bins) #TODO 2024-08-07 19:11: - [ ] This might be off, as they are the edges of the epoch but maybe not "real" edges?
                        reduced_time_bin_centers = np.asarray([(epoch.start + epoch.stop) / 2]) # And the bin center is just the middle of the epoch
                        actual_window_size = float(epoch.stop - epoch.start) # the actual (variable) bin size
                        variable_extents = (epoch.start, epoch.stop)
                        manual_center_info = BinningInfo(variable_extents=variable_extents, step=actual_window_size, num_bins=1) # num_bins == 1, just like when (len(reduced_time_bin_edges) == 2)
                        bin_container = BinningContainer(edges=reduced_time_bin_edges, centers=reduced_time_bin_centers, center_info=manual_center_info) # have to manually provide center_info because it doesn't work with two or less entries.   
                        print(f'\t ERROR (cont.): even after this hack `slide_view` is not updated, so the returned spkcount is not valid and has the old (wrong, way too many) number of bins. This results in decoded posteriors/postitions/etc with way too many bins downstream. see `SOLUTION 2024-08-07 20:08: - [ ] Recompute the Invalid Quantities with the known correct number of time bins` for info.')                     
                    elif n_reduced_edges == 2:
                        # have to manually build center_info from subsampled `bins` because it doesn't work with two or less entries.
                        reduced_time_bin_edges = deepcopy(bins)
                        # And the bin center is just the middle of the epoch
                        reduced_time_bin_centers = np.asarray([(reduced_time_bin_edges[0] + reduced_time_bin_edges[1]) / 2]) # just a single element?
                        actual_window_size = float(reduced_time_bin_edges[1] - reduced_time_bin_edges[0]) # the actual (variable) bin size... #TODO 2024-08-07 18:50: - [ ] this might be the subsampled bin size
                        manual_center_info = BinningInfo(variable_extents=(reduced_time_bin_edges[0], reduced_time_bin_edges[-1]), step=actual_window_size, num_bins=len(reduced_time_bin_centers))
                        bin_container = BinningContainer(edges=reduced_time_bin_edges, centers=reduced_time_bin_centers, center_info=manual_center_info) # have to manually provide center_info because it doesn't work with two or less entries.
                    else:
                        # can do it like normal:
                        ## automatically computes reduced_time_bin_centers and both infos:
                        bin_container = BinningContainer(edges=reduced_time_bin_edges)
                        reduced_time_bin_centers = deepcopy(bin_container.centers)                 

                except Exception as err:
                    print(f'ERROR: epochs_spkcount(...): epoch[{i}], nbins[{i}]: while building time bins, encountered exception err: {err}.')
                    raise                
            
            if debug_print:
                num_bad_time_bins = len(bins)
                print(f'num_bad_time_bins: {num_bad_time_bins}')
                if not use_single_time_bin_per_epoch:
                    print(f'reduced_slide_by_amount: {reduced_slide_by_amount}')
                print(f'reduced_time_bin_edges.shape: {reduced_time_bin_edges.shape}') # reduced_time_bin_edges.shape: (20717,)
                print(f'reduced_time_bin_centers.shape: {reduced_time_bin_centers.shape}') # reduced_time_bin_centers.shape: (20716,)

            assert len(reduced_time_bin_centers) == nbins[i], f"The length of the produced reduced_time_bin_centers and the nbins[i] should be the same, but len(reduced_time_bin_centers): {len(reduced_time_bin_centers)} and nbins[i]: {nbins[i]}!"
            # time_bin_centers_list.append(reduced_time_bin_centers)
            time_bin_containers_list.append(bin_container)
            
        spkcount.append(slide_view)

    return spkcount, nbins, time_bin_containers_list


class Decode1d:
    n_jobs = 8

    def __init__(self, neurons: core.Neurons, ratemap: core.Ratemap, epochs: core.Epoch=None, time_bin_size=0.5, slideby=None):
        self.ratemap = ratemap
        self._events = None
        self.posterior = None
        self.neurons = neurons
        self.time_bin_size = time_bin_size
        self.decodingtime = None
        self.time_bin_centers = None
        
        self.decoded_position = None
        self.epochs = epochs
        self.slideby = slideby
        self.score = None
        self.shuffle_score = None

        self._estimate()

    def _decoder(self, spkcount, ratemaps):
        """
        ===========================
        Probability is calculated using this formula
        prob = (1 / nspike!)* ((tau * frate)^nspike) * exp(-tau * frate)
        where,
            tau = binsize
        ===========================
        """
        tau = self.time_bin_size
        nCells = spkcount.shape[0]
        cell_prob = np.zeros((ratemaps.shape[1], spkcount.shape[1], nCells))
        for cell in range(nCells):
            cell_spkcnt = spkcount[cell, :][np.newaxis, :]
            cell_ratemap = ratemaps[cell, :][:, np.newaxis]

            coeff = 1 / (factorial(cell_spkcnt))
            # broadcasting
            cell_prob[:, :, cell] = (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (
                np.exp(-tau * cell_ratemap)
            )

        posterior = np.prod(cell_prob, axis=2)
        posterior /= np.sum(posterior, axis=0)

        return posterior

    def _estimate(self):
        """Estimates position within each"""

        tuning_curves = self.ratemap.tuning_curves
        bincntr = self.ratemap.xbin_centers

        if self.epochs is not None:
            spkcount, nbins, time_bin_centers_list = epochs_spkcount(self.neurons, self.epochs, self.time_bin_size, self.slideby)
            posterior = self._decoder(np.hstack(spkcount), tuning_curves)
            decodedPos = bincntr[np.argmax(posterior, axis=0)]
            cum_nbins = np.cumsum(nbins)[:-1]

            self.decodingtime = None # time bins are invalid for this mode
            self.time_bin_centers = None

            self.decoded_position = np.hsplit(decodedPos, cum_nbins)
            self.posterior = np.hsplit(posterior, cum_nbins)
            self.spkcount = spkcount
            self.nbins_epochs = nbins
            self.score, _ = self.score_posterior(self.posterior)

        else:
            flat_filtered_neurons = self.neurons.get_binned_spiketrains(bin_size=self.time_bin_size)
            spkcount = flat_filtered_neurons.spike_counts
            neuropy_decoder_time_bins = flat_filtered_neurons.time
            self.decodingtime = neuropy_decoder_time_bins # get the time_bins (bin edges)
            self.time_bin_centers = self.decodingtime[:-1] + np.diff(self.decodingtime) / 2.0
            # spkcount = self.neurons.get_binned_spiketrains(bin_size=self.bin_size).spike_counts

            self.posterior = self._decoder(spkcount, tuning_curves)
            self.decoded_position = bincntr[np.argmax(self.posterior, axis=0)]
            self.score = None

    def calculate_shuffle_score(self, n_iter=100, method="column"):
        """Shuffling and decoding epochs"""

        # print(f"Using {kind} shuffle")

        if method == "neuron_id":
            posterior, score = [], []
            for i in range(n_iter):
                tuning_curves = self.ratemap.tuning_curves.copy()
                np.random.shuffle(tuning_curves)
                post_ = self._decoder(np.hstack(self.spkcount), tuning_curves)
                cum_nbins = np.cumsum(self.nbins_epochs)[::-1]
                posterior.extend(np.hsplit(post_, cum_nbins))

            score = self.score_posterior(posterior)[0]
            score = score.reshape(n_iter, len(self.spkcount))

        if method == "column":

            def col_shuffle(mat):
                shift = np.random.randint(1, mat.shape[1], mat.shape[1])
                direction = np.random.choice([-1, 1], size=mat.shape[1])
                shift = shift * direction

                mat = np.array([np.roll(mat[:, i], sh) for i, sh in enumerate(shift)])
                return mat.T

            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = lambda x: x # NO-OP alternative to tqdm progress bar when not using tqdm

            score = []
            for i in tqdm(range(n_iter)):
                evt_shuff = [col_shuffle(arr) for arr in self.posterior]
                score.append(self._score_events(evt_shuff)[0])

        # score = np.concatenate(score)
        self.shuffle_score = np.array(score)

    def score_posterior(self, p):
        """Scoring of epochs

        Returns
        -------
        [type]
            [description]

        References
        ----------
        1) Kloosterman et al. 2012
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(radon_transform)(epoch) for epoch in p
        )
        score = [res[0] for res in results]
        slope = [res[1] for res in results]

        return np.asarray(score), np.asarray(slope)

    @property
    def p_value(self):
        shuff_score = self.shuffle_score
        n_iter = shuff_score.shape[0]
        diff_score = shuff_score - np.tile(self.score, (n_iter, 1))
        chance = np.where(diff_score > 0, 1, 0).sum(axis=0)
        return (chance + 1) / (n_iter + 1)

    def plot_in_bokeh(self):
        pass

    def plot_replay_epochs(self, pval=0.05, speed_thresh=True, cmap="hot"):
        pval_events = self.p_val_events
        replay_ind = np.where(pval_events < pval)[0]
        posterior = [self.posterior[_] for _ in replay_ind]
        sort_ind = np.argsort(self.score[replay_ind])[::-1]
        posterior = [posterior[_] for _ in sort_ind]
        events = self.events.iloc[replay_ind].reset_index(drop=True)
        events["score"] = self.score[replay_ind]
        events["slope"] = self.slope[replay_ind]
        events.sort_values(by=["score"], inplace=True, ascending=False)

        spikes = Spikes(self._obj)
        spks = spikes.pyr
        pf1d_obj = self.ratemaps

        mapinfo = pf1d_obj.ratemaps
        ratemaps = np.asarray(mapinfo["ratemaps"])

        # ----- removing cells that fire < 1 HZ --------
        good_cells = np.where(np.max(ratemaps, axis=1) > 1)[0]
        spks = [spks[_] for _ in good_cells]
        ratemaps = ratemaps[good_cells, :]

        # --- sorting the cells according to pf location -------
        sort_ind = np.argsort(np.argmax(ratemaps, axis=1))
        spks = [spks[_] for _ in sort_ind]
        ratemaps = ratemaps[sort_ind, :]

        figure = Fig()
        fig, gs = figure.draw(grid=(6, 12), hspace=0.34)

        for i, epoch in enumerate(events.itertuples()):
            gs_ = figure.subplot2grid(gs[i], grid=(2, 1), hspace=0.1)
            ax = plt.subplot(gs_[0])
            spikes.plot_raster(
                spks, ax=ax, period=[epoch.start, epoch.end], tstart=epoch.start
            )
            ax.set_title(
                f"Score = {np.round(epoch.score,2)},\n Slope = {np.round(epoch.slope,2)}",
                loc="left",
            )
            ax.set_xlabel("")
            ax.tick_params(length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            axdec = plt.subplot(gs_[1], sharex=ax)
            axdec.pcolormesh(
                np.arange(posterior[i].shape[1] + 1) * self.binsize,
                self.ratemaps.bin - np.min(self.ratemaps.bin),
                posterior[i],
                cmap=cmap,
                vmin=0,
                vmax=0.5,
            )
            axdec.set_ylabel("Position")

            if i % 12:
                ax.set_ylabel("")
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(axdec.get_yticklabels(), visible=False)
                axdec.set_ylabel("")

            if i > (5 * 6 - 1):
                axdec.set_xlabel("Time (ms)")


class Decode2d:
    """ 2D Decoder 
    
    """
    def __init__(self, pf2d_obj: PfND):
        assert isinstance(pf2d_obj, PfND)
        self.pf = pf2d_obj
        self.ratemap = self.pf.ratemap

        self._all_positions_matrix = None
        self._original_data_shape = None
        self._flat_all_positions_matrix = None
        
        self.time_bin_size = None
        self.decodingtime = None
        self.time_bin_centers = None
        
        self.actualbin = None
        self.posterior = None
        self.actualpos = None
        self.decoded_position = None

    def _decoder(self, spkcount, ratemaps):
        """
        ===========================
        Probability is calculated using this formula
        prob = (1 / nspike!)* ((tau * frate)^nspike) * exp(-tau * frate)
        where,
            tau = binsize
        ===========================
        """
        tau = self.time_bin_size
        nCells = spkcount.shape[0]
        # nSpikes = spkcount.shape[1] 
        # nFlatPositionBins = ratemaps.shape[1]
        cell_prob = np.zeros((ratemaps.shape[1], spkcount.shape[1], nCells))
        for cell in range(nCells):
            cell_spkcnt = spkcount[cell, :][np.newaxis, :]
            cell_ratemap = ratemaps[cell, :][:, np.newaxis]

            coeff = 1 / (factorial(cell_spkcnt))
            # broadcasting
            cell_prob[:, :, cell] = (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (
                np.exp(-tau * cell_ratemap)
            )

        posterior = np.prod(cell_prob, axis=2)
        posterior /= np.sum(posterior, axis=0)

        return posterior
    
    def estimate_behavior(self, spikes_df, t_start_end, time_bin_size=0.25, smooth=1, plot=True):
        """ 
        Updates:
            ._all_positions_matrix
            ._original_data_shape
            ._flat_all_positions_matrix
            .bin_size
            .decodingtime
            .time_bin_centers
            .actualbin
            .posterior
            .actualpos
            .decodedPos
        """
        ratemap_cell_ids = self.pf.cell_ids
        # spks = Spikes(self._obj).get_cells(ids=ratemap_cell_ids)
        spk_dfs = spikes_df.spikes.get_split_by_unit(included_neuron_ids=ratemap_cell_ids)
        spk_times = [cell_df[spikes_df.spikes.time_variable_name].to_numpy() for cell_df in spk_dfs]
        
        # ratemaps = self.pf.ratemap
        # tuning_curves = self.pf.ratemap.tuning_curves
        tuning_curves = self.ratemap.tuning_curves
        
        speed = self.pf.speed
        xgrid = self.pf.xbin
        ygrid = self.pf.ybin
        # gridbin = self.pf.gridbin
        
        # gridbin = (self.pf.bin_info['xstep'], self.pf.bin_info['ystep'])
        gridbin_x = self.pf.bin_info['xstep']
        gridbin_y = self.pf.bin_info['ystep']
        
        # gridcenter = self.pf.gridcenter
        # gridcenter = self.pf.gridcenter
        self._all_positions_matrix, self._flat_all_positions_matrix, self._original_data_shape = build_spanning_grid_matrix(x_values=self.pf.xbin_centers, y_values=self.pf.ybin_centers, debug_print=False)
        # len(self._flat_all_positions_matrix) # 1066
        
        # --- average position in each time bin and which gridbin it belongs to ----
        t = self.pf.t
        x = self.pf.x
        y = self.pf.y
        assert t_start_end is not None and isinstance(t_start_end, tuple)
        # t_start_end = self.pf.period
        tmz = np.arange(t_start_end[0], t_start_end[1], time_bin_size)
        self.time_bin_size = time_bin_size
        self.decodingtime = tmz # time_bin_edges
        self.time_bin_centers = tmz[:-1] + np.diff(tmz) / 2.0
        
        actualposx = stats.binned_statistic(t, values=x, bins=tmz)[0]
        actualposy = stats.binned_statistic(t, values=y, bins=tmz)[0]
        actualpos = np.vstack((actualposx, actualposy))
        self.actualpos = actualpos

        actualbin_x = xgrid[np.digitize(actualposx, bins=xgrid) - 1] + gridbin_x / 2
        actualbin_y = ygrid[np.digitize(actualposy, bins=ygrid) - 1] + gridbin_y / 2
        self.actualbin = np.vstack((actualbin_x, actualbin_y))

        # ---- spike counts and linearize 2d ratemaps -------
        spkcount = np.asarray([np.histogram(cell, bins=tmz)[0] for cell in spk_times])
        spkcount = gaussian_filter1d(spkcount, sigma=3, axis=1)
        # ratemaps = np.asarray([ratemap.flatten() for ratemap in ratemaps])
        tuning_curves = np.asarray([ratemap.flatten() for ratemap in tuning_curves]) # note .flatten() returns a deepcopy, np.ravel(a) returns a shallow copy

        print(f'tuning_curves.shape: {np.shape(tuning_curves)}')
        print(f'spkcount.shape: {np.shape(spkcount)}')
        
        nCells = spkcount.shape[0]
        nTimeBins = spkcount.shape[1]
        nFlatPositionBins = tuning_curves.shape[1]
        print(f'\nnCells: {nCells}, nTimeBins: {nTimeBins}, nFlatPositionBins: {nFlatPositionBins}') # nCells: 66, nTimeBins: 3529, nFlatPositionBins: 1066
        
        self.posterior = self._decoder(spkcount=spkcount, ratemaps=tuning_curves) # self.posterior.shape: (nFlatPositionBins, nTimeBins)
        print(f'self.posterior.shape: {np.shape(self.posterior)}') # self.posterior.shape: (1066, 3529)
        
        # Compute the decoded position from the posterior:
        _test_most_likely_position_flat_idxs = np.argmax(self.posterior, axis=0)
        # _test_most_likely_position_flat_idxs.shape # (3529,)
        _test_most_likely_positions = np.array([self._flat_all_positions_matrix[a_pos_idx] for a_pos_idx in _test_most_likely_position_flat_idxs])
        # _test_most_likely_positions.shape # (3529, 2)
        self.decoded_position = _test_most_likely_positions
        # _test_most_likely_position = np.argmax(self.posterior, axis=0)
        # print(f'_test_most_likely_position: {_test_most_likely_position}')        
        # self.decodedPos = gridcenter[:, _test_most_likely_position]
        
        # if plot:
        #     _, gs = Fig().draw(grid=(4, 4), size=(15, 6))
        #     axposx = plt.subplot(gs[0, :3])
        #     axposx.plot(self.actualbin[0, :], "k")
        #     axposx.set_ylabel("Actual position")

        #     axdecx = plt.subplot(gs[1, :3], sharex=axposx)
        #     axdecx.plot(self.decodedPos[0, :], "gray")
        #     axdecx.set_ylabel("Decoded position")

        #     axposy = plt.subplot(gs[2, :3], sharex=axposx)
        #     axposy.plot(self.actualpos_gridcntr[1, :], "k")
        #     axposy.set_ylabel("Actual position")

        #     axdecy = plt.subplot(gs[3, :3], sharex=axposx)
        #     axdecy.plot(
        #         # self.decodedPos,
        #         self.decodedPos[1, :],
        #         "gray",
        #     )
        #     axdecy.set_ylabel("Decoded position")

    def decode_events(self, binsize=0.02, slideby=0.005):
        """Decodes position within events which are set using self.events

        Parameters
        ----------
        binsize : float, seconds, optional
            size of binning withing each events, by default 0.02
        slideby : float, seconds optional
            sliding by this much, by default 0.005

        Returns
        -------
        [type]
            [description]
        """

        events = self.events
        ratemap_cell_ids = self.pf.cell_ids
        spks = Spikes(self._obj).get_cells(ids=ratemap_cell_ids)
        nCells = len(spks)
        print(f"Number of cells/ratemaps in pf2d: {nCells}")

        ratemaps = self.pf.ratemaps
        gridcenter = self.pf.gridcenter

        nbins, spkcount, time_bin_centers_list = epochs_spkcount(binsize, slideby, events, spks)

        # ---- linearize 2d ratemaps -------
        ratemaps = np.asarray([ratemap.flatten() for ratemap in ratemaps])

        posterior = self._decoder(spkcount=spkcount, ratemaps=ratemaps)
        decodedPos = gridcenter[:, np.argmax(posterior, axis=0)]

        # --- splitting concatenated time bins into separate arrays ------
        cum_nbins = np.cumsum(nbins)[:-1]
        self.posterior = np.hsplit(posterior, cum_nbins)
        self.decoded_position = np.hsplit(decodedPos, cum_nbins)

        return decodedPos, posterior

    def plot(self):

        # decodedPos = gaussian_filter1d(self.decodedPos, sigma=1, axis=1)
        decodedPos = self.decoded_position
        posterior = self.posterior
        decodingtime = self.decodingtime[1:]
        actualPos = self.actualPos
        speed = self.speed
        error = np.sqrt(np.sum((decodedPos - actualPos) ** 2, axis=0))

        plt.clf()
        fig = plt.figure(1, figsize=(10, 15))
        gs = gridspec.GridSpec(3, 6, figure=fig)
        fig.subplots_adjust(hspace=0.3)

        ax = fig.add_subplot(gs[0, :])
        # ax.pcolormesh(decodingtime, np.arange(npos), posterior, cmap="binary")
        ax.plot(decodingtime, actualPos[0, :], "#4FC3F7")
        ax.plot(decodingtime, decodedPos[0, :], "#F48FB1")
        ax.set_ylabel("X coord")
        ax.set_title("Bayesian position estimation (only pyr cells)")

        ax = fig.add_subplot(gs[1, :], sharex=ax)
        # ax.pcolormesh(decodingtime, np.arange(npos), posterior, cmap="binary")
        ax.plot(decodingtime, actualPos[1, :], "#4FC3F7")
        ax.plot(decodingtime, decodedPos[1, :], "#F48FB1")
        ax.set_ylabel("Y coord")
        ax.set_title("Bayesian position estimation (only pyr cells)")

        ax = fig.add_subplot(gs[2, :], sharex=ax)
        # ax.pcolormesh(decodingtime, np.arange(npos), posterior, cmap="binary")
        ax.plot(decodingtime, speed, "k")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("speed (cm/s)")
        # ax.set_title("Bayesian position estimation (only pyr cells)")
        ax.set_ylim([0, 120])
        ax.spines["right"].set_visible(True)

        axerror = ax.twinx()
        axerror.plot(decodingtime, gaussian_filter1d(error, sigma=1), "#05d69e")
        axerror.set_ylabel("error (cm)")
