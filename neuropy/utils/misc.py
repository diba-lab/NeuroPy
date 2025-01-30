import numpy as np
import math
from collections.abc import Iterable
from itertools import chain


def find_nearest(array, value):
    """https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return array[idx - 1]
    else:
        return array[idx]


def arg_find_nearest(array, value):
    """https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def get_interval(period, nwindows):

    interval = np.linspace(period[0], period[1], nwindows + 1)
    interval = [[interval[i], interval[i + 1]] for i in range(nwindows)]
    return interval


def flatten(list_in):
    """Flatten a ragged list of different sized lists into one continuous list.
    Unlike `flatten_all` this only flattens the top level."""
    return list(chain.from_iterable(list_in))

def flatten_all(xs):
    """Completely flattens an iterable of iterables into one long generator"""
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_all(x)
        else:
            yield x


def nan_helper(y: np.ndarray):
    """Helper to handle indices and logical indices of NaNs.
    From https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interp_nans(y: np.ndarray):
    """interpolate nans based on values on either side of them in an array! In case of 2d array
    will move along rows"""

    if y.ndim == 2:
        for idr, yrow in enumerate(y):
            y[idr] = interp_nans(yrow)
    else:

        nans, x = nan_helper(y)
        if not np.all(np.isnan(y)):
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y
