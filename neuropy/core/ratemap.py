import numpy as np
from . import DataWriter
from scipy import stats, interpolate


class Ratemap(DataWriter):
    """Ratemap class to hold tuning curves of neurons. For example, place field ratemaps"""

    def __init__(
        self,
        tuning_curves: np.ndarray,
        x=None,
        y=None,
        occupancy=None,
        neuron_ids=None,
        metadata=None,
    ) -> None:
        """Inputs for Ratemap class

        Parameters
        ----------
        tuning_curves : np.ndarray, (neurons x nx) or (neurons x nx x ny)
            numpy array for firing rates
        x : np.array or float/int,
            values defining the x coordinates in cms, if a scalar value is provided it is assumed as the x spacing, by default None which creates x with spacing 1
        y : np.array or float/int, optional
            bins defining the ygrid in cms, if a scalar value is provided it is assumed as the y spacing, by default None which creates y with spacing 1
        occupancy : np.array, optional
            occupancy map for the tuning curves, by default None
        neuron_ids : np.array, optional
            neuron_ids of the neurons, by default None
        metadata : _type_, optional
            _description_, by default None
        """
        super().__init__(metadata=metadata)

        self.tuning_curves = tuning_curves
        self.x = x
        self.y = y
        self.occupancy = occupancy
        self.neuron_ids = neuron_ids

    @property
    def tuning_curves(self):
        return self._tuning_curves

    @tuning_curves.setter
    def tuning_curves(self, val):
        val = np.asarray(val)
        assert val.ndim <= 3, "tuning_curves shape should be <= 3"
        self._tuning_curves = val

    @property
    def occupancy(self):
        return self._occupancy

    @occupancy.setter
    def occupancy(self, arr):
        if arr is not None:
            arr = np.asarray(arr)
            assert (
                arr.shape == self.tuning_curves.shape[1:]
            ), "Occupancy shape should be of same shape as tuning_curves"

        self._occupancy = arr

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        if isinstance(val, (int, float)):
            val = np.arange(0, val * self.tuning_curves.shape[1], val)
        assert (
            len(val) == self.tuning_curves.shape[1]
        ), "length of x should be equal to tuning_curve.shape[1]"
        assert np.allclose(m := np.diff(val), m[0]), "x should be equally spaced"
        self._x = val

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        if val is not None:
            assert self.ndim == 3, "Cannot set ybin for 1D ratemap"
            if isinstance(val, (int, float)):
                val = np.arange(0, val * self.tuning_curves.shape[2], val)
            assert (
                len(val) == self.tuning_curves.shape[2]
            ), "length of ybin should be equal to tuning_curves.shape[2]"
            assert np.allclose(m := np.diff(val), m[0]), "y should be equally spaced"
        self._y = val

    @property
    def neuron_ids(self):
        return self._neuron_ids

    @neuron_ids.setter
    def neuron_ids(self, arr):
        if arr is not None:
            assert (
                len(arr) == self.tuning_curves.shape[0]
            ), "The length of neuron_ids should match the tuning_curves.shape[0]"
            self._neuron_ids = np.asarray(arr)
        else:
            self._neuron_ids = np.arange(self.tuning_curves.shape[0])

    def copy(self):
        return Ratemap(
            tuning_curves=self.tuning_curves,
            x=self.x,
            y=self.y,
            neuron_ids=self.neuron_ids,
            occupancy=self.occupancy,
            metadata=self.metadata,
        )

    @property
    def x_binsize(self):
        return np.diff(self.xbin)[0]

    @property
    def y_binsize(self):
        if self.y is not None:
            return np.diff(self.ybin)[0]

    @property
    def n_neurons(self):
        return self.tuning_curves.shape[0]

    @property
    def ndim(self):
        return self.tuning_curves.ndim - 1

    def get_frate_normalized(self):
        pass

    def resample(self, nbins):
        """Resample the ratemap with nbins

        Parameters
        ----------
        nbins : int, optional
            the number of bins in new ratemap

        Returns
        -------
        Ratemap
            new ratemap
        """
        assert self.ndim == 1, "Only allowed for 1 dimensional ratemaps"
        f_tc = interpolate.interp1d(self.x, self.tuning_curves)
        x_new = np.linspace(self.x[0], self.x[-1], nbins)
        tc_new = f_tc(x_new)  # Interpolated tuning curve

        ratemap_new = self.copy()
        ratemap_new.tuning_curves = tc_new
        ratemap_new.x = x_new

        return ratemap_new

    def peak_locations(self, by="index"):
        sort_ind = np.argmax(stats.zscore(self.tuning_curves, axis=1), axis=1)
        if by == "index":
            return sort_ind
        if by == "position":
            return self.xbin[sort_ind]

    def get_sort_order(self, by="index"):
        """Return sorting order tuning curves by position in ascending order

        Parameters
        ----------
        by : str, optional
            'index' returns row-index location of tuning_curves, 'neuron_id' returns id of neurons that will sort the location of peaks tuning curves, by default "index"

        Returns
        -------
        [type]
            [description]
        """
        sort_ind = np.argsort(self.peak_locations(by="index"))
        if by == "neuron_id":
            return self.neuron_ids[sort_ind]
        if by == "index":
            return sort_ind

    def peak_firing_rate(self):
        return np.max(self.tuning_curves, axis=1)
