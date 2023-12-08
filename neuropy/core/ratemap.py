import numpy as np
from . import DataWriter
from scipy import stats, interpolate


class Ratemap(DataWriter):
    """Ratemap class to hold tuning curves of neurons. For example, place field ratemaps"""

    def __init__(
        self,
        tuning_curves: np.ndarray,
        coords: np.ndarray,
        occupancy=None,
        neuron_ids=None,
        metadata=None,
    ) -> None:
        """Inputs for Ratemap class

        Parameters
        ----------
        tuning_curves : np.ndarray, (neurons, nx) or (neurons, nx, ny)
            numpy array for firing rates
        coords : float, array, [float,float] or [array,array]
            values defining the coordinates in cms,
                * If float, the spacing for all dimensions.
                * If array, the coordinates for all dimensions.
                    (x_bins=y_bins=coords).
                * If [float, float], the spacing in each dimension
                    (x_binsize, y_binsize = bins).
                * If [array, array], the bin edges in each dimension
                    (x_edges, y_edges = bins).
                * A combination [int, array] or [array, int], where int
                    is the number of bins and array is the bin edges.
        occupancy : np.array, optional
            occupancy map for the tuning curves, by default None
        neuron_ids : np.array, optional
            neuron_ids of the neurons, by default None
        metadata : _type_, optional
            _description_, by default None
        """
        super().__init__(metadata=metadata)

        self.tuning_curves = tuning_curves
        self.coords = coords
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
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, val):
        if self.ndim == 1:
            if isinstance(val, (int, float)):
                val = np.arange(0, val * self.tuning_curves.shape[1], val)

            assert (
                len(val) == self.tuning_curves.shape[1]
            ), "length of coords should be equal to tuning_curve.shape[1]"
            assert np.allclose(m := np.diff(val), m[0]), "x should be equally spaced"
            val = val.reshape(1, -1)

        if self.ndim == 2:
            if isinstance(val, (int, float)):
                x = np.arange(0, val * self.tuning_curves.shape[1], val)
                val = np.vstack([x, x])

            if len(val) == 2:
                x, y = val
                if isinstance(x, (int, float)):
                    x = np.arange(0, x * self.tuning_curves.shape[1], x)
                if isinstance(y, (int, float)):
                    y = np.arange(0, y * self.tuning_curves.shape[1], y)

                assert len(x) == self.tuning_curves.shape[1], "improper coords"
                assert len(y) == self.tuning_curves.shape[2], "improper coords"
                val = np.vstack([x, y])

        assert val.ndim == 2, "Improper coords"

        self._coords = val

    def x_coords(self):
        return self.coords[0]

    def y_coords(self):
        assert self.ndim == 2, "Can't return y for 1D ratemaps"
        return self.coords[1]

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
        return np.diff(self.x_coords())[0]

    @property
    def y_binsize(self):
        if self.y is not None:
            return np.diff(self.y_coords())[0]

    @property
    def n_neurons(self):
        return self.tuning_curves.shape[0]

    @property
    def ndim(self):
        return self.tuning_curves.ndim - 1

    def get_frate_normalized(self):
        pass

    def resample_1D(self, nbins):
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
        f_tc = interpolate.interp1d(self.x_coords, self.tuning_curves)
        x_new = np.linspace(self.x_coords[0], self.x_coords[-1], nbins)
        tc_new = f_tc(x_new)  # Interpolated tuning curve

        ratemap_new = self.copy()
        ratemap_new.tuning_curves = tc_new
        ratemap_new.coords = x_new

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
