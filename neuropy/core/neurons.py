import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from .datawriter import DataWriter
from . import Epoch
from copy import deepcopy


class Neurons(DataWriter):
    """Class to hold a group of spiketrains and their labels, ids etc."""

    # TODO: Contemplate adding implicit support for noisy_epochs, such that firing_rate, get_binned_spiketrains, get_mua etc. deletes/ignores these time points for more accurate estimations

    def __init__(
        self,
        spiketrains: np.ndarray,
        t_stop,
        t_start=0.0,
        sampling_rate=1,
        neuron_ids=None,
        neuron_type=None,
        waveforms=None,
        waveforms_amplitude=None,
        peak_channels=None,
        shank_ids=None,
        metadata=None,
    ) -> None:
        """Initializes the Neurons instance

        Parameters
        ----------
        spiketrains : np.array/list of numpy arrays
            each array contains spiketimes in seconds, 5 arrays for 5 neurons
        t_stop : float
            time when the recording was stopped
        t_start : float, optional
            start time for the recording/spike trains, by default 0.0
        sampling_rate : int, optional
            at what sampling rate the spike times were recorded, by default 1
        neuron_ids : array, optional
            id for each spiketrain/neuron, by default None
        neuron_type : array of strings, optional
            what neuron type, by default None
        waveforms : (n_neurons x n_channels x n_timepoints), optional
            waveshape for each neuron, by default None
        waveforms_amplitude : list/array of arrays, optional
            the number of arrays should match spiketrains, each value gives scaling factor used for template waveform to extract that spike, by default None
        peak_channels : array, optional
            peak channel for waveform, by default None
        shank_ids : array of int, optional
            which shank of the probe each spiketrain was recorded from, by default None
        metadata : dict, optional
            any additional metadata, by default None
        """
        super().__init__(metadata=metadata)

        self.spiketrains = np.array(spiketrains, dtype="object")
        if neuron_ids is None:
            self.neuron_ids = np.arange(len(self.spiketrains))
        else:
            self.neuron_ids = neuron_ids

        if waveforms is not None:
            assert (
                waveforms.shape[0] == self.n_neurons
            ), "Waveforms first dimension should match number of neurons"

        if waveforms_amplitude is not None:
            assert len(waveforms_amplitude) == len(
                self.spiketrains
            ), "length should match"
            self.waveforms_amplitude = waveforms_amplitude
        else:
            self.waveforms_amplitude = None

        self.waveforms = waveforms
        self.shank_ids = shank_ids
        self.neuron_type = neuron_type
        self.peak_channels = peak_channels
        self._sampling_rate = sampling_rate
        self.t_start = t_start
        self.t_stop = t_stop

    def __getitem__(self, i):
        # copy object
        spiketrains = self.spiketrains[i]
        if self.neuron_type is not None:
            neuron_type = self.neuron_type[i]
        else:
            neuron_type = self.neuron_type

        if self.waveforms is not None:
            waveforms = self.waveforms[i]
        else:
            waveforms = self.waveforms

        if self.waveforms_amplitude is not None:
            waveforms_amplitude = self.waveforms_amplitude[i]
        else:
            waveforms_amplitude = self.waveforms_amplitude

        if self.peak_channels is not None:
            peak_channels = self.peak_channels[i]
        else:
            peak_channels = self.peak_channels

        if self.shank_ids is not None:
            shank_ids = self.shank_ids[i]
        else:
            shank_ids = self.shank_ids

        return Neurons(
            spiketrains=spiketrains,
            t_start=self.t_start,
            t_stop=self.t_stop,
            sampling_rate=self.sampling_rate,
            neuron_ids=self.neuron_ids[i],
            neuron_type=neuron_type,
            waveforms=waveforms,
            waveforms_amplitude=waveforms_amplitude,
            peak_channels=peak_channels,
            shank_ids=shank_ids,
        )

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def n_neurons(self):
        return len(self.spiketrains)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n n_neurons: {self.n_neurons}\n t_start: {self.t_start}\n t_stop: {self.t_stop}\n neuron_type: {np.unique(self.neuron_type)}"

    def time_slice(self, t_start=None, t_stop=None):

        t_start, t_stop = super()._time_slice_params(t_start, t_stop)
        neurons = deepcopy(self)
        spiketrains = [t[(t >= t_start) & (t <= t_stop)] for t in neurons.spiketrains]

        return Neurons(
            spiketrains=spiketrains,
            t_stop=t_stop,
            t_start=t_start,
            sampling_rate=neurons.sampling_rate,
            neuron_ids=neurons.neuron_ids,
            neuron_type=neurons.neuron_type,
            waveforms=neurons.waveforms,
            peak_channels=neurons.peak_channels,
            shank_ids=neurons.shank_ids,
        )

    def get_neuron_type(self, neuron_type):
        if isinstance(neuron_type, str):
            indices = self.neuron_type == neuron_type
        if isinstance(neuron_type, list):
            indices = np.any(
                np.vstack([ntype == self.neuron_type for ntype in neuron_type]), axis=0
            )
        return self[indices]

    def _check_integrity(self):
        assert isinstance(self.spiketrains, np.ndarray)
        # n_neurons = self.n_neurons
        # assert all(
        #     len(arr) == n_neurons
        #     for arr in [
        #         self.shankid,
        #         self.labels,
        #         self.ids,
        #         self.waveforms,
        #         self.instfiring,
        #     ]
        # )

    def __str__(self) -> str:
        return f"# neurons = {self.n_neurons}"

    def __len__(self):
        return self.n_neurons

    def add_metadata(self):
        pass

    def get_all_spikes(self):
        return np.concatenate(self.spiketrains)

    @property
    def n_spikes(self):
        "number of spikes within each spiketrain"
        return np.asarray([len(_) for _ in self.spiketrains])

    @property
    def firing_rate(self):
        return self.n_spikes / (self.t_stop - self.t_start)

    def get_above_firing_rate(self, thresh: float):
        """Return neurons which have firing rate above thresh"""
        indices = self.firing_rate > thresh
        return self[indices]

    def get_by_id(self, ids):
        """Returns neurons object with neuron_ids equal to ids"""
        indices = np.isin(self.neuron_ids, ids)
        return self[indices]

    def get_isi(self, bin_size=0.001, n_bins=200):
        """Interspike interval

        Parameters
        ----------
        bin_size : float, optional
            [description], by default 0.001
        n_bins : int, optional
            [description], by default 200

        Returns
        -------
        [type]
            [description]
        """
        bins = np.arange(n_bins + 1) * bin_size
        return np.asarray(
            [np.histogram(np.diff(spktrn), bins=bins)[0] for spktrn in self.spiketrains]
        )

    def get_waveform_similarity(self):
        waveforms = np.reshape(self.waveforms, (self.n_neurons, -1)).astype(float)
        similarity = np.corrcoef(waveforms)
        np.fill_diagonal(similarity, 0)
        return similarity

    def get_binned_spiketrains(self, bin_size=0.25, ignore_epochs: Epoch = None):

        """Get binned spike counts

        Parameters
        ----------
        bin_size : float, optional
            bin size in seconds, by default 0.25

        Returns
        -------
        neuropy.core.BinnedSpiketrains

        """
        duration = self.t_stop - self.t_start
        n_bins = np.floor(duration / bin_size)
        # bins = np.arange(self.t_start, self.t_stop + bin_size, bin_size)
        bins = np.arange(n_bins + 1) * bin_size + self.t_start
        spike_counts = np.asarray(
            [np.histogram(_, bins=bins)[0] for _ in self.spiketrains]
        ).astype("float")
        if ignore_epochs is not None:
            ignore_bins = ignore_epochs.flatten()
            ignore_indices = np.digitize(bins, ignore_bins) % 2 == 1
            spike_counts[:, ignore_indices] = np.nan

        return BinnedSpiketrain(
            spike_counts,
            t_start=self.t_start,
            bin_size=bin_size,
            neuron_ids=self.neuron_ids,
            peak_channels=self.peak_channels,
            shank_ids=self.shank_ids,
        )

    def get_mua(self, bin_size=0.001):
        """Get mua between two time points

        Parameters
        ----------
        bin_size : float, optional
            [description], by default 0.001

        Returns
        -------
        MUA object
            [description]
        """

        all_spikes = self.get_all_spikes()
        bins = np.arange(self.t_start, self.t_stop, bin_size)
        spike_counts = np.histogram(all_spikes, bins=bins)[0]
        return Mua(spike_counts.astype("int"), t_start=self.t_start, bin_size=bin_size)

    def add_jitter(self):
        pass

    # def get_psth(self, t, bin_size, window=0.25):
    #     """Get peri-stimulus time histograms w.r.t time points in t"""

    #     time_diff = [np.histogram(spktrn - t) for spktrn in self.spiketrains]

    def get_modulation_in_epochs(self, epochs: Epoch, n_bins):
        """Total number of across all epochs where each epoch is divided into equal number of bins

        Parameters
        ----------
        epochs : Epoch
            epochs for calculation
        n_bins : int
            number of bins to divide each epoch

        Returns
        -------
        2d array: n_neurons x n_bins
            total number of spikes within each bin across all epochs
        """
        assert epochs.is_overlapping == False, "epochs should be non-overlapping"
        assert isinstance(n_bins, int), "n_bins can only be integer"
        starts = epochs.starts.reshape(-1, 1)
        bin_size = (epochs.durations / n_bins).reshape(-1, 1)

        # create 2D-array (n_epochs x n_bins+1) with bin_size spacing along columns
        bins = np.arange(n_bins + 1) * bin_size

        epoch_bins = (starts + bins).flatten()

        # calculate spikes on flattened epochs and delete bins which represent spike counts between (not within) epochs and then sums across all epochs for each bin
        counts = [
            np.delete(
                np.histogram(_, epoch_bins)[0],
                np.arange(n_bins, epoch_bins.size, n_bins + 1)[:-1],
            )
            .reshape(-1, n_bins)
            .sum(axis=0)
            for _ in self.spiketrains
        ]

        return np.asarray(counts)

    def get_spikes_in_epochs(self, epochs: Epoch, bin_size=0.01, slideby=None):
        """A list of 2D arrays containing spike counts

        Parameters
        ----------
        epochs : Epoch
            start and stop times of epochs
        bin_size : float, optional
            bin size to be used to within each epoch, by default 0.01
        slideby : [type], optional
            if spike counts should have sliding window, by default None

        Returns
        -------
        spkcount, nbins
            list of arrays, number of bins within each epoch
        """
        spkcount = []
        nbins = np.zeros(epochs.n_epochs, dtype="int")

        # ----- little faster but requires epochs to be non-overlapping ------

        if (~epochs.is_overlapping) and (slideby is None):
            bins_epochs = []
            for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
                bins = np.arange(epoch.start, epoch.stop, bin_size)
                nbins[i] = len(bins) - 1
                bins_epochs.extend(bins)
            spkcount = np.asarray(
                [np.histogram(_, bins=bins_epochs)[0] for _ in self.spiketrains]
            )

            # deleting unwanted columns that represent time between events
            cumsum_nbins = np.cumsum(nbins)
            del_columns = cumsum_nbins[:-1] + np.arange(len(cumsum_nbins) - 1)
            spkcount = np.delete(spkcount, del_columns.astype(int), axis=1)
            spkcount = np.hsplit(spkcount, cumsum_nbins[:-1])

        else:
            if slideby is None:
                slideby = bin_size
            for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
                # first dividing in 1ms
                bins = np.arange(epoch.start, epoch.stop, 0.001)
                spkcount_ = np.asarray(
                    [np.histogram(_, bins=bins)[0] for _ in self.spiketrains]
                )

                # if signficant portion at end of epoch is not included then append zeros
                # if (frac := epoch.duration / bin_size % 1) > 0.7:
                #     extra_columns = int(100 * (1 - frac))
                #     spkcount_ = np.hstack(
                #         (spkcount_, np.zeros((neurons.n_neurons, extra_columns)))
                #     )

                slide_view = np.lib.stride_tricks.sliding_window_view(
                    spkcount_, int(bin_size * 1000), axis=1
                )[:, :: int(slideby * 1000), :].sum(axis=2)

                nbins[i] = slide_view.shape[1]
                spkcount.append(slide_view)

        return spkcount, nbins


class BinnedSpiketrain(DataWriter):
    """Class to hold binned spiketrains"""

    def __init__(
        self,
        spike_counts: np.ndarray,
        bin_size: float,
        t_start=0.0,
        neuron_ids=None,
        peak_channels=None,
        shank_ids=None,
        metadata=None,
    ) -> None:
        super().__init__()
        self.spike_counts = spike_counts
        self.bin_size = bin_size
        self.t_start = t_start
        self.peak_channels = peak_channels
        self.shank_ids = shank_ids
        if neuron_ids is None:
            self.neuron_ids = np.arange(self.n_neurons)

        self.metadata = metadata

    @staticmethod
    def from_neurons(neurons: Neurons, t_start=None, t_stop=None, bin_size=0.25):
        pass

    @property
    def spike_counts(self):
        return self._spike_counts

    @spike_counts.setter
    def spike_counts(self, arr):
        self._spike_counts = arr

    @property
    def n_neurons(self):
        return self.spike_counts.shape[0]

    @property
    def n_bins(self):
        return self.spike_counts.shape[1]

    @property
    def duration(self):
        return self.n_bins * self.bin_size

    @property
    def t_stop(self):
        return self.t_start + self.duration

    def add_metadata(self):
        pass

    @property
    def time(self):
        return np.arange(self.n_bins) * self.bin_size + self.t_start

    def _get_nan_bins(self):
        return np.isnan(self.spike_counts).any(axis=0)

    def get_pairwise_corr(self, pairs_bool=None, return_pair_id=False):
        """Pairwise correlation between pairs of binned of spiketrains

        Parameters
        ----------
        pairs_bool : 2D bool/logical array, optional
            Only these pairs are returned, by default None which means all pairs
        return_pair_id : bool, optional
            If true pair_ids are returned, by default False

        Returns
        -------
        corr
            1d vector of pairwise correlations
        """

        assert self.n_neurons > 1, "Should have more than 1 neuron"
        corr = np.corrcoef(self.spike_counts[:, ~self._get_nan_bins()])

        if pairs_bool is not None:
            assert (
                pairs_bool.shape[0] == pairs_bool.shape[1]
            ), "pairs_bool should be sqare shpae"
            assert (
                pairs_bool.shape[0] == self.n_neurons
            ), f"pairs_bool should be of {corr.shape} shape"
            pairs_bool = pairs_bool.astype("bool")
        else:
            pairs_bool = np.ones(corr.shape).astype("bool")

        pairs_bool = np.tril(pairs_bool, k=-1)

        return corr[pairs_bool]

    @property
    def firing_rate(self):
        return self.spike_counts / self.bin_size


class Mua(DataWriter):
    def __init__(
        self,
        spike_counts: np.ndarray,
        bin_size: float,
        t_start: float = 0.0,
        metadata=None,
    ) -> None:

        super().__init__()
        self.spike_counts = spike_counts
        self.t_start = t_start
        self.bin_size = bin_size
        self.metadata = metadata

    @property
    def spike_counts(self):
        return self._spike_counts

    @spike_counts.setter
    def spike_counts(self, arr: np.ndarray):
        assert arr.ndim == 1, "only 1 dimensional arrays are allowed"
        self._spike_counts = arr

    @property
    def bin_size(self):
        return self._bin_size

    @bin_size.setter
    def bin_size(self, val):
        self._bin_size = val

    @property
    def n_bins(self):
        return len(self._spike_counts)

    @property
    def duration(self):
        return self.n_bins * self.bin_size

    @property
    def t_stop(self):
        return self.t_start + self.duration

    @property
    def time(self):
        return np.arange(self.n_bins) * self.bin_size + self.t_start

    @property
    def firing_rate(self):
        return self.spike_counts / self.bin_size

    def get_smoothed(self, sigma=0.02, truncate=4.0):
        t_gauss = np.arange(-truncate * sigma, truncate * sigma, self.bin_size)
        gaussian = np.exp(-(t_gauss**2) / (2 * sigma**2))
        gaussian /= np.sum(gaussian)

        # numpy convolve is much faster than scipy
        spike_counts = np.convolve(self._spike_counts, gaussian, mode="same")
        # frate = gaussian_filter1d(self._frate, sigma=sigma, **kwargs)
        return Mua(spike_counts, t_start=self.t_start, bin_size=self.bin_size)

    # def _gaussian(self):
    #     # TODO fix gaussian smoothing binsize
    #     """Gaussian function for generating instantenous firing rate

    #     Returns:
    #         [array] -- [gaussian kernel centered at zero and spans from -1 to 1 seconds]
    #     """

    #     sigma = 0.020  # 20 ms
    #     binSize = 0.001  # 1 ms
    #     t_gauss = np.arange(-1, 1, binSize)
    #     gaussian = np.exp(-(t_gauss ** 2) / (2 * sigma ** 2))
    #     gaussian /= np.sum(gaussian)

    #     return gaussian

    def time_slice(self, t_start, t_stop):
        indices = (self.time >= t_start) & (self.time <= t_stop)

        return Mua(
            spike_counts=self.spike_counts[indices],
            bin_size=self.bin_size,
            t_start=t_start,
        )

    def to_dataframe(self):
        return pd.DataFrame({"time": self.time, "spike_counts": self.spike_counts})
