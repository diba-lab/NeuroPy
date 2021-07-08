import numpy as np
import pandas as pd
import scipy.signal as sg
from .datawriter import DataWriter


class Neurons(DataWriter):
    """Class to hold a group of spiketrains and their labels, ids etc."""

    def __init__(
        self,
        spiketrains: np.ndarray,
        t_stop,
        t_start=0.0,
        sampling_rate=1,
        neuron_ids=None,
        neuron_type=None,
        waveforms=None,
        peak_channels=None,
        shank_ids=None,
        filename=None,
        metadata=None,
    ) -> None:
        super().__init__(filename=filename)

        self.spiketrains = spiketrains
        if neuron_ids is None:
            self.neuron_ids = np.arange(len(self.spiketrains))
        assert (
            waveforms.shape[0] == self.n_neurons
        ), "Waveforms first dimension should match number of neurons"
        self.waveforms = waveforms
        self.shank_ids = shank_ids
        self.neuron_type = neuron_type
        self.peak_channels = peak_channels
        self.instfiring = None
        self._sampling_rate = sampling_rate
        self.t_start = t_start
        self.t_stop = t_stop
        self.metadata = {}

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def n_neurons(self):
        return len(self.spiketrains)

    def get_spiketrains(self, ids):
        spiketrains = [self.spiketrains[_] for _ in ids]
        return spiketrains

    def time_slice(self):
        pass

    def neuron_slice(self, ids):
        pass

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

    def _time_check(self, t_start, t_stop):
        if t_start is None:
            t_start = self.t_start

        if t_stop is None:
            t_stop = self.t_stop

        return t_start, t_stop

    def __str__(self) -> str:
        return f"# neurons = {self.n_neurons}"

    # def load(self):
    #     data = super().load()
    #     if data is not None:
    #         for key in data:
    #             setattr(self, key, data[key])

    def to_dict(self):

        # self._check_integrity()

        data = {
            "spiketrains": self.spiketrains,
            "t_stop": self.t_stop,
            "t_start": self.t_start,
            "sampling_rate": self.sampling_rate,
            "neuron_ids": self.neuron_ids,
            "neuron_type": self.neuron_type,
            "waveforms": self.waveforms,
            "peak_channels": self.peak_channels,
            "shank_ids": self.shank_ids,
            "filename": self.filename,
            "metadata": self.metadata,
        }
        return data

    @staticmethod
    def from_dict(d):

        neurons = Neurons(
            d["spiketrains"],
            d["t_stop"],
            d["t_start"],
            d["sampling_rate"],
            d["neuron_ids"],
            d["neuron_type"],
            d["waveforms"],
            d["peak_channels"],
            d["shank_ids"],
            d["filename"],
            d["metadata"],
        )
        return neurons

    def add_metadata(self):
        pass

    def get_all_spikes(self):
        return np.concatenate(self.spiketrains)

    def n_spikes(self, t_start=None, t_stop=None):
        if t_start is None:
            t_start = self.t_start

        if t_stop is None:
            t_stop = self.t_stop

        return np.concatenate(
            [np.histogram(_, bins=[t_start, t_stop])[0] for _ in self.spiketrains]
        )

    def firing_rate(self, t_start=None, t_stop=None):
        return self.n_spikes(t_start, t_stop) / (t_stop - t_start)

    def get_binned_spiketrains(self, t_start=None, t_stop=None, bin_size=0.25):
        if t_start is None:
            t_start = self.t_start

        if t_stop is None:
            t_stop = self.t_stop

        """Get binned spike counts within a period for the given cells"""
        bins = np.arange(t_start, t_stop + bin_size, bin_size)
        spike_counts = np.asarray(
            [np.histogram(_, bins=bins)[0] for _ in self.spiketrains]
        )

        return BinnedSpiketrain(
            spike_counts,
            t_start,
            bin_size,
            self.neuron_ids,
            self.peak_channels,
            self.shank_ids,
        )

    def get_mua(self, t_start, t_stop, bin_size):
        all_spikes = self.get_all_spikes()
        bins = np.arange(t_start, t_stop, bin_size)
        frate = np.histogram(all_spikes, bins=bins)[0]
        return Mua(frate, t_start, bin_size)

    def add_jitter(self):
        pass


class BinnedSpiketrain(DataWriter):
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

    def to_dict(self):
        return {
            "spike_counts": self.spike_counts,
            "t_start": self.t_start,
            "bin_size": self.bin_size,
            "neuron_ids": self.neuron_ids,
            "peak_channels": self.peak_channels,
            "shank_ids": self.shank_ids,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d):
        return BinnedSpiketrain(
            d["spike_counts"],
            d["t_start"],
            d["bin_size"],
            d["neuron_ids"],
            d["peak_channels"],
            d["shank_ids"],
            d["metadata"],
        )

    def get_pairwise_corr(self, cross_shanks=False, return_pair_id=False):

        shank_ids = self.shank_ids
        corr = np.corrcoef(self.spike_counts)

        selected_pairs = np.tril_indices(self.n_neurons, k=-1)
        if cross_shanks:
            selected_pairs = np.nonzero(
                np.tril(shank_ids.reshape(-1, 1) - shank_ids.reshape(1, -1))
            )

        corr = corr[selected_pairs]

        return corr


class Mua(DataWriter):
    def __init__(self, frate: np.array, t_start: float, bin_size: float) -> None:

        super().__init__()
        self.frate = frate
        self.t_start = t_start
        self.bin_size = bin_size

    @property
    def frate(self):
        return self._frate

    @frate.setter
    def frate(self, arr: np.ndarray):
        assert arr.ndim == 1, "only 1 dimensional arrays are allowed"
        self._frate = arr

    @property
    def bin_size(self):
        return self._bin_size

    @bin_size.setter
    def bin_size(self, val):
        self._bin_size = val

    @property
    def n_bins(self):
        return len(self.frate)

    @property
    def duration(self):
        return self.n_bins * self.bin_size

    @property
    def t_stop(self):
        return self.t_start + self.duration

    @property
    def time(self):
        return np.linspace(self.t_start, self.t_stop, self.n_bins)

    def smooth(self, sigma):
        gaussKernel = self._gaussian()
        self._frate = sg.convolve(
            self._frate, gaussKernel, mode="same", method="direct"
        )

    def _gaussian(self):
        # TODO fix gaussian smoothing binsize
        """Gaussian function for generating instantenous firing rate

        Returns:
            [array] -- [gaussian kernel centered at zero and spans from -1 to 1 seconds]
        """

        sigma = 0.020
        binSize = 0.001
        t_gauss = np.arange(-1, 1, binSize)
        A = 1 / np.sqrt(2 * np.pi * sigma ** 2)
        gaussian = A * np.exp(-(t_gauss ** 2) / (2 * sigma ** 2))

        return gaussian

    def to_dict(self):
        return {
            "frate": self._frate,
            "t_start": self.t_start,
            "bin_size": self.bin_size,
        }

    @staticmethod
    def from_dict(d):
        return Mua(d["frate"], d["t_start"], d["bin_size"])

    def to_dataframe(self):
        return pd.DataFrame({"time": self.time, "frate": self.frate})
