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
        filename=None,
    ) -> None:
        super().__init__(filename=filename)

        self.spiketrains = spiketrains
        if neuron_ids is None:
            self.neuron_ids = np.arange(len(self.spiketrains))
        self.waveforms = waveforms
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
        assert isinstance(self.spiketrains, list)
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

    def load(self):
        data = super().load()
        if data is not None:
            for key in data:
                setattr(self, key, data[key])

    def to_dict(self):

        self._check_integrity()

        data = {
            "spiketrains": self.spiketrains,
            "neuron_type": self.neuron_type,
            "neuron_ids": self.neuron_ids,
            "t_start": self.t_start,
            "t_stop": self.t_stop,
            "peak_channels": self.peak_channels,
            "waveforms": self.waveforms,
            "sampling_rate": self.sampling_rate,
            "filename": self.filename,
            "metadata": self.metadata,
        }
        return data

    @staticmethod
    def from_dict(d):

        spiketrains = d["spiketrains"]
        neuron_type = d["neuron_type"]

        neurons = Neurons(spiketrains)
        return Neurons

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

        return BinnedSpiketrain(spike_counts, t_start, bin_size, self.neuron_ids)

    def get_mua(self, t_start, t_stop, bin_size):
        all_spikes = self.get_all_spikes()
        bins = np.arange(t_start, t_stop, bin_size)
        frate = np.histogram(all_spikes, bins=bins)[0]
        return Mua(frate, t_start, bin_size)

    def add_jitter(self):
        pass


class BinnedSpiketrain(DataWriter):
    def __init__(
        self, spike_counts: np.ndarray, t_start, bin_size=0.5, neuron_ids=None
    ) -> None:
        super().__init__()
        self.spike_counts = spike_counts
        self.bin_size = bin_size
        self.t_start = t_start
        if neuron_ids is None:
            self.neuron_ids = np.arange(self.n_neurons)

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

    def to_dict(self):
        return {
            "spike_counts": self.spike_counts,
            "t_start": self.t_start,
            "bin_size": self.bin_size,
            "neuron_ids": self.neuron_ids,
        }

    @staticmethod
    def from_dict(d):
        return BinnedSpiketrain(
            d["spike_counts"], d["t_start"], d["bin_size"], d["neuron_ids"]
        )


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
