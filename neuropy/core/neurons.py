import numpy as np
import pandas as pd
import scipy.signal as sg
from .datawriter import DataWriter
from ..utils.ccg import correlograms
from pathlib import Path


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
            "ids": self.ids,
            "t_start": self.t_start,
            "t_stop": self.t_stop,
            "peak_channels": self.peak_channels,
            "waveforms": self.waveforms,
            "instfiring": self.instfiring,
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

    def n_spikes(self, t_start=None, t_stop=None, ids=None):
        spiketrains = self.get_spiketrains(ids)
        if t_start is None:
            t_start = self.t_start

        if t_stop is None:
            t_stop = self.t_stop

        return np.concatenate(
            [np.histogram(_, bins=[t_start, t_stop])[0] for _ in spiketrains]
        )

    def firing_rate(self, t_start=None, t_stop=None, ids=None):
        return self.n_spikes(t_start, t_stop, ids) / (t_stop - t_start)

    def binned_spiketrains(self, ids, period, binsize=0.25):
        """Get binned spike counts within a period for the given cells"""
        bins = np.arange(period[0], period[1] + binsize, binsize)
        return np.asarray(
            [np.histogram(self.spiketrains[_], bins=bins)[0] for _ in ids]
        )

    def _gaussian(self):
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

    def calculate_instfiring(self):
        spkall = np.concatenate(self.spiketrains)
        bins = np.arange(spkall.min(), spkall.max(), 0.001)
        spkcnt = np.histogram(spkall, bins=bins)[0]
        gaussKernel = self._gaussian()
        instfiring = sg.convolve(spkcnt, gaussKernel, mode="same", method="direct")
        self.instfiring = pd.DataFrame({"time": bins[1:], "frate": instfiring})

    def add_jitter(self):
        pass
