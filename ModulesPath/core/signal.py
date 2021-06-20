import numpy as np


class Signal:
    def __init__(
        self,
        array,
        sampling_rate,
        t_start,
        t_stop,
    ) -> None:
        self.array = array
        self._sampling_rate = int(sampling_rate)

    @property
    def n_channels(self):
        return self.signal.shape[1]

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, srate):
        self._sampling_rate = srate

    def time_slice(self):
        pass
