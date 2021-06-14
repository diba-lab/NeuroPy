import numpy as np


class Analogsignal:
    def __init__(self) -> None:
        self._sampling_rate = None

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, srate):
        self._sampling_rate = srate
