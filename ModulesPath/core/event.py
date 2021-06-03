import numpy as np
from .datawriter import DataWriter


class Event(DataWriter):
    def __init__(self, times=None, labels=None) -> None:

        self.times = times
        self.labels = labels

    def to_dataframe(self):
        pass

    def add_event(self):
        pass

    def remove_event(self):
        pass