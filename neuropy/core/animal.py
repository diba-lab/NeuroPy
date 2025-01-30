import pandas as pd
from . import DataWriter
from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class Animal(DataWriter):
    name: str = None
    alias: str = None
    sex: str = None
    weight: float = None  # weight of the animal in grams during the recording session
    day: str = None  # day of experiment
    start: pd.Timestamp = None  # start time of experiment
    stop: pd.Timestamp = None  # end time of experiment
    experimenter: str = None  # Experimenter's name
    experiment: str = None  # Experiments name
    track: str = None  # what type track was used
    brain_region: str = None  # which brain regions were implanted
    probe: str = None  # which probes were used
    metadata: dict = None  # any additional info

    def __post_init__(self):
        super().__init__()

    def to_dataframe(self):
        # return pd.DataFrame(index=self.to_dict().keys())
        return pd.DataFrame.from_dict(self.to_dict(), orient="index")

    def write_csv(self):
        pass

    # @staticmethod
    # def from_dict(d):
    #     return Animal(**d)


if __name__ == "__main__":
    print('test')
