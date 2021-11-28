import pandas as pd
from . import DataWriter
from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class Animal(DataWriter):
    name: str = None
    alias: str = None
    tag: str = None
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

    def to_dict(self, recurrsively=False):
        return {
            "name": self.name,
            "alias": self.alias,
            "tag": self.tag,
            "day": self.day,
            "start": self.start,
            "stop": self.stop,
            "sex": self.sex,
            "weight": self.weight,
            "experimenter": self.experimenter,
            "experiment": self.experiment,
            "probe": self.probe,
            "track": self.track,
            "brain_region": self.brain_region,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d):
        return Animal(
            name=d["name"],
            alias=d["alias"],
            tag=d["tag"],
            sex=d["sex"],
            weight=d["weight"],
            day=d["day"],
            date=d["date"],
            experimenter=d["experimenter"],
            experiment=d["experiment"],
            track=d["track"],
            brain_region=d["brain_region"],
            metadata=d["metadata"],
        )

    @staticmethod
    def from_file(f):
        d = DataWriter.from_file(f)
        if d is not None:
            return Animal.from_dict(d)
        else:
            return None

