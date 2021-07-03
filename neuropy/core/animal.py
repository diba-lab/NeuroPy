import pandas as pd
from . import DataWriter
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Animal(DataWriter):
    filename: Path = None
    name: str = None
    alias: str = None
    tag: str = None
    sex: str = None
    day = None  # day of experiment
    date = None  # date of experiment
    experimenter: str = None  # Experimenter's name
    experiment: str = None  # Experiments name
    weight = None  # weight of the animal during the recording session
    track = None  # what type track was used
    brain_region = None  # which brain regions were implanted
    Notes: dict = {}

    def __post_init__(self):
        super().__init__(filename=self.filename)

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict())

    def write_csv(self):
        pass

    def to_dict(self):
        return {
            "name": self.name,
            "alias": self.alias,
            "tag": self.tag,
            "sex": self.sex,
            "weight": self.weight,
            "experimenter": self.experimenter,
            "experiment": self.experiment,
            "day": self.day,
            "date": self.date,
            "probe": self.probe,
            "track": self.track,
            "brain_region": self.brain_region,
        }

    @staticmethod
    def from_dict(d):
        pass
