import pandas as pd
from . import DataWriter
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Animal(DataWriter):
    name: str = None
    alias: str = None
    tag: str = None
    sex: str = None
    weight = None  # weight of the animal during the recording session
    day = None  # day of experiment
    date = None  # date of experiment
    experimenter: str = None  # Experimenter's name
    experiment: str = None  # Experiments name
    track = None  # what type track was used
    brain_region = None  # which brain regions were implanted
    metadata = None  # any additional info

    def __post_init__(self):
        super().__init__()

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict())

    def write_csv(self):
        pass

    def to_dict(self):
        return {
            "name": self.name,
            "alias": self.alias,
            "tag": self.tag,
            "day": self.day,
            "date": self.date,
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
