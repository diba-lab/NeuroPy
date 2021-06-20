import pandas as pd
from . import DataWriter
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Animal(DataWriter):
    name = None
    alias = None
    tag = None
    sex = None
    day = None
    date = None  # date of experiment
    experimenter = None
    weight = None
    track = None
    brain_region = None
    Notes: dict = {}

    # def __init__(self) -> None:
    #     """
    #     Name:  name of the animal e.g. Roy, Ted, Kevin
    #     Alias:  code names as per lab standard e.g, RatN, Rat3254,
    #     Tag:  helps identifying category for analyses, e.g, sd, nsd, control, rollipramSD
    #     Sex: sex of animal
    #     Experimenter: Experimenter's name
    #     weight  : wieght of the animal during this recording
    #     Probe :  what probes were used
    #     BrainRegion :  which brain regions were implanted
    #     Day  : The number of experiments : Day1, Day2
    #     Date :  Date of experiment
    #     Experiment  : any name for experiment
    #     Track  : type of maze used, linear, l-shpaed
    #     files :
    #     StartTime : startime of the experiment in this format, 2019-10-11_03-58-54
    #     nFrames :
    #     deletedStart (minutes):  deleted from raw data to eliminate noise, required       for synching video tracking
    #     deletedEnd (minutes): deleted tiem from raw data
    #     Notes:  Any important notes

    #     """

    #     if metadatafile.is_file():
    #         metadata = pd.read_csv(metadatafile)
    #         self.name = metadata.Name[0]
    #         self.alias = metadata.Alias[0]
    #         self.tag = metadata.Tag[0]
    #         self.sex = metadata.Sex[0]
    #         self.day = metadata.Day[0]
    #         self.date = metadata.Date[0]

    #     else:
    #         metadata = pd.DataFrame(
    #             columns=[
    #                 "Name",
    #                 "Alias",
    #                 "Tag",
    #                 "Sex",
    #                 "Experimenter",
    #                 "weight",
    #                 "Probe",
    #                 "BrainRegion",
    #                 "Day",
    #                 "Date",
    #                 "Experiment",
    #                 "Track",
    #                 "files",
    #                 "StartTime",
    #                 "nFrames",
    #                 "deletedStart (minutes)",
    #                 "deletedEnd (minutes)",
    #                 "Notes",
    #             ]
    #         )
    #         metadata.to_csv(metadatafile)
    #         print(
    #             f"Template metadata file {metadatafile.name} created. Please fill it accordingly."
    #         )

    def to_dict(self):
        return
