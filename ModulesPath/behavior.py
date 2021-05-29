import time
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .getPosition import ExtractPosition
from .parsePath import Recinfo
from .core import WritableEpoch

# from ModulesPath.core import Epoch


class behavior_epochs(WritableEpoch):
    """Class for epochs within a session. Such as pre, maze and post."""

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ---- defining filenames --------
        filePrefix = self._obj.files.filePrefix
        filename = Path(str(filePrefix) + ".epochs.npy")
        super().__init__(filename=filename)
        self.load()
        if not self._epochs.empty:
            for epoch in self._epochs.itertuples():
                setattr(self, epoch.label, [epoch.start, epoch.stop])

    def add_epochs(self, epochs: pd.DataFrame):

        self._check_epochs(epochs)
        self.epochs = self.epochs.append(epochs)
        self._epochs.drop_duplicates(
            subset=["label"], keep="last", inplace=True, ignore_index=True
        )
        self.save()
        print("epochs updated/added")

    def __getitem__(self, label_name):
        label_pos = np.where(self._epochs["label"] == label_name)[0]
        epoch_label = self._epochs[self._epochs["label"] == label_name]
        return epoch_label[["start", "stop"]].values.tolist()[0]

    def all_maze(self):
        """Make the entire session a maze epoch if that's all you are analyzing"""
        position = ExtractPosition(self._obj.basePath)
        maze_time = np.array([position.t[0], position.t[-1]])
        epoch_times = {"MAZE": maze_time}

        np.save(self._obj.files.epochs, epoch_times)

        self._load()  # load these in for immediate use
