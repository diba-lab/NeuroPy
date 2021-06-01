import numpy as np
import pandas as pd

from .parsePath import Recinfo
from .core import Epoch


class behavior_epochs(Epoch):
    """Class for epochs within a session. Such as pre, maze and post."""

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ---- defining filenames --------
        filePrefix = self._obj.files.filePrefix
        filename = filePrefix.with_suffix(".epochs.npy")
        super().__init__(filename=filename)
        self.load()
        if self._epochs is not None:
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
        epoch_label = self._epochs[self._epochs["label"] == label_name]
        return epoch_label[["start", "stop"]].values.tolist()[0]

    def all_maze(self):
        """Make the entire session a maze epoch if that's all you are analyzing"""
        maze_time = [0, self._obj.duration]
        self.add_epochs(
            pd.DataFrame({"start": [0], "stop": [maze_time], "label": ["maze"]})
        )
