import time
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .getPosition import ExtractPosition
from .parsePath import Recinfo

# from ModulesPath.core import Epoch


class behavior_epochs:
    """Class for epochs within a session which comprises of pre, maze and post

    Attributes:
        pre -- [seconds] timestamps for pre sleep
        maze -- [seconds] timestamps for MAZE period when the animal is on the track
        post -- [seconds] timestamps for sleep following maze exploration
        totalduration -- entire duration excluding brief peiods between epochs
    """

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ---- defining filenames --------
        filePrefix = self._obj.files.filePrefix
        self.filename = Path(str(filePrefix) + "_epochs.npy")
        # Epoch.__init__(filename)

        self._load()

    def _load(self):
        """Loads epochs files if that exists in the basepath"""

        if (f := self.filename).is_file():
            epochs = np.load(f, allow_pickle=True).item()

            totaldur = []
            self.times = pd.DataFrame(epochs)
            for (epoch, times) in epochs.items():  # alternative list(epochs)
                setattr(self, epoch.lower(), times)  # .lower() will be removed

                totaldur.append(np.diff(times))

            self.totalduration = np.sum(np.asarray(totaldur))

        else:
            print(f"Epochs file does not exist for {self.filename}")

    def __repr__(self):
        if (f := self.filename).is_file():
            epochs = np.load(f, allow_pickle=True).item()
            epoch_string = [f"{key}: {epochs[key]}\n" for key in epochs]
        else:
            epoch_string = "No epochs exist"

        return f"Epochs (seconds) \n" + "".join(epoch_string)

    def __getitem__(self, name):
        assert name in self.times, f"Epoch {name} does not exist"
        return self.times[name].to_list()

    # def __getattr__(self, name):
    #     return self[name]

    def make_epochs(self, new_epochs: dict):
        """Adds epochs to the sessions at given timestamps. If epoch file already exists then new epochs are merged.
        NOTE: If new_epochs have names common to previous existing epochs, values will be updated with new one.

        Parameters
        ----------
        new_epochs : dict
            'dict_key' is meaningful string, 'dict_value' should be 2 element array/list of epoch start/end times in
            seconds.
            Example: if you have a session with pre-sleep, maze1 running, maze2 running, and then post-sleep,
            you would enter:
            {'pre': [t1, t2], 'maze1': [t3, t4], 'maze2': [t5, t6], 'post': [t7, t8]}
        """

        assert isinstance(new_epochs, dict), "Dictionaries are only valid argument"
        length_epochs = np.array([len(new_epochs[_]) for _ in new_epochs])
        assert np.all(length_epochs == 2), "epochs can only have length of 2"

        if (f := self.filename).is_file():
            epochs = np.load(f, allow_pickle=True).item()
            epochs = {**epochs, **new_epochs}
        else:
            epochs = new_epochs

        np.save(self.filename, epochs)
        self._load()

    def getfromPosition(self):
        """user defines epoch boundaries from the positons by selecting a rectangular region in the plot"""
        pass

    def all_maze(self):
        """Make the entire session a maze epoch if that's all you are analyzing"""
        position = ExtractPosition(self._obj.basePath)
        maze_time = np.array([position.t[0], position.t[-1]])
        epoch_times = {"MAZE": maze_time}

        np.save(self._obj.files.epochs, epoch_times)

        self._load()  # load these in for immediate use
