import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from getPosition import ExtractPosition
from parsePath import Recinfo


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

        self.pre = None
        self.maze = None
        self.post = None

        if Path(self._obj.files.epochs).is_file():
            epochs = np.load(self._obj.files.epochs, allow_pickle=True).item()

            totaldur = []
            self.times = pd.DataFrame(epochs)
            for (epoch, times) in epochs.items():  # alternative list(epochs)
                setattr(self, epoch.lower(), times)  # .lower() will be removed

                totaldur.append(np.diff(times))

            self.totalduration = np.sum(np.asarray(totaldur))

        else:
            print("Epochs file does not exist...did not load epochs")

    def __str__(self):
        return "This creates behavioral epochs by loading positons and letting the user select a period which most likely represents maze"

    def getfromPosition(self):
        """user defines epoch boundaries from the positons by selecting a rectangular region in the plot
        """

        position = ExtractPosition(self._obj.basePath)

        def tellme(s):
            print(s)
            plt.title(s, fontsize=16)
            plt.draw()

        # Copy to clipboard

        # Define a rectangle by clicking two points

        t = position.t
        y = position.y
        x = position.x

        plt.clf()
        plt.setp(plt.gca(), autoscale_on=True)
        plt.plot(t[::4], y[::4])

        tellme("You will define a rectangle for track, click to begin")

        plt.waitforbuttonpress()

        while True:
            pts = []
            while len(pts) < 2:
                tellme("Select 2 edges with mouse")
                pts = np.asarray(plt.ginput(2, timeout=-1))
                if len(pts) < 2:
                    tellme("Too few points, starting over")
                    time.sleep(1)  # Wait a second

                pts = np.asarray(
                    [[pts[0, 0], 400], [pts[0, 0], 0], [pts[1, 0], 0], [pts[1, 0], 400]]
                )

            ph = plt.fill(pts[:, 0], pts[:, 1], "r", lw=2, alpha=0.6)

            tellme("Happy? Key click for yes, mouse click for no")

            if plt.waitforbuttonpress():
                break

            # Get rid of fill
            for p in ph:
                p.remove()
        maze_start = pts[0][0]  # in seconds
        maze_end = pts[2][0]  # in seconds

        pre_time = np.array([0, maze_start - 1])
        maze_time = np.array([maze_start, maze_end])
        post_time = np.array([maze_end + 1, t[-1]])
        epoch_times = {"PRE": pre_time, "MAZE": maze_time, "POST": post_time}

        np.save(self._obj.files.epochs, epoch_times)

    def getfromCSV(self):
        file = Path(str(self._obj.files.filePrefix) + "_epochs.csv")
        epochs = pd.read_csv(file, index_col=0)
        epochs_name = epochs.index.values.tolist()

        epochs_times = {}
        for name in epochs_name:
            epochs_times[name] = np.asarray(epochs.loc[name])

        np.save(self._obj.files.epochs, epochs_times)
