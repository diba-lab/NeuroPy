import csv
import linecache
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from behavior import behavior_epochs
from getPosition import ExtractPosition
from mathutil import contiguous_regions
from parsePath import Recinfo


class Track:
    def __init__(self, basepath: Recinfo) -> None:
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ------- defining file names ---------
        filePrefix = self._obj.files.filePrefix

        @dataclass
        class files:
            trackinfo: str = filePrefix.with_suffix(".tracksinfo.npy")

        self.files = files()

        self._load()

    def _load(self):
        if (f := self.files.trackinfo).is_file():
            tracks = np.load(f, allow_pickle=True).item()

            for name in tracks["mazes"]:
                setattr(self, name, TrackProcess(name, self._obj))

    def create(self, epoch_names):

        position = ExtractPosition(self._obj)
        assert hasattr(position, "x"), "First extract position"
        epochs = behavior_epochs(self._obj)
        periods = None
        if isinstance(epoch_names, str):
            periods = [epochs.times[epoch_names].to_list()]
            epoch_names = [epoch_names]
        elif all(isinstance(name, str) for name in epoch_names):
            periods = [epochs.times[_].to_list() for _ in epoch_names]

        posdata = position.data

        mazes = {}
        for name, epch in zip(epoch_names, periods):
            mazes[name] = epch
            maze_data = {
                "data": posdata[
                    (posdata.time > epch[0]) & (posdata.time < epch[1])
                ].reset_index(drop=True),
                "tracking_sRate": position.tracking_sRate,
            }
            np.save(
                self._obj.files.filePrefix.with_suffix(".tracks." + name + ".npy"),
                maze_data,
            )

        info = {"mazes": mazes, "nMazes": len(mazes)}
        np.save(self.files.trackinfo, info)


class TrackProcess:
    def __init__(self, name, obj: Recinfo) -> None:

        filePrefix = obj.files.filePrefix

        @dataclass
        class files:
            track: str = filePrefix.with_suffix(".tracks." + name + ".npy")

        self.files = files()
        self._load()

    def _load(self):
        if (f := self.files.track).is_file():
            posdata = np.load(f, allow_pickle=True).item()
            self.data = posdata["data"]
            self.tracking_sRate = posdata["tracking_sRate"]

    def linearize_position(self, sample_sec=3, method="isomap", plot=True):
        """linearize trajectory. Use method='PCA' for off-angle linear track, method='ISOMAP' for any non-linear track.

        Parameters
        ----------
        sample_sec : int, optional
            sample a point every sample_sec seconds for training ISOMAP, by default 3. Lower it if inaccurate results
        method : str, optional
            by default "PCA" (for straight tracks) or 'ISOMAP' (for any continuous track, untested on t-maze as of 12/22/2020)


        """
        xpos = self.data.x
        ypos = self.data.y
        position = np.vstack((xpos, ypos)).T
        xlinear = None
        if method == "pca":
            pca = PCA(n_components=1)
            xlinear = pca.fit_transform(position).squeeze()
        elif method == "isomap":
            imap = Isomap(n_neighbors=5, n_components=2)
            pos_ds = position[
                0 : -1 : np.round(int(self.tracking_sRate) * sample_sec)
            ]  # downsample points to reduce memory load and time
            imap.fit(pos_ds)
            iso_pos = imap.transform(position)
            # Keep iso_pos here in case we want to use 2nd dimension (transverse to track) in future...
            if iso_pos.std(axis=0)[0] < iso_pos.std(axis=0)[1]:
                iso_pos[:, [0, 1]] = iso_pos[:, [1, 0]]
            xlinear = iso_pos[:, 1]
        if plot:
            fig, ax = plt.subplots()
            fig.set_size_inches([28, 8.6])
            ax.plot(xlinear)
            ax.set_xlabel("Frame #")
            ax.set_ylabel("Linear Position")
            ax.set_title(method + " Sanity Check Plot")

        self.data["linear"] = xlinear
        posdata = np.load(self.files.track, allow_pickle=True).item()
        posdata["data"] = self.data

        np.save(self.files.track, posdata)

        self._load()

    def laps(self):
        pass
