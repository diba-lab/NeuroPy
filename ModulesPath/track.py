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
            trackinfo: str = filePrefix.with_suffix(".tracks.info.npy")
            laps: str = filePrefix.with_suffix(".tracks.laps.npy")

        self.files = files()
        self.names = None

        self._load()

    def _load(self):
        if (f := self.files.trackinfo).is_file():
            tracks = np.load(f, allow_pickle=True).item()
            self.names = list(tracks.keys())
            self.data = tracks

            # for name in tracks["mazes"]:
            #     setattr(self, name, TrackProcess(name, self._obj))

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

        maze_data = {}
        for name, epch in zip(epoch_names, periods):
            maze_data[name] = posdata[
                (posdata.time > epch[0]) & (posdata.time < epch[1])
            ].reset_index(drop=True)

        np.save(self.files.trackinfo, maze_data)
        self._load()

    def __getitem__(self, track_name):
        return self.data[track_name]

    def __len__(self):
        return len(self.data)

    def linearize_position(
        self, track_name=None, sample_sec=3, method="isomap", plot=True
    ):
        """linearize trajectory. Use method='PCA' for off-angle linear track, method='ISOMAP' for any non-linear track.

        Parameters
        ----------
        sample_sec : int, optional
            sample a point every sample_sec seconds for training ISOMAP, by default 3. Lower it if inaccurate results
        method : str, optional
            by default "PCA" (for straight tracks) or 'ISOMAP' (for any continuous track, untested on t-maze as of 12/22/2020)

        """
        posinfo = ExtractPosition(self._obj)
        tracking_sRate = posinfo.tracking_sRate

        if track_name is None:
            track_name = self.names

        # ---- loading the data ----------
        alldata = np.load(self.files.trackinfo, allow_pickle=True).item()

        for name in track_name:
            xpos = alldata[name].x
            ypos = alldata[name].y
            position = np.vstack((xpos, ypos)).T
            xlinear = None
            if method == "pca":
                pca = PCA(n_components=1)
                xlinear = pca.fit_transform(position).squeeze()
            elif method == "isomap":
                imap = Isomap(n_neighbors=5, n_components=2)
                # downsample points to reduce memory load and time
                pos_ds = position[0 : -1 : np.round(int(tracking_sRate) * sample_sec)]
                imap.fit(pos_ds)
                iso_pos = imap.transform(position)
                # Keep iso_pos here in case we want to use 2nd dimension (transverse to track) in future...
                if iso_pos.std(axis=0)[0] < iso_pos.std(axis=0)[1]:
                    iso_pos[:, [0, 1]] = iso_pos[:, [1, 0]]
                xlinear = iso_pos[:, 0]
            if plot:
                fig, ax = plt.subplots()
                fig.set_size_inches([28, 8.6])
                ax.plot(xlinear)
                ax.set_xlabel("Frame #")
                ax.set_ylabel("Linear Position")
                ax.set_title(method + " Sanity Check Plot")

            alldata[name]["linear"] = xlinear

        # ---- saving the updated data -----------
        np.save(self.files.trackinfo, alldata)

        self._load()

    def plot(self, track_name=None):

        if track_name is None:
            track_name = self.names

        fig, ax = plt.subplots(1, len(track_name))

        for ind, name in enumerate(track_name):
            posdata = self[name]
            ax[ind].plot(posdata.x, posdata.y)
            ax[ind].set_title(name)

    def laps(self):
        """Divide track session into laps"""
        pass
