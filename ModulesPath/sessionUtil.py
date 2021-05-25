import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from numpy.core.fromnumeric import var
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.preprocessing import normalize

from . import signal_process
from .mathutil import threshPeriods
from .parsePath import Recinfo


class SessionUtil:

    binSize = 0.001  # in seconds
    gauss_std = 0.025  # in seconds

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def export2Neuroscope(self, times, suffix="evt"):
        times = times * 1000  # convert to ms
        file_neuroscope = self._obj.files.filePrefix.with_suffix(f".evt.{suffix}")
        with file_neuroscope.open("w") as a:
            for beg, stop in times:
                a.write(f"{beg} start\n{stop} end\n")

    def getinterval(self, period, nwindows):

        interval = np.linspace(period[0], period[1], nwindows + 1)
        interval = [[interval[i], interval[i + 1]] for i in range(nwindows)]
        return interval

    def spectrogram(self, chan, period=None, window=4, overlap=2):

        eegSrate = self._obj.lfpSrate
        lfp = self._obj.geteeg(chans=chan, timeRange=period)

        specgram = signal_process.spectrogramBands(
            lfp, sampfreq=eegSrate, window=window, overlap=overlap
        )

        return specgram

    def savefile(self, variable, namesuffix):
        """Save files in the basepath for variables created by the user

        Parameters
        ----------
        variable : dict/array/list/dataframe
            [description]
        namesuffix : string
            name and suffix e.g,  coherence.npy , coherence.pkl
        """
        fileprefix = self._obj.files.filePrefix
        filename = fileprefix.with_suffix(namesuffix)
        np.save(filename, variable)
