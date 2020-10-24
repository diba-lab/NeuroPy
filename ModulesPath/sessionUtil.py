import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats
import scipy.signal as sg
from pathlib import Path
import matplotlib.gridspec as gridspec
import signal_process
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.preprocessing import normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib as mpl
from mathutil import threshPeriods
from parsePath import Recinfo


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
        lfp = self.geteeg(chans=chan, timeRange=period)

        specgram = signal_process.spectrogramBands(
            lfp, sampfreq=eegSrate, window=window, overlap=overlap
        )

        return specgram

