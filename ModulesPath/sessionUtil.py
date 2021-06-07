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
    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def getinterval(self, period, nwindows):

        interval = np.linspace(period[0], period[1], nwindows + 1)
        interval = [[interval[i], interval[i + 1]] for i in range(nwindows)]
        return interval
