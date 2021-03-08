import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from parsePath import Recinfo


class trace_behavior:
    """Class to analyze trace_conditioning behavioral data"""

    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)
