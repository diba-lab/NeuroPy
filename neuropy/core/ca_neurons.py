import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from .datawriter import DataWriter
from copy import deepcopy


class Neurons(DataWriter):
    """Class to hold calcium imaging data and their labels, raw traces, etc."""

    def __init__(self):
        pass
