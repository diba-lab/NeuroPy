import sys
import numpy as np
import h5py
import hdf5storage # conda install hdf5storage
from pathlib import Path

from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter


def import_mat_file(mat_import_file='data/RoyMaze1/positionAnalysis.mat'):
	with ProgressMessagePrinter(mat_import_file, 'Loading', 'matlab import file'):
		data = hdf5storage.loadmat(mat_import_file, appendmat=False)
	return data