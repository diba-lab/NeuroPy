import sys
import numpy as np
import h5py
import hdf5storage # conda install hdf5storage
from pathlib import Path


def import_mat_file(mat_import_file='data/RoyMaze1/positionAnalysis.mat'):
	print('Loading matlab import file: {}...'.format(mat_import_file))
	data = hdf5storage.loadmat(mat_import_file, appendmat=False)
	print('done.')
	return data