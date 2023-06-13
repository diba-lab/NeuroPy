import unittest
import numpy as np
import pandas as pd
# import the package
import sys, os
from pathlib import Path
from scipy import stats

# Add Neuropy to the path as needed
tests_folder = Path(os.path.dirname(__file__))

try:
    import neuropy
except ModuleNotFoundError as e:    
    root_project_folder = tests_folder.parent
    print('root_project_folder: {}'.format(root_project_folder))
    neuropy_folder = root_project_folder.joinpath('neuropy')
    print('neuropy_folder: {}'.format(neuropy_folder))
    sys.path.insert(0, str(root_project_folder))
finally:
    from neuropy.core.position import Position
    from neuropy.core.position_old import PositionOld


class SimulatedData:
    def __init__(self, basepath) -> None:
        self.basepath = basepath

    def test_ripple_detection(self):
        pass

    def test_position(self):
        self.t = np.array([-0.888858 , -0.855297 , -0.822507 , -0.789218 , -0.755506 , -0.722752 , -0.688706 , -0.656252 , -0.622272 , -0.589054 , -0.555391 , -0.522546 , -0.489066 , -0.455254 , -0.420878 , -0.387854 , -0.354451 , -0.322581 , -0.287982 , -0.255434])
        self.x = np.array([104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278])
        self.y  = np.array([100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569])
        self.lin_pos = np.array([np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN])

        ## Loaded from the PositionTestingVariables.npz file saved out to the neuropy dir:
        test_variables_path = tests_folder.joinpath('{}.npz'.format('PositionTestingVariables'))
        loaded = np.load(test_variables_path)
        self.t = loaded['t']
        self.x = loaded['x']
        self.y = loaded['y']
        self.lin_pos = loaded['lin_pos']
        self.speed = loaded['speed']
    

    def test_swa_detection(self):
        pass

    def test_pbe_detection(self):
        pass

    def test_explained_variance(self):
        pass

    
