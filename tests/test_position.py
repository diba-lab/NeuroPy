import unittest
import numpy as np
import pandas as pd
# import the package
import sys, os
from pathlib import Path

# Add Neuropy to the path as needed
try:
    import neuropy
except ModuleNotFoundError as e:
    tests_folder = Path(os.path.dirname(__file__))
    root_project_folder = tests_folder.parent
    print('root_project_folder: {}'.format(root_project_folder))
    neuropy_folder = root_project_folder.joinpath('neuropy')
    print('neuropy_folder: {}'.format(neuropy_folder))
    sys.path.insert(0, str(root_project_folder))
finally:
    from neuropy.core.position import Position




class TestPositionMethods(unittest.TestCase):

    def setUp(self):
        self.t = [-0.888858 , -0.855297 , -0.822507 , -0.789218 , -0.755506 , -0.722752 , -0.688706 , -0.656252 , -0.622272 , -0.589054 , -0.555391 , -0.522546 , -0.489066 , -0.455254 , -0.420878 , -0.387854 , -0.354451 , -0.322581 , -0.287982 , -0.255434]
        self.x = [104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278 , 104.287278]
        self.y  = [100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569 , 100.981569]
        self.lin_pos = [np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN , np.NaN]
        self.position = Position.from_separate_arrays(self.t, self.x, self.y, lin_pos=self.lin_pos)

    def tearDown(self):
        self.position=None
        


    def test_traces(self):
        print('np.size(self.position.traces): {}'.format(np.size(self.position.traces)))
        self.assertGreater(np.size(self.position.traces), 0)
     
    def test_traces_shape(self):
        print('np.shape(self.position.traces): {}'.format(np.shape(self.position.traces)))
        self.assertGreater(np.shape(self.position.traces)[0], 0)
        
        
     
    # def test_isupper(self):
    # 	self.assertTrue('FOO'.isupper())
    # 	self.assertFalse('Foo'.isupper())
        
        
    # def test_init_from_separate_arrays(self):
        

    # def test_split(self):
    # 	s = 'hello world'
    # 	self.assertEqual(s.split(), ['hello', 'world'])
    # 	# check that s.split fails when the separator is not a string
    # 	with self.assertRaises(TypeError):
    # 		s.split(2)

if __name__ == '__main__':
    unittest.main()
    