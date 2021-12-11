import unittest
import numpy as np
import pandas as pd
# import the package
import sys, os
from pathlib import Path

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



class TestPositionMethods(unittest.TestCase):

    def setUp(self):
        # Hardcoded:
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

        ## New Position Objects:
        self.position = Position.from_separate_arrays(self.t, self.x, y=self.y, lin_pos=self.lin_pos)
        self.position1D = Position.from_separate_arrays(self.t, self.x, lin_pos=self.lin_pos)
        
        # self.position_old = PositionOld.from_separate_arrays(self.t, self.x, self.y, lin_pos=self.lin_pos)
        # self.position1D_old = PositionOld.from_separate_arrays(self.t, self.x, lin_pos=self.lin_pos)                
        # print('np.shape(np.vstack((self.x, self.y))): {}'.format(np.shape(np.vstack((self.x, self.y)))))
        # print('np.shape(np.vstack((self.x))): {}'.format(np.shape(np.vstack((self.x)))))
        # print('np.shape(self.x): {}'.format(np.shape(self.x))) # (20,)
        
        ## Old Position Objects:
        self.position_old = PositionOld(np.vstack((self.x, self.y)))
        self.position1D_old = PositionOld(self.x)


    def tearDown(self):
        self.position=None
        


    def test_traces(self):
        # print('np.size(self.position.traces): {}'.format(np.size(self.position.traces)))
        self.assertGreater(np.size(self.position.traces), 0)
     
    def test_traces_shape_xy(self):
        # print('np.shape(self.position.traces): {}'.format(np.shape(self.position.traces)))
        old_traces = np.vstack((self.x, self.y))
        # print('np.shape(old_traces): {}'.format(np.shape(old_traces)))
        # self.assertGreater(np.shape(self.position.traces)[0], 0)
        self.assertEqual(np.shape(old_traces), np.shape(self.position.traces))
        
    # def test_traces_shape_x(self):
    #     print('np.shape(self.position1D.traces): {}'.format(np.shape(self.position1D.traces)))
    #     old_traces = np.vstack((self.x))
    #     print('np.shape(old_traces): {}'.format(np.shape(old_traces)))
    #     # self.assertGreater(np.shape(self.position.traces)[0], 0)
    #     self.assertEqual(np.shape(old_traces), np.shape(self.position1D.traces))

    # def test_comparison_new_old_xy(self):
    #     self.assertAlmostEquals(self.position.traces, self.position_old.traces)

    # def test_comparison_new_old_x(self):
    #     self.assertAlmostEquals(self.position1D.traces, self.position1D_old.traces)
    
    def test_comparison_speed2D(self):
        print('self.position.speed: {}'.format(self.position.speed))
        print('self.position_old.speed: {}'.format(self.position_old.speed))
        self.assertListEqual(list(self.position.speed), list(self.position_old.speed))
        
            
    def test_comparison_speed(self):
        print('self.position1D.speed: {}'.format(self.position1D.speed))
        print('self.position1D_old.speed: {}'.format(self.position1D_old.speed))
        self.assertListEqual(list(self.position1D.speed), list(self.position1D_old.speed))
        
    def test_df_extraction(self):
        xpos = self.position.x
        ypos = self.position.y
        
        pos_df = self.position.to_dataframe()
        # xy_pos = pos_df[['x','y']].to_numpy()
        # xy_pos = np.vstack((xpos, ypos)).T
        self.assertEqual(np.shape(pos_df[['x','y']].to_numpy()), np.shape(np.vstack((xpos, ypos)).T))
        # self.assertEquals(pos_df[['x','y']].to_numpy(), np.vstack((xpos, ypos)).T)
     
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
    