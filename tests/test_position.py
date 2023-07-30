import os
import unittest
import numpy as np
# from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose
import pandas as pd
# import the package
import sys, os
from pathlib import Path

# testing extensions
# from tests.unittesting_extensions.numpy_helpers import NumpyTestCase
from unittesting_extensions.numpy_helpers import NumpyTestCase

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



class TestPositionMethods(NumpyTestCase):

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
        

        self.hdf_test_file = 'temp_unittest_hdf.h5'
        # Create a sample DataFrame for testing
        self.pos_df = pd.DataFrame({
            't': np.arange(10),
            'x': np.random.rand(10),
            'y': np.random.rand(10)
        })

        # Create a Position object
        self.hdf_tests_position = Position(self.pos_df)
        
    def tearDown(self):
        self.position=None
        # Clean up the test file
        os.remove(self.hdf_test_file)

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
        # self.assert_array_almost_equal(self.position.speed, self.position_old.speed)
        # self.assertAllClose(self.position.speed, self.position_old.speed)
            
    def test_comparison_speed(self):
        print('self.position1D.speed: {}'.format(self.position1D.speed))
        print('self.position1D_old.speed: {}'.format(self.position1D_old.speed))
        self.assertListEqual(list(self.position1D.speed), list(self.position1D_old.speed))
        # self.assert_array_almost_equal(self.position1D.speed, self.position1D_old.speed)
        

    def test_df_extraction(self):
        xpos = self.position.x
        ypos = self.position.y
        
        pos_df = self.position.to_dataframe()
        # xy_pos = pos_df[['x','y']].to_numpy()
        # xy_pos = np.vstack((xpos, ypos)).T
        self.assertEqual(np.shape(pos_df[['x','y']].to_numpy()), np.shape(np.vstack((xpos, ypos)).T))
        # self.assertEquals(pos_df[['x','y']].to_numpy(), np.vstack((xpos, ypos)).T)
     

    def test_grid_bin_bounds(self):
        from neuropy.utils.mathutil import compute_grid_bin_bounds
        bounds = compute_grid_bin_bounds(self.x, None)
        print(f'bounds: {bounds}')
        self.assertEqual(len(bounds), 2)
        self.assertEqual(len(bounds[0]), 2)
        self.assertIsNone(bounds[1])
        

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


    def test_to_hdf(self):
        # Write to HDF5
        self.hdf_tests_position.to_hdf(self.hdf_test_file, 'pos')

        # Now try to read the file and check it matches the original DataFrame
        read_df = pd.read_hdf(self.hdf_test_file, 'pos')
        pd.testing.assert_frame_equal(read_df, self.pos_df)

    def test_read_hdf(self):
        # Write to HDF5
        self.hdf_tests_position.to_hdf(self.hdf_test_file, 'pos')

        # Now try to read back the Position object
        read_position = Position.read_hdf(self.hdf_test_file, 'pos')

        # Check that the data matches the original
        pd.testing.assert_frame_equal(read_position._data, self.pos_df)

        # Check metadata, modify as needed
        self.assertEqual(read_position.metadata, self.hdf_tests_position.metadata)



if __name__ == '__main__':
    unittest.main()
    