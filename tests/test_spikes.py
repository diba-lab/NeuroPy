import unittest
import numpy as np
import pandas as pd
# import the package
import sys, os
from pathlib import Path
from copy import deepcopy

from neuropy.core.epoch import Epoch

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
    from neuropy.core import Position, Neurons
    from neuropy.core.neurons import NeuronType
    from neuropy.core.flattened_spiketrains import SpikesAccessor, FlattenedSpiketrains
    from neuropy.utils.debug_helpers import print_aligned_columns
    
class TestSpikesMethods(unittest.TestCase):
    """ 
    Testing files created with:

    TESTING:
        from neuropy.analyses.placefields import PfND
        from neuropy.core.epoch import Epoch
        from pandas.api.types import CategoricalDtype
        from neuropy.core.position import Position
        from neuropy.core.flattened_spiketrains import SpikesAccessor
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData

        # Write out files to neuropy tests:
        neuropy_testing_path = Path('./NeuroPy/tests')
        assert neuropy_testing_path.exists()
        hdf5_neuropy_tests_output_path_pkl: Path = neuropy_testing_path.joinpath('testing_spikes_df.pkl')
        hdf5_neuropy_tests_output_path_hdf: Path = neuropy_testing_path.joinpath('testing_spikes_df.h5')
        print(f'hdf5_neuropy_tests_output_path_pkl: {hdf5_neuropy_tests_output_path_pkl}')
        print(f'hdf5_neuropy_tests_output_path_hdf: {hdf5_neuropy_tests_output_path_hdf}')

        # Copy the objects that will be serialized (INCOMPLETE, objects computed in notebook):
        _pfnd_obj: PfND = long_one_step_decoder_1D.pf # self.position, self.spikes_df, self.epochs
        _spikes_df = deepcopy(_pfnd_obj.spikes_df)
        saveData(hdf5_neuropy_tests_output_path_pkl, _spikes_df) # Save out .pkl format
        _spikes_df.spikes.to_hdf(hdf5_neuropy_tests_output_path_hdf, key=f'/spikes_df') # note the .spikes accessor

    """

    def setUp(self):
        """ Corresponding load for Neuropy Testing file 'NeuroPy/tests/neuropy_pf_testing.h5': 
            ## Save for NeuroPy testing:
            finalized_testing_file='../NeuroPy/tests/neuropy_pf_testing.h5'
            sess_identifier_key='sess'
            spikes_df.to_hdf(finalized_testing_file, key=f'{sess_identifier_key}/spikes_df')
            active_pos.to_dataframe().to_hdf(finalized_testing_file, key=f'{sess_identifier_key}/pos_df', format='table')
        """
        self.enable_debug_plotting = False
        self.enable_debug_printing = True

        # self.old_hdf_testing_file = tests_folder.joinpath('neuropy_pf_testing.h5')
        self.hdf5_neuropy_tests_output_path_pkl: Path = tests_folder.joinpath('testing_spikes_df.pkl')
        self.hdf5_neuropy_tests_output_path_hdf: Path = tests_folder.joinpath('testing_spikes_df.h5')
        self.temp_hdf_tests_file = 'temp_unittest_hdf.h5' # temp file read/write for testing

        sess_identifier_key = 'sess'
        # Load the saved .h5 spikes_df and active_pos dataframes for testing:
        # self.old_spikes_df = pd.read_hdf(self.old_hdf_testing_file, key=f'{sess_identifier_key}/spikes_df')        
        self.spikes_df = SpikesAccessor.read_hdf(self.hdf5_neuropy_tests_output_path_hdf, key=f'/spikes_df')

    def tearDown(self):
        # Clean up the test file
        if os.path.exists(self.temp_hdf_tests_file):
            os.remove(self.temp_hdf_tests_file)
        # pass

    def test_from_hdf(self):
        loaded_spikes_df = SpikesAccessor.read_hdf(self.hdf5_neuropy_tests_output_path_hdf, key=f'/spikes_df')
        print(f'{loaded_spikes_df.dtypes}')
        self.assertIsInstance(loaded_spikes_df["cell_type"].values[0], NeuronType)
        self.assertTrue(True, "loaded successfully")

    def test_to_hdf(self):
        # Write to HDF5 using the accessor
        _spikes_df = deepcopy(self.spikes_df)
        _spikes_df.spikes.to_hdf(self.temp_hdf_tests_file, 'spikes')

        # Now try to read the file and check it matches the original DataFrame
        loaded_spikes_df = SpikesAccessor.read_hdf(self.temp_hdf_tests_file, 'spikes')
        self.assertIsInstance(loaded_spikes_df["cell_type"].values[0], NeuronType)
        pd.testing.assert_frame_equal(loaded_spikes_df, self.spikes_df)
        self.assertTrue(True, f'to_hdf was successful')






if __name__ == '__main__':
    unittest.main()