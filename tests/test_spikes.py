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

        self.finalized_testing_file = tests_folder.joinpath('neuropy_pf_testing.h5')
        sess_identifier_key = 'sess'
        # Load the saved .h5 spikes_df and active_pos dataframes for testing:
        self.old_spikes_df = pd.read_hdf(self.finalized_testing_file, key=f'{sess_identifier_key}/spikes_df')
        
        self.temp_finalized_testing_file = Path(r'W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\output\test_data_new.h5')
        key: str = 'pfnd_new'
        self.spikes_df = SpikesAccessor.read_hdf(self.temp_finalized_testing_file, key=f'{key}/spikes')

    def test_get_by_neuron_id(self):
        # Test excluding certain neurons from the placefield
        spikes_df = deepcopy(self.spikes_df)
        _subset_spikes_df = spikes_df.spikes.sliced_by_neuron_type('pyramidal')

        self.assertTrue(len(_subset_spikes_df) <= len(self.spikes_df))
        # original_pf_neuron_ids = original_pf.included_neuron_IDs.copy()
        # subset_included_neuron_IDXs = np.arange(10) # only get the first 10 neuron_ids
        # subset_included_neuron_ids = original_pf_neuron_ids[subset_included_neuron_IDXs] # only get the first 10 neuron_ids
        # if self.enable_debug_printing:
        #     print(f'{original_pf_neuron_ids = }\n{subset_included_neuron_ids = }')
        # neuron_sliced_pf = deepcopy(original_pf)
        # neuron_sliced_pf = neuron_sliced_pf.get_by_id(subset_included_neuron_ids)
        # neuron_sliced_pf_neuron_ids = neuron_sliced_pf.included_neuron_IDs
        # if self.enable_debug_printing:
        #     print(f'{neuron_sliced_pf_neuron_ids = }')

        # self.assertTrue(np.all(neuron_sliced_pf_neuron_ids == subset_included_neuron_ids)) # ensure that the returned neuron ids actually equal the desired subset
        # self.assertTrue(np.all(np.array(neuron_sliced_pf.ratemap.neuron_ids) == subset_included_neuron_ids)) # ensure that the ratemap neuron ids actually equal the desired subset
        # self.assertTrue(len(neuron_sliced_pf.ratemap.tuning_curves) == len(subset_included_neuron_ids)) # ensure one output tuning curve for each neuron_id
        # self.assertTrue(np.all(np.isclose(neuron_sliced_pf.ratemap.tuning_curves, [original_pf.ratemap.tuning_curves[idx] for idx in subset_included_neuron_IDXs]))) # ensure that the tuning curves built for the neuron_slided_pf are the same as those subset as retrieved from the  original_pf



    def tearDown(self):
        # # Clean up the test file
        # if os.path.exists(self.hdf_tests_file):
        #     os.remove(self.hdf_tests_file)
        pass




if __name__ == '__main__':
    unittest.main()