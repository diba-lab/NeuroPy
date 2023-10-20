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
    from neuropy.analyses.placefields import PlacefieldComputationParameters
    from neuropy.analyses.placefields import PfND
    from neuropy.core.neuron_identities import NeuronType
    from neuropy.core.flattened_spiketrains import SpikesAccessor, FlattenedSpiketrains
    from neuropy.utils.debug_helpers import debug_print_placefield, debug_print_subsession_neuron_differences
    from neuropy.utils.debug_helpers import debug_print_ratemap, debug_print_spike_counts, debug_plot_2d_binning, compare_placefields_info
    from neuropy.utils.debug_helpers import parameter_sweeps, _plot_parameter_sweep
    from neuropy.utils.debug_helpers import print_aligned_columns
    
def _compute_parameter_sweep(spikes_df, active_pos, all_param_sweep_options: dict) -> dict:
    """ Computes the PfNDs for all the swept parameters (combinations of grid_bin, smooth, etc)
    
    Usage:
        smooth_options = [(None, None), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
        grid_bin_options = [(1,1),(5,5),(10,10)]
        all_param_sweep_options = cartesian_product(smooth_options, grid_bin_options)
        param_sweep_option_n_values = dict(smooth=len(smooth_options), grid_bin=len(grid_bin_options)) 
        output_pfs = _compute_parameter_sweep(spikes_df, active_pos, all_param_sweep_options)

    """
    output_pfs = {} # empty dict

    for a_sweep_dict in all_param_sweep_options:
        a_sweep_tuple = frozenset(a_sweep_dict.items())
        output_pfs[a_sweep_tuple] = PfND.from_config_values(deepcopy(spikes_df).spikes.sliced_by_neuron_type('pyramidal'), deepcopy(active_pos.linear_pos_obj), **a_sweep_dict) # grid_bin=, etc
        
    return output_pfs





class TestPlacefieldsMethods(unittest.TestCase):

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

        finalized_testing_file = tests_folder.joinpath('neuropy_pf_testing.h5')
        sess_identifier_key='sess'
        # Load the saved .h5 spikes_df and active_pos dataframes for testing:
        self.spikes_df = pd.read_hdf(finalized_testing_file, key=f'{sess_identifier_key}/spikes_df')
        active_pos_df = pd.read_hdf(finalized_testing_file, key=f'{sess_identifier_key}/pos_df')
        self.active_pos = active_pos_df.position.to_Position_obj() # convert back to a full position object
        self.epochs = None # Epoch(...) # Create an Epoch object as needed
        
        # Create a PfND object
        self.config = PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(2, 2), grid_bin_bounds=((29.16, 261.7), (130.23, 150.99)), smooth=(2.0, 2.0), frate_thresh=1.0)
        self.pfnd = PfND(self.spikes_df, self.active_pos, self.epochs, config=self.config, position_srate=self.active_pos.sampling_rate)
        self.hdf_tests_file = 'test_pfnd.h5'



    def tearDown(self):
        # Clean up the test file
        if os.path.exists(self.hdf_tests_file):
            os.remove(self.hdf_tests_file)


    def assertMonotonicallyDecreasing(self, a_list):
        """ Assert that the list is monotonically decreasing """
        self.assertTrue(all(a_list[i] >= a_list[i+1] for i in range(len(a_list)-1)))
        


    def test_monotonically_decreasing_spikes_with_speed_filter(self):
        """ show that the number of spikes and active cells decrease monotonically with higher speed filter values. """
        speed_thresh_options = [0.0, 1.0, 25.0, 50.0, 100.0, 200.0]
        all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(speed_thresh=speed_thresh_options)
        output_pfs = _compute_parameter_sweep(self.spikes_df, self.active_pos, all_param_sweep_options)

        num_good_placefield_neurons_list, num_total_spikes_list, num_spikes_per_spiketrain_list = compare_placefields_info(output_pfs)
        if self.enable_debug_printing:
            print_aligned_columns(['speed_thresh', 'num_good_neurons', 'num_total_spikes'], 
                        [speed_thresh_options, num_good_placefield_neurons_list, num_total_spikes_list])
        if self.enable_debug_plotting:
            fig, axs = _plot_parameter_sweep(output_pfs, param_sweep_option_n_values) # Visual Debugging


        self.assertMonotonicallyDecreasing(num_total_spikes_list)
        self.assertMonotonicallyDecreasing(num_good_placefield_neurons_list)


    def test_monotonically_decreasing_spikes_with_increasing_frate_thresh(self):
        """ show that the number of spikes and active cells decrease monotonically with higher firing rate threshold (frate_thresh) values. """
        frate_thresh_options = [0.0, 0.1, 1.0, 5.0, 10.0, 100.0]
        all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(frate_thresh=frate_thresh_options)
        output_pfs = _compute_parameter_sweep(self.spikes_df, self.active_pos, all_param_sweep_options)

        num_good_placefield_neurons_list, num_total_spikes_list, num_spikes_per_spiketrain_list = compare_placefields_info(output_pfs)
        if self.enable_debug_printing:
            print_aligned_columns(['frate_thresh', 'num_good_neurons', 'num_total_spikes'], 
                        [frate_thresh_options, num_good_placefield_neurons_list, num_total_spikes_list])
        if self.enable_debug_plotting:
            fig, axs = _plot_parameter_sweep(output_pfs, param_sweep_option_n_values) # Visual Debugging

        self.assertMonotonicallyDecreasing(num_total_spikes_list)
        self.assertMonotonicallyDecreasing(num_good_placefield_neurons_list)


    def test_conform_to_position_bins(self):
        ## Generate Placefields with varying bin-sizes:
        ### Here we use frate_thresh=0.0 which ensures that differently binned ratemaps don't have different numbers of spikes or cells.
        smooth_options = [(None, None)]
        grid_bin_options = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
        all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(grid_bin=grid_bin_options, smooth=smooth_options, frate_thresh=[0.0])
        output_pfs = _compute_parameter_sweep(self.spikes_df, self.active_pos, all_param_sweep_options)
        num_good_placefield_neurons_list, num_total_spikes_list, num_spikes_per_spiketrain_list = compare_placefields_info(output_pfs)
        if self.enable_debug_printing:
            print_aligned_columns(['grid_bin x smooth', 'num_good_neurons', 'num_total_spikes'], [all_param_sweep_options, num_good_placefield_neurons_list, num_total_spikes_list], enable_checking_all_values_width=True)
        fine_binned_pf = list(output_pfs.values())[0]
        coarse_binned_pf = list(output_pfs.values())[-1]

        if self.enable_debug_printing:
            print(f'{coarse_binned_pf.bin_info = }\n{fine_binned_pf.bin_info = }')
        rebinned_fine_binned_pf = deepcopy(fine_binned_pf)
        rebinned_fine_binned_pf.conform_to_position_bins(coarse_binned_pf, force_recompute=True)
        self.assertTrue(rebinned_fine_binned_pf.bin_info == coarse_binned_pf.bin_info) # the bins must be equal after conforming

        num_good_placefield_neurons_list, num_total_spikes_list, num_spikes_per_spiketrain_list = compare_placefields_info(dict(zip(['coarse', 'original', 'rebinned'],[coarse_binned_pf, fine_binned_pf, rebinned_fine_binned_pf])))
        if self.enable_debug_printing:
            print_aligned_columns(['pf', 'num_good_neurons', 'num_total_spikes'], [['coarse', 'original', 'rebinned'], num_good_placefield_neurons_list, num_total_spikes_list], enable_checking_all_values_width=True)

        self.assertTrue(num_good_placefield_neurons_list[0] == num_good_placefield_neurons_list[-1]) # require the rebinned pf to have the same number of good neurons as the one that it conformed to
        self.assertTrue(num_total_spikes_list[0] == num_total_spikes_list[-1]) # require the rebinned pf to have the same number of total spikes as the one that it conformed to
        #  self.assertTrue(assert num_spikes_per_spiketrain_list[0] == num_spikes_per_spiketrain_list[-1]) # require the rebinned pf to have the same number of spikes in each spiketrain as the one that it conformed to


    def test_get_by_neuron_id(self):
        # Test excluding certain neurons from the placefield
        original_pf = PfND.from_config_values(spikes_df=deepcopy(self.spikes_df).spikes.sliced_by_neuron_type('pyramidal'), position=deepcopy(self.active_pos.linear_pos_obj), frate_thresh=0.0) # all other settings default
        original_pf_neuron_ids = original_pf.included_neuron_IDs.copy()
        subset_included_neuron_IDXs = np.arange(10) # only get the first 10 neuron_ids
        subset_included_neuron_ids = original_pf_neuron_ids[subset_included_neuron_IDXs] # only get the first 10 neuron_ids
        if self.enable_debug_printing:
            print(f'{original_pf_neuron_ids = }\n{subset_included_neuron_ids = }')
        neuron_sliced_pf = deepcopy(original_pf)
        neuron_sliced_pf = neuron_sliced_pf.get_by_id(subset_included_neuron_ids)
        neuron_sliced_pf_neuron_ids = neuron_sliced_pf.included_neuron_IDs
        if self.enable_debug_printing:
            print(f'{neuron_sliced_pf_neuron_ids = }')

        self.assertTrue(np.all(neuron_sliced_pf_neuron_ids == subset_included_neuron_ids)) # ensure that the returned neuron ids actually equal the desired subset
        self.assertTrue(np.all(np.array(neuron_sliced_pf.ratemap.neuron_ids) == subset_included_neuron_ids)) # ensure that the ratemap neuron ids actually equal the desired subset
        self.assertTrue(len(neuron_sliced_pf.ratemap.tuning_curves) == len(subset_included_neuron_ids)) # ensure one output tuning curve for each neuron_id
        self.assertTrue(np.all(np.isclose(neuron_sliced_pf.ratemap.tuning_curves, [original_pf.ratemap.tuning_curves[idx] for idx in subset_included_neuron_IDXs]))) # ensure that the tuning curves built for the neuron_slided_pf are the same as those subset as retrieved from the  original_pf

    # def test_pf(self):
        ## Unfinished.
        # t = np.linspace(0, 1, 240000)
        # y = np.sin(2 * np.pi * 12 * t) * 100

        # pos = Position(traces=y.reshape(1, -1), t_start=0, sampling_rate=120)

        # spktrns = []
        # for i in range(-100, 100, 30):
        #     indices = np.where((pos.x >= i) & (pos.x <= i + 20))[0]
        #     indices = np.random.choice(indices, 4000)
        #     spktrns.append(indices / 120)
        # spktrns = np.array(spktrns)

        # neurons = Neurons(spiketrains=spktrns, t_start=0, t_stop=2000)
        # pf1d = Pf1D(neurons=neurons, position=pos, speed_thresh=0.1, grid_bin=5)
        


    def test_computation_config_hashing(self):
        ## Hash testing:
        obj1 = PlacefieldComputationParameters(speed_thresh=15.0, grid_bin=None, smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5)
        obj2 = PlacefieldComputationParameters(speed_thresh=15.0, grid_bin=None, smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5)
        self.assertEqual(obj1, obj2, f'The hashes of two objects with the same values should be equal, but: hash(obj1): {hash(obj1)}, hash(obj2): {hash(obj2)}!')
        # self.assertEqual(hash(obj1), hash(obj2), f'The hashes of two objects with the same values should be equal, but: hash(obj1): {hash(obj1)}, hash(obj2): {hash(obj2)}!')
        # self.assertTrue(hash(obj1) == hash(obj2), f'Two objects with the same values should be equal, but they are not!')
        # self.assertTrue(obj1 == obj2, f'Two objects with the same values should be equal, but they are not!')
        # hash(obj1): 2090320457320539818, hash(obj2): 2090320457320539818!


    def test_to_hdf(self):
        # Write to HDF5
        self.pfnd.to_hdf(self.hdf_tests_file, 'test_pfnd')

        # Read back the DataFrames
        read_position = Position.read_hdf(self.hdf_tests_file, 'test_pfnd/pos')
        try:
            read_epochs = Epoch.read_hdf(self.hdf_tests_file, 'test_pfnd/epochs')
        except KeyError as e:
            # epochs can be None, in which case the serialized object will not contain the f'{key}/epochs' key.  'No object named test_pfnd/epochs in the file'
            read_epochs = None
        except Exception as e:
            # epochs can be None, in which case the serialized object will not contain the f'{key}/epochs' key
            print(f'Unhandled exception {e}')
            raise e

        read_spikes = SpikesAccessor.read_hdf(self.hdf_tests_file, 'test_pfnd/spikes')
        # Check that the data matches the original
        pd.testing.assert_frame_equal(read_position.df, self.active_pos.df)
        pd.testing.assert_frame_equal(read_spikes, self.spikes_df) # AssertionError: Attributes of DataFrame.iloc[:, 14] (column name="cell_type") are different
        if read_epochs is None:
            self.assertIsNone(self.epochs)
        else:
            pd.testing.assert_frame_equal(read_epochs, self.epochs)
        # Add checks for epochs and spikes as needed

    def test_read_hdf(self):
        # Write to HDF5
        self.pfnd.to_hdf(self.hdf_tests_file, 'test_pfnd')

        # Read back the PfND object
        read_pfnd = PfND.read_hdf(self.hdf_tests_file, 'test_pfnd')

        # Check that the data matches the original
        pd.testing.assert_frame_equal(read_pfnd.position.df, self.active_pos.df) # almost equal but datatype of column different
        pd.testing.assert_frame_equal(read_pfnd.spikes_df, self.spikes_df) # AssertionError: Attributes of DataFrame.iloc[:, 14] (column name="cell_type") are different
        if read_pfnd.epochs is None:
            self.assertIsNone(self.epochs)
        else:
            pd.testing.assert_frame_equal(read_pfnd.epochs, self.epochs)
            
        # Add checks for epochs and config as needed
        #TODO 2023-07-30 11:34: - [ ] Partially tested, need to test `read_pfnd.config``



if __name__ == '__main__':
    unittest.main()
    