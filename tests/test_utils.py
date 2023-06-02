import unittest
from neuropy.utils.misc import split_array
import numpy as np
import pandas as pd
# import the package
import sys, os
from pathlib import Path

from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter

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
    from neuropy.utils.misc import is_iterable
    from neuropy.utils.result_context import IdentifyingContext


class TestUtilityMethods(unittest.TestCase):

    def test_isiterable(self):
        self.assertFalse(is_iterable(0), "integer should return false for is_iterable")
        self.assertFalse(is_iterable(0.011), "float should return false for is_iterable")
        self.assertFalse(is_iterable('string_value'), "a string literal should return false for is_iterable")
        
        self.assertTrue(is_iterable(['string_value', 'string_value_2']), "a list of string literals should return true for is_iterable")
        self.assertTrue(is_iterable((0, 1, 2)), "tuple should return true for is_iterable")
        self.assertTrue(is_iterable([0, 1, 2]), "list should return true for is_iterable")
        self.assertTrue(is_iterable(set()), "set should return true for is_iterable")
        self.assertTrue(is_iterable(dict()), "dict should return true for is_iterable")
        
        self.assertTrue(is_iterable(np.array([0, 1, 2])), "ndArray should return true for is_iterable")
        self.assertTrue(is_iterable(pd.DataFrame({'col':[0, 1, 2]})), "Pandas DataFrame should return true for is_iterable")
        
        

    def test_progress_message_printer(self):
        # TODO: not sure how to test this
        mat_import_file = 'data/RoyMaze1/positionAnalysis.mat'
        with ProgressMessagePrinter(mat_import_file, 'Loading', 'matlab import file', returns_string=False):
            print('\t inside the with statement', end=' ', file=sys.stdout)

        # TODO: check the output buffer for something like this:
        # 'Loading matlab import file results to data/RoyMaze1/positionAnalysis.mat... 	 inside the with statement done.'
        # assert is_iterable([0, 1, 2]), "list should return true for is_iterable"
        # assert is_iterable(np.array([0, 1, 2])), "ndArray should return true for is_iterable"
        # assert is_iterable(pd.DataFrame({'col', [0, 1, 2]})), "Pandas DataFrame should return true for is_iterable"

    # def test_split_array(self):
    #     arr = np.array([1, 2, 3, 4, 5, 6])
    #     sub_element_lengths = np.array([2, 1, 3])
    #     # expected_output = [[1, 2], [3], [4, 5, 6]]
    #     expected_output = [np.array([1, 2]), np.array([3]), np.array([4, 5, 6])]
    #     output = split_array(arr, sub_element_lengths)
    #     print(f'expected_output: {expected_output}\n output: {output}')
    #     # self.assertEqual([a == b] in zip(output, expected_output)])
    #     # self.assertListEqual(output, expected_output)
    #     # self.assertTrue(np.allclose(output, expected_output))
        
    def test_sum_of_sub_element_lengths_not_equal_to_N(self):
        arr = np.array([1, 2, 3, 4, 5, 6])
        sub_element_lengths = np.array([2, 1, 2])
        with self.assertRaises(ValueError):
            split_array(arr, sub_element_lengths)
            
    def test_return_type(self):
        arr = np.array([1, 2, 3, 4, 5, 6])
        sub_element_lengths = np.array([2, 1, 3])
        self.assertIsInstance(split_array(arr, sub_element_lengths), list)
        
    # def test_empty_array(self):
    #     arr = np.array([])
    #     sub_element_lengths = np.array([])
    #     expected_output = []
    #     self.assertListEqual(split_array(arr, sub_element_lengths), expected_output).all()
        
    # def test_sub_element_lengths_containing_zeros(self):
    #     arr = np.array([1, 2, 3, 4, 5, 6])
    #     sub_element_lengths = np.array([2, 0, 3, 1])
    #     with self.assertRaises(ValueError):
    #         split_array(arr, sub_element_lengths)


    def test_identity_context_merge(self):
        """ tests merging two IdentityContext objects """        
        active_identifying_session_ctx = IdentifyingContext.init_from_dict({'format_name': 'kdiba', 'animal': 'gor01', 'exper_name': 'two', 'session_name': '2006-6-09_22-24-40'})
        display_context = IdentifyingContext(display_fn='DecodedEpochSlices', epochs='replays', decoder='long_results_obj')
        # Tests:
        self.assertEqual((active_identifying_session_ctx | display_context), active_identifying_session_ctx.adding_context('display_', **display_context.to_dict()))
        self.assertEqual((active_identifying_session_ctx | display_context), (active_identifying_session_ctx.merging_context('display_', display_context)))

    def test_identity_context_merge_overlapping_and_conflicting_keys(self):
        """ tests merging two IdentityContext objects """        
        active_identifying_session_ctx = IdentifyingContext.init_from_dict({'format_name': 'kdiba', 'animal': 'gor01', 'conflicting_key_name': 'two', 'session_name': '2006-6-09_22-24-40'})
        overlapping_and_conflicting_context = IdentifyingContext(display_fn='DecodedEpochSlices', epochs='replays', decoder='long_results_obj', conflicting_key_name='one') # conflicting_key_name='one' is the overlapping key but is the same as the key from `active_identifying_session_ctx`
        
        # Tests:
        with self.assertRaises(AssertionError):
            (active_identifying_session_ctx | overlapping_and_conflicting_context)

    def test_identity_context_merge_overlapping_but_not_conflicting_keys(self):
        """ tests merging two IdentityContext objects with non-conflicting keys (same value). Still should raise an AssertionError """        
        active_identifying_session_ctx = IdentifyingContext.init_from_dict({'format_name': 'kdiba', 'animal': 'gor01', 'conflicting_key_name': 'two', 'session_name': '2006-6-09_22-24-40'})
        overlapping_and_non_conflicting_context = IdentifyingContext(display_fn='DecodedEpochSlices', epochs='replays', decoder='long_results_obj', conflicting_key_name='two') # conflicting_key_name='two' is the overlapping key and is different from the key from `active_identifying_session_ctx`
        
        # Tests:
        with self.assertRaises(AssertionError):
            (active_identifying_session_ctx | overlapping_and_non_conflicting_context)

if __name__ == '__main__':
    unittest.main()
