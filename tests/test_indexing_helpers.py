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
    from neuropy.utils.indexing_helpers import flatten, find_desired_sort_indicies, union_of_arrays, paired_incremental_sorting
    

class TestPairedIncrementalSorting(unittest.TestCase):

    def test_empty_lists(self):
        """Test with empty lists."""
        neuron_IDs_lists = []
        sortable_values_lists = []
        result = paired_incremental_sorting(neuron_IDs_lists, sortable_values_lists)
        self.assertEqual(result, [])

    def test_single_list(self):
        """Test with a single list."""
        neuron_IDs_lists = [[1, 2, 3]]
        sortable_values_lists = [[30, 20, 10]]
        result = paired_incremental_sorting(neuron_IDs_lists, sortable_values_lists)
        expected = [np.array([3, 2, 1])]
        np.testing.assert_array_equal(result, expected)

    def test_multiple_disjoint_lists(self):
        """Test with multiple lists."""
        neuron_IDs_lists = [[1, 2], [3, 4]]
        sortable_values_lists = [[20, 10], [30, 40]]
        result = paired_incremental_sorting(neuron_IDs_lists, sortable_values_lists)
        expected = [np.array([2, 1]), np.array([3, 4])]
        np.testing.assert_array_equal(result, expected)

    # def test_incremental_nature(self):
    #     """Test the incremental nature of sorting."""
    #     neuron_IDs_lists = [[1, 2], [2, 3], [1, 4]]
    #     sortable_values_lists = [[20, 30], [25, 15], [10, 5]]
    #     result = paired_incremental_sorting(neuron_IDs_lists, sortable_values_lists)
    #     expected = [np.array([1, 2]), np.array([2, 3, 1]), np.array([4, 2, 3, 1])]
    #     np.testing.assert_array_equal(result, expected)

    def test_length_mismatch(self):
        """Test with mismatch in length of ID and value lists."""
        neuron_IDs_lists = [[1, 2, 3], [4, 5]]
        sortable_values_lists = [[30, 20, 10], ]
        with self.assertRaises(AssertionError):
            paired_incremental_sorting(neuron_IDs_lists, sortable_values_lists)

    def test_realistic_identical_lists(self):
        """Test with multiple lists."""
        test_neuron_IDs_lists = [np.array([  5,   7,   9,  31,  32,  39,  41,  45,  46,  48,  50,  55,  61,  62,  64,  69,  72,  75,  76,  78,  79,  83,  84,  86,  88,  90,  91,  92,  93,  95,  99, 100, 101, 108]),
            np.array([  5,   7,   9,  31,  32,  39,  41,  45,  46,  48,  50,  55,  61,  62,  64,  69,  72,  75,  76,  78,  79,  83,  84,  86,  88,  90,  91,  92,  93,  95,  99, 100, 101, 108]),
            np.array([  5,   7,   9,  31,  32,  39,  41,  45,  46,  48,  50,  55,  61,  62,  64,  69,  72,  75,  76,  78,  79,  83,  84,  86,  88,  90,  91,  92,  93,  95,  99, 100, 101, 108]),
            np.array([  5,   7,   9,  31,  32,  39,  41,  45,  46,  48,  50,  55,  61,  62,  64,  69,  72,  75,  76,  78,  79,  83,  84,  86,  88,  90,  91,  92,  93,  95,  99, 100, 101, 108])]

        test_sortable_values_lists = [np.array([17,  6, 54, 30, 54, 22, 45,  7, 54, 51, 21, 29, 42, 52, 41, 24,  0, 45, 29, 20, 14, 19,  3,  9, 15, 24,  4, 11, 15,  4, 14, 50, 30, 48]), 
            np.array([21,  8, 58, 25, 28, 12, 31,  9,  8, 56, 31, 35,  9, 55, 40, 45, 29, 45, 39, 44, 45, 35,  8, 49, 57, 36,  8, 12, 23,  9, 27, 49, 30, 57]), 
            np.array([21, 31, 46, 32, 31, 29, 44, 45, 45, 44, 25, 29, 43, 44, 41, 27, 41, 38, 32, 26, 43, 21, 13, 38, 45, 30, 31, 21, 34, 18, 41, 39, 33, 45]),
            np.array([45, 17, 50, 24, 28, 16, 28, 16, 23, 48, 28, 21, 17, 34, 38, 33, 23, 34, 38, 38, 48, 33, 15, 39, 27, 32, 18, 23, 22, 51, 26, 39, 30, 48])]

        result = paired_incremental_sorting(test_neuron_IDs_lists, test_sortable_values_lists)
        expected = [np.array([ 72,  84,  91,  95,   7,  45,  86,  92,  79,  99,  88,  93,   5,  83,  78,  50,  39,  69,  90,  55,  76,  31, 101,  64,  61,  41,  75, 108, 100,  48,  62,   9,  32,  46]),
            np.array([ 72,  84,  91,  95,   7,  45,  86,  92,  79,  99,  88,  93,   5,  83,  78,  50,  39,  69,  90,  55,  76,  31, 101,  64,  61,  41,  75, 108, 100,  48,  62,   9,  32,  46]),
            np.array([ 72,  84,  91,  95,   7,  45,  86,  92,  79,  99,  88,  93,   5,  83,  78,  50,  39,  69,  90,  55,  76,  31, 101,  64,  61,  41,  75, 108, 100,  48,  62,   9,  32,  46]),
            np.array([ 72,  84,  91,  95,   7,  45,  86,  92,  79,  99,  88,  93,   5,  83,  78,  50,  39,  69,  90,  55,  76,  31, 101,  64,  61,  41,  75, 108, 100,  48,  62,   9,  32,  46])]
        np.testing.assert_array_equal(result, expected)


    # #TODO 2023-11-28 03:23: - [ ] Finish testing these:
    # test_case_identical_lists = [np.array([2, 9, 14, 19, 22]), np.array([2, 9, 14, 19, 22]), np.array([2, 9, 14, 19, 22]), np.array([2, 9, 14, 19, 22])]
    # test_case_identical_list_sort_weights = [np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4])]

    # test_case_alternating_disjoint_lists = [np.array([2, 9, 14, 19, 22]), np.array([1, 11, 20, 21, 23]), np.array([2, 9, 14, 19, 22]), np.array([1, 11, 20, 21, 23])]
    # test_case_identical_list_sort_weights = [np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4])]


    # test_case_all_unique_lists = [np.array([2, 9, 14, 19, 22]), np.array([1, 2, 11, 20, 21, 23]), np.array([5, 9, 15, 16]), np.array([1, 9, 19, 20, 21, 23])]
    # test_case_identical_list_sort_weights = [np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4, 5]), np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3, 4, 5])]
    # test_case_all_unique_sorted_lists = paired_incremental_sorting(test_case_all_unique_lists, test_case_identical_list_sort_weights)
    # test_case_all_unique_sorted_lists


# class TestIndexing(unittest.TestCase):
# 	def test_flatten(self):
# 		self.fail()

# 	def test_find_desired_sort_indicies(self):
# 		self.fail()

# 	def test_union_of_arrays(self):
# 		self.fail()

# 	def test_paired_incremental_sorting(self):
# 		self.fail()


if __name__ == '__main__':
    unittest.main()
