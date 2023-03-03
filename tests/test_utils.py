import unittest
import numpy as np
try:
    import modin.pandas as pd # modin is a drop-in replacement for pandas that uses multiple cores
except ImportError:
    import pandas as pd # fallback to pandas when modin isn't available
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


if __name__ == '__main__':
    unittest.main()
