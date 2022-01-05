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
        
        





# assert is_iterable([0, 1, 2]), "list should return true for is_iterable"
# assert is_iterable(np.array([0, 1, 2])), "ndArray should return true for is_iterable"
# assert is_iterable(pd.DataFrame({'col', [0, 1, 2]})), "Pandas DataFrame should return true for is_iterable"


if __name__ == '__main__':
    unittest.main()
