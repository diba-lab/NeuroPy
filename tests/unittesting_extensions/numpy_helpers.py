import unittest
import numpy as np
# from numpy.testing import assert_array_equal, assert_array_almost_equal, \
#     assert_array_less, assert_array_almost_equal_nulp, \
#     assert_array_max_ulp, assert_equal, assert_almost_equal, \
#     assert_approx_equal, assert_raises, assert_warns, \
#     assert_no_warnings, assert_allclose, assert_string_equal

from numpy.testing import *

class NumpyTestCase(unittest.TestCase):
    """ 

    Usage:
        from tests.unittesting_extensions.numpy_helpers import NumpyTestCase

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_assertion_methods()

    def _add_assertion_methods(self):
        np_testing_methods = [name for name in dir(np.testing) if name.startswith("assert")]
        for method_name in np_testing_methods:
            method = getattr(np.testing, method_name)
            if callable(method):
                setattr(self, method_name, self._create_assertion_method(method))

    def _create_assertion_method(self, method):
        def assertion_method(*args, **kwargs):
            try:
                method(*args, **kwargs)
            except AssertionError as e:
                if len(args) > 1:
                    msg = kwargs.get("msg", None)
                    if msg is not None:
                        e.args = (msg,) + e.args
                raise e

        return assertion_method
    


# class NumpyTestCase(unittest.TestCase):
#     """ 
    
#     Usage:
#         from tests.unittesting_extensions.numpy_helpers import NumpyTestCase
    
#     """
#     def assertArrayEqual(self, arr1, arr2, msg=None):
#         try:
#             assert_array_equal(arr1, arr2)
#         except AssertionError as e:
#             if msg is not None:
#                 e.args = (msg,) + e.args
#             raise e

#     def assertArrayAlmostEqual(self, arr1, arr2, decimal=6, msg=None):
#         try:
#             assert_array_almost_equal(arr1, arr2, decimal=decimal)
#         except AssertionError as e:
#             if msg is not None:
#                 e.args = (msg,) + e.args
#             raise e

#     def assertArrayLess(self, arr1, arr2, msg=None):
#         try:
#             assert_array_less(arr1, arr2)
#         except AssertionError as e:
#             if msg is not None:
#                 e.args = (msg,) + e.args
#             raise e

#     def assertArrayAlmostEqualNulp(self, arr1, arr2, nulp=1, msg=None):
#         try:
#             assert_array_almost_equal_nulp(arr1, arr2, nulp=nulp)
#         except AssertionError as e:
#             if msg is not None:
#                 e.args = (msg,) + e.args
#             raise e

#     def assertArrayMaxUlp(self, arr1, arr2, maxulp=1, dtype=None, msg=None):
#         try:
#             assert_array_max_ulp(arr1, arr2, maxulp=maxulp, dtype=dtype)
#         except AssertionError as e:
#             if msg is not None:
#                 e.args = (msg,) + e.args
#             raise e

#     def assertEqual(self, first, second, msg=None):
#         try:
#             assert_equal(first, second)
#         except AssertionError as e:
#             if msg is not None:
#                 e.args = (msg,) + e.args
#             raise e

#     def assertAlmostEqual(self, first, second, places=7, msg=None, delta=None):
#         try:
#             assert_almost_equal(first, second, places=places, msg=msg, delta=delta)
#         except AssertionError as e:
#             if msg is not None:
#                 e.args = (msg,) + e.args
#             raise e
        
#     def assertAlmostEqual(self, first, second, places=7, msg=None, delta=None):
#         try:
#             assert_almost_equal(first, second, places=places, msg=msg, delta=delta)
#         except AssertionError as e:
#             if msg is not None:
#                 e.args = (msg,) + e.args
#             raise e

#     def assertApproxEqual(self, first, second, rel_tol=None, abs_tol=None, msg=None):
#         try:
#             assert_approx_equal(first, second, rel_tol=rel_tol, abs_tol=abs_tol)
#         except AssertionError as e:
#             if msg is not None:
#                 e.args = (msg,) + e.args
#             raise e

#     def assertRaises(self, excClass, callableObj=None, *args, **kwargs):
#         try:
#             assert_raises(excClass, callableObj, *args, **kwargs)
#         except AssertionError as e:
#             if callableObj is not None:
#                 name = getattr(callableObj, '__name__', '')
#                 if not name:
#                     name = callableObj.__class__.__name__
#                 msg = f"{name} did not raise {excClass.__name__}"
#                 if msg is not None:
#                     e.args = (msg,) + e.args
#             raise e

#     def assertWarns(self, expected_warning, *args, **kwargs):
#         try:
#             assert_warns(expected_warning, *args, **kwargs)
#         except AssertionError as e:
#             raise e

#     def assertNoWarnings(self, *args, **kwargs):
#         try:
#             assert_no_warnings(*args, **kwargs)
#         except AssertionError as e:
#             raise e

#     def assertAllClose(self, actual, desired, rtol=1e-7, atol=0, equal_nan=False, err_msg='', verbose=True):
#         try:
#             assert_allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg, verbose=verbose)
#         except AssertionError as e:
#             raise e

#     def assertStringEqual(self, first, second, msg=None):
#         try:
#             assert_string_equal(first, second)
#         except AssertionError as e:
#             if msg is not None:
#                 e.args = (msg,) + e.args
#             raise e
        
#     # Add other assertion methods from numpy.testing as needed
#     # ...

# Example usage:
# class YourTestCase(NumpyTestCase):
#     def test_array_equality(self):
#         # Assume self.position and self.position_old are NumPy arrays
#         self.assertArrayEqual(self.position.speed, self.position_old.speed)
#         self.assertArrayAlmostEqual(self.position.values, self.expected_values, decimal=4)
#         # ...

    # def test_array_equality(self):
    #     # Assume self.position and self.position_old are NumPy arrays
    #     self.assert_array_equal(self.position.speed, self.position_old.speed)
    #     self.assert_array_almost_equal(self.position.values, self.expected_values, decimal=4)
    #     self.assert_array_less(self.position.speed, self.position_old.speed_max)
    #     self.assert_array_almost_equal_nulp(self.position.values, self.expected_values, nulp=2)
    #     self.assert_array_max_ulp(self.position.values, self.expected_values, maxulp=10, dtype=np.float64)
    #     self.assert_equal(self.position.count, 10)
    #     self.assert_almost_equal(self.position.accuracy, 0.123, places=3)
    #     self.assert_approx_equal(self.position.distance, 1000, rel_tol=0.1, abs_tol=10)
    #     self.assert_raises(ValueError, int, 'a')
    #     self.assert_warns(UserWarning, np.sqrt, -1)
    #     self.assert_no_warnings(np.sqrt, 1)
    #     self.assert_allclose(self.actual_values, self.expected_values, rtol=1e-5, atol=0.01)
    #     self.assert_string_equal(self.greeting, "Hello, World!")
    #     # ...



if __name__ == '__main__':
    unittest.main()
