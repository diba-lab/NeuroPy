import unittest
from unittest.mock import MagicMock
import dill
import copy
from neuropy.utils.result_context import IdentifyingContext

class TestIdentifyingContext(unittest.TestCase):

    def setUp(self):
        # Create test instances of IdentifyingContext
        self.context1 = IdentifyingContext(format_name='kdiba', session_name='2006-6-08_14-26-15')
        self.context2 = IdentifyingContext(format_name='kdiba', session_name='2006-6-08_14-26-15')

    def test_add_context(self):
        self.context1.add_context(collision_prefix='prefix', additional_key='value')
        self.assertTrue(hasattr(self.context1, 'additional_key'))
        self.assertEqual(self.context1.additional_key, 'value')

    def test_adding_context(self):
        new_context = self.context1.adding_context(collision_prefix='prefix', additional_key='value')
        self.assertTrue(hasattr(new_context, 'additional_key'))
        self.assertEqual(new_context.additional_key, 'value')
        self.assertFalse(hasattr(self.context1, 'additional_key'))  # Original context should remain unchanged

    def test_merging_context(self):
        additional_context = IdentifyingContext(session_name='2006-6-08_14-26-15', filter_name='maze1_PYR')
        merged_context = self.context1.merging_context(collision_prefix='prefix', additional_context=additional_context)
        self.assertTrue(hasattr(merged_context, 'filter_name'))
        self.assertEqual(merged_context.filter_name, 'maze1_PYR')
        self.assertFalse(hasattr(self.context1, 'filter_name'))  # Original context should remain unchanged

    def test_or_operator(self):
        additional_context = IdentifyingContext(session_name='2006-6-08_14-26-15', filter_name='maze1_PYR')
        merged_context = self.context1 | additional_context
        self.assertTrue(hasattr(merged_context, 'filter_name'))
        self.assertEqual(merged_context.filter_name, 'maze1_PYR')
        self.assertFalse(hasattr(self.context1, 'filter_name'))  # Original context should remain unchanged

    def test_get_description(self):
        description = self.context1.get_description()
        self.assertEqual(description, 'kdiba_2006-6-08_14-26-15')

    def test_get_initialization_code_string(self):
        code_string = self.context1.get_initialization_code_string()
        self.assertEqual(code_string, "IdentifyingContext(format_name='kdiba',session_name='2006-6-08_14-26-15')")

    def test_hash_equality(self):
        self.assertEqual(hash(self.context1), hash(self.context2))
        self.assertEqual(self.context1, self.context2)

    def test_subtract(self):
        subtracted_context = IdentifyingContext.subtract(self.context1, self.context2)
        self.assertIsInstance(subtracted_context, IdentifyingContext)
        self.assertFalse(hasattr(subtracted_context, 'format_name'))
        self.assertFalse(hasattr(subtracted_context, 'session_name'))


    def test_pickling_unpickling(self):
        context = IdentifyingContext(name="test")
        context = context.add_context("test", key1="value1", key2="value2")

        # Pickle the context
        pickled_context = dill.dumps(context)

        # Unpickle the context
        unpickled_context = dill.loads(pickled_context)

        # Check if the unpickled context is equal to the original context
        self.assertEqual(context, unpickled_context)

        # Check if the unpickled context is a different instance
        self.assertIsNot(context, unpickled_context)

        # Check if the attributes of the unpickled context are equal to the original context
        self.assertEqual(context.name, unpickled_context.name)
        self.assertEqual(context.key1, unpickled_context.key1)
        self.assertEqual(context.key2, unpickled_context.key2)

        # Check if modifying the unpickled context does not affect the original context
        modified_context = copy.deepcopy(unpickled_context)
        modified_context.key1 = "modified"
        self.assertNotEqual(context.key1, modified_context.key1)
        

    # Testing Hashability by Value:
    def test_dictionary_key(self):
        # Arrange
        obj1 = IdentifyingContext(name='test1')
        obj2 = IdentifyingContext(name='test2')

        # Act
        my_dict = {obj1: "value1", obj2: "value2"}

        # Assert
        self.assertEqual(my_dict[obj1], "value1")
        self.assertEqual(my_dict[obj2], "value2")

    def test_identity_by_value(self):
        # Arrange
        obj1 = IdentifyingContext(name="test1")
        obj2 = IdentifyingContext(name="test2")

        # Assert
        self.assertNotEqual(obj1, obj2)

    def test_identity_by_value_with_same_values(self):
        # Arrange
        obj1 = IdentifyingContext(name="test")
        obj2 = IdentifyingContext(name="test")

        # Assert
        self.assertEqual(obj1, obj2)



if __name__ == '__main__':
    unittest.main()
