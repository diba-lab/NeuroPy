import unittest
from unittest.mock import MagicMock
import dill
import copy
from neuropy.utils.result_context import IdentifyingContext, CollisionOutcome

class TestIdentifyingContext(unittest.TestCase):

    def setUp(self):
        # Create test instances of IdentifyingContext
        self.context1 = IdentifyingContext(format_name='kdiba', session_name='2006-6-08_14-26-15')
        self.context2 = IdentifyingContext(format_name='kdiba', session_name='2006-6-08_14-26-15')
        # Create some IdentifyingContext instances for testing
        self.ic1 = IdentifyingContext(format_name='kdiba', animal='gor01', exper_name='one', session_name='2006-6-08_14-26-15')
        self.ic2 = IdentifyingContext(format_name='kdiba', animal='vvp01', exper_name='one', session_name='2006-4-09_17-29-30')
        self.ic3 = IdentifyingContext(format_name='kdiba', animal='vvp01', exper_name='two', session_name='2006-4-09_16-40-54')
        

        # An example dictionary containing dictionaries with values for each of the IdentifyingContexts:
        self.identifying_context_dict = { 
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'):{'grid_bin_bounds':((22.397021260868584, 245.6584673739576), (133.66465594522782, 155.97244934208123))},
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'):dict(grid_bin_bounds=(((17.01858788173554, 250.2171441367766), (135.66814125966783, 154.75073313142283)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'):{'grid_bin_bounds':(((29.088604852961407, 251.70402561515647), (138.496638485457, 154.30675703402517)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'):{'grid_bin_bounds':(((29.16, 261.7), (133.87292045454544, 150.19888636363635)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'):{'grid_bin_bounds':((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'):{'grid_bin_bounds':(((19.639345624112345, 248.63934562411234), (134.21607306829767, 154.57926689187622)))},
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'):dict(grid_bin_bounds=(((36.58620390950715, 248.91627658974846), (132.81136363636367, 149.2840909090909)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'):{'grid_bin_bounds':((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915))},
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_3-23-37'):{'grid_bin_bounds':(((29.64642522460817, 257.8732552112081), (106.68603845428224, 146.71219371189815)))},
        }
            
        
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


    def test_identity_is_unordered(self):
        # Arrange
        obj1 = IdentifyingContext(k1='a', k2='b')
        obj2 = IdentifyingContext(k2='b', k1='a')
        # Assert
        self.assertEqual(obj1, obj2)
        

    def test_identity_by_value_with_same_values(self):
        # Arrange
        obj1 = IdentifyingContext(name="test")
        obj2 = IdentifyingContext(name="test")

        # Assert
        self.assertEqual(obj1, obj2)


    def test_dictionary_key_not_ordered(self):
        """ test that values can be accessed in dict regardless of key order. """
        a_dict = {IdentifyingContext(k1='a', k2='b'): 'good'}
        self.assertEqual(a_dict[IdentifyingContext(k2='b', k1='a')], 'good')
        

    def test_query(self):
        # Test 1: Query for a single attribute
        self.assertTrue(self.ic1.query({'exper_name': 'one'}))
        self.assertFalse(self.ic1.query({'exper_name': 'two'}))
        
        # Test 2: Query for multiple attributes
        self.assertTrue(self.ic1.query({'exper_name': 'one', 'animal': 'gor01'}))
        self.assertFalse(self.ic1.query({'exper_name': 'one', 'animal': 'vvp01'}))
        
        # Test 3: Query for non-existent attribute
        self.assertFalse(self.ic1.query({'nonexistent_attribute': 'value'}))
    
    def test_init_from_dict(self):
        # Test: Check that init_from_dict initializes an IdentifyingContext instance correctly
        ic = IdentifyingContext.init_from_dict({'format_name': 'kdiba', 'animal': 'gor01', 'exper_name': 'one', 'session_name': '2006-6-08_14-26-15'})
        self.assertEqual(ic.format_name, 'kdiba')
        self.assertEqual(ic.animal, 'gor01')
        self.assertEqual(ic.exper_name, 'one')
        self.assertEqual(ic.session_name, '2006-6-08_14-26-15')

    def test_query_one(self):
        # Test 1: To find any relevant entries for the 'exper_name' == 'one'
        relevant_entries = [ic for ic in self.identifying_context_dict.keys() if ic.query({'exper_name': 'one'})]
        self.assertEqual(len(relevant_entries), 5, F"{relevant_entries}")

    def test_query_two(self):
        # Test 2: To find any relevant entries for the 'animal' == 'vvp01'
        relevant_entries = [ic for ic in self.identifying_context_dict.keys() if ic.query({'animal': 'vvp01'})]
        self.assertEqual(len(relevant_entries), 3)



if __name__ == '__main__':
    unittest.main()
