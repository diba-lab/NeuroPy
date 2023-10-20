import unittest
import dill
from attrs import define, field, Factory
from neuropy.utils.mixins.dict_representable import SubsettableDictRepresentable


@define(slots=False)
class ImplementingClass(SubsettableDictRepresentable):
    key1:str = field(default="value1")
    key2:str = field(default="value2")
    key3:str = field(default="value3")
    

@define(slots=False)
class HashableImplementingClass(SubsettableDictRepresentable):
    key1:str = field(default="value1")
    key2:str = field(default="value2")
    key3:str = field(default="value3")

    def __eq__(self, other):
        if isinstance(other, HashableImplementingClass):
            return self.to_dict() == other.to_dict()
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.to_dict().items())))



class SubsettableDictRepresentableTests(unittest.TestCase):
    def test_to_dict_all_keys(self):
        # Arrange
        implementing_class = ImplementingClass()

        # Act
        result = implementing_class.to_dict()

        # Assert
        expected_result = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        self.assertEqual(result, expected_result)

    def test_to_dict_subset_includelist(self):
        # Arrange
        implementing_class = ImplementingClass()
        includelist = ['key1', 'key3']

        # Act
        result = implementing_class.to_dict(subset_includelist=includelist)

        # Assert
        expected_result = {'key1': 'value1', 'key3': 'value3'}
        self.assertEqual(result, expected_result)

    def test_to_dict_subset_excludelist(self):
        # Arrange
        implementing_class = ImplementingClass()
        excludelist = ['key2']

        # Act
        result = implementing_class.to_dict(subset_excludelist=excludelist)

        # Assert
        expected_result = {'key1': 'value1', 'key3': 'value3'}
        self.assertEqual(result, expected_result)

    def test_keys_all_keys(self):
        # Arrange
        implementing_class = ImplementingClass()

        # Act
        result = implementing_class.keys()

        # Assert
        expected_result = ['key1', 'key2', 'key3']
        self.assertEqual(result, expected_result)

    def test_keys_subset_includelist(self):
        # Arrange
        implementing_class = ImplementingClass()
        includelist = ['key1', 'key3']

        # Act
        result = implementing_class.keys(subset_includelist=includelist)

        # Assert
        expected_result = ['key1', 'key3']
        self.assertEqual(result, expected_result)

    def test_keys_subset_excludelist(self):
        # Arrange
        implementing_class = ImplementingClass()
        excludelist = ['key2']

        # Act
        result = implementing_class.keys(subset_excludelist=excludelist)

        # Assert
        expected_result = ['key1', 'key3']
        self.assertEqual(result, expected_result)

    def test_keypaths_all_keys(self):
        # Arrange
        implementing_class = ImplementingClass()

        # Act
        result = implementing_class.keypaths()

        # Assert
        expected_result = ['key1', 'key2', 'key3']
        self.assertEqual(result, expected_result)

    def test_keypaths_subset_includelist(self):
        # Arrange
        implementing_class = ImplementingClass()
        includelist = ['key1', 'key3']

        # Act
        result = implementing_class.keypaths(subset_includelist=includelist)

        # Assert
        expected_result = ['key1', 'key3']
        self.assertEqual(result, expected_result)

    def test_keypaths_subset_excludelist(self):
        # Arrange
        implementing_class = ImplementingClass()
        excludelist = ['key2']

        # Act
        result = implementing_class.keypaths(subset_excludelist=excludelist)

        # Assert
        expected_result = ['key1', 'key3']
        self.assertEqual(result, expected_result)

    def test_pickle_unpickle(self):
        # Arrange
        original_obj = ImplementingClass()
        pickled_obj = None
        unpickled_obj = None

        # Act
        try:
            # Pickle the object
            pickled_obj = dill.dumps(original_obj)

            # Unpickle the object
            unpickled_obj = dill.loads(pickled_obj)
        except Exception as e:
            self.fail(f"Pickling/unpickling failed: {str(e)}")

        # Assert
        self.assertEqual(original_obj.to_dict(), unpickled_obj.to_dict())

    # Testing Hashability by Value:
    def test_dictionary_key(self):
        # Arrange
        obj1 = HashableImplementingClass()
        # obj2 = HashableImplementingClass()
        obj2 = HashableImplementingClass(key1='new_value1')
        
        # Act
        my_dict = {obj1: "value1", obj2: "value2"}

        # Assert
        self.assertEqual(my_dict[obj1], "value1")
        self.assertEqual(my_dict[obj2], "value2")

    def test_identity_by_value(self):
        # Arrange
        obj1 = HashableImplementingClass()
        # obj2 = HashableImplementingClass()
        obj2 = HashableImplementingClass(key1='new_value1')        
        self.assertNotEqual(obj1, obj2)

    def test_identity_by_value(self):
        # Arrange
        obj1 = HashableImplementingClass(key1='new_value_1_same')
        # obj2 = HashableImplementingClass()
        obj2 = HashableImplementingClass(key1='new_value_1_same')        
        self.assertEqual(obj1, obj2)


if __name__ == "__main__":
    unittest.main()
