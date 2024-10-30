from typing import Dict, List, Tuple, Optional, Callable, Union, Any


class GetAccessibleMixin:
    """ Implementors provide a default `get('an_attribute', a_default)` option so they can be accessed like dictionaries via passthrough instead of having to use getattr(....) 
    
    from neuropy.utils.mixins.gettable_mixin import GetAccessibleMixin
    
    History: 2024-10-23 11:36 Refactored from pyphocorehelpers.mixins.gettable_mixin.GetAccessibleMixin 

    """
    def get(self, attribute_name: str, default: Optional[Any] = None) -> Optional[Any]:
        """ Use the getattr built-in function to retrieve attributes """
        # If the attribute doesn't exist, return the default value
        return getattr(self, attribute_name, default)


class KeypathsAccessibleMixin:
    """ implementors support benedict-like keypath indexing and updating

    from neuropy.utils.mixins.gettable_mixin import KeypathsAccessibleMixin
    

    """
    def get_by_keypath(self, keypath: str) -> Any:
        """Gets the value at the specified keypath.
        Usage:
            # Get a value using keypath
            value = params.get_by_keypath('directional_train_test_split.training_data_portion')
            print(value)  # Output: 0.8333333333333334
                        
        """
        keys = keypath.split('.')
        value = self
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            else:
                raise AttributeError(f"Attribute '{key}' not found in '{value}'")
        return value

    def set_by_keypath(self, keypath: str, new_value: Any):
        """Sets the value at the specified keypath.
        Usage:
            # Set a value using keypath
            params.set_by_keypath('directional_train_test_split.training_data_portion', 0.9)
        """            
            
        keys = keypath.split('.')
        obj = self
        for key in keys[:-1]:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            else:
                raise AttributeError(f"Attribute '{key}' not found in '{obj}'")
        if hasattr(obj, keys[-1]):
            setattr(obj, keys[-1], new_value)
        else:
            raise AttributeError(f"Attribute '{keys[-1]}' not found in '{obj}'")

    def keypaths(self) -> list:
        """Returns all keypaths in the nested attrs classes.

        Usage:
            # Get all keypaths
            all_keypaths = params.keypaths()
            print(all_keypaths)     
        """
        paths = []
        def _collect_paths(obj, prefix=''):
            for field in getattr(obj, '__attrs_attrs__', []):
                key = field.name
                if prefix:
                    full_key = f'{prefix}.{key}'
                else:
                    full_key = key
                paths.append(full_key)
                value = getattr(obj, key)
                if hasattr(value, '__attrs_attrs__'):
                    _collect_paths(value, full_key)
        _collect_paths(self)
        return paths


    @classmethod
    def keypath_dict_to_nested_dict(cls, keypath_dict: Dict[str, Any]) -> Dict:
        """ converts a flat dict of keypath:value -> nested dicts of keys """
        nested_dict = {}
        for keypath, value in keypath_dict.items():
            keys = keypath.split('.')
            current_level = nested_dict
            for key in keys[:-1]:
                if key not in current_level:
                    current_level[key] = {}
                current_level = current_level[key]
            current_level[keys[-1]] = value
        return nested_dict
