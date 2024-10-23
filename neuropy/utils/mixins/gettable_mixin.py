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

