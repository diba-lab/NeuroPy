
from typing import Optional, List, Dict, Tuple, Union
from attr import define, field, Factory, asdict, astuple
from neuropy.utils.mixins.gettable_mixin import GetAccessibleMixin, KeypathsAccessibleMixin
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_attribute_field, serialized_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_SerializationMixin, HDF_Converter


@define(slots=False)
class BaseConfig(KeypathsAccessibleMixin, GetAccessibleMixin):
    """ 2023-10-24 - Base class to enable successful unpickling from old pre-attrs-based classes (based on `DynamicParameters`) to attrs-based classes.`

    from neuropy.core.parameters import BaseConfig
    
    History: 2024-10-23 11:36 Refactored from {pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs.BaseConfig}
    
    """

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes (_mapping and _keys_at_init). Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        # del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        if ('_mapping' in state) and ('_keys_at_init' in state):
            # unpickling from the old DynamicParameters-based ComputationResult
            print(f'unpickling from old DynamicParameters-based computationResult')
            self.__dict__.update(state['_mapping'])
        else:
             # typical update
            self.__dict__.update(state)


    def get(self, attribute_name, default=None):
        """ Use the getattr built-in function to retrieve attributes """
        # If the attribute doesn't exist, return the default value
        return getattr(self, attribute_name, default)



# ==================================================================================================================== #
# ParametersContainer                                                                                                  #
# ==================================================================================================================== #
@define(slots=False, repr=False, eq=False)
class ParametersContainer(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseConfig):
    """ 
    from neuropy.core.parameters import ParametersContainer
    
	from neuropy.core.session.Formats.BaseDataSessionFormats.ParametersContainer
    from neuropy.core.session.Formats.BaseDataSessionFormats import ParametersContainer
    
    """
    epoch_estimation_parameters: dict = serialized_field(serialization_fn=(lambda f, k, v: HDF_Converter._convert_dict_to_hdf_attrs_fn(f, k, v))) # , serialization_fn=(lambda f, k, v: HDF_Converter._convert_dict_to_hdf_attrs_fn(f, k, v)))

    def to_dict(self) -> Dict:
        return self.__dict__.copy()
    
    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes  """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        # content = ",\n\t".join([f"{a.name}: {strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
        # return f"{type(self).__name__}({content}\n)"
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"
    
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)
        ## TODO: need to override





