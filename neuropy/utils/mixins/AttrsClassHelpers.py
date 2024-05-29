from functools import wraps, partial, total_ordering
from enum import Enum, unique
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, List, Optional, Union, Dict, Tuple
import pandas as pd
import numpy as np

from attrs import define as original_define
from attrs import field, Factory, fields, fields_dict, asdict

""" 
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr

"""

def keys_only_repr(instance):
    """ specifies that this field only prints its .keys(), not its values.
    
    # Usage (within attrs class):
        computed_data: Optional[DynamicParameters] = serialized_field(default=None, repr=keys_only_repr)
        accumulated_errors: Optional[DynamicParameters] = non_serialized_field(default=Factory(DynamicParameters), is_computable=True, repr=keys_only_repr)
    
    """
    if (isinstance(instance, dict) or hasattr(instance, 'keys')):
        return f"keys={list(instance.keys())}"
    return repr(instance)


## Custom __repr__ for attrs-classes:

# def __repr__(self):
#     """ 
#     TrackTemplates(long_LR_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#         long_RL_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#         short_LR_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#         short_RL_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#         shared_LR_aclus_only_neuron_IDs: numpy.ndarray,
#         is_good_LR_aclus: NoneType,
#         shared_RL_aclus_only_neuron_IDs: numpy.ndarray,
#         is_good_RL_aclus: NoneType,
#         decoder_LR_pf_peak_ranks_list: list,
#         decoder_RL_pf_peak_ranks_list: list
#     )
#     """
#     # content = ", ".join( [f"{a.name}={v!r}" for a in self.__attrs_attrs__ if (v := getattr(self, a.name)) != a.default] )
#     # content = ", ".join([f"{a.name}:{strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
#     content = ",\n\t".join([f"{a.name}: {strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
#     # content = ", ".join([f"{a.name}" for a in self.__attrs_attrs__]) # 'TrackTemplates(long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder, shared_LR_aclus_only_neuron_IDs, is_good_LR_aclus, shared_RL_aclus_only_neuron_IDs, is_good_RL_aclus, decoder_LR_pf_peak_ranks_list, decoder_RL_pf_peak_ranks_list)'
#     return f"{type(self).__name__}({content}\n)"



@unique
class HDF_SerializationType(Enum):
    """ Specifies how a serialized field is stored, as an HDF5 Dataset or Attribute """
    DATASET = 0
    ATTRIBUTE = 1

    @property
    def required_tag(self):
        return HDF_SerializationType.requiredClassTags()[self.value]
        

    # Static properties
    @classmethod
    def requiredClassTags(cls):
        return np.array(['dataset','attribute'])

# ==================================================================================================================== #
# 2023-07-30 `attrs`-based classes Helper Mixin                                                                        #
# ==================================================================================================================== #
class AttrsBasedClassHelperMixin:
    """ heleprs for classes defined with `@define(slots=False, ...)` 
    
    from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define


    hdf_fields = BasePositionDecoder.get_serialized_dataset_fields('hdf')

    """
    @classmethod
    def get_fields_with_tag(cls, tag:str='hdf', invert:bool=False) -> Tuple[List, Callable]:
        def _fields_matching_query_filter_fn(an_attr, attr_value):
            """ return attributes only if they have serialization.{serialization_format} in their shape metadata. Captures `tag`. """
            return (tag in an_attr.metadata.get('tags', []))
            
        found_fields = []
        for attr_field in fields(cls):
            # attrs.Attribute
            query_condition: bool = _fields_matching_query_filter_fn(attr_field, None)            
            if invert:
                query_condition = (not query_condition)
            if query_condition:
                found_fields.append(attr_field) # .name
        return found_fields, _fields_matching_query_filter_fn

    @classmethod
    def get_serialized_fields(cls, serializationType: "HDF_SerializationType", serialization_format:str='hdf') -> Tuple[List, Callable]:
        """ general function for getting the list of fields with a certain serializationType as a list of attrs attributes and a filter to select them useful for attrs.asdict(...) filtering. """
        def _serialized_attribute_fields_filter_fn(an_attr, attr_value):
            """ return attributes only if they have serialization.{serialization_format} in their shape metadata. Captures `serialization_format` and `serializationType`. """
            return (an_attr.metadata.get('serialization', {}).get(serialization_format, False) and (serializationType.required_tag in an_attr.metadata.get('tags', [])))

        hdf_fields = []
        for attr_field in fields(cls):
            if _serialized_attribute_fields_filter_fn(attr_field, None): # pass None for value because it doesn't matter
                hdf_fields.append(attr_field) # attr_field.name
        # hdf_fields = [attr_field for attr_field in fields(cls) if _serialized_attribute_fields_filter_fn(attr_field, None)] # list comprehension is more concise
        return hdf_fields, _serialized_attribute_fields_filter_fn
    

    @classmethod
    def get_serialized_dataset_fields(cls, serialization_format:str='hdf') -> Tuple[List, Callable]:
        return cls.get_serialized_fields(serialization_format=serialization_format, serializationType=HDF_SerializationType.DATASET)

    @classmethod
    def get_serialized_attribute_fields(cls, serialization_format:str='hdf') -> Tuple[List, Callable]:
        return cls.get_serialized_fields(serialization_format=serialization_format, serializationType=HDF_SerializationType.ATTRIBUTE)
    

    def to_dict(self) -> Dict:
        return asdict(self)
    

    @classmethod
    def _test_find_fields_by_shape_metadata(cls, desired_keys_subset=None):
        """ tries to get all the fields that match the shape criteria. Not completely implemented, but seems to work.
        
        indices_fields_n_epochs = [field.name for field in class_fields if hasattr(field.metadata, 'shape') and field.metadata['shape'][0] == 'n_epochs']
        # # Get the values at epoch_IDX from a particular instance `active_result`:
        # epoch_IDX: int = 0
        # # values = [getattr(active_result, field)[epoch_IDX] for field in indices_fields_n_epochs]
        # # values = [getattr(active_result, field) for field in indices_fields_n_epochs]
        # values_dict = {field:getattr(active_result, field)[epoch_IDX] for field in indices_fields_n_epochs if field in desired_keys}
        # values_dict

        Usage:
            desired_keys_subset = ['most_likely_positions_list', 'p_x_given_n_list', 'marginal_x_list', 'marginal_y_list', 'most_likely_position_indicies_list', 'nbins', 'time_bin_containers', 'time_bin_edges']
        

        """
        class_fields = cls.__attrs_attrs__
        # indices_fields_n_epochs = [field.name for field in class_fields if hasattr(field.metadata, 'shape') and field.metadata['shape'][0] == 'n_epochs']
        indices_fields_n_epochs = [field.name for field in class_fields if 'shape' in field.metadata and field.metadata['shape'][0] == 'n_epochs']
        # print(f'indices_fields_n_epochs: {indices_fields_n_epochs}') # ['most_likely_positions_list', 'p_x_given_n_list', 'marginal_x_list', 'marginal_y_list', 'most_likely_position_indicies_list', 'spkcount', 'nbins', 'time_bin_containers', 'time_bin_edges', 'epoch_description_list']
        # desired_keys = ['most_likely_positions_list', 'p_x_given_n_list', 'marginal_x_list', 'marginal_y_list', 'most_likely_position_indicies_list', 'nbins', 'time_bin_containers', 'time_bin_edges']
        return [a_field for a_field in indices_fields_n_epochs if a_field in desired_keys]



# ==================================================================================================================== #
# Custom `@define` that automatically makes class inherit from `AttrsBasedClassHelperMixin`                            #
# ==================================================================================================================== #


custom_define = partial(original_define, slots=False)

# def custom_define(slots=False, **kwargs):
#     """ replaces the `@define` for classes to cause the class to inherity from `AttrsBasedClassHelperMixin` automatically and use slots=False by default!
    
#     from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define
#     @custom_define()
#     class AClass:
#         pass
    
#     """
#     mixin_cls = AttrsBasedClassHelperMixin

#     def wrap(cls):
#         # Apply the original attrs.define
#         new_cls = original_define(cls, slots=slots, **kwargs)

#         # If the class doesn't already inherit from the mixin, add it to its bases
#         if not issubclass(new_cls, mixin_cls):
#             new_cls.__bases__ = (mixin_cls,) + new_cls.__bases__

#         return new_cls

#     return wrap



def merge_metadata(default_metadata: Dict[str, Any], additional_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if additional_metadata:
        for key, value in additional_metadata.items():
            if key in default_metadata and isinstance(default_metadata[key], dict):
                default_metadata[key].update(value)
            else:
                default_metadata[key] = value
    return default_metadata



# ==================================================================================================================== #
# Custom `field`s                                                                                                      #
# ==================================================================================================================== #

# currently I'm indicating whether a field must be provided or whether it can be computed by setting the metadata['tags'] += ['computed']
def _mark_field_metadata_computable(metadata: Optional[Dict[str, Any]] = None):
    """ merely adds the `metadata['tags'] += ['computed']` to indicate that a field can be computed or whether it must be provided for a complete object. """
    return merge_metadata({'tags': ['computed']}, metadata)

def _mark_field_metadata_is_handled_custom(metadata: Optional[Dict[str, Any]] = None):
    """ merely adds the `metadata['tags'] += ['custom_hdf_implementation']` to indicate that a field will be handled in an overriden to_hdf implementation. """
    return merge_metadata({'tags': ['custom_hdf_implementation']}, metadata)



# For HDF serializable fields, they can either be serialized as a dataset or an attribute on the group or dataset.

def non_serialized_field(default: Optional[Any] = None, is_computable:bool=True, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> field:
    default_metadata = {
        'serialization': {'hdf': False, 'csv': False, 'pkl': True}
    }
    if is_computable:
        default_metadata['tags'] = ['computed']
    else:
        if metadata is not None:
            assert ('computed' not in metadata.get('tags', [])), f"'computed' is in the user-provided metadata but the user set is_computable=False!"
    return field(default=default, metadata=merge_metadata(default_metadata, metadata), **kwargs)

def serialized_field(default: Optional[Any] = None, is_computable:bool=False, serialization_fn: Optional[Callable]=None, is_hdf_handled_custom:bool=False, hdf_metadata: Optional[Dict]=None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> field:
    default_metadata = {
        'tags': ['dataset'],
        'serialization': {'hdf': True},
        'custom_serialization_fn': serialization_fn,
        'hdf_metadata': (hdf_metadata or {}),
    }
    if is_hdf_handled_custom:
        default_metadata = _mark_field_metadata_is_handled_custom(metadata=default_metadata)
    if is_computable:
        default_metadata = _mark_field_metadata_computable(metadata=default_metadata)
    return field(default=default, metadata=merge_metadata(default_metadata, metadata), **kwargs)


def serialized_attribute_field(default: Optional[Any] = None, is_computable:bool=False, serialization_fn: Optional[Callable]=None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> field:
    """ marks a specific field to be serialized as an HDF5 attribute on the group for this object """
    default_metadata = {
        'tags': ['attribute'],
        'serialization': {'hdf': True},
        'custom_serialization_fn': serialization_fn,
    }
    # if serialization_fn is not None:
    #     default_metadata['custom_serialization_fn'] = serialization_fn
        
    if is_computable:
        default_metadata = _mark_field_metadata_computable(metadata=default_metadata)
    return field(default=default, metadata=merge_metadata(default_metadata, metadata), **kwargs)



"""
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field

"""


# def to_dict(self):
#     # Excluded from serialization: ['_included_thresh_neurons_indx', '_peak_frate_filter_function']
#     # filter_fn = filters.exclude(fields(PfND)._included_thresh_neurons_indx, int)
#     filter_fn = lambda attr, value: attr.name not in ["_included_thresh_neurons_indx", "_peak_frate_filter_function"]
#     return asdict(self, filter=filter_fn) # serialize using attrs.asdict but exclude the listed properties

# ==================================================================================================================== #
# 2023-06-22 13:24 `attrs` auto field exploration                                                                      #
# ==================================================================================================================== #

# from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
# from attrs import asdict, fields, evolve

# ## For loop version:
# for a_field in fields(type(subset)):
# 	if 'n_epochs' in a_field.metadata.get('shape', ()):
# 		# is a field indexed by epochs
# 		print(a_field.name)
# 		print(a_field.value)

# # Find all fields that contain a 'n_neurons':
# epoch_indexed_attributes = [a_field for a_field in fields(type(subset)) if ('n_epochs' in a_field.metadata.get('shape', ()))]
# epoch_indexed_attributes

# # neuron_shape_index_for_attributes = [a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes]
# epoch_shape_index_for_attribute_name_dict = {a_field.name:a_field.metadata['shape'].index('n_epochs') for a_field in epoch_indexed_attributes} # need the actual attributes so that we can get the .metadata['shape'] from them and find the n_epochs index location
# epoch_shape_index_for_attribute_name_dict
# _temp_obj_dict = {k:v.take(indices=is_included_in_subset, axis=epoch_shape_index_for_attribute_name_dict[k]) for k, v in _temp_obj_dict.items()} # filter the n_epochs axis containing items to get a reduced dictionary
# evolve(subset, **_temp_obj_dict)

# def sliced_by_aclus(self, aclus):
#     """ returns a copy of itself sliced by the aclus provided. """
#     from attrs import asdict, fields, evolve
#     aclu_is_included = np.isin(self.original_1D_decoder.neuron_IDs, aclus)  #.shape # (104, 63)
#     def _filter_obj_attribute(an_attr, attr_value):
#         """ return attributes only if they have n_neurons in their shape metadata """
#         return ('n_neurons' in an_attr.metadata.get('shape', ()))            
#     _temp_obj_dict = asdict(self, filter=_filter_obj_attribute)
#     # Find all fields that contain a 'n_neurons':
#     neuron_indexed_attributes = [a_field for a_field in fields(type(self)) if ('n_neurons' in a_field.metadata.get('shape', ()))]
#     # neuron_shape_index_for_attributes = [a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes]
#     neuron_shape_index_for_attribute_name_dict = {a_field.name:a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes} # need the actual attributes so that we can get the .metadata['shape'] from them and find the n_neurons index location
#     _temp_obj_dict = {k:v.take(indices=aclu_is_included, axis=neuron_shape_index_for_attribute_name_dict[k]) for k, v in _temp_obj_dict.items()} # filter the n_neurons axis containing items to get a reduced dictionary
#     return evolve(self, **_temp_obj_dict)


# `attrs` object shape specifications, updating `LeaveOneOutDecodingAnalysisResult`
# from attrs import fields, fields_dict, asdict
# from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import LeaveOneOutDecodingAnalysisResult, TimebinnedNeuronActivity, LeaveOneOutDecodingResult

# LeaveOneOutDecodingAnalysisResult.__annotations__

# def _filter_obj_attribute(an_attr, attr_value):
# 	""" return attributes only if they have n_neurons in their shape metadata """
# 	return ('n_neurons' in an_attr.metadata.get('shape', ()))

# # Find all fields that contain a 'n_neurons':
# neuron_indexed_attributes = [a_field for a_field in fields(type(long_results_obj)) if ('n_neurons' in a_field.metadata.get('shape', ()))]
# # neuron_shape_index_for_attributes = [a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes]
# neuron_shape_index_for_attribute_name_dict = {a_field.name:a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes} # need the actual attributes so that we can get the .metadata['shape'] from them and find the n_neurons index location
# neuron_shape_index_for_attribute_name_dict
# shape_specifying_fields = {a_field.name:a_field.metadata.get('shape', None) for a_field in fields(type(long_results_obj)) if a_field.metadata.get('shape', None) is not None}
# shape_specifying_fields

# _temp_obj_dict = asdict(long_results_obj, filter=_filter_obj_attribute)
# _temp_obj_dict = {k:v.take(indices=aclu_is_included, axis=neuron_shape_index_for_attribute_name_dict[k]) for k, v in _temp_obj_dict.items()} # filter the n_neurons axis containing items to get a reduced dictionary



class SimpleFieldSizesReprMixin:
    """ Defines the __repr__ for implementors that only renders the implementors fields and their sizes

    from neuropy.utils.mixins.AttrsClassHelpers import SimpleFieldSizesReprMixin

    
    """

    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
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
