""" result_context.py

The general goal of these context objects is to be able to generate distinct identifiers for a given result so that it can be serialized to disk and analyzed later.

To do this systematically, we need to be able to identify



For human purposes, it's also incredibly useful to be able to generate minimal-difference versions of these contexts.

For example, if you generated two figures, one with a spatial bin size of 0.5 and another of 1.0 when generating titles for the two plots those two levels of the spatial bin size variable are all that you'd want to keep straight. The rest of the variables aren't changing, and so the minimal-difference required to discriminate between the two cases is only the levels of that variables.
{'bin_size': [0.5, 1.0]}

Let's say you save these out to disk and then a week later see that applying the exact same analyses (with those same two levels) to a different dataset produces an unexpected result at a value of 0.5. You don't remember such a wonky looking graph of 0.5 for the previous dataset, and want to figure out what's going wrong. 

To do this you'll now want to compare the 4 graphs, meaning you'll need to keep track:

Now you have two different datasets: 'earlier' and 'now', each with a minimal-difference of:
    'earlier': {'bin_size': [0.5, 1.0]}
    'now': {'bin_size': [0.5, 1.0]}

To uniquely identify them you'll probably want to visualize them as:

'earlier' (row):    | 'bin_size'=0.5 | 'bin_size'=1.0 | 
'now' (row):        | 'bin_size'=0.5 | 'bin_size'=1.0 |

Ultimately you only have two axes over which things can be compared (rows and columns)
There's one hidden one: 'tabs', 'windows'

Humans need things with distinct, visual groupings. Inclusion Sets, Exceptions (a single outlier rendered in juxtaposition to an inclusion set of the norms)

"""
import re # used in try_extract_date_from_session_name
import copy
from typing import Any, List, Dict, Optional, Union, Protocol
from enum import Enum
from functools import wraps # used for decorators
from attrs import define, field, Factory
from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage
from collections import defaultdict
from neuropy.utils.mixins.diffable import OrderedSet
from copy import deepcopy

import numpy as np
import pandas as pd # used for find_unique_values

from neuropy.utils.indexing_helpers import convert_to_dictlike
from neuropy.utils.mixins.diffable import DiffableObject
from neuropy.utils.mixins.dict_representable import SubsettableDictRepresentable
from neuropy.utils.mixins.gettable_mixin import GetAccessibleMixin

""" 
user_args, user_kwds = _keygen(func, ignored, *args, **kwds)

# Function arguments are converted into a key via:
```python
_args, _kwds = rounded_args(*args, **kwds)
_args, _kwds = _keygen(user_function, ignore, *_args, **_kwds)
key = keymap(*_args, **_kwds)
```

keymap: a tool for converting a function's input signature to an unique key

"""


class CollisionOutcome(Enum):
    """Describes how to update the context when a key that already exists is present."""
    IGNORE_UPDATED = "ignore_updated" # do nothing, ignoring the new value and keeping the existing one
    FAIL_IF_DIFFERENT = "fail" # throws an exception if the two values are not equal
    REPLACE_EXISTING = "replace_existing" # replaces the existing value for that key with the new one provided
    APPEND_USING_KEY_PREFIX = "append_using_key_prefix" # uses the collision_prefix provided to generate a new unique key and assigns that key the new value

@define(slots=False, eq=False) # eq=False makes hashing and equality by identity, which is appropriate for this type of object
class IdentifyingContext(GetAccessibleMixin, DiffableObject, SubsettableDictRepresentable):
    """ a general extnsible base context that allows additive member creation
    
        Should not hold any state or progress-related variables. 
    
        # SubsettableDict: provides `to_dict`, `keys`, `keypaths`
        
        
        Usage:
            from neuropy.utils.result_context import IdentifyingContext, CollisionOutcome
            
        The user should be able to query a list of IdentifyingContext items and find all that match a certain criteria.
        For an example dictionary containing dictionaries with values for each of the IdentifyingContexts:
            _specific_session_override_dict = { 
                IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'):{'grid_bin_bounds':((29.16, 261.70), (130.23, 150.99))},
                IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'):{'grid_bin_bounds':((22.397021260868584, 245.6584673739576), (133.66465594522782, 155.97244934208123))},
                IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'):dict(grid_bin_bounds=(((17.01858788173554, 250.2171441367766), (135.66814125966783, 154.75073313142283)))),
                IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'):{'grid_bin_bounds':(((29.088604852961407, 251.70402561515647), (138.496638485457, 154.30675703402517)))},
                IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'):{'grid_bin_bounds':(((29.16, 261.7), (133.87292045454544, 150.19888636363635)))},
                IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'):{'grid_bin_bounds':((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915))},
                IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'):{'grid_bin_bounds':(((19.639345624112345, 248.63934562411234), (134.21607306829767, 154.57926689187622)))},
                IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'):dict(grid_bin_bounds=(((36.58620390950715, 248.91627658974846), (132.81136363636367, 149.2840909090909)))),
                IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_3-23-37'):{'grid_bin_bounds':(((29.64642522460817, 257.8732552112081), (106.68603845428224, 146.71219371189815)))},
            }
            # Example Query 1: To find any relevent entries for the 'exper_name'=='one':
                    IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'):{'grid_bin_bounds':((29.16, 261.70), (130.23, 150.99))},
                    IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'):{'grid_bin_bounds':(((29.16, 261.7), (133.87292045454544, 150.19888636363635)))},
                    IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'):{'grid_bin_bounds':((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915))},
                    IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'):dict(grid_bin_bounds=(((36.58620390950715, 248.91627658974846), (132.81136363636367, 149.2840909090909)))),
                    IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_3-23-37'):{'grid_bin_bounds':(((29.64642522460817, 257.8732552112081), (106.68603845428224, 146.71219371189815)))},

                    # Test 1: To find any relevant entries for the 'exper_name' == 'one'
                    relevant_entries = [ic for ic, _ in identifying_context_list if ic.query({'exper_name': 'one'})]

            
            # Example Query 2: To find any relevent entries for the 'animal'=='vvp01':
                IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'):{'grid_bin_bounds':(((29.16, 261.7), (133.87292045454544, 150.19888636363635)))},
                IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'):{'grid_bin_bounds':((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915))},
                IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'):{'grid_bin_bounds':(((19.639345624112345, 248.63934562411234), (134.21607306829767, 154.57926689187622)))},
        
                # Test 2: To find any relevant entries for the 'animal' == 'vvp01'
                relevant_entries = [ctxt for ctxt, _ in self.identifying_context_list if ctxt.query({'animal': 'vvp01'})]
        
                
        Query Usage:
            # Test 2: To find any relevant entries for the 'animal' == 'vvp01'
            relevant_entries = [ctxt for ctxt, _ in self.identifying_context_list if ctxt.query({'animal': 'vvp01'})]

            # Test 2: To find any relevant entries for the 'animal' == 'vvp01'
            relevant_entries = {ctxt:v for ctxt,v in identifying_context_dict.items() if ctxt.query({'animal': 'vvp01'})]    
                
    """
    def __init__(self, **kwargs):
        super().__init__()
        # super(self.__class__, self).__init__()
        # super(IdentifyingContext, self).__init__()
        ## Sets attributes dnymically:
        for name, value in kwargs.items():
            setattr(self, name, value)
        
    @classmethod
    def try_init_from_session_key(cls, session_str: str) -> "IdentifyingContext":
        """ Tries to create a valid IdentifyingContext from the known session_context_keys and a session string like 'kdiba_pin01_one_fet11-01_12-58-54'
        """
        split_keys = session_str.split('_', maxsplit=3)
        assert len(split_keys) == 4, f"session_str: '{session_str}' could not be parsed into a valid IdentifyingContext. split_keys: {split_keys}"
        return cls(**dict(zip(cls._get_session_context_keys(), split_keys)))
    
    # Comparing/Resolving Functions ______________________________________________________________________________________ #

    @classmethod
    def matching(cls, context_iterable: Union[Dict["IdentifyingContext", Any], List["IdentifyingContext"]], criteria: Union[Dict[str, Any], "IdentifyingContext"]) -> Union[Dict["IdentifyingContext", Any], List["IdentifyingContext"]]:
        """ 
        Queries the iterable (either list or dict with IdentifyingContext as keys) and returns the values matching the criteria
        criteria={'animal': 'vvp01'}
        
        Usage:
        
        annotations_man = UserAnnotationsManager()
        identifying_context_dict = annotations_man.get_hardcoded_specific_session_override_dict()

        relevant_entries = IdentifyingContext.matching(identifying_context_dict, criteria={'animal': 'vvp01'})
        
        """
        if isinstance(context_iterable, (list, tuple)):
            relevant_entries = [ctxt for ctxt, _ in context_iterable if ctxt.query(criteria)]

        elif hasattr(context_iterable, 'items'):
            relevant_entries = {ctxt:v for ctxt,v in context_iterable.items() if ctxt.query(criteria)}
        else:
            raise ValueError

        return relevant_entries

    def query(self, criteria: Union[Dict[str, Any], "IdentifyingContext"]) -> bool:
        """
        Checks if the IdentifyingContext instance matches the given criteria.

        Parameters
        ----------
        criteria : Dict[str, Any]
            A dictionary where keys are attribute names and values are attribute values that an
            IdentifyingContext instance should have to match the criteria.

        Returns
        -------
        bool
            True if the IdentifyingContext instance matches the criteria, False otherwise.
        """
        for key, value in criteria.items():
            if not hasattr(self, key) or getattr(self, key) != value:
                return False
        return True

    @classmethod
    def find_unique_values(cls, context_iterable: List["IdentifyingContext"]) -> dict:
        """
        Takes a list of IdentifyingContext objects and finds all of the unique values
        for each of their shared keys.

        Parameters
        ----------
        ic_list : list
            A list of IdentifyingContext objects.

        Returns
        -------
        dict
            A dictionary where keys are attribute names shared by all IdentifyingContext objects in the list,
            and values are sets of unique attribute values.
            
        Example:
        
        {'format_name': {'kdiba'},
        'animal': {'gor01', 'pin01', 'vvp01'},
        'exper_name': {'one', 'two'},
        'session_name': {'11-02_17-46-44','2006-4-09_16-40-54', '2006-4-09_17-29-30', '2006-4-10_12-25-50', '2006-6-07_16-40-19', '2006-6-08_14-26-15', '2006-6-08_21-16-25', '2006-6-09_1-22-43', '2006-6-09_22-24-40', '2006-6-12_15-55-31', '2006-6-12_16-53-46', 'fet11-01_12-58-54'}
        }
        """
        unique_values = defaultdict(OrderedSet) # defaultdict provides a default value when a key is accessed and it doesn't exist. the `(set)` part here defines that the default value should be a new empty `set` type object. default int factory function which returns 0

        for ic in context_iterable:
            for key, value in ic.to_dict().items():
                unique_values[key].add(value) # if [key] doesn't exist, a new empty `set` is created before the .add call

        # Remove keys that are not shared by all IdentifyingContext objects
        shared_keys = OrderedSet.intersection(*(OrderedSet(ic.to_dict().keys()) for ic in context_iterable))
        unique_values = {key:list(values) for key, values in unique_values.items() if key in shared_keys}

        return unique_values

    @classmethod
    def find_longest_common_context(cls, context_iterable: Union[Dict["IdentifyingContext", Any], List["IdentifyingContext"]]) -> "IdentifyingContext":
        """ returns the context common to all entries in the provided iterable. 
        """
        unique_values_dict = IdentifyingContext.find_unique_values(context_iterable)
        non_leaf_unique_values = {k:v[0] for k, v in unique_values_dict.items() if len(v) == 1}
        common_context = IdentifyingContext(**non_leaf_unique_values)
        return common_context
            
    @classmethod
    def converting_to_relative_contexts(cls, common_context: "IdentifyingContext", context_iterable: Union[Dict["IdentifyingContext", Any], List["IdentifyingContext"]]):
        """ returns the iterable contexts relative to the provided common_context

        Useage:
            unique_values_dict = IdentifyingContext.find_unique_values(context_iterable)
            non_leaf_unique_values = {k:v[0] for k, v in unique_values_dict.items() if len(v) == 1}
            common_context = IdentifyingContext(**non_leaf_unique_values)
        
            common_context = find_longest_common_context(user_annotations)

            common_context_user_annotations = converting_to_relative_contexts(common_context, user_annotations)
            common_context_user_annotations

        """


        ## Convert to relative contexts
        # matching_entries = IdentifyingContext.matching(context_iterable, criteria=non_leaf_unique_values)

        if isinstance(context_iterable, list):
            relative_contexts_list = []
            for a_ctx in context_iterable:
                a_relative_context = a_ctx - common_context
                relative_contexts_list.append(a_relative_context)
            return relative_contexts_list
        else:
            relative_contexts_dict = {}
            for a_ctx, v in context_iterable.items():
                a_relative_context = a_ctx - common_context
                relative_contexts_dict[a_relative_context] = v

            return relative_contexts_dict
                    

    
                
    @staticmethod
    def _get_session_context_keys() -> List[str]:
        return ['format_name','animal','exper_name', 'session_name']

    @classmethod
    def resolve_key(cls, duplicate_ctxt: "IdentifyingContext", name:str, value, collision_prefix:str, strategy:CollisionOutcome=CollisionOutcome.APPEND_USING_KEY_PREFIX):
        """ensures no collision between attributes occur, and if they do resolve them according to strategy. e.g. rename them with an identifying prefix
        Returns the resolved key (str) or None.
        """
        if hasattr(duplicate_ctxt, name):
            # Check whether the existing context already has that key:
            if (getattr(duplicate_ctxt, name) == value):
                # Check whether it is the same value or not:
                # the existing context has the same value for the overlapping key as the current one. Permit the merge.
                final_name = name # this will not change the result
            else:
                # the keys exist on both and they have differing values. Try to resolve with the `collision_prefix`:                
                if strategy.name == CollisionOutcome.IGNORE_UPDATED.name:
                    final_name = None # by returning None the caller will know the keep the existing.
                elif strategy.name == CollisionOutcome.FAIL_IF_DIFFERENT.name:
                    print(f'ERROR: namespace collision in resolve_key! attr with name "{name}" already exists and the value differs: existing_value: {getattr(duplicate_ctxt, name)} new_value: {value}. Using collision_prefix to add new name: "{final_name}"!')
                    raise KeyError
                elif strategy.name == CollisionOutcome.REPLACE_EXISTING.name:
                    final_name = name # this will overwrite the old value because the returned key is the same.
                elif strategy.name == CollisionOutcome.APPEND_USING_KEY_PREFIX.name:
                    ## rename the current attribute to be set by appending a prefix
                    assert collision_prefix is not None, f"namespace collision in `adding_context(...)`! attr with name '{name}' already exists but the value differs: existing_value: {getattr(duplicate_ctxt, name)} new_value: {value}! Furthermore 'collision_prefix' is None!"
                    final_name = f'{collision_prefix}{name}'
                    print(f'WARNING: namespace collision in resolve_key! attr with name "{name}" already exists and the value differs: existing_value: {getattr(duplicate_ctxt, name)} new_value: {value}. Using collision_prefix to add new name: "{final_name}"!')
        else:
            final_name = name
        return final_name

    # Adding/Updating Functions __________________________________________________________________________________________ #
    
    def add_context(self, collision_prefix:str, strategy:CollisionOutcome=CollisionOutcome.APPEND_USING_KEY_PREFIX, **additional_context_items):
        """ adds the additional_context_items to self 
        collision_prefix: only used when an attr name in additional_context_items already exists for this context and the values of that attr are different
        
        """
        for name, value in additional_context_items.items():
            # ensure no collision between attributes occur, and if they do rename them with an identifying prefix
            final_name = self.resolve_key(self, name, value, collision_prefix, strategy=strategy)
            if final_name is not None:
                # Set the new attr
                setattr(self, final_name, value)
        
        return self

    def adding_context(self, collision_prefix:str, strategy:CollisionOutcome=CollisionOutcome.APPEND_USING_KEY_PREFIX, **additional_context_items) -> "IdentifyingContext":
        """ returns a new IdentifyingContext that results from adding additional_context_items to a copy of self 
        collision_prefix: only used when an attr name in additional_context_items already exists for this context and the values of that attr are different
        
        """
        # assert isinstance(collision_prefix, str), f"collision_prefix must be provided as a string! Did you forget to provide it?"
        duplicate_ctxt = copy.deepcopy(self)
        
        for name, value in additional_context_items.items():
            # ensure no collision between attributes occur, and if they do rename them with an identifying prefix
            final_name = self.resolve_key(duplicate_ctxt, name, value, collision_prefix, strategy=strategy)
            if final_name is not None:
                # Set the new attr
                setattr(duplicate_ctxt, final_name, value)
        
        return duplicate_ctxt
    
    # Helper methods that don't require a collision_prefix and employ a fixed strategy. All call self.adding_context(...) with the appropriate arguments:
    def adding_context_if_missing(self, **additional_context_items) -> "IdentifyingContext":
        return self.adding_context(None, strategy=CollisionOutcome.IGNORE_UPDATED, **additional_context_items)
    def overwriting_context(self, **additional_context_items) -> "IdentifyingContext":
        return self.adding_context(None, strategy=CollisionOutcome.REPLACE_EXISTING, **additional_context_items)

    def __add__(self, other) -> "IdentifyingContext":
        """ Allows adding contexts using the `+` operator
        """
        other_dict = convert_to_dictlike(other)
        return copy.deepcopy(self).overwriting_context(**other_dict)


    # Merging Functions __________________________________________________________________________________________________ #
    def merging_context(self, collision_prefix:str, additional_context: "IdentifyingContext") -> "IdentifyingContext":
        """ returns a new IdentifyingContext that results from adding the items in additional_context to a copy of self 
            collision_prefix: only used when an attr name in additional_context_items already exists for this context and the values of that attr are different    
        """
        return self.adding_context(collision_prefix, **additional_context.to_dict())
    
    def __or__(self, other):
        """ Used with vertical bar operator: |
        Usage:
            (_test_complete_spike_analysis_config | _test_partial_spike_analysis_config)    
        """
        return self.merging_context(None, other) # due to passing None as the collision context, this will fail if there are collisions

    # String/Printing Functions __________________________________________________________________________________________ #
    def get_description(self, subset_includelist=None, subset_excludelist=None, separator:str='_', include_property_names:bool=False, replace_separator_in_property_names:str='-', key_value_separator=None, prefix_items=[], suffix_items=[])->str:
        """ returns a simple text descriptor of the context
        
        include_property_names: str - whether to include the keys/names of the properties in the output string or just the values
        replace_separator_in_property_names: str = replaces occurances of the separator with the str specified for the included property names. has no effect if include_property_names=False
        key_value_separator: Optional[str] = if None, uses the same separator between key{}value as used between items.
        
        Outputs:
            a str like 'sess_kdiba_2006-6-07_11-26-53'
        """
        ## Build a session descriptor string:
        if include_property_names:
            if key_value_separator is None:
                key_value_separator = separator # use same separator between key{}value as pairs of items.
            # the double .replace(...).replace(...) below is to make sure the name string doesn't contain either separator, which may be different.
            descriptor_array = [[name.replace(separator, replace_separator_in_property_names).replace(key_value_separator, replace_separator_in_property_names), str(val)] for name, val in self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist).items()] # creates a list of [name, val] list items
            if key_value_separator != separator:
                # key_value_separator is different from separator. Join the pairs into strings [(k0, v0), (k1, v1), ...] -> [f"{k0}{key_value_separator}{v0}", f"{k1}{key_value_separator}{v1}", ...]
                descriptor_array = [key_value_separator.join(sublist) for sublist in descriptor_array]
            else:
                # old way, just flattens [(k0, v0), (k1, v1), ...] -> [k0, v0, k1, v1, ...]
                descriptor_array = [item for sublist in descriptor_array for item in sublist] # flat descriptor array
        else:
            descriptor_array = [str(val) for val in list(self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist).values())] # ensures each value is a string
            
        if prefix_items is not None:
            descriptor_array.extend(prefix_items)
        if suffix_items is not None:
            descriptor_array.extend(suffix_items)
        
        descriptor_string = separator.join(descriptor_array)
        return descriptor_string
    
    def get_initialization_code_string(self, subset_includelist=None, subset_excludelist=None, class_name_override=None) -> str:
        """ returns the string that contains valid code to initialize a matching object. """
        class_name_override = class_name_override or "IdentifyingContext"
        init_args_list_str = ",".join([f"{k}='{v}'" for k,v in self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist).items()]) # "format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'"
        return f"{class_name_override}({init_args_list_str})" #"IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')"

    def __str__(self) -> str:
        """ 'kdiba_2006-6-08_14-26-15_maze1_PYR' """
        return self.get_description()

    def __repr__(self) -> str:
        """ 
            "IdentifyingContext({'format_name': 'kdiba', 'session_name': '2006-6-08_14-26-15', 'filter_name': 'maze1_PYR'})" 
            "IdentifyingContext<('kdiba', '2006-6-08_14-26-15', 'maze1_PYR')>"
        """
        return f"IdentifyingContext<{self.as_tuple().__repr__()}>"

    # _ipython_key_completions_
    def _repr_pretty_(self, p, cycle):
        """  just adds non breaking text to the output, p.breakable() either adds a whitespace or breaks here. If you pass it an argument it is used instead of the default space. 
        https://ipython.readthedocs.io/en/stable/api/generated/IPython.lib.pretty.html#module-IPython.lib.pretty

        Can test with:
            from IPython.lib.pretty import pprint
            pprint(active_context)

        """
        if cycle:
            p.text('Context(...)')
        else:
            with p.group(8, 'Context(', ')'):
                dict_rep = self.to_dict()
                for name, val in dict_rep.items():
                    # name = name.replace(separator, replace_separator_in_property_names).replace(key_value_separator, replace_separator_in_property_names)
                    # val = p.pretty(val)
                    # p.text(f"{val}")
                    # ## Spaces:
                    # p.text(f"{name}")
                    # p.text(': ')
                    # p.pretty(val)
                    # if name != list(dict_rep.keys())[-1]:
                    #     p.breakable(sep=", ") # Add a breakable separator to the output. This does not mean that it will automatically break here. If no breaking on this position takes place the sep is inserted which default to one space.
                    ## Functional (no line breaks):
                    p.text(f"{name}")
                    p.text('= ')
                    p.pretty(val)
                    if name != list(dict_rep.keys())[-1]:
                        p.text(', ')
                        


    def __hash__(self):
        """ custom hash function that allows use in dictionary just based off of the values and not the object instance. """
        dict_rep = self.to_dict()
        sorted_dict_rep = dict(sorted(dict_rep.items())) # sort the dict rep's keys so the the comparisons are ultimately independent of order, meaning IdentifyingContext(k1='a', k2='b') == IdentifyingContext(k2='b', k1='a')
        member_names_tuple = list(sorted_dict_rep.keys())
        values_tuple = list(sorted_dict_rep.values())
        combined_tuple = tuple(member_names_tuple + values_tuple)
        return hash(combined_tuple)
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, IdentifyingContext):
            # return self.to_dict() == other.to_dict() # Python's dicts use element-wise comparison by default, so this is what we want.
            return dict(sorted(self.to_dict().items())) == dict(sorted(other.to_dict().items())) 
        else:
            raise NotImplementedError

    
    @classmethod
    def init_from_dict(cls, a_dict):
        return cls(**a_dict) # expand the dict as input args.
    

    def get_subset(self, subset_includelist=None, subset_excludelist=None) -> "IdentifyingContext":
        """ returns a proper subset of self given the incldued/exclude lists. """
        return IdentifyingContext.init_from_dict(self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist))

    # Differencing and Set Operations ____________________________________________________________________________________ #
    def subtracting(self, rhs) -> "IdentifyingContext":
        return self.subtract(self, rhs)

    def __sub__(self, other) -> "IdentifyingContext":
        """ implements the `-` subtraction operator """
        return self.subtract(self, other)
    

    @classmethod
    def subtract(cls, lhs, rhs):
        """ Returns the lhs less the keys that are in the rhs.
        Example:
            non_primary_desired_files = FileList.subtract(found_any_pickle_files, (found_default_session_pickle_files + found_global_computation_results_files))        
        """
        return cls.init_from_dict(lhs.to_dict(subset_excludelist=rhs.keys()))

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries. del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        
        
    # Context manager Methods ____________________________________________________________________________________________ #
    # """ These methods allow nested usage as a context manager like so:
    # with IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',user_annotation='selections') as ctx:
    #     print(f'ctx: {ctx}')
    #     inner_ctx = ctx.overwriting_context(epochs='ripple', decoder='short_RL')
    #     print(f'inner_ctx: {inner_ctx}')
        
    # """
    def __enter__(self):
        # This is where you can set up any resources if necessary
        # print(f"Entering context: {self}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # This is where you clean up resources if necessary
        # print(f"Exiting context: {self}")
        pass


            
    # ==================================================================================================================== #
    # BADLY PLACED METHODS (TO REFACTOR)                                                                                   #
    # ==================================================================================================================== #

    @classmethod
    def try_extract_date_from_session_name(cls, session_name: str, assumed_year_if_missing:str="2009", debug_print:bool=False): # Optional[Union[pd.Timestamp, NaTType]]
        """ 2023-08-24 - Attempts to determine at least the relative recording date for a given session from the session's name alone.
        From the 'session_name' column in the provided data, we can observe two different formats used to specify the date:

        Format 1: Dates with the pattern YYYY-M-D_H-M-S (e.g., "2006-6-07_11-26-53").
        Format 2: Dates with the pattern MM-DD_H-M-S (e.g., "11-02_17-46-44").
        
        """
        # Remove any non-digit prefixes or suffixes before parsing. Handles 'fet11-01_12-58-54'

        # Check for any non-digit prefix
        if re.match(r'^\D+', session_name):
            if debug_print:
                print(f"WARN: Removed prefix from session_name: {session_name}")
            session_name = re.sub(r'^\D*', '', session_name)

        # Check for any non-digit suffix
        if re.search(r'\D+$', session_name):
            if debug_print:
                print(f"WARN: Removed suffix from session_name: {session_name}")
            session_name = re.sub(r'\D*$', '', session_name)


        # Try Format 1 (YYYY-M-D_H-M-S)
        date_match1 = re.search(r'\d{4}-\d{1,2}-\d{1,2}_\d{1,2}-\d{1,2}-\d{1,2}', session_name)
        if date_match1:
            date_str1 = date_match1.group().replace('_', ' ')
            return pd.to_datetime(date_str1, format='%Y-%m-%d %H-%M-%S', errors='coerce')

        # Try Format 2 (MM-DD_H-M-S)
        date_match2 = re.search(r'\d{1,2}-\d{1,2}_\d{1,2}-\d{1,2}-\d{1,2}', session_name)
        if date_match2:
            date_str2 = f"{assumed_year_if_missing}-" + session_name.split('_')[0] # Assuming year provided in `assumed_year_if_missing`
            time_str2 = session_name.split('_')[1].replace('-', ':')
            full_str2 = date_str2 + ' ' + time_str2
            return pd.to_datetime(full_str2, format='%Y-%m-%d %H:%M:%S', errors='coerce')

        if debug_print:
            print(f"WARN: Could not parse date from session_name: {session_name} for any known format.")
        return None
    
        

# ==================================================================================================================== #
# Decorators                                                                                                           #
# ==================================================================================================================== #
def print_args(func):
    """ A decorator that extracts the arguments from its decorated function and prints them 

    Example:

        @print_args
        def add(x, y):
            return x + y

        >> add(2, 5)
            Arguments: (2, 5)
            Keyword arguments: {}

    """
    def wrapper(*args, **kwargs):
        print(f"Arguments: {args}")
        print(f"Keyword arguments: {kwargs}")
        return func(*args, **kwargs)
    return wrapper




def overwriting_display_context(**additional_context_kwargs):
    """Adds to the context of the function or class

    Usage:
        from neuropy.utils.result_context import overwriting_display_context
        @overwriting_display_context('tag1', 'tag2')
        def my_function():
            ...

        Access via:
            my_function.active_context

    """
    def decorator(func):
        """ I didn't realize that decorators are limited to having only one argument (the function being decorated), so if we want to pass multiple arguments we have to do this triple-nested function thing. """
        @wraps(func)
        def wrapper(*args, **kwargs):
            incomming_context = kwargs.get('active_context', IdentifyingContext()) or IdentifyingContext()
            updated_context = incomming_context.overwriting_context(**additional_context_kwargs)
            kwargs['active_context'] = updated_context ## set the updated context
            ## Also set the context as an attribute of the function in addition to passing it in the kwargs to the function.
            func.active_context = updated_context
            try:
                result = func(*args, **kwargs)
            except TypeError as e:
                # TypeError: __init__() got an unexpected keyword argument 'active_context'
                del kwargs['active_context'] # remove from the **kwargs, it can't handle it
                result = func(*args, **kwargs)
            except Exception as e:
                raise e

            ## TODO: update the returned context poentially?
            return result
        return wrapper
    return decorator


def providing_context(**additional_context_kwargs):
    """Specifies additional (narrowing) context for use in the function or class
    
    Usage:
        from neuropy.utils.result_context import overwriting_display_context
        @overwriting_display_context('tag1', 'tag2')
        def my_function():
            ...

        Access via:
            my_function.active_context

    """
    def decorator(func):
        """ I didn't realize that decorators are limited to having only one argument (the function being decorated), so if we want to pass multiple arguments we have to do this triple-nested function thing. """
        @wraps(func)
        def wrapper(*args, **kwargs):
            incomming_context = kwargs.get('active_context', IdentifyingContext()) or IdentifyingContext()
            updated_context = incomming_context.adding_context_if_missing(**additional_context_kwargs)
            kwargs['active_context'] = updated_context ## set the updated context
            ## Also set the context as an attribute of the function in addition to passing it in the kwargs to the function.
            func.active_context = updated_context
            try:
                result = func(*args, **kwargs)
            except TypeError as e:
                # TypeError: __init__() got an unexpected keyword argument 'active_context'
                del kwargs['active_context'] # remove from the **kwargs, it can't handle it
                result = func(*args, **kwargs)
            except Exception as e:
                raise e
            ## TODO: update the returned context poentially?
            return result
        return wrapper
    return decorator



### EXAMPLE of `overwriting_display_context` decorator usage:
# # def my_function(**kwargs):
# # 	print(f'test: {kwargs}')
	
# # my_function = overwriting_display_context(my_function, tag1='value1', tag2='value2')
# # my_function(active_context=IdentifyingContext(coolest_value='NEATO!'))

# @overwriting_display_context(tag1='value1', tag2='value2')
# def my_function(**kwargs):
# 	print(f'test: {kwargs}')
	
# # my_function = overwriting_display_context(my_function, tag1='value1', tag2='value2')
# my_function(active_context=IdentifyingContext(coolest_value='NEATO!'))




# *args, **kwargs





# ==================================================================================================================== #
# OLD (Pre 2023-07-04                                                                                                  #
# ==================================================================================================================== #
# class context_extraction:
#     """ function decorator for getting the context changes generated by a functional call by capturing its args and kwargs.
#     Inspired by klepto.safe's cache decorators

#     keymap = cache key encoder (default is keymaps.stringmap(flat=False))
#     ignore = function argument names and indicies to 'ignore' (default is None)
#     tol = integer tolerance for rounding (default is None)
#     deep = boolean for rounding depth (default is False, i.e. 'shallow')

#     If *keymap* is given, it will replace the hashing algorithm for generating
#     cache keys.  Several hashing algorithms are available in 'keymaps'. The
#     default keymap does not require arguments to the cached function to be
#     hashable.  If a hashing error occurs, the cached function will be evaluated.

#     If the keymap retains type information, then arguments of different types
#     will be cached separately.  For example, f(3.0) and f(3) will be treated
#     as distinct calls with distinct results.  Cache typing has a memory penalty,
#     and may also be ignored by some 'keymaps'.

#     If *ignore* is given, the keymap will ignore the arguments with the names
#     and/or positional indicies provided. For example, if ignore=(0,), then
#     the key generated for f(1,2) will be identical to that of f(3,2) or f(4,2).
#     If ignore=('y',), then the key generated for f(x=3,y=4) will be identical
#     to that of f(x=3,y=0) or f(x=3,y=10). If ignore=('*','**'), all varargs
#     and varkwds will be 'ignored'.  Ignored arguments never trigger a
#     recalculation (they only trigger cache lookups), and thus are 'ignored'.
#     When caching class methods, it may be useful to ignore=('self',).


#     Usage:
#         from neuropy.utils.result_context import context_extraction


#     """
#     def __init__(self, maxsize=None, cache=None, keymap=None, ignore=None, tol=None, deep=False, purge=False):
#        #if maxsize is not None: raise ValueError('maxsize cannot be set')
#         maxsize = None #XXX: allow maxsize to be given but ignored ?
#         purge = False #XXX: allow purge to be given but ignored ?
#         if cache is None: cache = archive_dict()
#         elif type(cache) is dict: cache = archive_dict(cache)

#         if keymap is None: keymap = stringmap(flat=False)
#         if ignore is None: ignore = tuple()

#         if deep: rounded = deep_round
#         else: rounded = simple_round
#        #else: rounded = shallow_round #FIXME: slow

#         @rounded(tol)
#         def rounded_args(*args, **kwds):
#             return (args, kwds)

#         # set state
#         self.__state__ = {
#             'maxsize': maxsize,
#             'cache': cache,
#             'keymap': keymap,
#             'ignore': ignore,
#             'roundargs': rounded_args,
#             'tol': tol,
#             'deep': deep,
#             'purge': purge,
#         }
#         return

#     def __call__(self, user_function):
#        #cache = dict()                  # mapping of args to results
#         stats = [0, 0, 0]               # make statistics updateable non-locally
#         HIT, MISS, LOAD = 0, 1, 2       # names for the stats fields
#        #_len = len                      # localize the global len() function
#        #lock = RLock()                  # linkedlist updates aren't threadsafe
#         maxsize = self.__state__['maxsize']
#         cache = self.__state__['cache']
#         keymap = self.__state__['keymap']
#         ignore = self.__state__['ignore']
#         rounded_args = self.__state__['roundargs']

#         def wrapper(*args, **kwds):
#             try:
#                 _args, _kwds = rounded_args(*args, **kwds)
#                 _args, _kwds = _keygen(user_function, ignore, *_args, **_kwds)
#                 key = keymap(*_args, **_kwds)
#             except: #TypeError
#                 result = user_function(*args, **kwds)
#                 stats[MISS] += 1
#                 return result

#             try:
#                 # get cache entry
#                 result = cache[key]
#                 raise KeyError # Disable loading from cache by raising a key error no matter what. TODO: potentially enable cache.
#                 stats[HIT] += 1
#             except KeyError:
#                 # if not in cache, look in archive
#                 if cache.archived():
#                     cache.load(key)
#                 try:
#                     result = cache[key]
#                     raise KeyError # Disable loading from cache by raising a key error no matter what. TODO: potentially enable cache.
#                     stats[LOAD] += 1
#                 except KeyError:
#                     # if not found, then compute
#                     result = user_function(*args, **kwds)
#                     cache[key] = result
#                     stats[MISS] += 1
#             except: #TypeError: # unhashable key
#                 result = user_function(*args, **kwds)
#                 stats[MISS] += 1
#             return result

#         def archive(obj):
#             """Replace the cache archive"""
#             if isinstance(obj, archive_dict): cache.archive = obj.archive
#             else: cache.archive = obj

#         def key(*args, **kwds):
#             """Get the cache key for the given *args,**kwds"""
#             _args, _kwds = rounded_args(*args, **kwds)
#             _args, _kwds = _keygen(user_function, ignore, *_args, **_kwds)
#             return keymap(*_args, **_kwds)

#         def lookup(*args, **kwds):
#             """Get the stored value for the given *args,**kwds"""
#             _args, _kwds = rounded_args(*args, **kwds)
#             _args, _kwds = _keygen(user_function, ignore, *_args, **_kwds)
#             return cache[keymap(*_args, **_kwds)]

#         def __get_cache():
#             """Get the cache"""
#             return cache

#         def __get_mask():
#             """Get the (ignore) mask"""
#             return ignore

#         def __get_keymap():
#             """Get the keymap"""
#             return keymap

#         def clear(keepstats=False):
#             """Clear the cache and statistics"""
#             cache.clear()
#             if not keepstats: stats[:] = [0, 0, 0]

#         def info():
#             """Report cache statistics"""
#             return CacheInfo(stats[HIT], stats[MISS], stats[LOAD], maxsize, len(cache))

#         # interface
#         wrapper.__wrapped__ = user_function
#         #XXX: better is handle to key_function=keygen(ignore)(user_function) ?
#         wrapper.info = info
#         wrapper.clear = clear
#         wrapper.load = cache.load
#         wrapper.dump = cache.dump
#         wrapper.archive = archive
#         wrapper.archived = cache.archived
#         wrapper.key = key
#         wrapper.lookup = lookup
#         wrapper.__cache__ = __get_cache
#         wrapper.__mask__ = __get_mask
#         wrapper.__map__ = __get_keymap
#        #wrapper._queue = None  #XXX
#         return update_wrapper(wrapper, user_function)

#     def __get__(self, obj, objtype):
#         """support instance methods"""
#         return partial(self.__call__, obj)

#     def __reduce__(self):
#         cache = self.__state__['cache']
#         keymap = self.__state__['keymap']
#         ignore = self.__state__['ignore']
#         tol = self.__state__['tol']
#         deep = self.__state__['deep']
#         return (self.__class__, (None, cache, keymap, ignore, tol, deep, False))



# class ResultContext(IdentifyingContext):
#     """result_context serves to uniquely identify the **context** of a given generic result.

#     Typically a result depends on several inputs:

#     - session: the context in which the original recordings were made.
#         Originally this includes the circumstances udner which the recording was performed (recording datetime, recording configuration (hardware, sampling rate, etc), experimenter name, animal identifer, etc)
#     - filter: the filtering performed to pre-process the loaded session data
#     - computation configuration: the specific computations performed and their parameters to transform the data to the result

#     Heuristically: If changing the value of that variable results in a changed result, it should be included in the result_context 
#     """
#     session_context: str = ''
#     filter_context: str = ''
#     computation_configuration: str = ''
    
 
#     def __init__(self, session_context, filter_context, computation_configuration):
#         super(ResultContext, self).__init__()
#         self.session_context = session_context
#         self.filter_context = filter_context
#         self.computation_configuration = computation_configuration
    
    
def print_identifying_context_array_code(included_session_contexts: List[IdentifyingContext], array_name: str='array_session_contexts') -> None:
    """ 
    Usage:
        from neuropy.utils.result_context import print_identifying_context_array_code
        
        print_identifying_context_array_code(included_session_contexts, array_name='included_session_contexts')
        
    """
    print(f'{array_name} = [')
    for a_session_context in included_session_contexts:
        print('\t' + a_session_context.get_initialization_code_string() + ',')
    print(f']')


class ContextFormatRenderingFn(Protocol):
    """ Functions implementing the protocol do not actually need to import/inherit from it it, it is just used for typehinting in the class below
        
        def _filename_formatting_fn(ctxt: DisplaySpecifyingIdentifyingContext, subset_includelist=None, subset_excludelist=None) -> str:
                return 'test_str'

    """
    def __call__( self, ctxt: "DisplaySpecifyingIdentifyingContext", subset_includelist: Optional[List[str]] = None, subset_excludelist: Optional[List[str]] = None) -> str:
        pass
    

    @classmethod
    def no_op(cls, ctxt: "DisplaySpecifyingIdentifyingContext", subset_includelist: Optional[List[str]] = None, subset_excludelist: Optional[List[str]]=None) -> str:
        """ does nothing """
        return ''

    @classmethod
    def merge_fns(cls, lhs_fn: "ContextFormatRenderingFn", rhs_fn: "ContextFormatRenderingFn") -> "ContextFormatRenderingFn":
        """ merges two functions """
        return (lambda ctxt, subset_includelist=None, subset_excludelist=None: lhs_fn(ctxt=ctxt, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist) + rhs_fn(ctxt=ctxt, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist))

    @classmethod
    def merge_specific_purpose_dicts(cls, lhs_dict: Dict[str, "ContextFormatRenderingFn"], rhs_dict: Dict[str, "ContextFormatRenderingFn"]) -> Dict[str, "ContextFormatRenderingFn"]:
        """ final_merged_output_dict = ContextFormatRenderingFn.merge_specific_purpose_dicts(lhs_dict= , rhs_dict= ) 
        """
        final_merged_output_dict = deepcopy(lhs_dict)             
        for a_display_purpose, rhs_fn in rhs_dict.items():
            if a_display_purpose not in final_merged_output_dict:
                # just add it
                final_merged_output_dict[a_display_purpose] = rhs_fn
            else:
                # it is in there, merge it
                lhs_fn = final_merged_output_dict.get(a_display_purpose, cls.no_op)
                final_fn = cls.merge_fns(lhs_fn=lhs_fn, rhs_fn=rhs_fn) ## return the combined fn
                final_merged_output_dict[a_display_purpose] = final_fn
        return final_merged_output_dict
        

@define(slots=False, eq=False)
class DisplaySpecifyingIdentifyingContext(IdentifyingContext):
    """ a class that extends IdentifyingContext to enable use-specific rendering of contexts.
    
    Primarily provides: `get_specific_purpose_description`

    """
    display_dict: Dict = field(default=None)
    specific_purpose_display_dict: Dict[str, ContextFormatRenderingFn] = field(default=None)

    def __init__(self, display_dict=None, specific_purpose_display_dict=None, **kwargs):
        # super(self.__class__, self).__init__(**kwargs)
        # super(self.__class__, self).__init__() # Replace super(self.__class__, self).__init__() with super().__init__() in both your base class and subclass. This ensures that the __init__ method of the immediate superclass is called correctly, respecting the MRO and preventing infinite recursion.
        super().__init__(**kwargs)
        # ## Sets attributes dnymically:
        # for name, value in kwargs.items():
        #     setattr(self, name, value)
        if display_dict is None:
            display_dict = {}
        if specific_purpose_display_dict is None:
            specific_purpose_display_dict = {}
        self.__attrs_init__(display_dict=display_dict, specific_purpose_display_dict=specific_purpose_display_dict)

            
        # self.display_dict = display_dict
        # self.specific_purpose_display_dict = specific_purpose_display_dict      
        # super(IdentifyingContext, self).__init__()
        ## Sets attributes dnymically:
        # for name, value in kwargs.items():
        #     setattr(self, name, value)
    
    
    @classmethod
    def init_from_context(cls, a_context: IdentifyingContext, display_dict=None, specific_purpose_display_dict=None) -> "DisplaySpecifyingIdentifyingContext":
        kwargs = copy.deepcopy(a_context.to_dict())
        if display_dict is None:
            display_dict = {}
        # _new_instance =  cls(**kwargs, display_dict=display_dict)
        _new_instance =  cls(display_dict=display_dict, specific_purpose_display_dict=specific_purpose_display_dict)
        # Set own attributes
        for name, value in kwargs.items():
            setattr(_new_instance, name, value)

        return _new_instance


    # String/Printing Functions __________________________________________________________________________________________ #
    def get_specific_purpose_description(self, specific_purpose:str, *extras_strings, subset_includelist=None, subset_excludelist=None)-> Optional[str]:
        """ returns a simple text descriptor of the context
        
        specific_purpose: str - the specific purpose name to get the formatted description of
        
        replace_separator_in_property_names: str = replaces occurances of the separator with the str specified for the included property names. has no effect if include_property_names=False
        key_value_separator: Optional[str] = if None, uses the same separator between key{}value as used between items.
        
        Outputs:
            a str like 'sess_kdiba_2006-6-07_11-26-53'
        """
        specific_purpose_render_fn = self.specific_purpose_display_dict.get(specific_purpose, None)  
        if (specific_purpose == 'flexitext_footer'):
            if np.all(self.has_keys(['format_name', 'animal', 'exper_name'])):
                first_portion_sess_ctxt_str = self.get_description(subset_includelist=['format_name', 'animal', 'exper_name'], separator=' | ')
                session_name_sess_ctxt_str = self.get_description(subset_includelist=['session_name'], separator=' | ') # 2006-6-08_14-26-15
                return (f"<color:silver, size:10>{first_portion_sess_ctxt_str} | <weight:bold>{session_name_sess_ctxt_str}</></>")
            else:
                # all keys
                bad_values_dict = {k:v for k, v in self.to_dict().items() if ((v is None) or (len(str(v)) == 0))}
                bad_values_keys = list(bad_values_dict.keys())
                subset_excludelist = ((subset_excludelist or []) + bad_values_keys) # exclude any keys that are bad (such as None/ zero-length/ etc)
                # good_values_dict = {k:v for k, v in self.to_dict().items() if ((v is not None) and (len(str(v)) > 0))}
                # good_values_keys = list(good_values_dict.keys())
                all_keys_ctxt_str = self.get_description(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist, separator=' | ')
                return (f"<color:silver, size:10>{all_keys_ctxt_str}</>")

            # return (f"<color:silver, size:10>{first_portion_sess_ctxt_str} | <weight:bold>{session_name_sess_ctxt_str}</></>")
        elif specific_purpose_render_fn is not None:
            return specific_purpose_render_fn(self)

        else:
            raise NotImplementedError(f'specific_purpose: {specific_purpose} does not match any known purpose. Known purposes: {list(self.specific_purpose_display_dict.keys())}')
            return None # has no specific purpose


    # String/Printing Functions __________________________________________________________________________________________ #
    def get_description(self, subset_includelist=None, subset_excludelist=None, separator:str='_', include_property_names:bool=False, replace_separator_in_property_names:str='-', key_value_separator=None, prefix_items=[], suffix_items=[])->str:
        """ returns a simple text descriptor of the context
        
        include_property_names: str - whether to include the keys/names of the properties in the output string or just the values
        replace_separator_in_property_names: str = replaces occurances of the separator with the str specified for the included property names. has no effect if include_property_names=False
        key_value_separator: Optional[str] = if None, uses the same separator between key{}value as used between items.
        
        Outputs:
            a str like 'sess_kdiba_2006-6-07_11-26-53'
            
        #TODO 2024-11-01 09:46: - [ ] We don't know the display purpose when get_descripton is called, so we just have to ignore it for now.
        
        
        """
        if subset_excludelist is None:
            subset_excludelist = []
        subset_excludelist = subset_excludelist + ['display_dict', 'specific_purpose_display_dict'] # always exclude the meta properties from printing

        def _get_formatted_str_value(name: str, val: Any):
            display_fn = self.display_dict.get(name, lambda n, v: str(v))
            return display_fn(name, val)

        ## Build a session descriptor string:
        if include_property_names:
            if key_value_separator is None:
                key_value_separator = separator # use same separator between key{}value as pairs of items.
            # the double .replace(...).replace(...) below is to make sure the name string doesn't contain either separator, which may be different.

            ## have to check formatting purpose:
            

            
            descriptor_array = [[name.replace(separator, replace_separator_in_property_names).replace(key_value_separator, replace_separator_in_property_names), _get_formatted_str_value(name, val)] for name, val in self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist).items()] # creates a list of [name, val] list items
            if key_value_separator != separator:
                # key_value_separator is different from separator. Join the pairs into strings [(k0, v0), (k1, v1), ...] -> [f"{k0}{key_value_separator}{v0}", f"{k1}{key_value_separator}{v1}", ...]
                descriptor_array = [key_value_separator.join(sublist) for sublist in descriptor_array]
            else:
                # old way, just flattens [(k0, v0), (k1, v1), ...] -> [k0, v0, k1, v1, ...]
                descriptor_array = [item for sublist in descriptor_array for item in sublist] # flat descriptor array
        else:
            # descriptor_array = [str(val) for val in list(self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist).values())] # ensures each value is a string
            descriptor_array = [_get_formatted_str_value(name, val) for name, val in self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist).items()] # ensures each value is a string
             
        if prefix_items is not None:
            descriptor_array.extend(prefix_items)
        if suffix_items is not None:
            descriptor_array.extend(suffix_items)
        
        descriptor_string = separator.join(descriptor_array)
        return descriptor_string
    

    def get_initialization_code_string(self, subset_includelist=None, subset_excludelist=None, class_name_override=None) -> str:
        """ returns the string that contains valid code to initialize a matching object. """
        class_name_override = class_name_override or "DisplaySpecifyingIdentifyingContext"
        init_args_list_str = ",".join([f"{k}='{v}'" for k,v in self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist).items()]) # "format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'"
        return f"{class_name_override}({init_args_list_str})" #"IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')"

    def __str__(self) -> str:
        """ 'kdiba_2006-6-08_14-26-15_maze1_PYR' """
        return self.get_description()

    def __repr__(self) -> str:
        """ 
            "DisplaySpecifyingIdentifyingContext({'format_name': 'kdiba', 'session_name': '2006-6-08_14-26-15', 'filter_name': 'maze1_PYR'})" 
            "DisplaySpecifyingIdentifyingContext<('kdiba', '2006-6-08_14-26-15', 'maze1_PYR')>"
        """
        return f"DisplaySpecifyingIdentifyingContext<{self.as_tuple().__repr__()}>"
        
    # _ipython_key_completions_
    def _repr_pretty_(self, p, cycle):
        """  just adds non breaking text to the output, p.breakable() either adds a whitespace or breaks here. If you pass it an argument it is used instead of the default space. 
        https://ipython.readthedocs.io/en/stable/api/generated/IPython.lib.pretty.html#module-IPython.lib.pretty

        Can test with:
            from IPython.lib.pretty import pprint
            pprint(active_context)

        """
        if cycle:
            p.text('DisplaySpecifyingIdentifyingContext(...)')
        else:
            with p.group(8, 'DisplaySpecifyingIdentifyingContext(', ')'):
                dict_rep = self.to_dict()
                for name, val in dict_rep.items():
                    ## Functional (no line breaks):
                    p.text(f"{name}")
                    p.text('= ')
                    p.pretty(val)
                    if name != list(dict_rep.keys())[-1]:
                        p.text(', ')
                        


    @classmethod
    def resolve_key(cls, duplicate_ctxt: "DisplaySpecifyingIdentifyingContext", name:str, value, collision_prefix:str, strategy:CollisionOutcome=CollisionOutcome.APPEND_USING_KEY_PREFIX):
        """ensures no collision between attributes occur, and if they do resolve them according to strategy. e.g. rename them with an identifying prefix
        Returns the resolved key (str) or None.
        """
        if hasattr(duplicate_ctxt, name):
            # Check whether the existing context already has that key:
            if (getattr(duplicate_ctxt, name) == value):
                # Check whether it is the same value or not:
                # the existing context has the same value for the overlapping key as the current one. Permit the merge.
                final_name = name # this will not change the result
            else:
                # the keys exist on both and they have differing values. Try to resolve with the `collision_prefix`:                
                if strategy.name == CollisionOutcome.IGNORE_UPDATED.name:
                    final_name = None # by returning None the caller will know the keep the existing.
                elif strategy.name == CollisionOutcome.FAIL_IF_DIFFERENT.name:
                    print(f'ERROR: namespace collision in resolve_key! attr with name "{name}" already exists and the value differs: existing_value: {getattr(duplicate_ctxt, name)} new_value: {value}. Using collision_prefix to add new name: "{final_name}"!')
                    raise KeyError
                elif strategy.name == CollisionOutcome.REPLACE_EXISTING.name:
                    final_name = name # this will overwrite the old value because the returned key is the same.
                elif strategy.name == CollisionOutcome.APPEND_USING_KEY_PREFIX.name:
                    ## rename the current attribute to be set by appending a prefix
                    assert collision_prefix is not None, f"namespace collision in `adding_context(...)`! attr with name '{name}' already exists but the value differs: existing_value: {getattr(duplicate_ctxt, name)} new_value: {value}! Furthermore 'collision_prefix' is None!"
                    final_name = f'{collision_prefix}{name}'
                    print(f'WARNING: namespace collision in resolve_key! attr with name "{name}" already exists and the value differs: existing_value: {getattr(duplicate_ctxt, name)} new_value: {value}. Using collision_prefix to add new name: "{final_name}"!')
        else:
            final_name = name
        return final_name


    def add_context(self, collision_prefix:str, strategy:CollisionOutcome=CollisionOutcome.APPEND_USING_KEY_PREFIX, **additional_context_items) -> "DisplaySpecifyingIdentifyingContext":
        """ adds the additional_context_items to self 
        collision_prefix: only used when an attr name in additional_context_items already exists for this context and the values of that attr are different
        
        """        
        for name, value in additional_context_items.items():
            # ensure no collision between attributes occur, and if they do rename them with an identifying prefix

            ## handle     display_dict, specific_purpose_display_dict
    
            if name == 'display_dict':
                # display_dict
                ## merge the display dict fns
                merged_dict = (self.display_dict | deepcopy(value)) ## return the combined fn
                # merged_dict = ContextFormatRenderingFn.merge_specific_purpose_dicts(lhs_dict=self.display_dict, rhs_dict=value)
                setattr(self, name, merged_dict)
            elif name == 'specific_purpose_display_dict':
                ## merge the display dict fns
                final_merged_output_dict = ContextFormatRenderingFn.merge_specific_purpose_dicts(lhs_dict=self.specific_purpose_display_dict, rhs_dict=value)
                # final_merged_output_dict = self.specific_purpose_display_dict                
                # for a_display_purpose, rhs_fn in value.items():
                #     if a_display_purpose not in final_merged_output_dict:
                #         # just add it
                #         final_merged_output_dict[a_display_purpose] = rhs_fn
                #     else:
                #         # it is in there, merge it
                #         self.specific_purpose_display_dict
                #         lhs_fn = final_merged_output_dict.get(a_display_purpose, ContextFormatRenderingFn.no_op)
                #         final_fn = ContextFormatRenderingFn.merge_fns(lhs_fn=lhs_fn, rhs_fn=rhs_fn) ## return the combined fn
                #         final_merged_output_dict[a_display_purpose] = final_fn
                setattr(self, name, final_merged_output_dict) # overwrite self.specific_purpose_display_dict
            else:
                final_name = self.resolve_key(self, name, value, collision_prefix, strategy=strategy)
                if final_name is not None:
                    # Set the new attr
                    setattr(self, final_name, value)
        
        return self

    def adding_context(self, collision_prefix:str, strategy:CollisionOutcome=CollisionOutcome.APPEND_USING_KEY_PREFIX, **additional_context_items) -> "DisplaySpecifyingIdentifyingContext":
        """ returns a new IdentifyingContext that results from adding additional_context_items to a copy of self 
        collision_prefix: only used when an attr name in additional_context_items already exists for this context and the values of that attr are different
        
        """
        # assert isinstance(collision_prefix, str), f"collision_prefix must be provided as a string! Did you forget to provide it?"
        duplicate_ctxt = copy.deepcopy(self)
        
        for name, value in additional_context_items.items():
            # ensure no collision between attributes occur, and if they do rename them with an identifying prefix
            if name == 'display_dict':
                # display_dict
                ## merge the display dict fns
                merged_dict = (duplicate_ctxt.display_dict | deepcopy(value)) ## return the combined fn
                setattr(duplicate_ctxt, name, merged_dict)
            elif name == 'specific_purpose_display_dict':
                ## merge the display dict fns
                final_merged_output_dict = ContextFormatRenderingFn.merge_specific_purpose_dicts(lhs_dict=duplicate_ctxt.specific_purpose_display_dict, rhs_dict=value)
                setattr(duplicate_ctxt, name, final_merged_output_dict)
            else:            
                final_name = self.resolve_key(duplicate_ctxt, name, value, collision_prefix, strategy=strategy)
                if final_name is not None:
                    # Set the new attr
                    setattr(duplicate_ctxt, final_name, value)
        
        return duplicate_ctxt

