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

import copy
from typing import Any
from enum import Enum
from functools import wraps # used for decorators
from attrs import define, field, Factory
from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage
from neuropy.utils.mixins.diffable import DiffableObject
from neuropy.utils.mixins.dict_representable import SubsettableDictRepresentable

from functools import update_wrapper, partial # for context_extraction
from klepto.archives import cache as archive_dict
from klepto.archives import file_archive, sql_archive, dict_archive
from klepto.keymaps import keymap, hashmap, stringmap
from klepto.tools import CacheInfo
from klepto.rounding import deep_round, simple_round
from klepto._inspect import _keygen

# _keymap = stringmap(flat=False)
# _keymap = keymap(flat=False)

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
class IdentifyingContext(DiffableObject, SubsettableDictRepresentable):
    """ a general extnsible base context that allows additive member creation
    
        Should not hold any state or progress-related variables. 
    
        # SubsettableDict: provides `to_dict`, `keys`, `keypaths`
    """
    def __init__(self, **kwargs):
        super(IdentifyingContext, self).__init__()
        ## Sets attributes dnymically:
        for name, value in kwargs.items():
            setattr(self, name, value)
        
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


    # Helper methods that don't require a collision_prefix and employ a fixed strategy. All call self.adding_context(...) with the appropriate arguments:
    def adding_context_if_missing(self, **additional_context_items) -> "IdentifyingContext":
        return self.adding_context(None, strategy=CollisionOutcome.IGNORE_UPDATED, **additional_context_items)
    def overwriting_context(self, **additional_context_items) -> "IdentifyingContext":
        return self.adding_context(None, strategy=CollisionOutcome.REPLACE_EXISTING, **additional_context_items)

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
        # if isinstance(other, (dict)):
        #     other_dict = other
        # elif isinstance(other, "IdentifyingContext"):
        #     other_dict = other.to_dict()
        # else:
        #     # try to convert the other type into a dict using all known available methods: IdentifyingContext
        #     try:
        #         other_dict = other.to_dict() # try to convert to dict using the .to_dict() method if possible
        #     except Exception as e:
        #         # If that failed, fallback to trying to access the object's .__dict__ property
        #         try:
        #             other_dict = dict(other.items())
        #         except Exception as e:
        #             # Give up, can't convert!                
        #             print(f'UNHANDLED TYPE: type(other): {type(other)}, other: {other}')         
        #             other_dict = None
        #             raise e

        #         pass # other_dict               
        
        # dict_or = self.to_dict().__or__(other_dict)
        # return IdentifyingContext.init_from_dict(dict_or)
        return self.merging_context(None, other) # due to passing None as the collision context, this will fail if there are collisions

    def get_description(self, subset_includelist=None, separator:str='_', include_property_names:bool=False, replace_separator_in_property_names:str='-', key_value_separator=None, prefix_items=[], suffix_items=[])->str:
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
            descriptor_array = [[name.replace(separator, replace_separator_in_property_names).replace(key_value_separator, replace_separator_in_property_names), str(val)] for name, val in self.to_dict(subset_includelist=subset_includelist).items()] # creates a list of [name, val] list items
            if key_value_separator != separator:
                # key_value_separator is different from separator. Join the pairs into strings [(k0, v0), (k1, v1), ...] -> [f"{k0}{key_value_separator}{v0}", f"{k1}{key_value_separator}{v1}", ...]
                descriptor_array = [key_value_separator.join(sublist) for sublist in descriptor_array]
            else:
                # old way, just flattens [(k0, v0), (k1, v1), ...] -> [k0, v0, k1, v1, ...]
                descriptor_array = [item for sublist in descriptor_array for item in sublist] # flat descriptor array
        else:
            descriptor_array = [str(val) for val in list(self.to_dict(subset_includelist=subset_includelist).values())] # ensures each value is a string
            
        if prefix_items is not None:
            descriptor_array.extend(prefix_items)
        if suffix_items is not None:
            descriptor_array.extend(suffix_items)
        
        descriptor_string = separator.join(descriptor_array)
        return descriptor_string
    
    def get_initialization_code_string(self) -> str:
        """ returns the string that contains valid code to initialize a matching object. """
        init_args_list_str = ",".join([f"{k}='{v}'" for k,v in self.to_dict().items()]) # "format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'"
        return f"IdentifyingContext({init_args_list_str})" #"IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')"

    def __str__(self) -> str:
        """ 'kdiba_2006-6-08_14-26-15_maze1_PYR' """
        return self.get_description()

    def __repr__(self) -> str:
        """ 
            "IdentifyingContext({'format_name': 'kdiba', 'session_name': '2006-6-08_14-26-15', 'filter_name': 'maze1_PYR'})" 
            "IdentifyingContext<('kdiba', '2006-6-08_14-26-15', 'maze1_PYR')>"
        """
        # return f"IdentifyingContext({self.get_description(include_property_names=True)})"
        # return f"IdentifyingContext({self.get_description(include_property_names=False)})"
        return f"IdentifyingContext<{self.as_tuple().__repr__()}>"
        # return f"IdentifyingContext({self.to_dict().__repr__()})"
        
    def __hash__(self):
        """ custom hash function that allows use in dictionary just based off of the values and not the object instance. """
        dict_rep = self.to_dict()
        member_names_tuple = list(dict_rep.keys())
        values_tuple = list(dict_rep.values())
        combined_tuple = tuple(member_names_tuple + values_tuple)
        return hash(combined_tuple)
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, IdentifyingContext):
            return self.to_dict() == other.to_dict() # Python's dicts use element-wise comparison by default, so this is what we want.
        else:
            raise NotImplementedError
        return NotImplemented # this part looks like a bug, yeah?
    
    @classmethod
    def init_from_dict(cls, a_dict):
        return cls(**a_dict) # expand the dict as input args.
        
    # Differencing and Set Operations ____________________________________________________________________________________ #
    def subtracting(self, rhs) -> "IdentifyingContext":
        return self.subtract(self, rhs)

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
        # Remove the unpicklable entries.
        # del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        
        
        

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
            incomming_context = kwargs.get('active_context', IdentifyingContext())
            updated_context = incomming_context.overwriting_context(**additional_context_kwargs)
            kwargs['active_context'] = updated_context ## set the updated context
            ## Also set the context as an attribute of the function in addition to passing it in the kwargs to the function.
            func.active_context = updated_context
            result = func(*args, **kwargs)
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



# # ==================================================================================================================== #
# # Function Attributes Decorators                                                                                       #
# # ==================================================================================================================== #
# _custom_metadata_attribute_names = dict(short_name=None, tags=None, creation_date=None,
#                                          input_requires=None, output_provides=None,
#                                          uses=None, used_by=None,
#                                          related_items=None, # references to items related to this definition
#                                          pyqt_signals_emitted=None # QtCore.pyqtSignal()s emitted by this object
# )


# def metadata_attributes(short_name=None, tags=None, creation_date=None, input_requires=None, output_provides=None, uses=None, used_by=None, related_items=None,  pyqt_signals_emitted=None):
#     """Adds generic metadata attributes to a function or class
#     Aims to generalize `pyphocorehelpers.function_helpers.function_attributes`

#     ```python
#         from pyphocorehelpers.programming_helpers import metadata_attributes

#         @metadata_attributes(short_name='pf_dt_sequential_surprise', tags=['tag1','tag2'], input_requires=[], output_provides=[], uses=[], used_by=[])
#         def _perform_time_dependent_pf_sequential_surprise_computation(computation_result, debug_print=False):
#             # function body
#     ```

#     func.short_name, func.tags, func.creation_date, func.input_requires, func.output_provides, func.uses, func.used_by
#     """
#     # decorator = function_attributes(func) # get the decorator provided by function_attributes

#     def decorator(func):
#         func.short_name = short_name
#         func.tags = tags
#         func.creation_date = creation_date
#         func.input_requires = input_requires
#         func.output_provides = output_provides
#         func.uses = uses
#         func.used_by = used_by
#         func.related_items = related_items
#         func.pyqt_signals_emitted = pyqt_signals_emitted
#         return func
#     return decorator




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
    
    