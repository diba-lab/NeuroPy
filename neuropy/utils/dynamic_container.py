
import collections
from collections.abc import MutableMapping
from neuropy.utils.mixins.diffable import DiffableObject
from neuropy.utils.mixins.gettable_mixin import KeypathsAccessibleMixin


def get_dict_subset(a_dict, included_keys=None, require_all_keys=False):
    """Gets a subset of a dictionary from a list of keys (included_keys)

    Args:
        a_dict ([type]): [description]
        included_keys ([type], optional): [description]. Defaults to None.
        require_all_keys: Bool, if True, requires all keys in included_keys to be in the dictionary (a_dict)

    Returns:
        [type]: [description]
    """
    if included_keys is not None:
        if require_all_keys:
            return {included_key:a_dict[included_key] for included_key in included_keys} # filter the dictionary for only the keys specified
        else:
            out_dict = {}
            for included_key in included_keys:
                if included_key in a_dict.keys():
                    out_dict[included_key] = a_dict[included_key]
            return out_dict
    else:
        return a_dict
    
    
def override_dict(lhs_dict, rhs_dict):
    """returns lhs_dict overriden with the values specified in rhs_dict (if they exist), otherwise returning the extant values.
    """
    limited_rhs_dict = get_dict_subset(rhs_dict, included_keys=lhs_dict.keys(), require_all_keys=False)  # restrict the other dict to the subset of keys in lhs_dict
    return lhs_dict.__or__(limited_rhs_dict) # now can perform normal __or__ using the restricted subset dict

def overriding_dict_with(lhs_dict, **kwargs):
    """returns lhs_dict overriden with the kwargs provided (if they exist), otherwise returning the extant values.
        Calls self.__ior__(other) under the hood.
    """
    return override_dict(lhs_dict, kwargs)
    

#TODO 2024-10-30 11:05: - [ ] There are VERY similar classes in `neuropy.utils.mixins.dict_representable.DictInitializable`, which is where these classes should probably be moved


class DictlikeInitializableMixin:
    """ Implementors can be initialized from a dict-like """
    # Helper initialization methods:    
    # For initialization from a different dictionary-backed object:
    @classmethod
    def init_from_dict(cls, a_dict):
        return cls(**a_dict) # expand the dict as input args.
    
    @classmethod
    def init_from_object(cls, an_object):
        # test to see if the object is dict-backed:
        obj_dict_rep = an_object.__dict__ ## could check for the object's .to_dict() or .items()
        return cls.init_from_dict(obj_dict_rep)
        
            
class DictlikeOverridableMixin:
    """ allows self to be overriden by a kwargs, a dict, or another DictlikeOverridableMixin (dict-like) 
    
    Usage:

        from neuropy.utils.dynamic_container import DictlikeInitializableMixin, DictlikeOverridableMixin
    
        
    """
    
    def to_dict(self):
        raise NotImplementedError(f'Implementor must override and implement')
        # return dict(self.items())

        
    def __ior__(self, other):
        """ Used with vertical bar equals operator: |=
        
        Unlike __or__(self, other), does not allow keys present ONLY in other to be added to self.
            Identically like __or__(self, other) though, if a key is present in both self and other the value in OTHER will be used. 
        
        Usage:
            # Explicit __ior__ usage:
            out = DynamicContainer(**{'s': 2, 'gamma': 0.2}).__ior__(kwargs)

            # Multi-line "override" update:
            out = DynamicContainer(**{'s': 2, 'gamma': 0.2})
            out|=kwargs
            
            # WARNING: this is wrong! Must first have a DynamicContainer to call __ior__ on, not a plain dict inside the DynamicContainer initializer
            out = DynamicContainer(**({'s': 2, 'gamma': 0.2}.__ior__(kwargs))) # |=

        Testing:
        def _test_xor(**kwargs):
            # want it only to pass 's' and 'gamma' keys to create the container despite more keys being present in kwargs
            # out = DynamicContainer(**{'s': 2, 'gamma': 0.2}).__ior__(kwargs) # |=
            out = DynamicContainer(**{'s': 2, 'gamma': 0.2}).override(kwargs)
            # out = DynamicContainer(**{'s': 2, 'gamma': 0.2})
            # out|=kwargs
            print(f'{out}')
            return out
            # dict_or = self.to_dict().__or__(other_dict)
            
        _test_xor(s=3) # DynamicContainer({'s': 3, 'gamma': 0.2})
        _test_xor(s=3, m='vet') # DynamicContainer({'s': 3, 'gamma': 0.2})
        _test_xor(s=3, m='vet', gamma=0.9) # DynamicContainer({'s': 3, 'gamma': 0.9})

        """
        assert isinstance(self, DictlikeInitializableMixin), f"self is not Dict-initializable!"
        
        if isinstance(other, (dict)):
            other_dict = other
        elif isinstance(other, DictlikeOverridableMixin):
            other_dict = other.to_dict()
        else:
            # try to convert the other type into a dict using all known available methods: DynamicContainer
            try:
                other_dict = other.to_dict() # try to convert to dict using the .to_dict() method if possible
            except Exception as e:
                # If that failed, fallback to trying to access the object's .__dict__ property
                try:
                    other_dict = dict(other.items())
                except Exception as e:
                    # Give up, can't convert!                
                    print(f'UNHANDLED TYPE: type(other): {type(other)}, other: {other}')
                    # raise NotImplementedError            
                    other_dict = None
                    raise e

                pass # other_dict               
        
        # restrict the other dict to the subset of keys that we have.
        limited_other_dict = get_dict_subset(other_dict, included_keys=self.to_dict().keys(), require_all_keys=False)
        # dict_or = self.to_dict().__ior__(other_dict)
        dict_or = self.to_dict().__or__(limited_other_dict) # now can perform normal __or__ using the restricted subset dict
        
        return self.init_from_dict(dict_or)
        # return DynamicContainer.init_from_dict(dict_or)
    
    
    def overriding_with(self, **kwargs):
        """returns self overriden with the kwargs provided (if they exist), otherwise returning the extant values.
            Calls self.__ior__(other) under the hood.
        """
        return self.override(kwargs)
     
    def override(self, other):
        """returns self overriden with the values specified in other (if they exist), otherwise returning the extant values.
            Calls self.__ior__(other) under the hood.
        """
        return self.__ior__(other)
        



#TODO 2024-10-30 10:30: - [ ] Add KeypathsAccessibleMixin conformance? Why not?

class DynamicContainer(KeypathsAccessibleMixin, DictlikeOverridableMixin, DictlikeInitializableMixin, DiffableObject, MutableMapping):
    """ A class that permits flexible prototyping of parameters and data needed for computations, while still allowing development-time guidance on available members.
        From https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/#When_making_a_custom_list_or_dictionary,_remember_you_have_options
        The UserDict class implements the interface that dictionaries are supposed to have, but it wraps around an actual dict object under-the-hood.
        The UserList and UserDict classes are for when you want something that acts almost identically to a list or a dictionary but you want to customize just a little bit of functionality.
        The abstract base classes in collections.abc are useful when you want something that’s a sequence or a mapping but is different enough from a list or a dictionary that you really should be making your own custom class.
    """
    debug_enabled = False
    outcome_on_item_not_found = None

    def __init__(self, **kwargs):
        self._mapping = {} # initialize the base dictionary object where things will be stored
        self._keys_at_init = list(kwargs.keys())
        self.update(kwargs)
       
    def __getitem__(self, key):
        return self._mapping[key] #@IgnoreException

    def __delitem__(self, key):
        del self._mapping[key]

    def __setitem__(self, key, value):
        self._mapping[key] = value

    def __iter__(self):
        return iter(self._mapping)
    def __len__(self):
        return len(self._mapping)
    def __repr__(self):
        return f"{type(self).__name__}({self._mapping})"

    # Extra/Extended
    def __dir__(self):
        return self.keys()


    def __or__(self, other):
        """ Used with vertical bar operator: |
        
        Usage:
            (_test_complete_spike_analysis_config | _test_partial_spike_analysis_config)    
        """
        if isinstance(other, (dict)):
            other_dict = other
        elif isinstance(other, DynamicContainer):
            other_dict = other.to_dict()
        else:
            # try to convert the other type into a dict using all known available methods: DynamicContainer
            try:
                other_dict = other.to_dict() # try to convert to dict using the .to_dict() method if possible
            except Exception as e:
                # If that failed, fallback to trying to access the object's .__dict__ property
                try:
                    other_dict = dict(other.items())
                except Exception as e:
                    # Give up, can't convert!                
                    print(f'UNHANDLED TYPE: type(other): {type(other)}, other: {other}')
                    # raise NotImplementedError            
                    other_dict = None
                    raise e

                pass # other_dict               
        
            
        dict_or = self.to_dict().__or__(other_dict)
        return DynamicContainer.init_from_dict(dict_or)
    
    
    def __getattr__(self, item):
        # Gets called when the item is not found via __getattribute__
        try:
            # try to return the value of the dictionary 
            return self[item]
        except KeyError as err:
            
            # if DynamicParameters.outcome_on_item_not_found:
            # return super(DynamicParameters, self).__setattr__(item, 'orphan')
            # Suggested to work around a deepcopy issue
            raise AttributeError(item)      #@IgnoreException 
            # raise
            
            
        # except AttributeError as err:
        #     print(f"AttributeError: {err}")
        #     return super(DynamicParameters, self).__setattr__(item, 'orphan')
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


    def __setattr__(self, attr, value):
        if attr == '__setstate__':
            return lambda: None
        elif ((attr == '_mapping') or (attr == '_keys_at_init')):
            # Access to the raw data variable
            # object.__setattr__(self, name, value)
            super(DynamicContainer, self).__setattr__(attr, value) # call super for valid properties
            # self._mapping = value # this would be infinitely recurrsive!
        else:
            self[attr] = value

    
    @property
    def all_attributes(self):
        """Any attributes on the object. """
        return list(self.keys())
    
    @property
    def original_attributes(self):
        """The attributes that were provided initially at init. """
        return self._keys_at_init
    
    @property
    def dynamically_added_attributes(self):
        """The attributes that were added dynamically post-init."""
        return list(set(self.all_attributes) - set(self.original_attributes))
    
    

    def __hash__(self):
        """ custom hash function that allows use in dictionary just based off of the values and not the object instance. """
        # return hash((self.age, self.name))
        member_names_tuple = list(self.keys())
        values_tuple = list(self.values())
        combined_tuple = tuple(member_names_tuple + values_tuple)
        return hash(combined_tuple)
    
        
    # For diffable parameters:
    def diff(self, other_object):
        return DiffableObject.compute_diff(self, other_object)


    def to_dict(self):
        return dict(self.items())
        
    # # Helper initialization methods:    
    # # For initialization from a different dictionary-backed object:
    # @classmethod
    # def init_from_dict(cls, a_dict):
    #     return cls(**a_dict) # expand the dict as input args.
    
    # @classmethod
    # def init_from_object(cls, an_object):
    #     # test to see if the object is dict-backed:
    #     obj_dict_rep = an_object.__dict__
    #     return cls.init_from_dict(obj_dict_rep)
    
    
    
    # ## For serialization/pickling:
    # def __getstate__(self):
    #     return self.to_dict()

    # def __setstate__(self, state):
    #     return self.init_from_dict(state)
        
        

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
        self.__dict__.update(state)
        
