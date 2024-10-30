from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage

class DictInitializable:
    """ Implementors can be initialized from a dict or dict-like
    """
    @staticmethod
    def from_dict(d: dict):
        raise NotImplementedError

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
    


class DictRepresentable(DictInitializable):
    def to_dict(self, recurrsively=False):
        raise NotImplementedError




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
    

class DictlikeOverridableMixin:
    """ allows self to be overriden by a kwargs, a dict, or another DictlikeOverridableMixin (dict-like) 
    
    Usage:

        from neuropy.utils.dynamic_container import DictlikeInitializableMixin, DictlikeOverridableMixin
    
        
    """
    
    def to_dict(self):
        raise NotImplementedError(f'Implementor must override and implement')
        # return dict(self.items())


    def __or__(self, other):
        """ Used with vertical bar operator: |
        
        Usage:
            (_test_complete_spike_analysis_config | _test_partial_spike_analysis_config)    
        """
        if isinstance(other, (dict)):
            other_dict = other
        elif hasattr(other, 'to_dict'):
        # elif isinstance(other, (DictRepresentable, DictlikeOverridableMixin)):
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
        return self.init_from_dict(dict_or)


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
        # assert isinstance(self, DictInitializable), f"self is not Dict-initializable!"
        
        if isinstance(other, (dict)):
            other_dict = other
        elif hasattr(other, 'to_dict'):
        # elif isinstance(other, (DictRepresentable, DictlikeOverridableMixin)):
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
        



# ==================================================================================================================== #
# Dictionary Subsetting                                                                                                #
# ==================================================================================================================== #

class SubsettableDictRepresentable(DictRepresentable):
    """ confomers can be subsettable
    requires `benedict` library: from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage

    # SubsettableDictRepresentable: provides `to_dict`, `keys`, `keypaths`, `as_tuple`,`has_keys`, `check_keys`
    """
    def to_dict(self, subset_includelist=None, subset_excludelist=None) -> benedict:
        """ 
        Inputs:
            subset_includelist:<list?> a list of keys that specify the subset of the keys to be returned. If None, all are returned.
        """
        if subset_excludelist is not None:
            # if we have a excludelist, assert that we don't have an includelist
            assert subset_includelist is None, f"subset_includelist MUST be None when a subset_excludelist is provided, but instead it's {subset_includelist}!"
            subset_includelist = self.keys(subset_excludelist=subset_excludelist) # get all but the excluded keys

        if subset_includelist is None:
            return benedict(self.__dict__)
        else:
            return benedict(self.__dict__).subset(subset_includelist)

    def keys(self, subset_includelist=None, subset_excludelist=None):
        if subset_includelist is None:
            return [a_key for a_key in benedict(self.__dict__).keys() if a_key not in (subset_excludelist or [])]
        else:
            assert subset_excludelist is None, f"subset_excludelist MUST be None when a subset_includelist is provided, but instead it's {subset_excludelist}!"
            return [a_key for a_key in benedict(self.__dict__).subset(subset_includelist).keys() if a_key not in (subset_excludelist or [])]

    def keypaths(self, subset_includelist=None, subset_excludelist=None): 
        if subset_includelist is None:
            return [a_key for a_key in benedict(self.__dict__).keys() if a_key not in (subset_excludelist or [])]
        else:
            assert subset_excludelist is None, f"subset_excludelist MUST be None when a subset_includelist is provided, but instead it's {subset_excludelist}!"
            return [a_key for a_key in benedict(self.__dict__).subset(subset_includelist).keys() if a_key not in (subset_excludelist or [])]
        

    # ==================================================================================================================== #
    # Extras: `as_tuple`, `has_keys`, `check_keys`,                                                                        #
    # ==================================================================================================================== #
    def as_tuple(self, subset_includelist=None, subset_excludelist=None, drop_missing:bool=False):
        """ returns a tuple of just its values 
        Inputs:
            subset_includelist:<list?> a list of keys that specify the subset of the keys to be returned. If None, all are returned.

        Usage:
        curr_sess_ctx_tuple = curr_sess_ctx.as_tuple(subset_includelist=['format_name','animal','exper_name', 'session_name'])
        curr_sess_ctx_tuple # ('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')

        """
        if drop_missing:
            return tuple([v for v in tuple(self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist).values()) if v is not None]) # Drops all 'None' values in the tuple
        else:
            return tuple(self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist).values())
            
    def has_keys(self, keys_list) -> list[bool]:
        """ returns a boolean array with each entry indicating whether that element in keys_list was found in the context """
        is_key_found = [(v is not None) for v in self.as_tuple(subset_includelist=keys_list)]
        return is_key_found

    def check_keys(self, keys_list, debug_print=False) -> tuple[bool, list, list]:
        """ checks whether it has the keys or not
        Usage:
            all_keys_found, found_keys, missing_keys = curr_sess_ctx.check_keys(['format_name','animal','exper_name', 'session_name'], debug_print=False)
        """
        is_key_found = self.has_keys(keys_list)

        found_keys = [k for k, is_found in zip(keys_list, is_key_found) if is_found]
        missing_keys = [k for k, is_found in zip(keys_list, is_key_found) if not is_found]

        all_keys_found = (len(missing_keys) == 0)
        if not all_keys_found:
            if debug_print:
                print(f'missing {len(missing_keys)} keys: {missing_keys}')
        else:
            if debug_print:
                print(f'found all {len(found_keys)} keys: {found_keys}')
        return all_keys_found, found_keys, missing_keys


            

