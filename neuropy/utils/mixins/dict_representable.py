from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage

class DictInitializable:
    """ Implementors can be initialized from a dict
    """
    @staticmethod
    def from_dict(d: dict):
        raise NotImplementedError


class DictRepresentable(DictInitializable):
    def to_dict(self, recurrsively=False):
        raise NotImplementedError


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


            

