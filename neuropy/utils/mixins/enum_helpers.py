from enum import Enum

""" 
    NOTE: HISTORY: MIRRORED from pyphocorehelpers.DataStructure.enum_helpers.ExtendedEnum on 2023-02-01
    Usage:

    === For an Enum that inherits ExtendedEnum defined below:
    ```python

        from neuropy.utils.mixins.enum_helpers import ExtendedEnum

        @unique
        class FileProgressAction(ExtendedEnum):
            LOADING = "Loading"
            SAVING = "Saving"
            GENERIC = "Generic"

            @classmethod
            def init(cls, name):
                if name.upper() == cls.LOADING.name.upper():
                    return cls.LOADING
                elif name.upper() == cls.SAVING.name.upper():
                    return cls.SAVING
                elif name.upper() == cls.GENERIC.name.upper():
                    return cls.GENERIC
                else:
                    return cls.GENERIC
                    # raise NotImplementedError
                
            @property
            def actionVerb(self):
                return FileProgressAction.actionVerbsList()[self]

            # Static properties
            @classmethod
            def actionVerbsList(cls):
                return cls.build_member_value_dict(['from','to',':'])

    ```

    >>> Output
        FileProgressAction.all_members() # [<FileProgressAction.LOADING: 'Loading'>, <FileProgressAction.SAVING: 'Saving'>, <FileProgressAction.GENERIC: 'Generic'>]
        FileProgressAction.all_member_names() # ['LOADING', 'SAVING', 'GENERIC']
        FileProgressAction.all_member_values() # ['Loading', 'Saving', 'Generic']
        FileProgressAction.build_member_value_dict(['from','to',':']) # {<FileProgressAction.LOADING: 'Loading'>: 'from', <FileProgressAction.SAVING: 'Saving'>: 'to', <FileProgressAction.GENERIC: 'Generic'>: ':'}

"""
class ExtendedEnum(Enum):
    """ Allows Inheritors to list their members, values, and names as lists

    MIRRORED from pyphocorehelpers.DataStructure.enum_helpers.ExtendedEnum on 2023-02-01
    Attribution:
        By blueFast answered Feb 28, 2019 at 5:58
        https://stackoverflow.com/a/54919285/9732163

    """
    @classmethod
    def all_members(cls) -> list:
        return list(cls)
    @classmethod
    def all_member_names(cls) -> list:
        return [member.name for member in cls]
    @classmethod
    def all_member_values(cls) -> list:
        return [member.value for member in cls]
    @classmethod
    def build_member_value_dict(cls, values_list) -> dict:
        assert len(values_list) == len(cls.all_members()), f"values_list must have one value for each member of the enum, but got {len(values_list)} values for {len(cls.all_members())} members."
        return dict(zip(cls.all_members(), values_list))

    # ==================================================================================================================== #
    # INIT Helpers                                                                                                         #
    # ==================================================================================================================== #
    @classmethod
    def _init_from_upper_name_dict(cls) -> dict:
        return dict(zip([a_name.upper() for a_name in cls.all_member_names()], cls.all_members()))
    @classmethod
    def _init_from_value_dict(cls) -> dict:
        return dict(zip(cls.all_member_values(), cls.all_members()))

    @classmethod
    def init(cls, name=None, value=None, fallback_value=None):
        """ Allows enum values to be initialized from either a name or value (but not both).
            Also allows passthrough of either name or value that are already of the correct type (of Enum type class) and those will just be returned.
                useful for unvalidated parameters

            e.g. FileProgressAction.init('lOaDing') # <FileProgressAction.LOADING: 'Loading'> 
        """
        assert (name is not None) or (value is not None), "You must specify either name or value, and the other will be returned"
        assert (name is None) or (value is None), "You cannot specify both name and value, as it would be ambiguous which takes priority. Please remove one of the two arguments."
        if name is not None:
            ## Name Mode:
            if isinstance(name, cls):
                return name # already the correct instance of class itself, return name (this allows passthrough of unvalidated parameters that will return the converted Enum or the original value if it already was of the correct class            
            if fallback_value is not None:
                return cls._init_from_upper_name_dict().get(name.upper(), fallback_value)
            else:
                return cls._init_from_upper_name_dict()[name.upper()]
        elif value is not None:
            ## Value Mode:
            if isinstance(value, cls):
                return value # already the correct instance of class itself, return value (this allows passthrough of unvalidated parameters that will return the converted Enum or the original value if it already was of the correct class
            if fallback_value is not None:
                return cls._init_from_value_dict().get(value, fallback_value)
            else:
                return cls._init_from_value_dict()[value]
        else:
            raise NotImplementedError # THIS SHOULD NOT EVEN BE POSSIBLE!
