# print_helpers.py
from typing import OrderedDict # for OrderedMeta
import numpy as np # for build_formatted_str_from_properties_dict

class SimplePrintable:
    """ Adds the default print method for classes that displays the class name and its dictionary. """
    def __repr__(self) -> str:
        ## TODO: this default printout is actually horrible for classes with any real content (like Pandas.DataFrame members, which spam the notebook)
        return f"<{self.__class__.__name__}: {self.__dict__};>"
    
    

class OrderedMeta(type):
    """ Replaces the inheriting object's dict of attributes with an OrderedDict that preserves enumeration order
    Reference: https://stackoverflow.com/questions/11296010/iterate-through-class-members-in-order-of-their-declaration
    Usage:
        # Set the metaclass property of your custom class to OrderedMeta
        class Person(metaclass=OrderedMeta):
            name = None
            date_of_birth = None
            nationality = None
            gender = None
            address = None
            comment = None
    
        # Can then enumerate members while preserving order
        for member in Person._orderedKeys:
            if not getattr(Person, member):
                print(member)
    """
    @classmethod
    def __prepare__(metacls, name, bases): 
        return OrderedDict()

    def __new__(cls, name, bases, clsdict):
        c = type.__new__(cls, name, bases, clsdict)
        c._orderedKeys = clsdict.keys()
        return c
    


from neuropy.utils.mixins.enum_helpers import ExtendedEnum

class FileProgressAction(ExtendedEnum):
    """Describes the type of file progress actions that can be performed to get the right verbage.
    Used by `print_file_progress_message(...)`
    """
    LOADING = "Loading"
    SAVING = "Saving"
    GENERIC = "Generic"

    @classmethod
    def init(cls, name=None, value=None, fallback_value=None):
        """ e.g. FileProgressAction.init('lOaDing') # <FileProgressAction.LOADING: 'Loading'> """
        return ExtendedEnum.init(name=name, value=value, fallback_value=(fallback_value or cls.GENERIC))

    @property
    def actionVerb(self):
        return FileProgressAction.actionVerbsList()[self]

    # Static properties
    @classmethod
    def actionVerbsList(cls):
        return cls.build_member_value_dict(['from','to',':'])
    

def print_file_progress_message(filepath, action: str, contents_description: str, print_line_ending=' ', returns_string=False):
    """[summary]
        
        print('Saving ripple epochs results to {}...'.format(ripple_epochs.filename), end=' ')
        ripple_epochs.save()
        print('done.')
        
    Args:
        filepath ([type]): [description]
        action (str): [description]
        contents_description (str): [description]
    """
    #  print_file_progress_message(ripple_epochs.filename, 'Saving', 'mua results') # replaces: print('Saving ripple epochs results to {}...'.format(ripple_epochs.filename), end=' ')
    parsed_action_type = FileProgressAction.init(action)
    if returns_string:
        out_string = f'{action} {contents_description} results {parsed_action_type.actionVerb} {str(filepath)}...'
        print(out_string, end=print_line_ending)
        return f'{out_string}{print_line_ending}'
    else:
        print(f'{action} {contents_description} results {parsed_action_type.actionVerb} {str(filepath)}...', end=print_line_ending)
    
    
class ProgressMessagePrinter(object):
    def __init__(self, filepath, action: str, contents_description: str, print_line_ending=' ', finished_message='done.', returns_string=False):
        self.filepath = filepath
        self.action = action
        self.contents_description = contents_description
        self.print_line_ending = print_line_ending
        self.finished_message = finished_message
        
        self.returns_string = returns_string
        if self.returns_string:
            self.returned_string = ''
        else:
            self.returned_string = None    
        
        
    def __enter__(self):
        self.returned_string = print_file_progress_message(self.filepath, self.action, self.contents_description, self.print_line_ending, returns_string=self.returns_string)
        
  
    def __exit__(self, *args):
        print(self.finished_message)        
        if self.returns_string:
            self.returned_string = f'{self.returned_string}{self.finished_message}\n'
            
            

def build_formatted_str_from_properties_dict(dict_items, param_sep_char=', ', key_val_sep_char=':') -> str:
    """ Builds a formatted output string from a dictionary of key:value pairs

    Args:
        dict_items (_type_): the dictionary of items to be built into an output string
        param_sep_char (_type_): the separatior string between each key:value pair in the dictionary
        key_val_sep_char (_type_): the value that separates individual key/value pairs.

    Returns:
        str: _description_
        
    Usage:
        from neuropy.utils.mixins.print_helpers import SimplePrintable, OrderedMeta, build_formatted_str_from_properties_dict
        
        
    """
    with np.printoptions(precision=3, suppress=True, threshold=5):
        properties_key_val_list = []
        for (name, val) in dict_items.items():
            try:
                if hasattr(val, 'str_for_concise_display'):
                    curr_string = f'{name}{key_val_sep_char}{val.str_for_concise_display()}'
                else:
                    curr_string = f'{name}{key_val_sep_char}{np.array(val)}'
            except TypeError:
                curr_string = f'{name}{key_val_sep_char}err'
            properties_key_val_list.append(curr_string)
    
    return param_sep_char.join(properties_key_val_list)

