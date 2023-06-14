# print_helpers.py
from typing import Optional, OrderedDict # for OrderedMeta
import numpy as np # for build_formatted_str_from_properties_dict
from neuropy.utils.misc import is_iterable
from neuropy.utils.mixins.indexing_helpers import get_dict_subset # for `build_formatted_str_from_properties_dict`


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
            
            

def build_formatted_str_from_properties_dict(dict_items, param_sep_char=', ', key_val_sep_char=':', subset_includelist:Optional[list]=None, subset_excludelist:Optional[list]=None, float_precision:int=3, array_items_threshold:int=5) -> str:
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
    # if subset include/exclude list are specified, use them to get the relevant dictionary subset before generating the output string.
    dict_items = get_dict_subset(dict_items, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist)
    with np.printoptions(precision=float_precision, suppress=True, threshold=array_items_threshold):
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





#TODO 2023-06-13 13:03: - [ ] Not yet complete. Not sure how to generalize.
class MultiItemStringRepresentationMixin:
    """ enables producing various customizable string representations from a dict-like item. 
    Initially from `neuropy.analyses.placefields.PlacefieldComputationParameters`
    
    Usage:
        from neuropy.utils.mixins.print_helpers import MultiItemStringRepresentationMixin
        
    """

    # print character options:
    _decimal_point_character: str=","
    _param_sep_char: str='-'

    # print precision options:
    _float_precision:int = 3
    _array_items_threshold:int = 5
    
    # variable names that must be provided by the specific class:
    _variable_names: list[str]=['speed_thresh', 'grid_bin', 'smooth', 'frate_thresh']
    _variable_inline_names: list[str]=['speedThresh', 'gridBin', 'smooth', 'frateThresh']
    # Note that I think it's okay to exclude `self.grid_bin_bounds` from these lists


    def _unlisted_parameter_strings(self) -> list[str]:
        """ returns the string representations of all key/value pairs that aren't normally defined. """
        # Dump all arguments into parameters.
        out_list: list[str] = []
        for key, value in self.__dict__.items():
            if (key is not None) and (key not in self._variable_names):
                if value is None:
                    out_list.append(f"{key}_None")
                else:
                    # non-None
                    if hasattr(value, 'str_for_filename'):
                        out_list.append(f'{key}_{value.str_for_filename()}')
                    elif hasattr(value, 'str_for_concise_display'):
                        out_list.append(f'{key}_{value.str_for_concise_display()}')
                    else:
                        # no special handling methods:
                        if isinstance(value, float):
                            out_list.append(f"{key}_{value:.2f}")
                        elif isinstance(value, np.ndarray):
                            out_list.append(f'{key}_ndarray[{np.shape(value)}]')
                        else:
                            # No special handling:
                            try:
                                out_list.append(f"{key}_{value}")
                            except Exception as e:
                                print(f'UNEXPECTED_EXCEPTION: {e}')
                                print(f'self.__dict__: {self.__dict__}')
                                raise e

        return out_list

    def str_for_filename(self, is_2D) -> str:
        """ returns a string that would be compatible for use in a filename across platforms. 
        This means it can't include any forbidden characters such as: ":", backslash, etc.
        """
        with np.printoptions(precision=self._float_precision, suppress=True, threshold=self._array_items_threshold):
            # score_text = f"score: " + str(np.array([epoch_score])).lstrip("[").rstrip("]") # output is just the number, as initially it is '[0.67]' but then the [ and ] are stripped.            
            extras_strings = self._unlisted_parameter_strings()
            if is_2D:
                return '-'.join([f"speedThresh_{self.speed_thresh:.2f}", f"gridBin_{self.grid_bin[0]:.2f}_{self.grid_bin[1]:.2f}", f"smooth_{self.smooth[0]:.2f}_{self.smooth[1]:.2f}", f"frateThresh_{self.frate_thresh:.2f}", *extras_strings])
            else:
                return '-'.join([f"speedThresh_{self.speed_thresh:.2f}", f"gridBin_{self.grid_bin_1D:.2f}", f"smooth_{self.smooth_1D:.2f}", f"frateThresh_{self.frate_thresh:.2f}", *extras_strings])

    def str_for_display(self, is_2D) -> str:
        """ For rendering in a title, etc """
        with np.printoptions(precision=self._float_precision, suppress=True, threshold=self._array_items_threshold):
            extras_string = ', '.join(self._unlisted_parameter_strings())
            if is_2D:
                return f"(speedThresh_{self.speed_thresh:.2f}, gridBin_{self.grid_bin[0]:.2f}_{self.grid_bin[1]:.2f}, smooth_{self.smooth[0]:.2f}_{self.smooth[1]:.2f}, frateThresh_{self.frate_thresh:.2f})" + extras_string
            else:
                return f"(speedThresh_{self.speed_thresh:.2f}, gridBin_{self.grid_bin_1D:.2f}, smooth_{self.smooth_1D:.2f}, frateThresh_{self.frate_thresh:.2f})" + extras_string


    def str_for_attributes_list_display(self, param_sep_char='\n', key_val_sep_char='\t', subset_includelist:Optional[list]=None, subset_excludelist:Optional[list]=None, override_float_precision:Optional[int]=None, override_array_items_threshold:Optional[int]=None):
        """ For rendering in attributes list like outputs
        # Default for attributes lists outputs:
        Example Output:
            speed_thresh	2.0
            grid_bin	[3.777 1.043]
            smooth	[1.5 1.5]
            frate_thresh	0.1
            time_bin_size	0.5
        """
        return build_formatted_str_from_properties_dict(self.to_dict(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist), param_sep_char, key_val_sep_char, float_precision=(override_float_precision or self.float_precision), array_items_threshold=(override_array_items_threshold or self.array_items_threshold))

