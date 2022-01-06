# print_helpers.py
from typing import OrderedDict # for OrderedMeta

class SimplePrintable:
    """ Adds the default print method for classes that displays the class name and its dictionary. """
    def __repr__(self) -> str:
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
    if returns_string:
        out_string = f'{action} {contents_description} results to {str(filepath)}...'
        print(out_string, end=print_line_ending)
        return f'{out_string}{print_line_ending}'
    else:
        print(f'{action} {contents_description} results to {str(filepath)}...', end=print_line_ending)
    
    
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
            