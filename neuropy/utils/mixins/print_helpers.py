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
    
