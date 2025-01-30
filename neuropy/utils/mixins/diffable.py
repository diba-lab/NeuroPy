import collections.abc
from typing import Set

class OrderedSet(collections.abc.MutableSet):
    """ a version of `set` that preserves (or at least doesn't arbitrarily sort/change) insertion order. 

    Usage:
        
        from neuropy.utils.mixins.diffable import OrderedSet

    """
    # def __init__(self, iterable=None):
    #     self._dict = dict.fromkeys(iterable, None) if iterable is not None else {}
    
    def __init__(self, iterable=None):
        self._dict = collections.OrderedDict()
        if iterable:
            self._dict.update((item, None) for item in iterable)
    
    def add(self, item):
        self._dict[item] = None

    def discard(self, item):
        self._dict.pop(item, None)

    def __contains__(self, item):
        return item in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self._dict)})"

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return list(self) == list(other)
        elif isinstance(other, collections.abc.Set):
            return set(self) == set(other)
        return NotImplemented
    
    def __str__(self):
        return "{" + ", ".join(repr(e) for e in self._dict.keys()) + "}"

    # Set operation methods with iterable checks:
    def __or__(self, other):
        if not isinstance(other, collections.abc.Iterable):
            return NotImplemented
        combined = list(self)
        combined.extend([item for item in other if item not in self._dict])
        return self.__class__(combined)

    def __and__(self, other):
        if not isinstance(other, collections.abc.Iterable):
            return NotImplemented
        common = [item for item in self if item in other]
        return self.__class__(common)

    def __sub__(self, other):
        if not isinstance(other, collections.abc.Iterable):
            return NotImplemented
        difference = [item for item in self if item not in other]
        return self.__class__(difference)

    def __xor__(self, other):
        if not isinstance(other, collections.abc.Iterable):
            return NotImplemented
        difference = [item for item in self if item not in other]
        difference.extend([item for item in other if item not in self._dict])
        return self.__class__(difference)

    def __le__(self, other):
        return all(item in other for item in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(item in self for item in other)

    def __gt__(self, other):
        return self >= other and self != other

    def isdisjoint(self, other):
        return not any(item in other for item in self)

    def copy(self):
        return self.__class__(self)

    def update(self, *others):
        for other in others:
            self._dict.update((item, None) for item in other)


    def intersection(self, *others):
        common_elements = self.__class__(self)
        for other in others:
            if not isinstance(other, collections.abc.Iterable):
                raise TypeError(f"Other must be an Iterable, got {type(other).__name__}")
            common_elements._dict = {
                item: None for item in common_elements if item in other
            }
        return common_elements
    
    def intersection_update(self, *others):
        for other in others:
            self._dict = {item: None for item in self if item in other}

    def difference_update(self, *others):
        for other in others:
            self._dict = {item: None for item in self if item not in other}

    def symmetric_difference_update(self, other):
        other_set = OrderedSet(other)
        for item in other_set:
            if item in self:
                self.discard(item)
            else:
                self.add(item)

    
class DiffableObject:
    """ Objects can be "diffed" to other objects of the same type or dictionaries, to see which members are the same and which differ. """
    def diff(self, other_object):
        return DiffableObject.compute_diff(self.__dict__, other_object.__dict__)
    
    @staticmethod
    def compute_diff(lhs_dict, rhs_dict) -> Set:
        """Returns a set of the properties that have changed between the two objects. Objects should be dictionary or related types.

        Args:
            lhs ([type]): [description]
            rhs ([type]): [description]

        Returns:
            [type]: [description]
        """
        set1 = OrderedSet(lhs_dict.items())
        set2 = OrderedSet(rhs_dict.items())
        return set(set1 ^ set2)



