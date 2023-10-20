class DiffableObject:
    """ Objects can be "diffed" to other objects of the same type or dictionaries, to see which members are the same and which differ. """
    def diff(self, other_object):
        return DiffableObject.compute_diff(self.__dict__, other_object.__dict__)
    
    @staticmethod
    def compute_diff(lhs_dict, rhs_dict):
        """Returns a set of the properties that have changed between the two objects. Objects should be dictionary or related types.

        Args:
            lhs ([type]): [description]
            rhs ([type]): [description]

        Returns:
            [type]: [description]
        """
        set1 = set(lhs_dict.items())
        set2 = set(rhs_dict.items())
        return set1 ^ set2
