""" result_context.py

The general goal of these context objects is to be able to generate distinct identifiers for a given result so that it can be serialized to disk and analyzed later.

To do this systematically, we need to be able to identify



For human purposes, it's also incredibly useful to be able to generate minimal-difference versions of these contexts.

For example, if you generated two figures, one with a spatial bin size of 0.5 and another of 1.0 when generating titles for the two plots those two levels of the spatial bin size variable are all that you'd want to keep straight. The rest of the variables aren't changing, and so the minimal-difference required to discriminate between the two cases is only the levels of that variables.
{'bin_size': [0.5, 1.0]}

Let's say you save these out to disk and then a week later see that applying the exact same analyses (with those same two levels) to a different dataset produces an unexpected result at a value of 0.5. You don't remember such a wonky looking graph of 0.5 for the previous dataset, and want to figure out what's going wrong. 

To do this you'll now want to compare the 4 graphs, meaning you'll need to keep track:

Now you have two different datasets: 'earlier' and 'now', each with a minimal-difference of:
    'earlier': {'bin_size': [0.5, 1.0]}
    'now': {'bin_size': [0.5, 1.0]}

To uniquely identify them you'll probably want to visualize them as:

'earlier' (row):    | 'bin_size'=0.5 | 'bin_size'=1.0 | 
'now' (row):        | 'bin_size'=0.5 | 'bin_size'=1.0 |

Ultimately you only have two axes over which things can be compared (rows and columns)
There's one hidden one: 'tabs', 'windows'

Humans need things with distinct, visual groupings. Inclusion Sets, Exceptions (a single outlier rendered in juxtaposition to an inclusion set of the norms)

"""

from neuropy.utils.mixins.diffable import DiffableObject


class IdentifyingContext(DiffableObject, object):
    """ a general extnsible base context that allows additive member creation
    
        Should not hold any state or progress-related variables. 
    
    """
    def __init__(self, **kwargs):
        super(IdentifyingContext, self).__init__()
        ## Sets attributes dnymically:
        for name, value in kwargs.items():
            setattr(self, name, value)
        
    def add_context(self, collision_prefix:str, **additional_context_items):
        """ adds the additional_context_items to self 
        collision_prefix: only used when an attr name in additional_context_items already exists for this context 
        
        """
        for name, value in additional_context_items.items():
            # TODO: ensure no collision between attributes occur, and if they do rename them with an identifying prefix
            if hasattr(self, name):
                print(f'WARNING: namespace collision in add_context! attr with name {name} already exists!')
                ## TODO: rename the current attribute to be set by appending a prefix
                final_name = f'{collision_prefix}{name}'
            else:
                final_name = name
            # Set the new attr
            setattr(self, final_name, value)
        
        return self
        
    def get_description(self, separator:str='_', include_property_names:bool=False, replace_separator_in_property_names:str='-', prefix_items=[], suffix_items=[])->str:
        """ returns a simple text descriptor of the context
        
        include_property_names: str - whether to include the keys/names of the properties in the output string or just the values
        replace_separator_in_property_names: str = replaces occurances of the separator with the str specified for the included property names. has no effect if include_property_names=False
        
        Outputs:
            a str like 'sess_kdiba_2006-6-07_11-26-53'
        """
        ## Build a session descriptor string:
        if include_property_names:
            descriptor_array = [[name.replace(separator, replace_separator_in_property_names), val]  for name, val in self.to_dict().items()] # creates a list of [name, val] list items
            descriptor_array = [item for sublist in descriptor_array for item in sublist] # flat descriptor array
        else:
            descriptor_array = list(self.to_dict().values())
            
        if prefix_items is not None:
            descriptor_array.extend(prefix_items)
        if suffix_items is not None:
            descriptor_array.extend(suffix_items)
        
        descriptor_string = separator.join(descriptor_array)
        return descriptor_string
    
    def __str__(self) -> str:
        return self.get_description()

    def __hash__(self):
        """ custom hash function that allows use in dictionary just based off of the values and not the object instance. """
        dict_rep = self.to_dict()
        member_names_tuple = list(dict_rep.keys())
        values_tuple = list(dict_rep.values())
        combined_tuple = tuple(member_names_tuple + values_tuple)
        return hash(combined_tuple)
    
    def to_dict(self):
        return self.__dict__
    @classmethod
    def init_from_dict(cls, a_dict):
        return cls(**a_dict) # expand the dict as input args.
        

# class SessionContext(IdentifyingContext):
#     """result_context serves to uniquely identify the **context** of a given generic result.

#     Typically a result depends on several inputs:

#     - session: the context in which the original recordings were made.
#         Originally this includes the circumstances udner which the recording was performed (recording datetime, recording configuration (hardware, sampling rate, etc), experimenter name, animal identifer, etc)
#     - filter: the filtering performed to pre-process the loaded session data
#     - computation configuration: the specific computations performed and their parameters to transform the data to the result

#     Heuristically: If changing the value of that variable results in a changed result, it should be included in the result_context 
#     """
#     format_name: str = ''
#     session_name: str = ''
    
#     # def __init__(self, session_context, filter_context, computation_configuration):
#     #     super(ResultContext, self).__init__()
#     #     self.session_context = session_context
#     #     self.filter_context = filter_context
#     #     self.computation_configuration = computation_configuration
        
        
# class ResultContext(IdentifyingContext):
#     """result_context serves to uniquely identify the **context** of a given generic result.

#     Typically a result depends on several inputs:

#     - session: the context in which the original recordings were made.
#         Originally this includes the circumstances udner which the recording was performed (recording datetime, recording configuration (hardware, sampling rate, etc), experimenter name, animal identifer, etc)
#     - filter: the filtering performed to pre-process the loaded session data
#     - computation configuration: the specific computations performed and their parameters to transform the data to the result

#     Heuristically: If changing the value of that variable results in a changed result, it should be included in the result_context 
#     """
#     session_context: str = ''
#     filter_context: str = ''
#     computation_configuration: str = ''
    
 
#     def __init__(self, session_context, filter_context, computation_configuration):
#         super(ResultContext, self).__init__()
#         self.session_context = session_context
#         self.filter_context = filter_context
#         self.computation_configuration = computation_configuration
    
    