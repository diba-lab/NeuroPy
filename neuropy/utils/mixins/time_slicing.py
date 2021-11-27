import numpy as np
# from typing import Protocol # NOTE: requires Python 3.8 or the typing-extensions package for Python 3.5+ ( which backports typing.Protocol )
# from abc import ABC, abstractmethod # ABC: abstract base class lib that prevents these mixins from being directly instantiated themselves

# class StartStopTimeProtocol:

class TimeSlicableIndiciesMixin:
    def safe_start_stop_times(self, t_start, t_stop):
        """ Returns t_start and t_stop while ensuring the values passed in aren't None. """
        if t_start is None:
            t_start = self.t_start
        if t_stop is None:
            t_stop = self.t_stop
        return t_start, t_stop
            
    def time_slice_indicies(self, t_start, t_stop):
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        return (self.time > t_start) & (self.time < t_stop)
    
# class TimeSlicableObjectProtocol(ABC):
# class TimeSlicableObjectProtocol(Protocol):
class TimeSlicableObjectProtocol:
    # @abstractmethod # for ABC approach
    def time_slice(self, t_start, t_stop):
        """ Implementors return a copy of themselves with each of their members sliced at the specified indicies """
        raise NotImplementedError


# class TimeSlicableObjectMixin(TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin, ABC):
#     def time_slice(self, t_start, t_stop):
#         """ Implementors return a copy of themselves with each of their members sliced at the specified indicies """
#         t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
#         indices = self.time_slice_indicies(t_start, t_stop)
#         raise NotImplementedError # should never happen because this is an @abstractmethod anyway, but just to be safe
#         return None
        
        