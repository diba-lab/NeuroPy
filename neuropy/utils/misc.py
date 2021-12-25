from enum import Enum, IntEnum, auto, unique
from itertools import islice
import numpy as np


class AutoNameEnum(Enum):
    """ Inheriting enums will be able to auto generate their name from a string value.

    Usage:
        class Ordinal(AutoNameEnum):
            NORTH = auto()
            SOUTH = auto()
            EAST = auto()
            WEST = auto()
    """
    def _generate_next_value_(name, start, count, last_values):
        return name


def chunks(iterable, size=10):
    """[summary]

    Args:
        iterable ([type]): [description]
        size (int, optional): [description]. Defaults to 10.

    Usage:
        laps_pages = [list(chunk) for chunk in _chunks(sess.laps.lap_id, curr_num_subplots)]
    """
    iterator = iter(iterable)
    for first in iterator:    # stops when iterator is depleted
        def chunk():          # construct generator for next chunk
            yield first       # yield element from for loop
            for more in islice(iterator, size - 1):
                yield more    # yield more elements from the iterator
        yield chunk()         # in outer generator, yield next chunk



def get_interval(self, period, nwindows):

    interval = np.linspace(period[0], period[1], nwindows + 1)
    interval = [[interval[i], interval[i + 1]] for i in range(nwindows)]
    return interval


def print_seconds_human_readable(seconds):
    """ prints the seconds arguments as a human-redable HH::MM:SS.FRACTIONAL time. """
    if isinstance(seconds, int):
        whole_seconds = seconds
        fractional_seconds = None
    else:    
        whole_seconds = int(seconds)
        fractional_seconds = seconds - whole_seconds
    
    m, s = divmod(whole_seconds, 60)
    h, m = divmod(m, 60)
    timestamp = '{0:02}:{1:02}:{2:02}'.format(h, m, s)
    if fractional_seconds is not None:
        frac_seconds_string = ('%f' % fractional_seconds).rstrip('0').rstrip('.').lstrip('0').lstrip('.') # strips any insignficant zeros from the right, and then '0.' string from the left.        
        timestamp = '{}:{}'.format(timestamp, frac_seconds_string) # append the fracitonal seconds string to the timestamp string
    print(timestamp) # print the timestamp
    return h, m, s, fractional_seconds