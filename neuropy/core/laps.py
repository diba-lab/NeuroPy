import numpy as np
import pandas as pd

from neuropy.utils.mixins.print_helpers import SimplePrintable
from .datawriter import DataWriter

from ..utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from ..utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol

## Import:
# from neuropy.core.laps import Laps

# TODO: implement: NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, TimeSlicableObjectProtocol
class Laps(SimplePrintable, DataWriter):
    """Class to hold computed info about laps and how they relate to other information like times, flat linear indicies, etc.
    
    ## TODO: Look at Epoch class for implementation guidance 
    """

    def __init__(
        self,
        lap_id: np.ndarray,
        laps_spike_counts=None,
        lap_start_stop_flat_idx=None,
        lap_start_stop_time=None,
        metadata=None,
    ) -> None:
        super().__init__(metadata=metadata)

        self.lap_id = np.array(lap_id)
        assert (len(laps_spike_counts) == len(lap_id)), "laps_spike_counts first dimension must match number of laps"
        self.laps_spike_counts = laps_spike_counts
        assert (lap_start_stop_flat_idx.shape[0] == len(lap_id)), "lap_start_stop_flat_idx first dimension must match number of laps"
        self.lap_start_stop_flat_idx = lap_start_stop_flat_idx
        assert (lap_start_stop_time.shape[0] == len(lap_id)), "lap_start_stop_time first dimension must match number of laps"
        self.lap_start_stop_time = lap_start_stop_time

    @property
    def n_laps(self):
        return len(self.lap_id)
        
    @staticmethod
    def from_dict(d: dict):
        return Laps(d['lap_id'], laps_spike_counts = d['laps_spike_counts'], lap_start_stop_flat_idx = d['lap_start_stop_flat_idx'],
                    lap_start_stop_time = d['lap_start_stop_time'], metadata = d.get('metadata', None))
        
        
    def to_dict(self, recurrsively=False):
        simple_dict = self.__dict__
        # if recurrsively:
        #     simple_dict['paradigm'] = simple_dict['paradigm'].to_dict()
        #     simple_dict['position'] = simple_dict['position'].to_dict()
        #     simple_dict['neurons'] = simple_dict['neurons'].to_dict()        
        return simple_dict
    

    # def time_slice(self, t_start=None, t_stop=None):
    #     # t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
    #     flattened_spiketrains = deepcopy(self)
    #     included_df = flattened_spiketrains.spikes_df[((flattened_spiketrains.spikes_df.t_seconds > t_start) & (flattened_spiketrains.spikes_df.t_seconds < t_stop))]
    #     return FlattenedSpiketrains(included_df, t_start=flattened_spiketrains.t_start, metadata=flattened_spiketrains.metadata)
        
        