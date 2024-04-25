"""
This type stub file was generated by pyright.
"""

from .datawriter import DataWriter

"""
This type stub file was generated by pyright.
"""
class Shank:
    def __init__(self) -> None:
        ...
    
    @staticmethod
    def auto_generate(columns=..., contacts_per_column=..., xpitch=..., ypitch=..., y_shift_per_column=..., channel_id=...):
        ...
    
    @staticmethod
    def from_library(probe_name):
        ...
    
    @staticmethod
    def set_contacts(positions, channel_ids):
        ...
    
    @property
    def x(self):
        ...
    
    @x.setter
    def x(self, arr):
        ...
    
    @property
    def y(self):
        ...
    
    @y.setter
    def y(self, arr):
        ...
    
    @property
    def contact_id(self):
        ...
    
    @property
    def channel_id(self):
        ...
    
    @channel_id.setter
    def channel_id(self, chan_ids):
        ...
    
    @property
    def connected(self):
        ...
    
    @connected.setter
    def connected(self, arr):
        ...
    
    @property
    def n_contacts(self):
        ...
    
    def to_dict(self, recurrsively=...):
        ...
    
    def from_dict(self):
        ...
    
    def set_disconnected_channels(self, channel_ids):
        ...
    
    def to_dataframe(self):
        ...
    
    def move(self, translation):
        ...
    


class Probe:
    def __init__(self, shanks, shank_pitch=...) -> None:
        ...
    
    @property
    def n_contacts(self):
        ...
    
    @property
    def n_shanks(self):
        ...
    
    @property
    def shank_id(self):
        ...
    
    @property
    def x(self):
        ...
    
    @property
    def x_max(self):
        ...
    
    @property
    def y(self):
        ...
    
    @property
    def channel_id(self):
        ...
    
    @property
    def connected(self):
        ...
    
    def add_shanks(self, shanks: Shank, shank_pitch=...):
        ...
    
    def to_dict(self, recurrsively=...):
        ...
    
    def to_dataframe(self):
        ...
    
    def move(self, translation):
        ...
    


class ProbeGroup(DataWriter):
    def __init__(self, metadata=...) -> None:
        ...
    
    @property
    def x(self):
        ...
    
    @property
    def x_min(self):
        ...
    
    @property
    def x_max(self):
        ...
    
    @property
    def y(self):
        ...
    
    @property
    def y_min(self):
        ...
    
    @property
    def y_max(self):
        ...
    
    @property
    def n_contacts(self):
        ...
    
    @property
    def channel_id(self):
        ...
    
    @property
    def shank_id(self):
        ...
    
    def get_channels(self, groupby=...):
        ...
    
    def get_shank_id_for_channels(self, channel_id):
        """Get shank ids for the channels.

        Parameters
        ----------
        channel_id : array
            channel_ids, can have repeated values

        Returns
        -------
        array
            shank_ids corresponding to the channels
        """
        ...
    
    def get_probe(self):
        ...
    
    def get_connected_channels(self, groupby=...):
        ...
    
    @property
    def probe_id(self):
        ...
    
    @property
    def n_probes(self):
        ...
    
    @property
    def n_shanks(self):
        ...
    
    @property
    def get_disconnected(self):
        ...
    
    def add_probe(self, probe: Probe):
        ...
    
    def to_dict(self, recurrsively=...):
        ...
    
    @staticmethod
    def from_dict(d: dict):
        ...
    
    def to_dataframe(self):
        ...
    
    def remove_probes(self, probe_id=...):
        ...
    

