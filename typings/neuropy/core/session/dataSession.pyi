"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import Sequence, Union
from pathlib import Path
from neuropy.core.epoch import Epoch, NamedTimerange
from neuropy.utils.mixins.concatenatable import ConcatenationInitializable
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
from neuropy.utils.mixins.panel import DataSessionPanelMixin
from neuropy.utils.mixins.HDF5_representable import HDF_SerializationMixin

"""
This type stub file was generated by pyright.
"""
class DataSession(HDF_SerializationMixin, DataSessionPanelMixin, NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, ConcatenationInitializable, TimeSlicableObjectProtocol):
    """ holds the collection of all data, both loaded and computed, related to an experimental recording session. Can contain multiple discontiguous time periods ('epochs') meaning it can represent the data collected over the course of an experiment for a single animal (across days), on a single day, etc.
    
    Provides methods for loading, accessing, and manipulating data such as neural spike trains, behavioral laps, etc.
        
    """
    def __init__(self, config, filePrefix=..., recinfo=..., eegfile=..., datfile=..., neurons=..., probegroup=..., position=..., paradigm=..., ripple=..., mua=..., laps=..., flattened_spiketrains=..., pbe=..., **kwargs) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def is_resolved(self):
        ...
    
    @property
    def basepath(self):
        ...
    
    @property
    def session_name(self):
        ...
    
    @property
    def name(self):
        ...
    
    @property
    def resolved_files(self):
        ...
    
    @property
    def position_sampling_rate(self):
        ...
    
    @property
    def neuron_ids(self):
        ...
    
    @property
    def n_neurons(self):
        ...
    
    @property
    def spikes_df(self):
        ...
    
    @property
    def epochs(self):
        """The epochs property is an alias for self.paradigm."""
        ...
    
    @epochs.setter
    def epochs(self, value):
        ...
    
    @property
    def t_start(self):
        ...
    
    @property
    def duration(self):
        ...
    
    @property
    def t_stop(self):
        ...
    
    @property
    def has_replays(self):
        """The has_replays property."""
        ...
    
    def time_slice(self, t_start, t_stop, enable_debug=...):
        """ Implementors return a copy of themselves with each of their members sliced at the specified indicies """
        ...
    
    def get_neuron_type(self, query_neuron_type):
        """ filters self by the specified query_neuron_type, only returning neurons that match. """
        ...
    
    def get_named_epoch_timerange(self, epoch_name):
        ...
    
    def filtered_by_time_slice(self, t_start=..., t_stop=...):
        ...
    
    def filtered_by_neuron_type(self, query_neuron_type):
        ...
    
    def filtered_by_epoch(self, epoch_specifier):
        ...
    
    def filtered_by_named_timerange(self, custom_named_timerange_obj: NamedTimerange):
        ...
    
    def get_by_id(self, ids):
        """Implementors return a copy of themselves with neuron_ids equal to ids"""
        ...
    
    def get_context(self):
        """ returns an IdentifyingContext for the session """
        ...
    
    def get_description(self) -> str:
        """ returns a simple text descriptor of the session
        Outputs:
            a str like 'sess_kdiba_2006-6-07_11-26-53'
        """
        ...
    
    def __str__(self) -> str:
        ...
    
    @staticmethod
    def from_dict(d: dict):
        ...
    
    def to_dict(self, recurrsively=...):
        ...
    
    def __sizeof__(self) -> int:
        """ Returns the approximate size in bytes for this object by getting the size of its dataframes. """
        ...
    
    def panel_dataframes_overview(self, max_page_items=...):
        ...
    
    def get_output_path(self, mkdir_if_needed: bool = ...) -> Path:
        """ Build a folder to store the temporary outputs of this session """
        ...
    
    def replace_session_replays_with_estimates(self, debug_print=..., **kwargs):
        """ 2023-04-20 - Backup the loaded replays if they exist for the session to `.replay_backup`, and then estimate them fresh and assign them to the `a_session.replay` """
        ...
    
    def replace_session_laps_with_estimates(self, **kwargs):
        """ 2023-05-02 - Estimates the laps and replaces the existing laps object. """
        ...
    
    @staticmethod
    def compute_neurons_ripples(session, save_on_compute=...):
        ...
    
    @staticmethod
    def compute_neurons_mua(session, save_on_compute=...):
        ...
    
    @staticmethod
    def compute_pbe_epochs(session, active_parameters=..., save_on_compute=...):
        """ 
            old_default_parameters = dict(sigma=0.02, thresh=(0, 3), min_dur=0.1, merge_dur=0.01, max_dur=1.0) # Default
            old_kamran_parameters = dict(sigma=0.02, thresh=(0, 1.5), min_dur=0.06, merge_dur=0.06, max_dur=2.3) # Kamran's Parameters
            new_papers_parameters = dict(sigma=0.030, thresh=(0, 1.5), min_dur=0.030, merge_dur=0.100, max_dur=0.300) # NewPaper's Parameters
            kamrans_new_parameters = dict(sigma=0.030, thresh=(0, 1.5), min_dur=0.030, merge_dur=0.100, max_dur=0.6) # 2023-10-05 Kamran's imposed Parameters, wants to remove the effect of the max_dur which was previously at 0.300
            
            new_pbe_epochs = sess.compute_pbe_epochs(sess, active_parameters=kamrans_new_parameters)

        """
        ...
    
    @staticmethod
    def compute_linear_position(session, debug_print=...):
        """ compute linear positions:
        TODO 2023-06-06: BUG: this is not correct. It should only compute one PCA, that for the global epoch, and then slice to the other two epochs.
        
        """
        ...
    
    def estimate_replay_epochs(self, require_intersecting_epoch=..., min_epoch_included_duration=..., max_epoch_included_duration=..., maximum_speed_thresh=..., min_inclusion_fr_active_thresh=..., min_num_unique_aclu_inclusions=..., save_on_compute=..., debug_print=...):
        """estimates replay epochs from PBE and Position data.

        Args:
            self (_type_): _description_
            min_epoch_included_duration (float, optional): all epochs shorter than min_epoch_included_duration will be excluded from analysis. Defaults to 0.06.
            maximum_speed_thresh (float, optional): epochs are only included if the animal's interpolated speed (as determined from the session's position dataframe) is below the speed. Defaults to 2.0 [cm/sec].
            save_on_compute (bool, optional): _description_. Defaults to False.`
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        ...
    
    @classmethod
    def perform_compute_estimated_replay_epochs(cls, a_session, require_intersecting_epoch=..., min_epoch_included_duration=..., max_epoch_included_duration=..., maximum_speed_thresh=..., min_inclusion_fr_active_thresh=..., min_num_unique_aclu_inclusions=..., save_on_compute=..., debug_print=...):
        """estimates replay epochs from PBE and Position data.

        Args:
            a_session (_type_): _description_
            min_epoch_included_duration (float, optional): all epochs shorter than min_epoch_included_duration will be excluded from analysis. Defaults to 0.06.
            max_epoch_included_duration (float, optional): all epochs longer than max_epoch_included_duration will be excluded from analysis. Defaults to 0.6.
            maximum_speed_thresh (float, optional): epochs are only included if the animal's interpolated speed (as determined from the session's position dataframe) is below the speed. Defaults to 2.0 [cm/sec].
            min_inclusion_fr_active_thresh: minimum firing rate (in Hz) for a unit to be considered "active" for inclusion.
            min_num_unique_aclu_inclusions: minimum number of unique active cells that must be included in an epoch to have it included.

            save_on_compute (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        ...
    
    @classmethod
    def filter_replay_epochs(cls, curr_replays, pos_df, spikes_df, **kwargs) -> Epoch:
        """filters the provided replay epochs by specified constraints.

        # require_intersecting_epoch:Epoch=None, min_epoch_included_duration=0.06, max_epoch_included_duration=0.6, maximum_speed_thresh=2.0, min_inclusion_fr_active_thresh=2.0, min_num_unique_aclu_inclusions=3, save_on_compute=False, debug_print=False

        Args:
            a_session (_type_): _description_
            min_epoch_included_duration (float, optional): all epochs shorter than min_epoch_included_duration will be excluded from analysis. Defaults to 0.06.
            max_epoch_included_duration (float, optional): all epochs longer than max_epoch_included_duration will be excluded from analysis. Defaults to 0.6.
            maximum_speed_thresh (float, optional): epochs are only included if the animal's interpolated speed (as determined from the session's position dataframe) is below the speed. Defaults to 2.0 [cm/sec].
            min_inclusion_fr_active_thresh: minimum firing rate (in Hz) for a unit to be considered "active" for inclusion.
            min_num_unique_aclu_inclusions: minimum number of unique active cells that must be included in an epoch to have it included.
            save_on_compute (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_

        """
        ...
    
    @classmethod
    def perform_replace_session_replays_with_estimates(cls, a_session, debug_print=..., **kwargs):
        """ 2023-04-20 - Backup the loaded replays if they exist for the session to `a_session.replay_backup`, and then estimate them fresh and assign them to the `a_session.replay` 
        Usage:
            long_replays, short_replays, global_replays = [replace_session_replays_with_estimates(a_session, debug_print=False) for a_session in [long_session, short_session, global_session]]
        """
        ...
    
    @classmethod
    def concat(cls, objList: Union[Sequence, np.array]):
        ...
    
    def split_by_laps(self):
        """ Returns a list containing separate copies of this session with all of its members filtered by the laps, for each lap
        """
        ...
    
    def filtered_by_laps(self, lap_indices=...):
        """ Returns a copy of this session with all of its members filtered by the laps.
        """
        ...
    
    def compute_position_laps(self):
        """ Adds the 'lap' and the 'lap_dir' columns to the position dataframe:
        Usage:
            laps_position_traces, curr_position_df = compute_position_laps(sess) """
        ...
    
    @staticmethod
    def compute_laps_position_df(position_df, laps_df):
        """ Adds a 'lap' column to the position dataframe:
            Also adds a 'lap_dir' column, containing 0 if it's an outbound trial, 1 if it's an inbound trial, and -1 if it's neither.
        Usage:
            laps_position_traces, curr_position_df = compute_position_laps(sess) """
        ...
    
    def compute_spikes_PBEs(self):
        """ Adds the 'PBE_id' column to the spikes dataframe:
        Usage:
            updated_spikes_df = sess.compute_spikes_PBEs()"""
        ...
    
    @staticmethod
    def compute_PBEs_spikes_df(spk_df, pbe_epoch_df):
        """ Adds a 'PBE_id' column to the spikes_df:
        Usage:
            spk_df = compute_PBEs_spikes_df(sess) """
        ...
    
    def plot_laps_2d(self):
        ...
    
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            file_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            _pfnd_obj.to_hdf(file_path, key='test_pfnd')
            
            
        `key` passed in should be the path to the session_root: '/kdiba/gor01/one/2006-6-08_14-26-15'
        """
        ...
    
    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> DataSession:
        """ Reads the data from the key in the hdf5 file at file_path
        """
        ...
    

