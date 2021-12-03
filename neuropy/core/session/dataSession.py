import sys
from typing import Sequence, Union
import numpy as np
import pandas as pd
from pathlib import Path
from neuropy.core import neurons
from neuropy.core.epoch import NamedTimerange
from neuropy.core.flattened_spiketrains import FlattenedSpiketrains
from neuropy.core.laps import Laps
from neuropy.core.position import Position
from neuropy.utils.mixins.concatenatable import ConcatenationInitializable

# Local imports:
## Core:
from neuropy.utils.mixins.print_helpers import SimplePrintable, OrderedMeta
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
        

class DataSession(NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, ConcatenationInitializable, TimeSlicableObjectProtocol):
    def __init__(self, config, filePrefix = None, recinfo = None,
                 eegfile = None, datfile = None,
                 neurons = None, probegroup = None, position = None, paradigm = None,
                 ripple = None, mua = None, laps= None, flattened_spiketrains = None):       
        self.config = config
        
        self.is_loaded = False
        self.filePrefix = filePrefix
        self.recinfo = recinfo
        
        self.eegfile = eegfile
        self.datfile = datfile
        
        self.neurons = neurons
        self.probegroup = probegroup
        self.position = position
        self.paradigm = paradigm
        self.ripple = ripple
        self.mua = mua
        self.laps = laps # core.laps.Laps
        self.flattened_spiketrains = flattened_spiketrains # core.FlattenedSpiketrains
        

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})"
    #######################################################
    ## Passthru Accessor Properties:
    @property
    def is_resolved(self):
        return self.config.is_resolved
    @property
    def basepath(self):
        return self.config.basepath
    @property
    def session_name(self):
        return self.config.session_name
    @property
    def name(self):
        return self.session_name
    @property
    def resolved_files(self):
        return (self.config.resolved_required_files + self.config.resolved_optional_files)

    @property
    def position_sampling_rate(self):
        return self.position.sampling_rate

    @property
    def neuron_ids(self):
        return self.neurons.neuron_ids
    
    @property
    def n_neurons(self):
        return self.neurons.n_neurons
    
    
    # @property
    # def n_total_spikes(self):
    #     return self.neurons.n_neurons
    # @property
    # def is_resolved(self):
    #     return self.config.is_resolved
    # @property
    # def is_resolved(self):
    #     return self.config.is_resolved


    @property
    def spikes_df(self):
        return self.flattened_spiketrains.spikes_df
    
    
    # @property
    # def is_loaded(self):
    #     """The epochs property is an alias for self.paradigm."""
    #     return self.paradigm
    
    @property
    def epochs(self):
        """The epochs property is an alias for self.paradigm."""
        return self.paradigm
    @epochs.setter
    def epochs(self, value):
        self.paradigm = value
        
    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop, enable_debug=True):
        """ Implementors return a copy of themselves with each of their members sliced at the specified indicies """
        active_epoch_times = [t_start, t_stop]
        if enable_debug: 
            print('Constraining to epoch with times (start: {}, end: {})'.format(active_epoch_times[0], active_epoch_times[1]))
        # make a copy of self:
        # should implement __deepcopy__() and __copy__()??
        copy_sess = DataSession.from_dict(self.to_dict())
        # update the copy_session's time_sliceable objects
        copy_sess.neurons = self.neurons.time_slice(active_epoch_times[0], active_epoch_times[1]) # active_epoch_session_Neurons: Filter by pyramidal cells only, returns a core.
        copy_sess.position = self.position.time_slice(active_epoch_times[0], active_epoch_times[1]) # active_epoch_pos: active_epoch_pos's .time and start/end are all valid
        copy_sess.flattened_spiketrains = self.flattened_spiketrains.time_slice(active_epoch_times[0], active_epoch_times[1]) # active_epoch_pos: active_epoch_pos's .time and start/end are all valid  
        
        # copy_sess.laps = self.      
        return copy_sess
    

    def get_neuron_type(self, query_neuron_type):
        """ filters self by the specified query_neuron_type, only returning neurons that match. """
        print('Constraining to units with type: {}'.format(query_neuron_type))
        # make a copy of self:
        copy_sess = DataSession.from_dict(self.to_dict())
        # update the copy_session's neurons objects
        copy_sess.neurons = self.neurons.get_neuron_type(query_neuron_type) # active_epoch_session_Neurons: Filter by pyramidal cells only, returns a core.
        copy_sess.flattened_spiketrains = self.flattened_spiketrains.get_neuron_type(query_neuron_type) # active_epoch_session_Neurons: Filter by pyramidal cells only, returns a core.
        return copy_sess
    def get_named_epoch_timerange(self, epoch_name):
        if isinstance(epoch_name, str):            
            # convert the epoch_name string to a NamedTimerange object, get its time from self.epochs
            active_custom_named_epoch = NamedTimerange(name=epoch_name, start_end_times=self.epochs[epoch_name])
            return active_custom_named_epoch
        return None
 
    

    ## filtered_by_* functions:
    
    def filtered_by_time_slice(self, t_start=None, t_stop=None):
        return self.time_slice(t_start, t_stop)
        
    def filtered_by_neuron_type(self, query_neuron_type):
        return self.get_neuron_type(query_neuron_type)
    
    def filtered_by_epoch(self, epoch_specifier):
        if isinstance(epoch_specifier, str):            
            # convert the epoch_name string to a NamedTimerange object, get its time from self.epochs
            active_custom_named_epoch = NamedTimerange(name=epoch_specifier, start_end_times=self.epochs[epoch_specifier])
            return self.filtered_by_named_timerange(active_custom_named_epoch)
        elif isinstance(epoch_specifier, NamedTimerange):
            return self.filtered_by_named_timerange(epoch_specifier)
        else:
            raise TypeError


    def filtered_by_named_timerange(self, custom_named_timerange_obj: NamedTimerange):
        return self.time_slice(custom_named_timerange_obj.t_start, custom_named_timerange_obj.t_stop)
    

    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids):
        """Implementors return a copy of themselves with neuron_ids equal to ids"""
        copy_sess = DataSession.from_dict(self.to_dict())
        copy_sess.neurons = self.neurons.get_by_id(ids)
        copy_sess.flattened_spiketrains = self.flattened_spiketrains.get_by_id(ids)
        return copy_sess

  
    @staticmethod
    def from_dict(d: dict):
        return DataSession(d['config'], filePrefix = d['filePrefix'], recinfo = d['recinfo'],
                 eegfile = d['eegfile'], datfile = d['datfile'],
                 neurons = d['neurons'], probegroup = d.get('probegroup', None), position = d['position'], paradigm = d['paradigm'],
                 ripple = d.get('ripple', None), mua = d.get('mua', None), 
                 laps= d.get('laps', None),
                 flattened_spiketrains = d.get('flattened_spiketrains', None))

        
    def to_dict(self, recurrsively=False):
        simple_dict = self.__dict__
        if recurrsively:
            simple_dict['paradigm'] = simple_dict['paradigm'].to_dict()
            simple_dict['position'] = simple_dict['position'].to_dict()
            simple_dict['neurons'] = simple_dict['neurons'].to_dict() 
            # simple_dict['flattened_spiketrains'] = simple_dict['flattened_spiketrains'].to_dict() ## TODO: implement .to_dict() for FlattenedSpiketrains object to make this work
        return simple_dict
        
    ## Linearize Position:
    @staticmethod
    def compute_linearized_position(session, epochLabelName='maze1', method='isomap'):
        # returns Position objects for active_epoch_pos and linear_pos
        from neuropy.utils import position_util
        active_epoch_times = session.epochs[epochLabelName] # array([11070, 13970], dtype=int64)
        acitve_epoch_timeslice_indicies = session.position.time_slice_indicies(active_epoch_times[0], active_epoch_times[1])
        active_epoch_pos = session.position.time_slice(active_epoch_times[0], active_epoch_times[1])
        linear_pos = position_util.linearize_position(active_epoch_pos, method=method)
        return acitve_epoch_timeslice_indicies, active_epoch_pos, linear_pos
    #  acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = compute_linearized_position(sess, 'maze1')
    #  acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = compute_linearized_position(sess, 'maze2')

    ## Ripple epochs
    #   To detect ripples one also needs probegroup.
    @staticmethod
    def compute_neurons_ripples(session):
        print('computing ripple epochs for session...\n')
        from neuropy.analyses import oscillations
        signal = session.eegfile.get_signal()
        ripple_epochs = oscillations.detect_ripple_epochs(signal, session.probegroup)
        ripple_epochs.filename = session.filePrefix.with_suffix('.ripple.npy')
        print('Saving ripple epochs results to {}...'.format(ripple_epochs.filename))
        ripple_epochs.save()
        print('done.\n')
        return ripple_epochs
    # sess.ripple = compute_neurons_ripples(sess)

    ## BinnedSpiketrain and Mua objects using Neurons
    @staticmethod
    def compute_neurons_mua(session):
        print('computing neurons mua for session...\n')
        mua = session.neurons.get_mua()
        mua.filename = session.filePrefix.with_suffix(".mua.npy")
        print('Saving mua results to {}...'.format(mua.filename))
        mua.save()
        print('done.\n')
        return mua    
    # sess.mua = compute_neurons_mua(sess) # Set the .mua field on the session object once complete

    @staticmethod
    def compute_pbe_epochs(session):
        from neuropy.analyses import detect_pbe_epochs
        print('computing PBE epochs for session...\n')
        smth_mua = session.mua.get_smoothed(sigma=0.02) # Get the smoothed mua from the session's mua
        pbe = detect_pbe_epochs(smth_mua)
        pbe.filename = session.filePrefix.with_suffix('.pbe.npy')
        print('Saving pbe results to {}...'.format(pbe.filename))
        pbe.save()
        print('done.\n')
        return pbe
    # sess.pbe = compute_pbe_epochs(sess)
    
    # ## TODO: needs neuropy! Specifically: from neuropy.analyses import Pf1D, Pf2D, perform_compute_placefields, plot_all_placefields
    # @staticmethod
    # def compute_placefields_as_needed(active_session, computation_config=None, active_epoch_placefields1D = None, active_epoch_placefields2D = None, should_force_recompute_placefields=False, should_display_2D_plots=False):
    #     if computation_config is None:
    #         computation_config = PlacefieldComputationParameters(speed_thresh=9, grid_bin=2, smooth=0.5)
    #     active_epoch_placefields1D, active_epoch_placefields2D = perform_compute_placefields(active_session.neurons, active_session.position, computation_config, active_epoch_placefields1D, active_epoch_placefields2D, should_force_recompute_placefields=True)
    #     # Plot the placefields computed and save them out to files:
    #     if should_display_2D_plots:
    #         ax_pf_1D, occupancy_fig, active_pf_2D_figures = plot_all_placefields(active_epoch_placefields1D, active_epoch_placefields2D, active_config.computation_config)
    #     else:
    #         print('skipping 2D placefield plots')
    #     return active_epoch_placefields1D, active_epoch_placefields2D


    # ConcatenationInitializable protocol:
    @classmethod
    def concat(cls, objList: Union[Sequence, np.array]):
        print('!! WARNING: Session.concat(...) is not yet fully implemented, meaning the returned session is not fully valid. Continue with caution.') 
        new_neurons = neurons.Neurons.concat([aSession.neurons for aSession in objList])
        new_position = Position.concat([aSession.position for aSession in objList])
        new_flattened_spiketrains = FlattenedSpiketrains.concat([aSession.flattened_spiketrains for aSession in objList])
        
        # TODO: eegfile, datfile, recinfo, filePrefix should all be checked to ensure they're the same, and if they aren't, should be set to None
        # TODO: probegroup, paradigm should be checked to ensure that they're the same for all, and an exception should occur if they differ
        # TODO: ripple, mua, and maybe others should be concatenated themselves once that functionality is implemented.
        
        return cls(objList[0].config, filePrefix = objList[0].filePrefix, recinfo = objList[0].recinfo,
                 eegfile = objList[0].eegfile, datfile = objList[0].datfile,
                 neurons = new_neurons, probegroup = objList[0].probegroup, position = new_position, paradigm = objList[0].paradigm,
                 ripple = None, mua = None, laps= objList[0].laps, flattened_spiketrains = new_flattened_spiketrains)
    
    
    def filtered_by_laps(self, lap_indices=None):
        """ Returns a copy of this session with all of its members filtered by the laps.
        """
        lap_specific_subsessions, lap_specific_dataframes, lap_spike_indicies, lap_spike_t_seconds = Laps.build_lap_specific_lists(self, include_empty_lists=True)

        if lap_indices is None:
            lap_indices = np.arange(1, len(lap_specific_subsessions)) # all laps by default, but exclude the 0 element since that's the -1 value
            
        print('filtering by laps: {}'.format(lap_indices))
        lap_specific_subsessions = [lap_specific_subsessions[i] for i in lap_indices] # filter by the desired number of laps 
        ## Effectively build the new session using only the lap-specific spiketimes:
        return DataSession.concat(lap_specific_subsessions)
        # raise NotImplementedError
    
# # Helper function that processed the data in a given directory
# def processDataSession(basedir='/Volumes/iNeo/Data/Bapun/Day5TwoNovel'):
#     # sess = DataSession(basedir)
#     curr_args_dict = dict()
#     curr_args_dict['basepath'] = basedir
#     curr_args_dict['session_obj'] = DataSession() # Create an empty session object
#     sess = DataSessionLoader._default_load_bapun_npy_session_folder(curr_args_dict)
#     return sess


# ## Main:
# if __name__ == "__main__":
#     # Now initiate the class
#     # basedir = '/data/Working/Opto/Jackie671/Jackie_placestim_day2/Jackie_TRACK_2020-10-07_11-21-39'  # fill in here
#     basedir = 'R:\data\Bapun\Day5TwoNovel'
#     # basedir = '/Volumes/iNeo/Data/Bapun/Day5TwoNovel'
#     sess = processDataSession(basedir)
#     print(sess.recinfo)
#     sess.epochs.to_dataframe()
#     sess.neurons.get_all_spikes()
#     sess.position.sampling_rate # 60
    
#     pass
