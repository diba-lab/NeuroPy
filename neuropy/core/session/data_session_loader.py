import numpy as np
import pandas as pd
from pathlib import Path

from pandas.core import base

# Local imports:
## Core:
# from .datawriter import DataWriter
# from .neurons import NeuronType, Neurons, BinnedSpiketrain, Mua
# from .probe import ProbeGroup
# from .position import Position
# from .epoch import Epoch #, NamedTimerange
# from .signal import Signal
# from .laps import Laps
# from .flattened_spiketrains import FlattenedSpiketrains

# from .. import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains

from neuropy.core import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains
from neuropy.core.session.dataSession import DataSession

from neuropy.io import NeuroscopeIO, BinarysignalIO 
# from ...io import NeuroscopeIO, BinarysignalIO # from neuropy.io import NeuroscopeIO, BinarysignalIO


from neuropy.utils.load_exported import import_mat_file
from neuropy.utils.mixins.print_helpers import SimplePrintable, OrderedMeta

class SessionConfig(SimplePrintable, metaclass=OrderedMeta):
    def __init__(self, basepath, session_spec, session_name):
        """[summary]
        Args:
            basepath (pathlib.Path): [description].
            session_spec (SessionFolderSpec): used to load the files
            session_name (str, optional): [description].
        """
        self.basepath = basepath
        self.session_name = session_name
        # Session spec:
        self.session_spec=session_spec
        self.is_resolved, self.resolved_required_files, self.resolved_optional_files = self.session_spec.validate(self.basepath)


class SessionFolderSpec():
    """ Documents the required and optional files for a given session format """
    def __init__(self, required = [], optional = [], additional_validation_requirements=[]) -> None:
        # additiona_validation_requirements: a list of callbacks that are passed the proposed_session_path on self.validate(...) and return True/False. All must return true for validate to succeed.
        self.required_files = required
        self.optional_files = optional
        self.additional_validation_requirements = additional_validation_requirements
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.__dict__};>"


    def resolved_paths(self, proposed_session_path):
        """ Gets whether the proposed_session_path meets the requirements and returns the resolved paths if it can.
            Does not check whether any of the files exist, it just builds the paths
        """
        proposed_session_path = Path(proposed_session_path)
        # build absolute paths from the proposed_session_path and the files
        resolved_required_files = [proposed_session_path.joinpath(a_path) for a_path in self.required_files]
        resolved_optional_files = [proposed_session_path.joinpath(a_path) for a_path in self.optional_files]
        return resolved_required_files, resolved_optional_files
        
    def validate(self, proposed_session_path):
        """Check whether the proposed_session_path meets this folder spec's requirements
        Args:
            proposed_session_path ([Path]): [description]

        Returns:
            [Bool]: [description]
        """
        resolved_required_files, resolved_optional_files = self.resolved_paths(proposed_session_path=proposed_session_path)
            
        meets_spec = False
        if not Path(proposed_session_path).exists():
            meets_spec = False # the path doesn't even exist, it can't be valid
        else:
            # the path exists:
            for a_required_file in resolved_required_files:
                if not a_required_file.exists():
                    print('Required File: {} does not exist.'.format(a_required_file))
                    meets_spec = False
                    break
            for a_required_validation_function in self.additional_validation_requirements:
                if not a_required_validation_function(Path(proposed_session_path)):
                    print('Required additional_validation_requirements[i]({}) returned False'.format(proposed_session_path))
                    meets_spec = False
                    break
            meets_spec = True # otherwise it exists
            
        return True, resolved_required_files, resolved_optional_files
    

# session_name = '2006-6-07_11-26-53'
# SessionFolderSpec(required=['{}.xml'.format(session_name),
#                             '{}.spikeII.mat'.format(session_name), 
#                             '{}.position_info.mat'.format(session_name),
#                             '{}.epochs_info.mat'.format(session_name), 
# ])




class DataSessionLoader:
    """ An extensible class that performs session data loading operations. 
        Data might be loaded into a Session object from many different source formats depending on lab, experimenter, and age of the data.
        Often this data needs to be reverse engineered and translated into the correct format, which is a tedious and time-consuming process.
        This class allows clearly defining and documenting the requirements of a given format once it's been reverse-engineered.
        
        Primary usage methods:
            DataSessionLoader.bapun_data_session(basedir)
            DataSessionLoader.kdiba_old_format_session(basedir)
    """
    # def __init__(self, load_function, load_arguments=dict()):        
    #     self.load_function = load_function
    #     self.load_arguments = load_arguments
        
    # def load(self, updated_load_arguments=None):
    #     if updated_load_arguments is not None:
    #         self.load_arguments = updated_load_arguments
                 
    #     return self.load_function(self.load_arguments)
    
    pix2cm = 287.7698 # constant conversion factor for spikeII and IIdata (KDiba) formats
    
    #######################################################
    ## Public Methods:
    #######################################################
    
    # KDiba Old Format:
    @staticmethod
    def bapun_data_session(basedir):
        def bapun_data_get_session_name(basedir):
            # Find the only .xml file to obtain the session name
            xml_files = sorted(basedir.glob("*.xml"))        
            assert len(xml_files) == 1, "Found more than one .xml file"
            file_prefix = xml_files[0].with_suffix("") # gets the session name (basically) without the .xml extension. (R:\data\Bapun\Day5TwoNovel\RatS-Day5TwoNovel-2020-12-04_07-55-09)   
            file_basename = xml_files[0].stem # file_basename: (RatS-Day5TwoNovel-2020-12-04_07-55-09)
            # print('file_prefix: {}\nfile_basename: {}'.format(file_prefix, file_basename))
            return file_basename # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
        def get_session_obj(config):
            curr_args_dict = dict()
            curr_args_dict['basepath'] = config.basepath
            curr_args_dict['session_obj'] = DataSession(config)
            return DataSessionLoader._default_load_bapun_npy_session_folder(curr_args_dict)
            
        session_name = bapun_data_get_session_name(basedir) # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
        session_spec = SessionFolderSpec(required=[fname.format(session_name) for fname in ['{}.xml','{}.neurons.npy','{}.probegroup.npy','{}.position.npy','{}.paradigm.npy']])
        session_config = SessionConfig(basedir, session_spec=session_spec, session_name=session_name)
        assert session_config.is_resolved, "active_sess_config could not be resolved!"
        return get_session_obj(session_config)
        
    # KDiba Old Format:
    def kdiba_old_format_session(basedir):
        def kdiba_old_format_get_session_name(basedir):
            return Path(basedir).parts[-1]
        def get_session_obj(config):
            curr_args_dict = dict()
            curr_args_dict['basepath'] = config.basepath
            curr_args_dict['session_obj'] = DataSession(config)
            return DataSessionLoader._default_kdiba_flat_spikes_load_session_folder(curr_args_dict)
        session_name = kdiba_old_format_get_session_name(basedir) # session_name = '2006-6-07_11-26-53'
        session_spec = SessionFolderSpec(required=[fname.format(session_name) for fname in ['{}.xml','{}.spikeII.mat','{}.position_info.mat','{}.epochs_info.mat']])
        session_config = SessionConfig(basedir, session_spec=session_spec, session_name=session_name)
        assert session_config.is_resolved, "active_sess_config could not be resolved!"
        return get_session_obj(session_config)
    
    #######################################################
    ## Internal Methods:
    #######################################################
    @staticmethod
    def _default_extended_postload(fp, session):
        # Computes Common Extended properties:
        ## Ripples:
        active_file_suffix = '.ripple.npy'
        found_datafile = Epoch.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.ripple = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.ripple = DataSession.compute_neurons_ripples(session)

        ## MUA:
        active_file_suffix = '.mua.npy'
        found_datafile = Mua.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.mua = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.mua = DataSession.compute_neurons_mua(session)

        ## PBE Epochs:
        active_file_suffix = '.pbe.npy'
        found_datafile = Epoch.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.pbe = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.pbe = DataSession.compute_pbe_epochs(session)
        # return the session with the upadated member variables    
        return session
    
    @staticmethod
    def _default_compute_spike_interpolated_positions_if_needed(session, spikes_df, time_variable_name='t_rel_seconds'):     
        ## Positions:
        active_file_suffix = '.interpolated_spike_positions.npy'
        found_datafile = FlattenedSpiketrains.from_file(session.filePrefix.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('\t Loading success: {}.'.format(active_file_suffix))
            session.flattened_spiketrains = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('\t Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            spikes_df = FlattenedSpiketrains.interpolate_spike_positions(spikes_df, session.position.time, session.position.x, session.position.y, position_linear_pos=session.position.linear_pos, position_speeds=session.position.speed, spike_timestamp_column_name=time_variable_name)
            session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=time_variable_name, t_start=0.0)
            print('\t Saving updated position results to {}...'.format(session.position.filename))
            session.flattened_spiketrains.save()
            print('\t done.\n')
    
        # return the session with the upadated member variables
        return session, spikes_df
    
    
    
    @staticmethod
    def _default_compute_linear_position_if_needed(session):
        # TODO: this is not general, this is only used for this particular flat kind of file:
            # Load or compute linear positions if needed:
        if (not session.position.has_linear_pos):
            # # compute linear positions:
            # print('computing linear positions for all active epochs for session...')
            # # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
            # session.position.computed_traces = np.full([1, session.position.traces.shape[1]], np.nan)
            # # acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, epochLabelName='maze', method='pca')
            # # session.position.computed_traces[0,  acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            # acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, epochLabelName='maze1', method='pca')
            # acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = DataSession.compute_linearized_position(session, epochLabelName='maze2', method='pca')
            # session.position.computed_traces[0,  acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            # session.position.computed_traces[0,  acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
            
            ## Positions:
            active_file_suffix = '.position.npy'
            found_datafile = Position.from_file(session.filePrefix.with_suffix(active_file_suffix))
            if found_datafile is not None:
                print('Loading success: {}.'.format(active_file_suffix))
                session.position = found_datafile
            else:
                # Otherwise load failed, perform the fallback computation
                print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
                session.position = DataSession.compute_linear_position(session)
            
            # session.position.filename = session.filePrefix.with_suffix(".position.npy")
            # print('Saving updated position results to {}...'.format(session.position.filename))
            # session.position.save()
            print('done.\n')
        else:
            print('linearized position loaded from file.')
            # return the session with the upadated member variables
        return session

    #######################################################
    ## Bapun Nupy Format Only Methods:
    @staticmethod
    def __default_compute_bapun_flattened_spikes(session, timestamp_scale_factor=(1/1E4)):
        # def __unpack_variables(active_session):
        #     # Spike variables: num_cells, spike_list, cell_ids, flattened_spikes
        #     num_cells = active_session.neurons.n_neurons
        #     spike_list = active_session.neurons.spiketrains
        #     cell_ids = active_session.neurons.neuron_ids
        #     # flattened_spikes = active_session.neurons.get_flattened_spikes() # get_flattened_spikes(..) returns a FlattenedSpiketrains object

        #     # session.neurons.get_flattened_spikes()
        #     # Gets the flattened spikes, sorted in ascending timestamp for all cells. Returns a FlattenedSpiketrains object
        #     flattened_spike_identities = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_ids[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_id for each spike that belongs to that neuron
        #     flattened_spike_times = np.concatenate(active_session.neurons.spiketrains)
        #     # Get the indicies required to sort the flattened_spike_times
        #     flattened_sort_indicies = np.argsort(flattened_spike_times)
        #     t_start = active_session.neurons.t_start

        #     # reverse_cellID_idx_lookup_map = build_cellID_reverse_lookup_map(cell_ids)
        #     reverse_cellID_idx_lookup_map = active_session.neurons.reverse_cellID_index_map

        #     # Position variables: t, x, y
        #     t = active_session.position.time
        #     x = active_session.position.x
        #     y = active_session.position.y
        #     linear_pos = active_session.position.linear_pos
        #     speeds = active_session.position.speed 

        #     return num_cells, spike_list, cell_ids, flattened_spike_identities, flattened_spike_times, flattened_sort_indicies, t_start, reverse_cellID_idx_lookup_map, t, x, y, linear_pos, speeds

        # num_cells, spike_list, cell_ids, flattened_spike_identities, flattened_spike_times, flattened_sort_indicies, t_start, reverse_cellID_idx_lookup_map, t, x, y, linear_pos, speeds = __unpack_variables(session)

        # # len(t): 2538347
        # # len(speeds): 2538347
        # ## Note, t, x, y, ... other position variables are not the positions per spike, but instead the tracked positions!
        # # num_flattened_spikes: 16318817

        # # flattened_spike_identities = flattened_spike_identities[flattened_sort_indicies],
        # # flattened_spike_times = flattened_spike_times[flattened_sort_indicies],

        # # Determine the x and y positions each spike occured for each cell
        # # spike_positions_list = build_spike_positions_list(session.neurons.spiketrains, t, x, y)
        # num_flattened_spikes = np.size(flattened_spike_times[flattened_sort_indicies])
        # print('len(t): {}'.format(len(t)))
        # print('len(speeds): {}'.format(len(speeds)))
        # print('num_flattened_spikes: {}'.format(num_flattened_spikes))

        # spikes_df = pd.DataFrame({'flat_spike_idx': np.arange(num_flattened_spikes),
        #     't_seconds':flattened_spike_times[flattened_sort_indicies],
        #     'aclu':flattened_spike_identities[flattened_sort_indicies],
        #     'unit_id': np.array([int(reverse_cellID_idx_lookup_map[original_cellID]) for original_cellID in flattened_spike_identities[flattened_sort_indicies]]),
        #     }
        # )
        
        # # flattened_spike_active_unitIdentities = np.array([int(reverse_cellID_idx_lookup_map[original_cellID]) for original_cellID in flattened_spike_identities[flattened_sort_indicies]]) # since flattened_spike_identities[flattened_sort_indicies] is already sorted, don't double sort
        
        # ## TODO: is flattened_spike_active_unitIdentities needed?
        # # spikes_df['active_unitIdentities'] = flattened_spike_active_unitIdentities
        
        # _temp = np.array([int(reverse_cellID_idx_lookup_map[original_cellID]) for original_cellID in session.neurons.neuron_ids])
        # spikes_df['cell_type'] = [session.neurons.neuron_type[_temp[a_spike_unit_id]] for a_spike_unit_id in spikes_df['aclu'].values]
        
        # # Determine the x and y positions each spike occured for each cell
        # print('__default_compute_bapun_flattened_spikes(session): interpolating {} position values over {} spike timepoints. This may take a minute...'.format(len(t), num_flattened_spikes))
        # ## TODO: spike_positions_list is in terms of cell_ids for some reason, maybe it's temporary?
        # num_cells = len(spike_list)
        # spike_positions_list = list()
        # for cell_id in np.arange(num_cells):
        #     spike_positions_list.append(np.vstack((np.interp(spike_list[cell_id], t, x), np.interp(spike_list[cell_id], t, y), np.interp(spike_list[cell_id], t, linear_pos), np.interp(spike_list[cell_id], t, speeds))))

        # # Gets the flattened spikes, sorted in ascending timestamp for all cells.
        # # Build the Active UnitIDs        
        # # reverse_cellID_idx_lookup_map: get the current filtered index for this cell given using reverse_cellID_idx_lookup_map
        # ## Build the flattened spike positions list
        # flattened_spike_positions_list = np.concatenate(tuple(spike_positions_list), axis=1) # needs tuple(...) to conver the list into a tuple, which is the format it expects
        # flattened_spike_positions_list = flattened_spike_positions_list[:, flattened_sort_indicies] # ensure the positions are ordered the same as the other flattened items so they line up
        # print('flattened_spike_positions_list: {}'.format(np.shape(flattened_spike_positions_list))) # (2, 19647)

        # spikes_df['x'] = flattened_spike_positions_list[0, :]
        # spikes_df['y'] = flattened_spike_positions_list[1, :]
        # spikes_df['linear_pos'] = flattened_spike_positions_list[2, :]
        # spikes_df['speed'] = flattened_spike_positions_list[3, :]
        
        # # spikes_df['qclu'] = session.neurons.neuron_type[reverse_cellID_idx_lookup_map[spikes_df['aclu']] ]

        # # flattened_spike_active_unitIdentities = np.array([int(reverse_cellID_idx_lookup_map[original_cellID]) for original_cellID in flattened_spike_identities[flattened_sort_indicies]]) # since flattened_spike_identities[flattened_sort_indicies] 

        # # still needs: 'cell_type', 'lap', and maybe 't' to match the Diba format.
        # # note that the 'x', 'y' columns in the Diba df seem to be bound between 0.0-1.0 but this currently produces values like -50.946354 for x
        # # spikes_df['cell_type'] = NeuronType.from_qclu_series(qclu_Series=spikes_df['qclu'])

        # spikes_df['t'] = spikes_df['t_seconds'] / timestamp_scale_factor
        # # spikes_df['flat_spike_idx'] = np.array(spikes_df.index)
        
        spikes_df = FlattenedSpiketrains.build_spike_dataframe(session)
        session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, t_start=session.neurons.t_start) # FlattenedSpiketrains(spikes_df)
        print('\t Done!')
        
        return session
    
    
    @staticmethod
    def _default_load_bapun_npy_session_folder(args_dict):        
        basepath = args_dict['basepath']
        session = args_dict['session_obj']
        
        basepath = Path(basepath)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        fp = xml_files[0].with_suffix("") # gets the session name (basically) without the .xml extension.
        session.filePrefix = fp
        session.recinfo = NeuroscopeIO(xml_files[0])

        # if session.recinfo.eeg_filename.is_file():
        try:
            session.eegfile = BinarysignalIO(
                session.recinfo.eeg_filename,
                n_channels=session.recinfo.n_channels,
                sampling_rate=session.recinfo.eeg_sampling_rate,
            )
        except ValueError:
            print('session.recinfo.eeg_filename exists ({}) but file cannot be loaded in the appropriate format. Skipping. \n'.format(session.recinfo.eeg_filename))
            session.eegfile = None

        if session.recinfo.dat_filename.is_file():
            session.datfile = BinarysignalIO(
                session.recinfo.dat_filename,
                n_channels=session.recinfo.n_channels,
                sampling_rate=session.recinfo.dat_sampling_rate,
            )
        else:
            session.datfile = None

        session.neurons = Neurons.from_file(fp.with_suffix(".neurons.npy"))
        session.probegroup = ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))
        session.position = Position.from_file(fp.with_suffix(".position.npy"))
        
        # ['.neurons.npy','.probegroup.npy','.position.npy','.paradigm.npy']
        #  [fname.format(session_name) for fname in ['{}.xml','{}.neurons.npy','{}.probegroup.npy','{}.position.npy','{}.paradigm.npy']]
        # session.paradigm = Epoch.from_file(fp.with_suffix(".paradigm.npy")) # "epoch" field of file
        session.paradigm = Epoch.from_file(fp.with_suffix(".paradigm.npy"))
        


        # session = DataSessionLoader.__default_compute_bapun_flattened_spikes(session)
        
        # Load or compute linear positions if needed:        
        if (not session.position.has_linear_pos):
            # compute linear positions:
            print('computing linear positions for all active epochs for session...')
            # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
            session.position.computed_traces = np.full([1, session.position.traces.shape[1]], np.nan)
            acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, 'maze1')
            acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = DataSession.compute_linearized_position(session, 'maze2')
            session.position.computed_traces[0,  acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            session.position.computed_traces[0,  acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
            session.position.filename = session.filePrefix.with_suffix(".position.npy")
            print('Saving updated position results to {}...'.format(session.position.filename))
            session.position.save()
            print('done.\n')
        else:
            print('linearized position loaded from file.')


        ## Load or compute flattened spikes since this format of data has the spikes ordered only by cell_id:
        ## flattened.spikes:
        active_file_suffix = '.flattened.spikes.npy'
        found_datafile = FlattenedSpiketrains.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.flattened_spiketrains = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session = DataSessionLoader.__default_compute_bapun_flattened_spikes(session) # sets session.flattened_spiketrains
            session.flattened_spiketrains.filename = session.filePrefix.with_suffix(active_file_suffix) # '.flattened.spikes.npy'
            print('\t Saving computed flattened spiketrains results to {}...'.format(session.flattened_spiketrains.filename))
            session.flattened_spiketrains.save()
            print('\t done.\n')
        
        
        
        # Common Extended properties:
        session = DataSessionLoader._default_extended_postload(fp, session)

        return session # returns the session when done





    #######################################################
    ## KDiba Old Format Only Methods:
    ## relies on _load_kamran_spikeII_mat, _default_spikeII_compute_laps_vars, __default_spikeII_compute_neurons, __default_load_kamran_exported_mats, _default_compute_linear_position_if_needed
    @staticmethod
    def _default_kdiba_flat_spikes_load_session_folder(args_dict):
        ## relies on _load_kamran_spikeII_mat, _default_spikeII_compute_laps_vars, __default_spikeII_compute_neurons, default_load_kamran_IIdata_mat, _default_compute_linear_position_if_needed
        basepath = args_dict['basepath']
        session = args_dict['session_obj']
        # timestamp_scale_factor = (1/1E6)
        # timestamp_scale_factor = (1/1E4)
        timestamp_scale_factor = 1.0
                     
        # active_time_variable_name = 't' # default
        # active_time_variable_name = 't_seconds' # use converted times (into seconds)
        active_time_variable_name = 't_rel_seconds' # use converted times (into seconds)

        
        basepath = Path(basepath)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        fp = xml_files[0].with_suffix("")
        session.filePrefix = fp
        session.recinfo = NeuroscopeIO(xml_files[0])

        try:
            session.eegfile = BinarysignalIO(
                session.recinfo.eeg_filename,
                n_channels=session.recinfo.n_channels,
                sampling_rate=session.recinfo.eeg_sampling_rate,
            )
        except ValueError:
            print('session.recinfo.eeg_filename exists ({}) but file cannot be loaded in the appropriate format. Skipping. \n'.format(session.recinfo.eeg_filename))
            session.eegfile = None

        if session.recinfo.dat_filename.is_file():
            session.datfile = BinarysignalIO(
                session.recinfo.dat_filename,
                n_channels=session.recinfo.n_channels,
                sampling_rate=session.recinfo.dat_sampling_rate,
            )
        else:
            session.datfile = None
            
        session_name = session.name
        print('\t basepath: {}\n\t session_name: {}'.format(basepath, session_name)) # session_name: 2006-6-08_14-26-15

        # *vt.mat file Position and Epoch:
        # session = DataSessionLoader.default_load_kamran_position_vt_mat(basepath, session_name, timestamp_scale_factor, spikes_df, session)
    
        # IIdata.mat file Position and Epoch:
        session = DataSessionLoader.__default_kdiba_exported_load_mats(basepath, session_name, session, time_variable_name=active_time_variable_name)
        
        ## .spikeII.mat file:
        try:
            spikes_df, flat_spikes_out_dict = DataSessionLoader.__default_kdiba_pho_exported_spikeII_load_mat(session, timestamp_scale_factor=timestamp_scale_factor)
            
        except FileNotFoundError as e:
            print('FileNotFoundError: {}.\n Trying to fall back to original .spikeII.mat file...'.format(e))
            spikes_df, flat_spikes_out_dict = DataSessionLoader.__default_kdiba_spikeII_load_mat(session, timestamp_scale_factor=timestamp_scale_factor)
            
        except Exception as e:
            # print('e: {}.\n Trying to fall back to original .spikeII.mat file...'.format(e))
            raise e
        else:
            pass
        
        
        # Load or compute linear positions if needed:
        try:
            session = DataSessionLoader._default_compute_linear_position_if_needed(session)
        except Exception as e:
            # raise e
            print('session.position linear positions could not be computed due to error {}. Skipping.'.format(e))
            session.position.computed_traces = np.full([1, session.position.traces.shape[1]], np.nan)
        else:
            # Successful!
            print('session.position linear positions computed!')
            pass
        
        
        ## Testing: Fixing spike positions
        spikes_df['x_loaded'] = spikes_df['x']
        spikes_df['y_loaded'] = spikes_df['y']
        session, spikes_df = DataSessionLoader._default_compute_spike_interpolated_positions_if_needed(session, spikes_df, time_variable_name=active_time_variable_name)
        # spikes_df = FlattenedSpiketrains.interpolate_spike_positions(spikes_df, session.position.time, session.position.x, session.position.y, position_linear_pos=session.position.linear_pos, position_speeds=session.position.speed, spike_timestamp_column_name=active_time_variable_name)
        
        ## Laps:
        try:
            session, laps_df = DataSessionLoader.__default_kdiba_spikeII_load_laps_vars(session, time_variable_name=active_time_variable_name)
        except Exception as e:
            # raise e
            print('session.laps could not be loaded from .spikes.mat due to error {}. Computing.'.format(e))
            session, spikes_df = DataSessionLoader.__default_kdiba_spikeII_compute_laps_vars(session, spikes_df, active_time_variable_name)
        else:
            # Successful!
            print('session.laps loaded successfully!')
            pass

        ## Neurons (by Cell):
        session = DataSessionLoader.__default_kdiba_spikeII_compute_neurons(session, spikes_df, flat_spikes_out_dict, active_time_variable_name)
        session.probegroup = ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))
                        
       
        
        # add the linear_pos to the spikes_df before building the FlattenedSpiketrains object:
        # spikes_df['linear_pos'] = session.position.linear_pos

        # add the flat spikes to the session so they don't have to be recomputed:
        session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=active_time_variable_name)
        
        # Common Extended properties:
        # session = DataSessionLoader.default_extended_postload(fp, session)
        session.is_loaded = True # indicate the session is loaded
        return session # returns the session when done

    @staticmethod
    def __default_kdiba_position_vt_load_mat(basepath, session_name, timestamp_scale_factor, spikes_df, session):
        # Loads a *vt.mat file that contains position and epoch information for the session
        session_position_mat_file_path = Path(basepath).joinpath('{}vt.mat'.format(session_name))
        # session.position = Position.from_vt_mat_file(position_mat_file_path=session_position_mat_file_path)
        position_mat_file = import_mat_file(mat_import_file=session_position_mat_file_path)
        tt = position_mat_file['tt'] # 1, 63192
        xx = position_mat_file['xx'] # 10 x 63192
        yy = position_mat_file['yy'] # 10 x 63192
        tt = tt.flatten()
        # tt_rel = tt - tt[0] # relative to start of position file timestamps
        # timestamps_conversion_factor = 1e6
        # timestamps_conversion_factor = 1e4
        # timestamps_conversion_factor = 1.0
        t = tt * timestamp_scale_factor  # (63192,)
        # t_rel = tt_rel * timestamp_scale_factor  # (63192,)
        position_sampling_rate_Hz = 1.0 / np.mean(np.diff(tt / 1e6)) # In Hz, returns 29.969777
        num_samples = len(t)
        x = xx[0,:].flatten() # (63192,)
        y = yy[0,:].flatten() # (63192,)
        # active_t_start = t[0] # absolute t_start
        # active_t_start = 0.0 # relative t_start
        active_t_start = (spikes_df.t.loc[spikes_df.x.first_valid_index()] * timestamp_scale_factor) # actual start time in seconds
        session.position = Position(traces=np.vstack((x, y)), computed_traces=np.full([1, num_samples], np.nan), t_start=active_t_start, sampling_rate=position_sampling_rate_Hz)
        
        # Range of the maze epoch (where position is valid):
        # t_maze_start = spikes_df.t.loc[spikes_df.x.first_valid_index()] # 1048
        # t_maze_end = spikes_df.t.loc[spikes_df.x.last_valid_index()] # 68159707

        t_maze_start = spikes_df.t.loc[spikes_df.x.first_valid_index()] * timestamp_scale_factor # 1048
        t_maze_end = spikes_df.t.loc[spikes_df.x.last_valid_index()] * timestamp_scale_factor # 68159707

        # Note needs to be absolute start/stop times: 
        # t_maze_start = session.position.t_start # 1048
        # t_maze_end = session.position.t_stop # 68159707 68,159,707
        
        # spikes_df.t.min() # 88
        # spikes_df.t.max() # 68624338
        epochs_df = pd.DataFrame({'start':[0.0, t_maze_start, t_maze_end],'stop':[t_maze_start, t_maze_end, session.neurons.t_stop],'label':['pre','maze','post']})
        session.paradigm = Epoch(epochs=epochs_df)  # "epoch" field 
        
        # return the session with the upadated member variables
        return session

    @staticmethod
    def __default_kdiba_exported_load_mats(basepath, session_name, session, time_variable_name='t_seconds'):
        """ Loads the *.epochs_info.mat & *.position_info.mat files that are exported by Pho Hale's 2021-11-28 Matlab script
            Adds the Epoch and Position information to the session, and returns the updated Session object
        """
        # Loads a IIdata.mat file that contains position and epoch information for the session
                
        # parent_dir = Path(basepath).parent() # the directory above the individual session folder
        # session_all_dataII_mat_file_path = Path(parent_dir).joinpath('IIdata.mat') # get the IIdata.mat in the parent directory
        # position_all_dataII_mat_file = import_mat_file(mat_import_file=session_all_dataII_mat_file_path)        
        
        ## Epoch Data is loaded first so we can define timestamps relative to the absolute start timestamp
        session_epochs_mat_file_path = Path(basepath).joinpath('{}.epochs_info.mat'.format(session_name))
        epochs_mat_file = import_mat_file(mat_import_file=session_epochs_mat_file_path)
        # ['epoch_data','microseconds_to_seconds_conversion_factor']
        epoch_data_array = epochs_mat_file['epoch_data'] # 
        n_epochs = np.shape(epoch_data_array)[0]
        
        session_absolute_start_timestamp = epoch_data_array[0,0].item()
        session.config.absolute_start_timestamp = epoch_data_array[0,0].item()


        if time_variable_name == 't_rel_seconds':
            epoch_data_array_rel = epoch_data_array - session_absolute_start_timestamp # convert to relative by subtracting the first timestamp
            epochs_df_rel = pd.DataFrame({'start':[epoch_data_array_rel[0,0].item(), epoch_data_array_rel[0,1].item()],'stop':[epoch_data_array_rel[1,0].item(), epoch_data_array_rel[1,1].item()],'label':['maze1','maze2']}) # Use the epochs starting at session_absolute_start_timestamp (meaning the first epoch starts at 0.0
            session.paradigm = Epoch(epochs=epochs_df_rel)
        elif time_variable_name == 't_seconds':
            epochs_df = pd.DataFrame({'start':[epoch_data_array[0,0].item(), epoch_data_array[0,1].item()],'stop':[epoch_data_array[1,0].item(), epoch_data_array[1,1].item()],'label':['maze1','maze2']})
            session.paradigm = Epoch(epochs=epochs_df)            
        else:
            raise ValueError
        
        ## Position Data loaded and zeroed to the same session_absolute_start_timestamp, which starts before the first timestamp in 't':
        session_position_mat_file_path = Path(basepath).joinpath('{}.position_info.mat'.format(session_name))
        position_mat_file = import_mat_file(mat_import_file=session_position_mat_file_path)
        # ['microseconds_to_seconds_conversion_factor','samplingRate', 'timestamps', 'x', 'y']
        t = position_mat_file['timestamps'].squeeze() # 1, 63192        
        
        x = position_mat_file['x'].squeeze() # 10 x 63192
        y = position_mat_file['y'].squeeze() # 10 x 63192
        position_sampling_rate_Hz = position_mat_file['samplingRate'].item() # In Hz, returns 29.969777
        microseconds_to_seconds_conversion_factor = position_mat_file['microseconds_to_seconds_conversion_factor'].item()
        num_samples = len(t)
        
        if time_variable_name == 't_rel_seconds':
            t_rel = position_mat_file['timestamps_rel'].squeeze()
            # t_rel = t - t[0] # relative to start of position file timestamps
            # t_rel = t - session_absolute_start_timestamp # relative to absolute start of the first epoch
            active_t_start = t_rel[0] # absolute to first epoch t_start
        elif time_variable_name == 't_seconds':
            # active_t_start = t_rel[0] # absolute to first epoch t_start         
            active_t_start = t[0] # absolute t_start
            # active_t_start = 0.0 # relative t_start
            # active_t_start = (spikes_df.t.loc[spikes_df.x.first_valid_index()] * timestamp_scale_factor) # actual start time in seconds
        else:
            raise ValueError
        
        session.position = Position(traces=np.vstack((x, y)), computed_traces=np.full([1, num_samples], np.nan), t_start=active_t_start, sampling_rate=position_sampling_rate_Hz)
        
        ## Extra files:
        
        
        # return the session with the upadated member variables
        return session
    
    @staticmethod
    def __default_kdiba_pho_exported_spikeII_load_mat(sess, timestamp_scale_factor=1):
        spike_mat_file = Path(sess.basepath).joinpath('{}.spikes.mat'.format(sess.session_name))
        if not spike_mat_file.is_file():
            print('ERROR: file {} does not exist!'.format(spike_mat_file))
            raise FileNotFoundError
        flat_spikes_mat_file = import_mat_file(mat_import_file=spike_mat_file)
        flat_spikes_data = flat_spikes_mat_file['spike']
        mat_variables_to_extract = ['t','t_seconds','t_rel_seconds', 'shank', 'cluster', 'aclu', 'qclu','x','y','speed','traj','lap','maze_relative_lap', 'maze_id']
        num_mat_variables = len(mat_variables_to_extract)
        flat_spikes_out_dict = dict()
        for i in np.arange(num_mat_variables):
            curr_var_name = mat_variables_to_extract[i]
            if curr_var_name == 'cluinfo':
                temp = flat_spikes_data[curr_var_name] # a Nx4 array
                temp = [tuple(temp[j,:]) for j in np.arange(np.shape(temp)[0])]
                flat_spikes_out_dict[curr_var_name] = temp
            else:
                # flat_spikes_out_dict[curr_var_name] = flat_spikes_data[curr_var_name][0,0].flatten() # TODO: do we want .squeeze() instead of .flatten()??
                flat_spikes_out_dict[curr_var_name] = flat_spikes_data[curr_var_name].flatten() # TODO: do we want .squeeze() instead of .flatten()??
                
        # print(flat_spikes_out_dict)
        spikes_df = pd.DataFrame(flat_spikes_out_dict) # 1014937 rows × 11 columns
        spikes_df[['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap','maze_relative_lap', 'maze_id']] = spikes_df[['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap','maze_relative_lap', 'maze_id']].astype('int') # convert integer calumns to correct datatype
        
        spikes_df['cell_type'] = NeuronType.from_qclu_series(qclu_Series=spikes_df['qclu'])
        # add times in seconds both to the dict and the spikes_df under a new key:
        # flat_spikes_out_dict['t_seconds'] = flat_spikes_out_dict['t'] * timestamp_scale_factor
        # spikes_df['t_seconds'] = spikes_df['t'] * timestamp_scale_factor
        # spikes_df['qclu']
        spikes_df['flat_spike_idx'] = np.array(spikes_df.index)
        spikes_df[['flat_spike_idx']] = spikes_df[['flat_spike_idx']].astype('int') # convert integer calumns to correct datatype
        return spikes_df, flat_spikes_out_dict 
    
    @staticmethod
    def __default_kdiba_spikeII_load_laps_vars(session, time_variable_name='t_seconds'):
        """ 
            time_variable_name = 't_seconds'
            sess, laps_df = __default_kdiba_spikeII_load_laps_vars(sess, time_variable_name=time_variable_name)
            laps_df
        """
        ## Get laps in/out
        session_laps_mat_file_path = Path(session.basepath).joinpath('{}.laps_info.mat'.format(session.name))
        laps_mat_file = import_mat_file(mat_import_file=session_laps_mat_file_path)
        mat_variables_to_extract = ['lap_id','maze_id','start_spike_index', 'end_spike_index', 'start_t', 'end_t', 'start_t_seconds', 'end_t_seconds', 'duration_seconds']
        num_mat_variables = len(mat_variables_to_extract)
        flat_var_out_dict = dict()
        for i in np.arange(num_mat_variables):
            curr_var_name = mat_variables_to_extract[i]
            flat_var_out_dict[curr_var_name] = laps_mat_file[curr_var_name].flatten() # TODO: do we want .squeeze() instead of .flatten()??
            
        laps_df = Laps.build_dataframe(flat_var_out_dict, time_variable_name=time_variable_name, absolute_start_timestamp=session.config.absolute_start_timestamp)  # 1014937 rows × 11 columns
        session.laps = Laps(laps_df) # new DataFrame-based approach
        
        # session.laps = Laps(laps_df['lap_id'].to_numpy(), laps_df['num_spikes'].to_numpy(), laps_df[['start_spike_index', 'end_spike_index']].to_numpy(), t_variable)
        
        return session, laps_df

    
    @staticmethod
    def __default_kdiba_spikeII_load_mat(sess, timestamp_scale_factor=(1/1E4)):
        spike_mat_file = Path(sess.basepath).joinpath('{}.spikeII.mat'.format(sess.session_name))
        if not spike_mat_file.is_file():
            print('ERROR: file {} does not exist!'.format(spike_mat_file))
            raise FileNotFoundError
        flat_spikes_mat_file = import_mat_file(mat_import_file=spike_mat_file)
        # print('flat_spikes_mat_file.keys(): {}'.format(flat_spikes_mat_file.keys())) # flat_spikes_mat_file.keys(): dict_keys(['__header__', '__version__', '__globals__', 'spike'])
        flat_spikes_data = flat_spikes_mat_file['spike']
        # print("type is: ",type(flat_spikes_data)) # type is:  <class 'numpy.ndarray'>
        # print("dtype is: ", flat_spikes_data.dtype) # dtype is:  [('t', 'O'), ('shank', 'O'), ('cluster', 'O'), ('aclu', 'O'), ('qclu', 'O'), ('cluinfo', 'O'), ('x', 'O'), ('y', 'O'), ('speed', 'O'), ('traj', 'O'), ('lap', 'O'), ('gamma2', 'O'), ('amp2', 'O'), ('ph', 'O'), ('amp', 'O'), ('gamma', 'O'), ('gammaS', 'O'), ('gammaM', 'O'), ('gammaE', 'O'), ('gamma2S', 'O'), ('gamma2M', 'O'), ('gamma2E', 'O'), ('theta', 'O'), ('ripple', 'O')]
        mat_variables_to_extract = ['t', 'shank', 'cluster', 'aclu', 'qclu', 'cluinfo','x','y','speed','traj','lap']
        num_mat_variables = len(mat_variables_to_extract)
        flat_spikes_out_dict = dict()
        for i in np.arange(num_mat_variables):
            curr_var_name = mat_variables_to_extract[i]
            if curr_var_name == 'cluinfo':
                temp = flat_spikes_data[curr_var_name][0,0] # a Nx4 array
                temp = [tuple(temp[j,:]) for j in np.arange(np.shape(temp)[0])]
                flat_spikes_out_dict[curr_var_name] = temp
            else:
                flat_spikes_out_dict[curr_var_name] = flat_spikes_data[curr_var_name][0,0].flatten() # TODO: do we want .squeeze() instead of .flatten()??
        spikes_df = pd.DataFrame(flat_spikes_out_dict) # 1014937 rows × 11 columns
        spikes_df[['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']] = spikes_df[['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']].astype('int') # convert integer calumns to correct datatype
        spikes_df['cell_type'] = NeuronType.from_qclu_series(qclu_Series=spikes_df['qclu'])
        # add times in seconds both to the dict and the spikes_df under a new key:
        flat_spikes_out_dict['t_seconds'] = flat_spikes_out_dict['t'] * timestamp_scale_factor
        spikes_df['t_seconds'] = spikes_df['t'] * timestamp_scale_factor
        # spikes_df['qclu']
        spikes_df['flat_spike_idx'] = np.array(spikes_df.index)
        spikes_df[['flat_spike_idx']] = spikes_df[['flat_spike_idx']].astype('int') # convert integer calumns to correct datatype
        return spikes_df, flat_spikes_out_dict 

    @staticmethod
    def __default_kdiba_spikeII_compute_laps_vars(session, spikes_df, time_variable_name='t_seconds'):
        """ 
        time_variable_name: (str) either 't' or 't_seconds', indicates which time variable to return in 'lap_start_stop_time'
        """
        spikes_df = spikes_df.copy() # duplicate spikes dataframe
        # Get only the rows with a lap != -1:
        # spikes_df = spikes_df[(spikes_df.lap != -1)] # 229887 rows × 13 columns
        # neg_one_indicies = np.argwhere((spikes_df.lap != -1))
        spikes_df['maze_relative_lap'] = spikes_df.loc[:, 'lap'] # the old lap is now called the maze-relative lap        
        spikes_df['maze_id'] = np.full_like(spikes_df.lap, np.nan)
        lap_ids = spikes_df.lap.to_numpy()
        
        # neg_one_indicies = np.argwhere(lap_ids == -1)
        
        neg_one_indicies = np.squeeze(np.where(lap_ids == -1))
        
        # spikes_df.laps[spikes_df.laps == -1] = np.Infinity
        # non_neg_one_indicies = np.argwhere(spikes_df.lap.values != -1)
        
        ## Deal with non-monotonically increasing lap numbers (such as when the lab_id is reset between epochs)
        # split_index = np.argwhere(np.logical_and((np.append(np.diff(spikes_df.lap), np.zeros((1,))) < 0), (spikes_df.lap != -1)))[0].item() + 1 # add one to account for the 1 less element after np.
            
        # split_index = np.argwhere(np.logical_and((np.append(np.diff(spikes_df.lap), np.zeros((1,))) < 0), (spikes_df.lap != -1)))[0].item() + 1      
        # split_index = np.argwhere(np.logical_and((np.insert(np.diff(spikes_df.lap), 0, 1) < 0), (spikes_df.lap != -1)))[0].item() + 1      
                    
        # way without removing the -1 entries:
        found_idx = np.argwhere((np.append(np.diff(lap_ids), 0) < 0))  
        # np.where(spikes_df.lap.values[found_idx] == 1)
        second_start_id_idx = np.argwhere(lap_ids[found_idx] == 1)[1]
        split_index = found_idx[second_start_id_idx[0]].item()
        # get the lap_id of the last lap in the pre-split
        pre_split_lap_idx = found_idx[second_start_id_idx[0]-1].item()
        # split_index = np.argwhere(np.diff(spikes_df.lap) < 0)[0].item() + 1 # add one to account for the 1 less element after np.
        max_pre_split_lap_id = lap_ids[pre_split_lap_idx].item()
        
        spikes_df.maze_id[0:split_index] = 1
        spikes_df.maze_id[split_index:] = 2 # maze 2
        spikes_df.maze_id[neg_one_indicies] = np.nan # make sure all the -1 entries are not assigned a maze
        
        lap_ids[split_index:] = lap_ids[split_index:] + max_pre_split_lap_id # adding the last pre_split lap ID means that the first lap starts at max_pre_split_lap_id + 1, the second max_pre_split_lap_id + 2, etc 
        lap_ids[neg_one_indicies] = -1 # re-set any negative 1 indicies from the beginning back to negative 1
        
        # set the lap column of the spikes_df with the updated values:
        spikes_df.lap = lap_ids

        # Group by the lap column:
        laps_only_spikes_df = spikes_df[(spikes_df.lap != -1)].copy()
        lap_grouped_spikes_df = laps_only_spikes_df.groupby(['lap']) #  as_index=False keeps the original index
        laps_first_spike_instances = lap_grouped_spikes_df.first()
        laps_last_spike_instances = lap_grouped_spikes_df.last()

        lap_id = np.array(laps_first_spike_instances.index) # the lap_id (which serves to index the lap), like 1, 2, 3, 4, ...
        laps_spike_counts = np.array(lap_grouped_spikes_df.size().values) # number of spikes in each lap

        # print('lap_number: {}'.format(lap_number))
        # print('laps_spike_counts: {}'.format(laps_spike_counts))
        first_indicies = np.array(laps_first_spike_instances.t.index)
        num_laps = len(first_indicies)

        lap_start_stop_flat_idx = np.empty([num_laps, 2])
        lap_start_stop_flat_idx[:, 0] = np.array(laps_first_spike_instances.flat_spike_idx.values)
        lap_start_stop_flat_idx[:, 1] = np.array(laps_last_spike_instances.flat_spike_idx.values)
        # print('lap_start_stop_flat_idx: {}'.format(lap_start_stop_flat_idx))

        lap_start_stop_time = np.empty([num_laps, 2])
        lap_start_stop_time[:, 0] = np.array(laps_first_spike_instances[time_variable_name].values)
        lap_start_stop_time[:, 1] = np.array(laps_last_spike_instances[time_variable_name].values)
        # print('lap_start_stop_time: {}'.format(lap_start_stop_time))
        
        # Build output Laps object to add to session
        print('setting laps object.')
        session.laps = Laps(lap_id, laps_spike_counts, lap_start_stop_flat_idx, lap_start_stop_time)
        
        # return lap_id, laps_spike_counts, lap_start_stop_flat_idx, lap_start_stop_time
        return session, spikes_df
        
        
        
    @staticmethod
    def __default_kdiba_spikeII_compute_neurons(session, spikes_df, flat_spikes_out_dict, time_variable_name='t_seconds'):
        ## Get unique cell ids to enable grouping flattened results by cell:
        unique_cell_ids = np.unique(flat_spikes_out_dict['aclu'])
        flat_cell_ids = [int(cell_id) for cell_id in unique_cell_ids]
        num_unique_cell_ids = len(flat_cell_ids)
        # print('flat_cell_ids: {}'.format(flat_cell_ids))
        # Group by the aclu (cluster indicator) column
        cell_grouped_spikes_df = spikes_df.groupby(['aclu'])
        spiketrains = list()
        shank_ids = np.zeros([num_unique_cell_ids, ]) # (108,) Array of float64
        cell_quality = np.zeros([num_unique_cell_ids, ]) # (108,) Array of float64
        cell_type = list() # (108,) Array of float64

        for i in np.arange(num_unique_cell_ids):
            curr_cell_id = flat_cell_ids[i] # actual cell ID
            #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
            curr_cell_dataframe = cell_grouped_spikes_df.get_group(curr_cell_id)
            spiketrains.append(curr_cell_dataframe[time_variable_name].to_numpy())
            shank_ids[i] = curr_cell_dataframe['shank'].to_numpy()[0] # get the first shank identifier, which should be the same for all of this curr_cell_id
            cell_quality[i] = curr_cell_dataframe['qclu'].mean() # should be the same for all instances of curr_cell_id, but use mean just to make sure
            cell_type.append(curr_cell_dataframe['cell_type'].to_numpy()[0])

        spiketrains = np.array(spiketrains, dtype='object')
        t_stop = np.max(flat_spikes_out_dict[time_variable_name])
        flat_cell_ids = np.array(flat_cell_ids)
        cell_type = np.array(cell_type)
        session.neurons = Neurons(spiketrains, t_stop, t_start=0,
            sampling_rate=session.recinfo.dat_sampling_rate,
            neuron_ids=flat_cell_ids,
            neuron_type=cell_type,
            shank_ids=shank_ids
        )
        ## Ensure we have the 'unit_id' field, and if not, compute it        
        try:
            test = spikes_df['unit_id']
        except KeyError as e:
            # build the valid key for unit_id:
            spikes_df['unit_id'] = np.array([int(session.neurons.reverse_cellID_index_map[original_cellID]) for original_cellID in spikes_df['aclu'].values])



        return session

