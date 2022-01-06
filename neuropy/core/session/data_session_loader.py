import traceback
import numpy as np
import pandas as pd
from pathlib import Path

from pandas.core import base
import param

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
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta

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


class DataSessionLoaderConfig(param.Parameterized):
    enable_save_cache_to_disk = param.Boolean(default=False, doc=' Whether the final loaded/computed data is re-written out to file on disk at the end of the load command. ')
    enable_load_cached_from_disk = param.Boolean(default=False, doc=' Whether previously cached final loaded/computed data is attempted to be loaded from disk if it is available. Otherwise only the raw data will be loaded, and the rest will be computed.')
    
    active_time_variable_name = param.Selector(default='t_rel_seconds', objects=['t', 't_seconds', 't_rel_seconds'], doc=' The time variable used as the primary timestamps. ')
    # active_time_variable_name = param.ListSelector(default='t_rel_seconds', objects=['t', 't_seconds', 't_rel_seconds'], doc=' The time variable used as the primary timestamps. ')
    # active_time_variable_name = 't' # default
    # active_time_variable_name = 't_seconds' # use converted times (into seconds)
    # active_time_variable_name = 't_rel_seconds' # use converted times (into seconds)
    

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
    def _default_compute_spike_interpolated_positions_if_needed(session, spikes_df, time_variable_name='t_rel_seconds', force_recompute=True):     
        ## Positions:
        active_file_suffix = '.interpolated_spike_positions.npy'
        if not force_recompute:
            found_datafile = FlattenedSpiketrains.from_file(session.filePrefix.with_suffix(active_file_suffix))
        else:
            found_datafile = None
        if found_datafile is not None:
            print('\t Loading success: {}.'.format(active_file_suffix))
            session.flattened_spiketrains = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('\t Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            spikes_df = FlattenedSpiketrains.interpolate_spike_positions(spikes_df, session.position.time, session.position.x, session.position.y, position_linear_pos=session.position.linear_pos, position_speeds=session.position.speed, spike_timestamp_column_name=time_variable_name)
            session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=time_variable_name, t_start=0.0)
            
            session.flattened_spiketrains.filename = session.filePrefix.with_suffix(active_file_suffix)
            # print('\t Saving updated interpolated spike position results to {}...'.format(session.flattened_spiketrains.filename), end='')
            with ProgressMessagePrinter(session.flattened_spiketrains.filename, '\t Saving', 'updated interpolated spike position results'):
                session.flattened_spiketrains.save()
            # print('\t done.\n')
    
        # return the session with the upadated member variables
        return session, spikes_df
    
    
    
    @staticmethod
    def _default_compute_linear_position_if_needed(session, force_recompute=True):
        # TODO: this is not general, this is only used for this particular flat kind of file:
            # Load or compute linear positions if needed:
        if (not session.position.has_linear_pos):
            ## compute linear positions: 
            ## Positions:
            active_file_suffix = '.position.npy'
            if not force_recompute:
                found_datafile = Position.from_file(session.filePrefix.with_suffix(active_file_suffix))
            else:
                found_datafile = None
            if found_datafile is not None:
                print('Loading success: {}.'.format(active_file_suffix))
                session.position = found_datafile
            else:
                # Otherwise load failed, perform the fallback computation
                print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
                session.position = DataSession.compute_linear_position(session)
            
            session.position.filename = session.filePrefix.with_suffix(active_file_suffix)
            # print('Saving updated position results to {}...'.format(session.position.filename), end='')
            with ProgressMessagePrinter(session.position.filename, 'Saving', 'updated position results'):
                session.position.save()
            # print('\t done.\n')
        else:
            print('\t linearized position loaded from file.')
            # return the session with the upadated member variables
        return session

    #######################################################
    ## Bapun Nupy Format Only Methods:
    @staticmethod
    def __default_compute_bapun_flattened_spikes(session, timestamp_scale_factor=(1/1E4)):
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

        session_name = session.name
        print('\t basepath: {}\n\t session_name: {}'.format(basepath, session_name)) # session_name: 2006-6-08_14-26-15
        
        session.neurons = Neurons.from_file(fp.with_suffix(".neurons.npy")) # Loads the Neurons from file if possible
        session.probegroup = ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))
        session.position = Position.from_file(fp.with_suffix(".position.npy"))
        
        # ['.neurons.npy','.probegroup.npy','.position.npy','.paradigm.npy']
        #  [fname.format(session_name) for fname in ['{}.xml','{}.neurons.npy','{}.probegroup.npy','{}.position.npy','{}.paradigm.npy']]
        session.paradigm = Epoch.from_file(fp.with_suffix(".paradigm.npy"))  # "epoch" field of file

        # session = DataSessionLoader.__default_compute_bapun_flattened_spikes(session)
        
        # Load or compute linear positions if needed:        
        if (not session.position.has_linear_pos):
            # compute linear positions:
            print('computing linear positions for all active epochs for session...')
            # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
            session.position.linear_pos = np.full_like(session.position.time, np.nan)
            acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, 'maze1')
            acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = DataSession.compute_linearized_position(session, 'maze2')
            session.position.linear_pos[acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            session.position.linear_pos[acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
            session.position.filename = session.filePrefix.with_suffix(".position.npy")
            # print('Saving updated position results to {}...'.format(session.position.filename))
            with ProgressMessagePrinter(session.position.filename, 'Saving', 'updated position results'):
                session.position.save()
            # print('done.\n')
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
            print('\t Saving computed flattened spiketrains results to {}...'.format(session.flattened_spiketrains.filename), end='')
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
            track = traceback.format_exc()
            print(track)
            raise e
        else:
            pass
        
        # Load or compute linear positions if needed:
        session = DataSessionLoader._default_compute_linear_position_if_needed(session)
        
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
        # spikes_df['lin_pos'] = session.position.linear_pos

        # add the flat spikes to the session so they don't have to be recomputed:
        session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=active_time_variable_name)
        
        # Common Extended properties:
        # session = DataSessionLoader.default_extended_postload(fp, session)
        session.is_loaded = True # indicate the session is loaded
        return session # returns the session when done

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
        
        
        session.config.position_sampling_rate_Hz = position_sampling_rate_Hz
        # session.position = Position(traces=np.vstack((x, y)), computed_traces=np.full([1, num_samples], np.nan), t_start=active_t_start, sampling_rate=position_sampling_rate_Hz)
        session.position = Position.from_separate_arrays(t_rel, x, y)
        
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
        """ Attempts to compute the Laps object from the loaded spikesII spikes, which have a 'lap' column.
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
        # lap_maze_id should give the maze_id for each of the laps. 
        lap_maze_id = np.full_like(lap_id, -1)
        lap_maze_id[0:split_index] = 1 # maze 1
        lap_maze_id[split_index:-1] = 2 # maze 2

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
        
        flat_var_out_dict = {'lap_id':lap_id,'maze_id':lap_maze_id,
                             'start_spike_index':np.array(laps_first_spike_instances.flat_spike_idx.values), 'end_spike_index': np.array(laps_last_spike_instances.flat_spike_idx.values),
                             'start_t':np.array(laps_first_spike_instances['t'].values), 'end_t':np.array(laps_last_spike_instances['t'].values),
                             'start_t_seconds':np.array(laps_first_spike_instances[time_variable_name].values), 'end_t_seconds':np.array(laps_last_spike_instances[time_variable_name].values)
                             }
        laps_df = Laps.build_dataframe(flat_var_out_dict, time_variable_name=time_variable_name, absolute_start_timestamp=session.config.absolute_start_timestamp)
        session.laps = Laps(laps_df) # new DataFrame-based approach
        
        
        # session.laps = Laps(lap_id, laps_spike_counts, lap_start_stop_flat_idx, lap_start_stop_time)
        
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

