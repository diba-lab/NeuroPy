from pathlib import Path
from typing import Dict
from neuropy.core.flattened_spiketrains import FlattenedSpiketrains
from neuropy.core.position import Position
from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec, SessionConfig

# For specific load functions:
from neuropy.core import Mua, Epoch
from neuropy.io import NeuroscopeIO, BinarysignalIO 
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter
from neuropy.utils.position_util import compute_position_grid_size



class DataSessionFormatRegistryHolder(type):
    """ a metaclass that automatically registers its conformers as a known loadable data session format.     
        
    Usage:
        from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
        
        DataSessionFormatRegistryHolder.get_registry()
        
    """
    REGISTRY: Dict[str, "DataSessionFormatRegistryHolder"] = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        """
            Here the name of the class is used as key but it could be any class
            parameter.
        """
        cls.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)
    
    @classmethod
    def get_registry_data_session_type_class_name_dict(cls):
        """ returns a dict<str, DataSessionFormatBaseRegisteredClass> with keys corresponding to the registered short-names of the data_session_type (like 'kdiba', or 'bapun') and values of DataSessionFormatBaseRegisteredClass. """
        return {a_class._session_class_name:a_class for a_class_name, a_class in cls.get_registry().items() if a_class_name != 'DataSessionFormatBaseRegisteredClass'}
    
    
    
    @classmethod
    def get_registry_known_data_session_type_dict(cls, override_data_basepath=None):
        """ returns a dict<str, KnownDataSessionTypeProperties> with keys corresponding to the registered short-names of the data_session_type (like 'kdiba', or 'bapun') and values of KnownDataSessionTypeProperties. """
        return {a_class._session_class_name:a_class.get_known_data_session_type_properties(override_data_basepath=override_data_basepath) for a_class_name, a_class in cls.get_registry().items() if a_class_name != 'DataSessionFormatBaseRegisteredClass'}
    
    

class DataSessionFormatBaseRegisteredClass(metaclass=DataSessionFormatRegistryHolder):
    """
    Any class that will inherits from DataSessionFormatBaseRegisteredClass will be included
    inside the dict RegistryHolder.REGISTRY, the key being the name of the
    class and the associated value, the class itself.
    
    The user specifies a basepath, which is the path containing a list of files:
    
    📦Day5TwoNovel
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.eeg
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.mua.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.neurons.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.paradigm.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.pbe.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.position.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.probegroup.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.ripple.npy
     ┗ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.xml
    
    
    By default it attempts to find the single *.xml file in the root of this basedir, from which it determines the `session_name` as the stem (the part before the extension) of this file:
        basedir: Path(r'R:\data\Bapun\Day5TwoNovel')
        session_name: 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
    
    From here, a list of known files to load from is determined:
        
    """
    _session_class_name = 'base'
    _session_default_relative_basedir = r'data\KDIBA\gor01\one\2006-6-07_11-26-53'
    _session_default_basedir = r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53'
    
    
    
    @classmethod
    def get_known_data_session_type_properties(cls, override_basepath=None):
        """ returns the KnownDataSessionTypeProperties for this class, which contains information about the process of loading the session."""
        if override_basepath is not None:
            basepath = override_basepath
        else:
            basepath = Path(cls._session_default_basedir)
        return KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: cls.get_session(basedir=a_base_dir)), basedir=basepath)
        
        
    @classmethod
    def build_default_filter_functions(cls, sess):
        all_epoch_names = list(sess.epochs.get_unique_labels()) # all_epoch_names # ['maze1', 'maze2']
        return {an_epoch_name:lambda a_sess, epoch_name=an_epoch_name: (a_sess.filtered_by_epoch(a_sess.epochs.get_named_timerange(epoch_name)), a_sess.epochs.get_named_timerange(epoch_name)) for an_epoch_name in all_epoch_names}
    
        
    @classmethod
    def compute_position_grid_bin_size(cls, x, y, num_bins=(64,64), debug_print=False):
        """ Compute Required Bin size given a desired number of bins in each dimension
        Usage:
            active_grid_bin = compute_position_grid_bin_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64)
        """
        out_grid_bin_size, out_bins, out_bins_infos = compute_position_grid_size(x, y, num_bins=num_bins)
        active_grid_bin = tuple(out_grid_bin_size)
        if debug_print:
            print(f'active_grid_bin: {active_grid_bin}') # (3.776841861770752, 1.043326930905373)
        return active_grid_bin

    @classmethod
    def build_default_computation_configs(cls, sess):
        """ _get_computation_configs(curr_kdiba_pipeline.sess) 
            # From Diba:
            # (3.777, 1.043) # for (64, 64) bins
            # (1.874, 0.518) # for (128, 128) bins
        """
        # active_grid_bin = compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64))
        # active_session_computation_config.computation_epochs = None # set the placefield computation epochs to None, using all epochs.
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(128, 128)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
        return [DynamicContainer(pf_params=PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=cls.compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=1.0, computation_epochs = None)
                     )]
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(3.777, 1.043), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(32, 32)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
        #         PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
        #         PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(128, 128)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
        #        ]
  
  
  
        
    @classmethod
    def get_session(cls, basedir):
        _test_session = cls.build_session(Path(basedir))
        _test_session, loaded_file_record_list = cls.load_session(_test_session)
        return _test_session
    
    
    @classmethod
    def find_session_name_from_sole_xml_file(cls, basedir, debug_print=False):
        """ By default it attempts to find the single *.xml file in the root of this basedir, from which it determines the `session_name` as the stem (the part before the extension) of this file
        Example:
            basedir: Path('R:\data\Bapun\Day5TwoNovel')
            session_name: 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
        """
        # Find the only .xml file to obtain the session name 
        xml_files = sorted(basedir.glob("*.xml"))
        assert len(xml_files) > 0, "Missing required .xml file!"
        assert len(xml_files) == 1, f"Found more than one .xml file. Found files: {xml_files}"
        file_prefix = xml_files[0].with_suffix("") # gets the session name (basically) without the .xml extension. (R:\data\Bapun\Day5TwoNovel\RatS-Day5TwoNovel-2020-12-04_07-55-09)   
        file_basename = xml_files[0].stem # file_basename: (RatS-Day5TwoNovel-2020-12-04_07-55-09)
        if debug_print:
            print('file_prefix: {}\nfile_basename: {}'.format(file_prefix, file_basename))
        return file_basename # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'

    @classmethod
    def get_session_name(cls, basedir):
        """ returns the session_name for this basedir, which determines the files to load. """
        raise NotImplementedError # innheritor must override

    @classmethod
    def get_session_spec(cls, session_name):
        raise NotImplementedError # innheritor must override
    
    @classmethod
    def build_session(cls, basedir):
        basedir = Path(basedir)
        session_name = cls.get_session_name(basedir) # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
        session_spec = cls.get_session_spec(session_name)
        session_config = SessionConfig(basedir, session_spec=session_spec, session_name=session_name)
        assert session_config.is_resolved, "active_sess_config could not be resolved!"
        session_obj = DataSession(session_config)
        return session_obj
    
    @classmethod
    def load_session(cls, session, debug_print=False):
        # .recinfo, .filePrefix, .eegfile, .datfile
        loaded_file_record_list = [] # Handled files list
        
        _test_xml_file_path, _test_xml_file_spec = list(session.config.resolved_required_filespecs_dict.items())[0]
        session = _test_xml_file_spec.session_load_callback(_test_xml_file_path, session)
        loaded_file_record_list.append(_test_xml_file_path)
        
        # Now have access to proper session.recinfo.dat_filename and session.recinfo.eeg_filename:
        session.config.session_spec.optional_files.insert(0, SessionFileSpec('{}'+session.recinfo.dat_filename.suffix, session.recinfo.dat_filename.stem, 'The .dat binary data file', cls._load_datfile))
        session.config.session_spec.optional_files.insert(0, SessionFileSpec('{}'+session.recinfo.eeg_filename.suffix, session.recinfo.eeg_filename.stem, 'The .eeg binary data file', cls._load_eegfile))
        session.config.validate()
        
        _eeg_file_spec = session.config.resolved_optional_filespecs_dict[session.recinfo.eeg_filename]
        session = _eeg_file_spec.session_load_callback(session.recinfo.eeg_filename, session)
        loaded_file_record_list.append(session.recinfo.eeg_filename)
        
        _dat_file_spec = session.config.resolved_optional_filespecs_dict[session.recinfo.dat_filename]
        session = _dat_file_spec.session_load_callback(session.recinfo.dat_filename, session)
        loaded_file_record_list.append(session.recinfo.dat_filename)
        
        return session, loaded_file_record_list
        
    #######################################################
    ## Internal Methods:
    #######################################################
    
    @classmethod
    def _default_compute_spike_interpolated_positions_if_needed(cls, session, spikes_df, time_variable_name='t_rel_seconds', force_recompute=True):     
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
            with ProgressMessagePrinter('spikes_df', 'Computing', 'interpolate_spike_positions columns'):
                spikes_df = FlattenedSpiketrains.interpolate_spike_positions(spikes_df, session.position.time, session.position.x, session.position.y, position_linear_pos=session.position.linear_pos, position_speeds=session.position.speed, spike_timestamp_column_name=time_variable_name)
                session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=time_variable_name, t_start=0.0)
            
            session.flattened_spiketrains.filename = session.filePrefix.with_suffix(active_file_suffix)
            # print('\t Saving updated interpolated spike position results to {}...'.format(session.flattened_spiketrains.filename), end='')
            with ProgressMessagePrinter(session.flattened_spiketrains.filename, '\t Saving', 'updated interpolated spike position results'):
                session.flattened_spiketrains.save()
            # print('\t done.\n')
    
        # return the session with the upadated member variables
        return session, spikes_df
    
    @classmethod
    def _default_add_spike_PBEs_if_needed(cls, session):
        with ProgressMessagePrinter('spikes_df', 'Computing', 'spikes_df PBEs column'):
            updated_spk_df = session.compute_spikes_PBEs()
        return session
    
    @classmethod
    def _default_add_spike_scISIs_if_needed(cls, session):
        with ProgressMessagePrinter('spikes_df', 'Computing', 'added spike scISI column'):
            updated_spk_df = session.spikes_df.spikes.add_same_cell_ISI_column()
        return session
    
    @classmethod
    def _default_extended_postload(cls, fp, session):
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
            try:
                session.ripple = DataSession.compute_neurons_ripples(session, save_on_compute=True)
            except ValueError as e:
                print(f'Computation failed. Skipping .ripple')
                session.ripple = None

        ## MUA:
        active_file_suffix = '.mua.npy'
        found_datafile = Mua.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.mua = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            try:
                session.mua = DataSession.compute_neurons_mua(session, save_on_compute=True)
            except ValueError as e:
                print(f'Computation failed. Skipping .mua')
                session.mua = None
                
        ## PBE Epochs:
        active_file_suffix = '.pbe.npy'
        found_datafile = Epoch.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.pbe = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            try:
                session.pbe = DataSession.compute_pbe_epochs(session, save_on_compute=True)
            except ValueError as e:
                print(f'Computation failed. Skipping .pbe')
                session.pbe = None
        
        # add PBE information to spikes_df from session.pbe
        cls._default_add_spike_PBEs_if_needed(session)
        cls._default_add_spike_scISIs_if_needed(session)
        # return the session with the upadated member variables
        return session
    
    
    @classmethod
    def _load_xml_file(cls, filepath, session):
        # .recinfo, .filePrefix:
        session.filePrefix = filepath.with_suffix("") # gets the session name (basically) without the .xml extension.
        session.recinfo = NeuroscopeIO(filepath)
        return session

    @classmethod
    def _load_eegfile(cls, filepath, session):
        # .eegfile
        try:
            session.eegfile = BinarysignalIO(filepath, n_channels=session.recinfo.n_channels, sampling_rate=session.recinfo.eeg_sampling_rate)
        except ValueError:
            print('session.recinfo.eeg_filename exists ({}) but file cannot be loaded in the appropriate format. Skipping. \n'.format(filepath))
            session.eegfile = None
        return session

    @classmethod
    def _load_datfile(cls, filepath, session):
        # .datfile
        if filepath.is_file():
            session.datfile = BinarysignalIO(filepath, n_channels=session.recinfo.n_channels, sampling_rate=session.recinfo.dat_sampling_rate)
        else:
            session.datfile = None   
        return session
