from pathlib import Path
from typing import Dict
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec, SessionConfig

# For specific load functions:
from neuropy.core import Mua, Epoch
from neuropy.io import NeuroscopeIO, BinarysignalIO 
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter

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


class DataSessionFormatBaseRegisteredClass(metaclass=DataSessionFormatRegistryHolder):
    """
    Any class that will inherits from DataSessionFormatBaseRegisteredClass will be included
    inside the dict RegistryHolder.REGISTRY, the key being the name of the
    class and the associated value, the class itself.
    
    The user specifies a basepath, which is the path containing a list of files:
    
    📦Day5TwoNovel
     ┣ 📂position
     ┃ ┣ 📜Take 2020-12-04 02.05.58 PM.csv
     ┃ ┣ 📜Take 2020-12-04 02.13.28 PM.csv
     ┃ ┣ 📜Take 2020-12-04 11.11.32 AM.csv
     ┃ ┗ 📜Take 2020-12-04 11.11.32 AM_001.csv
     ┣ 📜phoLoadBapunData.py
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.eeg
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.flattened.spikes.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.flattened.spikes.npy.bak
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.maze1.linear.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.maze2.linear.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.mua.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.neurons.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.nrs
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.paradigm.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.pbe.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.position.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.probegroup.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.ripple.npy
     ┗ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.xml
    
    
    By default it attempts to find the single *.xml file in the root of this basedir, from which it determines the `session_name` as the stem (the part before the extension) of this file:
        basedir: Path('R:\data\Bapun\Day5TwoNovel')
        session_name: 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
    
    From here, a list of known files to load from is determined:
    
    
        
    """
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
    def _default_add_spike_PBEs_if_needed(cls, session):
        updated_spk_df = session.compute_spikes_PBEs()
        return session
    
    @classmethod
    def _default_add_spike_scISIs_if_needed(cls, session):
        with ProgressMessagePrinter('filepath?', 'Computing', 'added spike scISI column'):
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
            session.ripple = DataSession.compute_neurons_ripples(session, save_on_compute=True)

        ## MUA:
        active_file_suffix = '.mua.npy'
        found_datafile = Mua.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.mua = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.mua = DataSession.compute_neurons_mua(session, save_on_compute=True)

        ## PBE Epochs:
        active_file_suffix = '.pbe.npy'
        found_datafile = Epoch.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.pbe = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.pbe = DataSession.compute_pbe_epochs(session, save_on_compute=True)
            
        
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
