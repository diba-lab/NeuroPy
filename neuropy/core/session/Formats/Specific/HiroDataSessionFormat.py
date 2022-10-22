import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from neuropy.core.epoch import NamedTimerange
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatBaseRegisteredClass
from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec

# For specific load functions:
from neuropy.core import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.load_exported import import_mat_file
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta

from neuropy.analyses.laps import estimation_session_laps # for estimation_session_laps
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs, drop_overlapping # Used for adding laps in KDiba mode


class FallbackRecInfo(object):
    """docstring for FallbackRecInfo."""
    def __init__(self):
        super(FallbackRecInfo, self).__init__()
        self.source_file = None
        self.channel_groups = None
        self.skipped_channels = None
        self.discarded_channels = None
        self.n_channels = None
        self.dat_sampling_rate = 30000.0
        self.eeg_sampling_rate = 1250.0

    

class HiroDataSessionFormatRegisteredClass(DataSessionFormatBaseRegisteredClass):
    """
    
    
    STATUS: Not quite working. Performs initial loading and session creation successfully, but encounters issues when performing even basic computation functions (like Pf2D). Also encounters issues that Rachel's data has with pf_colors missing and that breaking the display function, but that actually might come from Pf2D.

    
    # Example Filesystem Hierarchy:
    📦analysesResults
    ┃...
    ┣ 📂RoyMaze1
    ┃ ┣ 📂ExportedData
    ┃ ┃ ┣ 📜BehavioralPeriodsManualExport.h5
    ┃ ┃ ┣ 📜extrasAnalysis.mat
    ┃ ┃ ┣ 📜positionAnalysis.mat     :: `PhoDibaLab_REM_HiddenMarkovModel\DEVELOPMENT\Pho2021\PhoDibaTest_PositionalAnalysis.m`
    ┃ ┃ ┣ 📜RippleIndexedSpikes.mat
    ┃ ┃ ┣ 📜RippleManualExport.h5
    ┃ ┃ ┣ 📜spikesAnalysis.mat    ::   `PhoDibaLab_REM_HiddenMarkovModel\DEVELOPMENT\Pho2021\PhoDibaConvert_SpikesToPython.m`
 
    From here, a list of known files to load from is determined:
        
    Usage:
    
        from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
        from neuropy.core.session.Formats.Specific.HiroDataSessionFormatDataSessionFormat import HiroDataSessionFormatRegisteredClass
        
        _test_session = HiroDataSessionFormatRegisteredClass.build_session(Path(r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53'))
        _test_session, loaded_file_record_list = HiroDataSessionFormatRegisteredClass.load_session(_test_session)
        _test_session

    """
    _session_class_name = 'hiro'
    _session_default_relative_basedir = r'PhoMatlabDataScripting\ExportedData\RoyMaze1' # Not quite right on this data
    _session_default_basedir = r'R:\rMBP Python Repos 2022-07-07\PhoNeuronGillespie2021CodeRepo\PhoMatlabDataScripting\ExportedData\RoyMaze1' # WINDOWS
    # _session_default_basedir = r'/run/media/halechr/MoverNew/data/KDIBA/gor01/one/2006-6-07_11-26-53'
    _time_variable_name = 't_seconds' # It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format/Hiro data
    
    @classmethod
    def get_known_data_session_type_properties(cls, override_basepath=None):
        """ returns the session_name for this basedir, which determines the files to load. """
        if override_basepath is not None:
            basepath = override_basepath
        else:
            basepath = Path(cls._session_default_basedir)
        return KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: cls.get_session(basedir=a_base_dir)), 
                                basedir=basepath, post_load_functions=None) # post_load_functions=[lambda a_loaded_sess: estimation_session_laps(a_loaded_sess, cls._time_variable_name)]


     # Pyramidal and Lap-Only:
    @classmethod
    def build_filters_track_only_pyramidal(cls, sess, epoch_name_whitelist=None):
        """ TODO: these filter functions are stupid and redundant """
        all_epoch_names = ['track'] # all_epoch_names # ['maze1', 'maze2']
        active_session_filter_configurations = {an_epoch_name:lambda a_sess, epoch_name=an_epoch_name: (a_sess.filtered_by_neuron_type('pyramidal').filtered_by_epoch(a_sess.epochs.get_named_timerange(epoch_name)), a_sess.epochs.get_named_timerange(epoch_name)) for an_epoch_name in all_epoch_names}
        if epoch_name_whitelist is not None:
            # if the whitelist is specified, get only the specified epochs
            active_session_filter_configurations = {name:filter_fn for name, filter_fn in active_session_filter_configurations.items() if name in epoch_name_whitelist}
        return active_session_filter_configurations

    
    @classmethod
    def build_track_only_filter_functions(cls, sess, **kwargs):
        """ filters only include the 'track', not the pre or post. """
        all_epoch_names = ['track'] # all_epoch_names # ['maze1', 'maze2']
        return {an_epoch_name:lambda a_sess, epoch_name=an_epoch_name: (a_sess.filtered_by_epoch(a_sess.epochs.get_named_timerange(epoch_name)), a_sess.epochs.get_named_timerange(epoch_name)) for an_epoch_name in all_epoch_names}


    @classmethod
    def build_default_computation_configs(cls, sess, **kwargs):
        """ _get_computation_configs(curr_kdiba_pipeline.sess) 
            # From Diba:
            # (3.777, 1.043) # for (64, 64) bins
            # (1.874, 0.518) # for (128, 128) bins
        """
        active_session_computation_configs = DataSessionFormatBaseRegisteredClass.build_default_computation_configs(sess, **kwargs)
        ## Non-restricted computation epochs:
        any_lap_specific_epochs = None
        # Track-restricted computation epochs:
        # track_only_specific_epochs = [sess.epochs.get_named_timerange('track')]
        for i in np.arange(len(active_session_computation_configs)):
            # active_session_computation_configs[i].pf_params.computation_epochs = track_only_specific_epochs # add the laps epochs to all of the computation configs.    
            active_session_computation_configs[i].pf_params.computation_epochs = any_lap_specific_epochs
        return active_session_computation_configs
        
    
    @classmethod
    def get_session_name(cls, basedir):
        """ returns the session_name for this basedir, which determines the files to load. """
        return Path(basedir).parts[-1] # session_name = '2006-6-07_11-26-53'

    @classmethod
    def get_session_spec(cls, session_name):
        # .joinpath('ExportedData','positionAnalysis.mat')
        # return SessionFolderSpec(required=[], optional=[]) # No session spec for testing
        return SessionFolderSpec(required=[
        #                                    SessionFileSpec('{}.spikeII.mat', session_name, 'The MATLAB data file containing information about neural spiking activity.', None),
        #                                    SessionFileSpec('{}.position_info.mat', session_name, 'The MATLAB data file containing the recorded animal positions (as generated by optitrack) over time.', None),
        #                                    SessionFileSpec('{}.epochs_info.mat', session_name, 'The MATLAB data file containing the recording epochs. Each epoch is defined as a: (label:str, t_start: float (in seconds), t_end: float (in seconds))', None)]
                                ], optional=[SessionFileSpec('{}.xml', session_name, 'The primary .xml configuration file', cls._load_xml_file)])
        
    @classmethod
    def load_session(cls, session, debug_print=False):
        timestamp_scale_factor = 1.0             
        # active_time_variable_name = 't' # default
        # active_time_variable_name = 't_seconds' # use converted times (into seconds)
        # active_time_variable_name = 't_rel_seconds' # use converted times (into seconds)
        
        # session, loaded_file_record_list = DataSessionFormatBaseRegisteredClass.load_session(session, debug_print=debug_print) # call the super class load_session(...) to load the common things (.recinfo, .filePrefix, .eegfile, .datfile)
        loaded_file_record_list = []
        session = HiroDataSessionFormatRegisteredClass.build_session(basedir=session.basepath)
        session = cls._fallback_recinfo(session.basepath.joinpath(session.name), session)
        
        # remaining_required_filespecs = {k: v for k, v in session.config.resolved_required_filespecs_dict.items() if k not in loaded_file_record_list}
        # if debug_print:
        #     print(f'remaining_required_filespecs: {remaining_required_filespecs}')
                
        # # Try to load from the FileSpecs:
        # for file_path, file_spec in remaining_required_filespecs.items():
        #     if file_spec.session_load_callback is not None:
        #         session = file_spec.session_load_callback(file_path, session)
        #         loaded_file_record_list.append(file_path)

        all_vars = HiroDataSessionFormatRegisteredClass._load_all_mats(parent_path=session.basepath)
        
        ## Adds Session.paradigm (Epochs)
        session_absolute_start_timestamp = all_vars.extras.behavioral_epochs.loc[0, 'start_seconds_absolute'] # 68368.714228
        session.config.absolute_start_timestamp = session_absolute_start_timestamp
        # 'start_seconds' and 'end_seconds' are relative to start
        if 'label' in all_vars.extras.behavioral_epochs.columns:
            epoch_labels = all_vars.extras.behavioral_epochs['label']
        else:
            num_rows = all_vars.extras.behavioral_epochs.shape[0]
            epoch_labels = [f'epoch{i}' for i in np.arange(num_rows)]
        epochs_df = pd.DataFrame({'label':epoch_labels, 'start':all_vars.extras.behavioral_epochs['start_seconds'].to_numpy(),'stop':all_vars.extras.behavioral_epochs['end_seconds'].to_numpy()})
        session.paradigm = Epoch(epochs=epochs_df)


        ## Adds Positions:
        position_sampling_rate_Hz = 1.0 / np.nanmean(np.diff(all_vars.pos.t)) # 1.0/0.03336651239320582 = 29.97016853950917 Hz
        session.config.position_sampling_rate_Hz = position_sampling_rate_Hz
        # session.position = Position(traces=np.vstack((x, y)), computed_traces=np.full([1, num_samples], np.nan), t_start=active_t_start, sampling_rate=position_sampling_rate_Hz)
        t_rel = all_vars.pos.t
        x = all_vars.pos.x
        y = all_vars.pos.y
        session.position = Position.from_separate_arrays(t_rel, x, y)
        # Load or compute linear positions if needed:
        session = cls._default_compute_linear_position_if_needed(session)
        
        
        ## Adds Spikes:
        
        ## spikes_cell_info_out_dict: neuron properties
        flat_cell_ids = all_vars.spikes.spikes_cell_info_out_dict.aclu
        flat_cell_ids = np.array(flat_cell_ids)
        cell_type = NeuronType.from_qclu_series(qclu_Series=all_vars.spikes.spikes_cell_info_out_dict.qclu)
        shank_ids = all_vars.spikes.spikes_cell_info_out_dict.shank
        cluster_ids = all_vars.spikes.spikes_cell_info_out_dict.cluster # NOT USED
        
        _test_neurons_properties_df = pd.DataFrame({'aclu': flat_cell_ids, 'qclu': all_vars.spikes.spikes_cell_info_out_dict.qclu, 'cell_type': cell_type, 'shank': shank_ids, 'cluster': cluster_ids})
        _test_neurons_properties_df[['aclu','qclu','shank','cluster']] = _test_neurons_properties_df[['aclu','qclu','shank','cluster']].astype('int') # convert integer calumns to correct datatype
        ## Spike trains:
        spiketrains = np.array(all_vars.spikes.spike_list, dtype='object')
        # t_stop = np.max(flat_spikes_out_dict[time_variable_name])
        t_stop = session.paradigm.t_stop
        
        # all_vars.spikes.spikes_cell_info_out_dict.speculated_unit_type

        dat_sampling_rate = 30000.0
        lfpSampleRate = 1250.0
        posSampleRate = 29.9700

        session.neurons = Neurons(spiketrains, t_stop, t_start=0,
            sampling_rate=dat_sampling_rate, # session.recinfo.dat_sampling_rate
            neuron_ids=flat_cell_ids,
            neuron_type=cell_type,
            shank_ids=shank_ids,
            extended_neuron_properties_df=_test_neurons_properties_df
        )
        

        ## Load or compute flattened spikes since this format of data has the spikes ordered only by cell_id:
        ## flattened.spikes:
        active_file_suffix = '.flattened.spikes.npy'
        # active_file_suffix = '.new.flattened.spikes.npy'
        found_datafile = FlattenedSpiketrains.from_file(session.filePrefix.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.flattened_spiketrains = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session = cls._default_compute_flattened_spikes(session, spike_timestamp_column_name=cls._time_variable_name) # sets session.flattened_spiketrains
        
            ## Testing: Fixing spike positions
            spikes_df = session.spikes_df

            
            ## TODO:
            # Want either 'shank_ids' or 'shank' to work as the column
            
                
            
            # if np.isin(['x','y'], spikes_df.columns).all():
            #     spikes_df['x_loaded'] = spikes_df['x']
            #     spikes_df['y_loaded'] = spikes_df['y']
            session, spikes_df = cls._default_compute_spike_interpolated_positions_if_needed(session, spikes_df, time_variable_name=cls._time_variable_name, force_recompute=True)  # TODO: we shouldn't need to force-recomputation, but when we don't pass True we're missing the 'speed' column mid computation
            cls._add_missing_spikes_df_columns(spikes_df, session.neurons) # add the missing columns to the dataframe
            session.flattened_spiketrains.filename = session.filePrefix.with_suffix(active_file_suffix) # '.flattened.spikes.npy'
            print('\t Saving computed flattened spiketrains results to {}...'.format(session.flattened_spiketrains.filename), end='')
            session.flattened_spiketrains.save()
            print('\t done.\n')
                    
        # ## Laps:
        # try:
        #     session, laps_df = cls.__default_kdiba_spikeII_load_laps_vars(session, time_variable_name=active_time_variable_name)
        # except Exception as e:
        #     # raise e
        #     print('session.laps could not be loaded from .spikes.mat due to error {}. Computing.'.format(e))
        #     session, spikes_df = cls.__default_kdiba_spikeII_compute_laps_vars(session, spikes_df, active_time_variable_name)
        # else:
        #     # Successful!
        #     print('session.laps loaded successfully!')
        #     pass

        ## TODO: Missing session.probegroup
        # session.probegroup = ProbeGroup.from_file(session.filePrefix.with_suffix(".probegroup.npy"))
        
        # # add the linear_pos to the spikes_df before building the FlattenedSpiketrains object:
        # # spikes_df['lin_pos'] = session.position.linear_pos
        
        # Common Extended properties:
        session = cls._default_extended_postload(session.filePrefix, session)
        session.is_loaded = True # indicate the session is loaded

        return session, loaded_file_record_list
 
    # ---------------------------------------------------------------------------- #
    #                     Extended Computation/Loading Methods                     #
    # ---------------------------------------------------------------------------- #
    
    #######################################################
    ## KDiba Old Format Only Methods:
    ## relies on _load_kamran_spikeII_mat, _default_spikeII_compute_laps_vars, __default_spikeII_compute_neurons, __default_load_kamran_exported_mats, _default_compute_linear_position_if_needed
    
    @staticmethod
    def extract_spike_timeseries(spike_cell):
        return spike_cell[:,1] # Extract only the first column that refers to the data.

    @staticmethod
    def process_positionalAnalysis_data(data):
        t = np.squeeze(data['positionalAnalysis']['track_position']['t'])
        x = np.squeeze(data['positionalAnalysis']['track_position']['x'])
        y = np.squeeze(data['positionalAnalysis']['track_position']['y'])
        speeds = np.squeeze(data['positionalAnalysis']['track_position']['speeds'])
        dt = np.squeeze(data['positionalAnalysis']['displacement']['dt'])
        dx = np.squeeze(data['positionalAnalysis']['displacement']['dx'])
        dy = np.squeeze(data['positionalAnalysis']['displacement']['dy'])
        return t,x,y,speeds,dt,dx,dy

    @classmethod
    def _fallback_recinfo(cls, filepath, session):
        """ called when the .xml-method fails. Implementor can override to provide a valid .recinfo and .filePrefix anyway. """
        dat_sampling_rate = 30000.0
        lfpSampleRate = 1250.0
        posSampleRate = 29.9700

        session.filePrefix = filepath.with_suffix("") # gets the session name (basically) without the .xml extension.
        
        session.recinfo = FallbackRecInfo()
        # session.recinfo = DynamicContainer(**{
        #     "source_file": None,
        #     "channel_groups": None,
        #     "skipped_channels": None,
        #     "discarded_channels": None,
        #     "n_channels": None,
        #     "dat_sampling_rate": 30000.0,
        #     "eeg_sampling_rate": 1250.0,
        # })
        return session
    
    
    @classmethod
    def _load_all_mats(cls, parent_path=r'R:\rMBP Python Repos 2022-07-07\PhoNeuronGillespie2021CodeRepo\PhoMatlabDataScripting\ExportedData\RoyMaze1'):
        """ 
            # RoyMaze1:
            # mat_import_parent_path = Path(r'R:\data\RoyMaze1')
            mat_import_parent_path = Path(r'R:\rMBP Python Repos 2022-07-07\PhoNeuronGillespie2021CodeRepo\PhoMatlabDataScripting\ExportedData\RoyMaze1')
            # mat_import_parent_path = Path(r'C:\Share\data\RoyMaze1') # Old one
            # mat_import_file = r'C:\Share\data\RoyMaze1\ExportedData.mat'
            all_vars = load_position_spikes_extras_mats(parent_path=mat_import_parent_path)
            all_vars
        """
        mat_import_parent_path = Path(parent_path)
        
        # Import the positions
        t,x,y,speeds,dt,dx,dy = cls.perform_import_positions(mat_import_parent_path=mat_import_parent_path)
        pos_vars = DynamicContainer(t=t,x=x,y=y,speeds=speeds,dt=dt,dx=dx,dy=dy)
        
        # Import the spikes: NOTE: Currently only using the 'spike_list' and not 'spike_matrix', 'spike_cells', etc.
        spike_cells, num_cells, spike_list, spike_positions_list, flat_cell_ids, reverse_cellID_idx_lookup_map, spikes_cell_info_out_dict = cls.perform_import_spikes(t, x, y, mat_import_parent_path=mat_import_parent_path)
        spikes_vars = DynamicContainer(spike_cells=spike_cells, num_cells=num_cells, spike_list=spike_list, spike_positions_list=spike_positions_list, flat_cell_ids=flat_cell_ids, reverse_cellID_idx_lookup_map=reverse_cellID_idx_lookup_map, spikes_cell_info_out_dict=DynamicContainer(**spikes_cell_info_out_dict))
        
        behavioral_periods, behavioral_epochs = cls.perform_import_extras(mat_import_parent_path=mat_import_parent_path)
        # behavioral_periods = all_results_data['active_processing/behavioral_periods_table']
        # print('spike_matrix: {}, spike_cells: {}'.format(np.shape(spike_matrix), np.shape(spike_cells)))
        num_periods = np.shape(behavioral_periods)[0]
        print('num_periods: {}'.format(num_periods))
        extras_vars = DynamicContainer(behavioral_periods=behavioral_periods, behavioral_epochs=behavioral_epochs, num_periods=num_periods)
        all_vars = DynamicContainer(pos=pos_vars, spikes=spikes_vars, extras=extras_vars)
        return all_vars
        
        


    
    
    @classmethod
    def _build_spike_positions_list(cls, spike_list, t, x, y):
        """Interpolate the positions that a spike occurred for each spike timestamp

        Args:
            spike_list ([type]): [description]
            t ([type]): [description]
            x ([type]): [description]
            y ([type]): [description]
        """
        # Determine the x and y positions each spike occured for each cell
        num_cells = len(spike_list)
        spike_positions_list = list()
        for cell_id in np.arange(num_cells):
            spike_positions_list.append(np.vstack((np.interp(spike_list[cell_id], t, x), np.interp(spike_list[cell_id], t, y))))
            # spike_positions_list.append(np.hstack(x[spike_list[cell_id]], y[spike_list[cell_id]]))
            # spike_speed = speeds[spike_list[cell_id]]
        return spike_positions_list

    ## TO REMOVE:
    @classmethod
    def _build_cellID_reverse_lookup_map(cls, cell_ids):
        # Allows reverse indexing into the linear imported array using the original cell ID indicies
        flat_cell_ids = [int(cell_id) for cell_id in cell_ids] # ensures integer indexes for IDs
        linear_flitered_ids = np.arange(len(cell_ids))
        return dict(zip(flat_cell_ids, linear_flitered_ids))

    @classmethod
    def perform_import_spikes(cls, t, x, y, mat_import_parent_path=Path(r'C:\Share\data\RoyMaze1'), debug_print=False):
        """ Import the spikes 
        
        The essential properties are:
        
        'spike_cells'
        'shank', 'cluster', 'aclu', 'qclu','speculated_unit_contamination_level','speculated_unit_type'
        
        """
        
        
        # spikes_mat_import_file = mat_import_parent_path.joinpath('spikesTable.mat')
        spikes_mat_import_file = mat_import_parent_path.joinpath('ExportedData', 'spikesAnalysis.mat')
        spikes_data = import_mat_file(mat_import_file=spikes_mat_import_file)
        # print(spikes_data.keys())
        
        try:
            # spike_matrix = spikes_data['spike_matrix']
            spike_cells = spikes_data['spike_cells'][0]
            cell_ids = spikes_data['spike_cells_ids'][:,0].T
            flat_cell_ids = [int(cell_id) for cell_id in cell_ids] 
        except KeyError as e:
            print(f'KeyError: {e}')
            print(f'\t Valid Keys: {list(spikes_data.keys())}')
            
        ## Pho 2022-07-08 Extra fields exported from `C:\Users\pho\repos\PhoDibaLab_REM_HiddenMarkovModel\DEVELOPMENT\NeuroPyExporting2022\PhoNeuroPyConvert_ExportSpikesToPython.m`    
        try:
            mat_variables_to_extract = ['shank', 'cluster', 'aclu', 'qclu','speculated_unit_contamination_level','speculated_unit_type']
            num_mat_variables = len(mat_variables_to_extract)
            spikes_cell_info_out_dict = dict()
            for i in np.arange(num_mat_variables):
                curr_var_name = mat_variables_to_extract[i]
                if curr_var_name == 'speculated_unit_type':
                    ## This is a cellstr (cell array of char-strings):
                    spikes_cell_info_out_dict[curr_var_name] = [str(an_item.item()[0]) for an_item in spikes_data[curr_var_name]] # ['pre_sleep', 'track', 'post_sleep']
                else:
                    # flat_spikes_out_dict[curr_var_name] = spikes_data[curr_var_name][0,0].flatten() # TODO: do we want .squeeze() instead of .flatten()??
                    spikes_cell_info_out_dict[curr_var_name] = spikes_data[curr_var_name].flatten() # TODO: do we want .squeeze() instead of .flatten()?? NO! .flatten() works really well.
                                        
            if debug_print:
                print(f'Successfully loaded the extra vars!')
                print(spikes_cell_info_out_dict)
            
        except KeyError as e:
            print(f'KeyError: {e}.')
            print(r'\t Did you export from Pho 2022-07-08 Extra fields exported from `C:\Users\pho\repos\PhoDibaLab_REM_HiddenMarkovModel\DEVELOPMENT\NeuroPyExporting2022\PhoNeuroPyConvert_ExportSpikesToPython.m`')
            print(f'\t Valid Keys: {list(spikes_data.keys())}')
            spikes_cell_info_out_dict = None
            

        # print('spike_matrix: {}, spike_cells: {}'.format(np.shape(spike_matrix), np.shape(spike_cells)))
        # num_cells = np.shape(spike_matrix)[0]        
        num_cells = len(cell_ids)
        
        # extract_spike_timeseries(spike_cells[8])
        spike_list = [cls.extract_spike_timeseries(spike_cell) for spike_cell in spike_cells]
        # print(spike_list[0])
        
    #     print('np.shape(cell_ids): {}, cell_ids: {}'.format(np.shape(cell_ids), cell_ids))
        # Determine the x and y positions each spike occured for each cell
        spike_positions_list = cls._build_spike_positions_list(spike_list, t, x, y)    
    #     print(np.shape(spike_positions_list[0])) # (2, 9297)
        
        # reverse_cellID_idx_lookup_map: Allows reverse indexing into the linear imported array using the original cell ID indicies
        reverse_cellID_idx_lookup_map = cls._build_cellID_reverse_lookup_map(cell_ids)

        return spike_cells, num_cells, spike_list, spike_positions_list, flat_cell_ids, reverse_cellID_idx_lookup_map, spikes_cell_info_out_dict

    # @classmethod
    # def _load_positionAnalysis_mat_file(cls, filepath, session):
    #     # .recinfo, .filePrefix:
    #     data = import_mat_file(mat_import_file=position_mat_import_file)
        
    #     session.filePrefix = filepath.with_suffix("") # gets the session name (basically) without the .xml extension.
    #     session.recinfo = NeuroscopeIO(filepath)
    #     return session

    @classmethod
    def perform_import_positions(cls, mat_import_parent_path=Path(r'C:\Share\data\RoyMaze1'), debug_print=False):
        position_mat_import_file = mat_import_parent_path.joinpath('ExportedData','positionAnalysis.mat')
        data = import_mat_file(mat_import_file=position_mat_import_file)
        # Get the position data:
        t,x,y,speeds,dt,dx,dy = cls.process_positionalAnalysis_data(data)
        if debug_print:
            print('shapes - t: {}, x: {}, y: {}'.format(np.shape(t), np.shape(x), np.shape(y))) 
        return t,x,y,speeds,dt,dx,dy

    @classmethod
    def perform_import_extras(cls, mat_import_parent_path=Path(r'C:\Share\data\RoyMaze1'), debug_print=False):
        extras_mat_import_file = mat_import_parent_path.joinpath('ExportedData','extrasAnalysis.mat')
        #source_data.behavior.RoyMaze1.list
        all_results_data = import_mat_file(mat_import_file=extras_mat_import_file)
    #   behavioral_periods = all_results_data['behavioral_periods_table']
        behavioral_periods = all_results_data['behavioral_periods']
        behavioral_epochs = all_results_data['behavioral_epochs']
        num_rows = behavioral_epochs.shape[0]
        behavioral_epoch_names = all_results_data.get('behavioral_epoch_names', None)
        if behavioral_epoch_names is None:
            # build artificial epoch names:
            behavioral_epoch_names = [f'epoch{i}' for i in np.arange(num_rows)]
        else:
            if debug_print:
                print(f'raw behavioral_epoch_names: {behavioral_epoch_names}') # [[array(['pre_sleep'], dtype='<U9')], [array(['track'], dtype='<U5')], [array(['post_sleep'], dtype='<U10')]]
            # behavioral_epoch_names = [str(an_item.item()[0]) for an_item in behavioral_epoch_names] # ['pre_sleep', 'track', 'post_sleep']
            behavioral_epoch_names = [str(an_item[0].item()) for an_item in behavioral_epoch_names] # ['pre_sleep', 'track', 'post_sleep']
            
            if debug_print:
                print(f'behavioral_epoch_names: {behavioral_epoch_names}')
            
        ## Convert the loaded dicts to dataframes:
        behavioral_epochs = pd.DataFrame(all_results_data['behavioral_epochs'], columns=['epoch_index','start_seconds_absolute','end_seconds_absolute','start_seconds','end_seconds','duration'])
        behavioral_epochs['label'] = behavioral_epoch_names
        
        
        #['pre_sleep','track','post_sleep']    
        behavioral_periods = pd.DataFrame(all_results_data['behavioral_periods'], columns=['period_index','epoch_start_seconds','epoch_end_seconds','duration','type','behavioral_epoch'])
        return behavioral_periods, behavioral_epochs



    #################################################### OLD

    
    @staticmethod
    def _default_compute_linear_position_if_needed(session, force_recompute=True):
        # this is not general, this is only used for this particular flat kind of file:
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
                # Only re-save after re-computation
                session.position.filename = session.filePrefix.with_suffix(active_file_suffix)
                # print('Saving updated position results to {}...'.format(session.position.filename), end='')
                with ProgressMessagePrinter(session.position.filename, 'Saving', 'updated position results'):
                    session.position.save()
            # print('\t done.\n')
        else:
            print('\t linearized position loaded from file.')
            # return the session with the upadated member variables
        return session
    
    