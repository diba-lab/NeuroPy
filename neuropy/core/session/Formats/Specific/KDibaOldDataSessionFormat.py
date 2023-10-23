from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.core.epoch import NamedTimerange
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatBaseRegisteredClass, find_local_session_paths
from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec, ParametersContainer

# For specific load functions:
from neuropy.core import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains
from neuropy.utils.load_exported import import_mat_file
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta

from neuropy.analyses.laps import estimate_session_laps # for estimation_session_laps
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs, drop_overlapping # Used for adding laps in KDiba mode
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.user_annotations import UserAnnotationsManager

class KDibaOldDataSessionFormatRegisteredClass(DataSessionFormatBaseRegisteredClass):
    """
    
    By default it attempts to find the single *.xml file in the root of this basedir, from which it determines the `session_name` as the stem (the part before the extension) of this file:
        basedir: Path('R:\data\KDIBA\gor01\one\2006-6-07_11-26-53')
        session_name: '2006-6-07_11-26-53'
    
    # Example Filesystem Hierarchy:
    📦gor01
    ┣ 📂one
    ┃ ┣ 📂2006-6-07_11-26-53
    ┃ ┃ ┣ 📂bak
    ┃ ┃ ┃ ┣ 📜2006-6-07_11-26-53.pbe.npy
    ┃ ┃ ┃ ┗ 📜2006-6-07_11-26-53.mua.npy
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.eeg
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.epochs_info.mat
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.1
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.10
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.11
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.12
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.2
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.3
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.4
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.5
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.6
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.7
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.8
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.fet.9
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.interpolated_spike_positions.npy     <-OPT-GEN
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.laps_info.mat
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.nrs
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.position.npy
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.position_info.mat
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.1
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.10
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.11
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.12
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.2
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.3
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.4
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.5
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.6
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.7
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.8
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.res.9
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.rpl.evt
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.seq.evt
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.session.mat
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spikeII.mat
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spikes.cellinfo.mat
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spikes.mat
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.1
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.10
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.11
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.12
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.2
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.3
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.4
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.5
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.6
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.7
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.8
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.spk.9
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.swr.evt
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.theta.1
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.whl
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.xml
    ┃ ┃ ┣ 📜2006-6-07_11-26-53IN.5.res
    ┃ ┃ ┣ 📜2006-6-07_11-26-53vt.mat
    ┃ ┃ ┣ 📜Events.Nev
    ┃ ┃ ┣ 📜RippleDatabase.mat
    ┃ ┃ ┣ 📜VT1.Nvt
    ┃ ┃ ┣ 📜data_NeuroScope2.mat
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.ripple.npy
    ┃ ┃ ┣ 📜2006-6-07_11-26-53.mua.npy
    ┃ ┃ ┗ 📜2006-6-07_11-26-53.pbe.npy
    ┃ ┣ 📜IIdata.mat
 
    From here, a list of known files to load from is determined:
        
    Usage:
    
        from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
        from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
        
        _test_session = KDibaOldDataSessionFormatRegisteredClass.build_session(Path(r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53'))
        _test_session, loaded_file_record_list = KDibaOldDataSessionFormatRegisteredClass.load_session(_test_session)
        _test_session

    """
    _session_class_name: str = 'kdiba'
    _session_default_relative_basedir: str = r'data/KDIBA/gor01/one/2006-6-07_11-26-53'
    _session_default_basedir: str = r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53' # WINDOWS
    # _session_default_basedir = r'/run/media/halechr/MoverNew/data/KDIBA/gor01/one/2006-6-07_11-26-53'
    _session_basepath_to_context_parsing_keys: list[str] = ['format_name','animal','exper_name', 'session_name']

    _time_variable_name: str = 't_rel_seconds' # It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format data
    
    ## Create a dictionary of overrides that have been specified manually for a given session:
    # Used in `build_lap_only_short_long_bin_aligned_computation_configs`
    @classmethod
    def get_specific_session_override_dict(cls) -> dict:
        return UserAnnotationsManager.get_hardcoded_specific_session_override_dict()


    @classmethod
    def get_known_data_session_type_properties(cls, override_basepath=None):
        """ returns the session_name for this basedir, which determines the files to load. """
        if override_basepath is not None:
            basepath = override_basepath
        else:
            basepath = Path(cls._session_default_basedir)
        return KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: cls.get_session(basedir=a_base_dir)), 
                                basedir=basepath, post_load_functions=[lambda a_loaded_sess: cls.POSTLOAD_estimate_laps_and_replays(a_loaded_sess)])  # post_load_functions=[lambda a_loaded_sess: estimate_session_laps(a_loaded_sess)]

    

    @classmethod
    def POSTLOAD_estimate_laps_and_replays(cls, sess):
        """ a POSTLOAD function: after loading, estimates the laps and replays objects (replacing those loaded). """
        print(f'POSTLOAD_estimate_laps_and_replays()...')
        
        # 2023-05-16 - Laps conformance function (TODO 2023-05-16 - factor out?)
        # lap_estimation_parameters = DynamicContainer(N=20, should_backup_extant_laps_obj=True) # Passed as arguments to `sess.replace_session_laps_with_estimates(...)`

        lap_estimation_parameters = sess.config.preprocessing_parameters.epoch_estimation_parameters.laps
        assert lap_estimation_parameters is not None

        use_direction_dependent_laps: bool = lap_estimation_parameters.pop('use_direction_dependent_laps', True)
        sess.replace_session_laps_with_estimates(**lap_estimation_parameters, should_plot_laps_2d=False) # , time_variable_name=None
        ## add `use_direction_dependent_laps` back in:
        lap_estimation_parameters.use_direction_dependent_laps = use_direction_dependent_laps

        ## Apply the laps as the limiting computation epochs:
        # computation_config.pf_params.computation_epochs = sess.laps.as_epoch_obj().get_non_overlapping().filtered_by_duration(1.0, 30.0)
        if use_direction_dependent_laps:
            print(f'.POSTLOAD_estimate_laps_and_replays(...): WARN: {use_direction_dependent_laps}')
            # TODO: I think this is okay here.


        # Get the non-lap periods using PortionInterval's complement method:
        non_running_periods = Epoch.from_PortionInterval(sess.laps.as_epoch_obj().to_PortionInterval().complement()) # TODO 2023-05-24- Truncate to session .t_start, .t_stop as currently includes infinity, but it works fine.
        

        # ## TODO 2023-05-19 - FIX SLOPPY PBE HANDLING
        PBE_estimation_parameters = sess.config.preprocessing_parameters.epoch_estimation_parameters.PBEs
        assert PBE_estimation_parameters is not None
        PBE_estimation_parameters.require_intersecting_epoch = non_running_periods # 2023-10-06 - Require PBEs to occur during the non-running periods, REQUIRED BY KAMRAN contrary to my idea of what PBE is.
        
        new_pbe_epochs = sess.compute_pbe_epochs(sess, active_parameters=PBE_estimation_parameters)
        sess.pbe = new_pbe_epochs
        updated_spk_df = sess.compute_spikes_PBEs()

        # 2023-05-16 - Replace loaded replays (which are bad) with estimated ones:
        
        
        
        # num_pre = session.replay.
        replay_estimation_parameters = sess.config.preprocessing_parameters.epoch_estimation_parameters.replays
        assert replay_estimation_parameters is not None
        ## Update the parameters with the session-specific values that couldn't be determined until after the session was loaded:
        replay_estimation_parameters.require_intersecting_epoch = non_running_periods
        replay_estimation_parameters.min_inclusion_fr_active_thresh = 1.0
        replay_estimation_parameters.min_num_unique_aclu_inclusions = 5
        sess.replace_session_replays_with_estimates(**replay_estimation_parameters)
        
        # ### Get both laps and existing replays as PortionIntervals to check for overlaps:
        # replays = sess.replay.epochs.to_PortionInterval()
        # laps = sess.laps.as_epoch_obj().to_PortionInterval() #.epochs.to_PortionInterval()
        # non_lap_replays = Epoch.from_PortionInterval(replays.difference(laps)) ## Exclude anything that occcurs during the laps themselves.
        # sess.replay = non_lap_replays.to_dataframe() # Update the session's replay epochs from those that don't intersect the laps.

        # print(f'len(replays): {len(replays)}, len(laps): {len(laps)}, len(non_lap_replays): {non_lap_replays.n_epochs}')
        

        # TODO 2023-05-22: Write the parameters somewhere:
        replays = sess.replay.epochs.to_PortionInterval()

        ## This is the inverse approach of the new method, which loads the parameters from `sess.config.preprocessing_parameters`
        # sess.config.preprocessing_parameters = DynamicContainer(epoch_estimation_parameters=DynamicContainer.init_from_dict({
        #     'laps': lap_estimation_parameters,
        #     'PBEs': PBE_estimation_parameters,
        #     'replays': replay_estimation_parameters
        # }))

        return sess



    # ==================================================================================================================== #
    # Filters                                                                                                              #
    # ==================================================================================================================== #

    # Pyramidal and Lap-Only:
    @classmethod
    def build_filters_pyramidal_epochs(cls, sess, epoch_name_includelist=None, filter_name_suffix='_PYR'):
        sess.epochs.t_start = 22.26 # exclude the first short period where the animal isn't on the maze yet
        active_session_filter_configurations = {'maze1': lambda x: (x.filtered_by_neuron_type('pyramidal').filtered_by_epoch(x.epochs.get_named_timerange('maze1')), x.epochs.get_named_timerange('maze1'), sess.get_context().adding_context('filter', filter_name=f'{"maze1"}{filter_name_suffix or ""}')),
                        'maze2': lambda x: (x.filtered_by_neuron_type('pyramidal').filtered_by_epoch(x.epochs.get_named_timerange('maze2')), x.epochs.get_named_timerange('maze2'), sess.get_context().adding_context('filter', filter_name=f'{"maze2"}{filter_name_suffix or ""}')),
                        'maze': lambda x: (x.filtered_by_neuron_type('pyramidal').filtered_by_epoch(NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]])), NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]]), sess.get_context().adding_context('filter', filter_name=f'{"maze"}{filter_name_suffix or ""}'))
                                        }
        
        if epoch_name_includelist is not None:
            # if the includelist is specified, get only the specified epochs
            active_session_filter_configurations = {name:filter_fn for name, filter_fn in active_session_filter_configurations.items() if name in epoch_name_includelist}
            
        if filter_name_suffix is not None:
            # if a filter_name_suffix is specified, change the keys of the returned dict to include the suffix
            active_session_filter_configurations = {f'{name}{filter_name_suffix}':filter_fn for name, filter_fn in active_session_filter_configurations.items()} 
            
        return active_session_filter_configurations
    
    
    # Any epoch on the maze, not limited to pyramidal cells, etc
    @classmethod
    def build_filters_any_maze_epochs(cls, sess):
        filter_name_suffix = None
        sess.epochs.t_start = 22.26 # exclude the first short period where the animal isn't on the maze yet
        # active_session_filter_configurations = {'maze1': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze1')), x.epochs.get_named_timerange('maze1')) } # just maze 1
        active_session_filter_configurations = {
                # 'maze1': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze1')), x.epochs.get_named_timerange('maze1')),
                #                                     'maze2': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze2')), x.epochs.get_named_timerange('maze2')),
                                            'maze': lambda x: (x.filtered_by_epoch(NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]])), NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]]), sess.get_context().adding_context('filter', filter_name=f'{"maze"}{filter_name_suffix or ""}'))
        }
        return active_session_filter_configurations


    @classmethod
    def build_default_filter_functions(cls, sess, epoch_name_includelist=None, filter_name_suffix=None, include_global_epoch=True):
        # all_epoch_names = list(sess.epochs.get_unique_labels()) # all_epoch_names # ['maze1', 'maze2']
        # default_filter_functions = DataSessionFormatBaseRegisteredClass.build_default_filter_functions(sess)
        ## TODO: currently hard-coded
        # active_session_filter_configurations = cls.build_pyramidal_epochs_filters(sess)
        # active_session_filter_configurations = cls.build_filters_any_maze_epochs(sess)
        return DataSessionFormatBaseRegisteredClass.build_default_filter_functions(sess, epoch_name_includelist=epoch_name_includelist, filter_name_suffix=filter_name_suffix, include_global_epoch=include_global_epoch)
        
    # ==================================================================================================================== #
    # Computation Configs                                                                                                  #
    # ==================================================================================================================== #
    
    @classmethod
    def build_lap_only_computation_configs(cls, sess, **kwargs):
        """ sets the computation intervals to only be performed on the laps """
        active_session_computation_configs = DataSessionFormatBaseRegisteredClass.build_default_computation_configs(sess, **kwargs)

        ## Lap-restricted computation epochs:
        use_direction_dependent_laps = True # whether to split the laps into left and right directions
        
        # Strangely many of the laps are overlapping. 82-laps in `sess.laps.as_epoch_obj()`, 77 in `sess.laps.as_epoch_obj().get_non_overlapping()`
        lap_specific_epochs = sess.laps.as_epoch_obj().get_non_overlapping().filtered_by_duration(1.0, 30.0) # laps specifically for use in the placefields with non-overlapping, duration, constraints: the lap must be at least 1 second long and at most 30 seconds long
        # Recover the lap information for the included epochs:
        is_epoch_included_after_filtering = np.isin(sess.laps.starts, lap_specific_epochs.starts)
        included_only_laps_dataframe = sess.laps.to_dataframe()[is_epoch_included_after_filtering]
        # Set the extended data properties:
        included_column_names = ['lap_id', 'lap_dir']
        lap_specific_epochs._df[included_column_names] = included_only_laps_dataframe[included_column_names].astype(int)
        ## Get the actual epochs that will be used:
        any_lap_specific_epochs = lap_specific_epochs
        #TODO 2023-06-30 14:34: - [ ] BUG: There is a bug of some sort here because 'lap_dir' and 'lap_id' are np.nan for some of the entries (like 6, 7, etc). Trace this
        is_even_lap = (lap_specific_epochs._df['lap_dir'].to_numpy() == 0)
        is_odd_lap = (lap_specific_epochs._df['lap_dir'].to_numpy() == 1)
        even_lap_specific_epochs = lap_specific_epochs.boolean_indicies_slice(is_even_lap)
        odd_lap_specific_epochs = lap_specific_epochs.boolean_indicies_slice(is_odd_lap)
        # lap_specific_epochs.labels: ['0', '1', ..., '79'] == ['0', ..., f'{len(sess.laps.lap_id)-1}]
        assert even_lap_specific_epochs.n_epochs + odd_lap_specific_epochs.n_epochs <= any_lap_specific_epochs.n_epochs
        if use_direction_dependent_laps:
            desired_computation_epochs = [even_lap_specific_epochs, odd_lap_specific_epochs, any_lap_specific_epochs]
        else:
            desired_computation_epochs = [any_lap_specific_epochs] # no directional laps version
        
        # Lap-restricted computation epochs:
        print(f'\tlen(active_session_computation_configs): {len(active_session_computation_configs)}')
        final_active_session_computation_configs = []
        
        # if len(active_session_computation_configs) < len(desired_computation_epochs):
        # Clone the configs for each epoch
        for a_restricted_lap_epoch in desired_computation_epochs:
            # for each lap to be used as a computation epoch:        
            for i in np.arange(len(active_session_computation_configs)):
                curr_config = deepcopy(active_session_computation_configs[i])
                curr_config.pf_params.computation_epochs = a_restricted_lap_epoch # add the laps epochs to all of the computation configs.
                final_active_session_computation_configs.append(curr_config)
        
        print(f'\tlen(final_active_session_computation_configs): {len(final_active_session_computation_configs)}')
        return final_active_session_computation_configs
    
    @classmethod
    def build_lap_only_short_long_bin_aligned_computation_configs(cls, sess, **kwargs):
        """ 2023-05-16 - sets the computation intervals to only be performed on the laps """
        active_session_computation_configs = DataSessionFormatBaseRegisteredClass.build_default_computation_configs(sess, **kwargs)
        
        # Need one computation config for each lap (even/odd)
        debug_print = kwargs.get('debug_print', False)
        if debug_print:
            print(f'build_lap_only_short_long_bin_aligned_computation_configs(...):')

        ## Lap-restricted computation epochs:
        use_direction_dependent_laps = True # whether to split the laps into left and right directions
        # use_direction_dependent_laps = True # whether to split the laps into left and right directions

        # Strangely many of the laps are overlapping. 82-laps in `sess.laps.as_epoch_obj()`, 77 in `sess.laps.as_epoch_obj().get_non_overlapping()`
        lap_specific_epochs = sess.laps.as_epoch_obj().get_non_overlapping().filtered_by_duration(1.0, 30.0) # laps specifically for use in the placefields with non-overlapping, duration, constraints: the lap must be at least 1 second long and at most 30 seconds long
        # Recover the lap information for the included epochs:
        is_epoch_included_after_filtering = np.isin(sess.laps.starts, lap_specific_epochs.starts)
        included_only_laps_dataframe = sess.laps.to_dataframe()[is_epoch_included_after_filtering]
        # Set the extended data properties:
        included_column_names = ['lap_id', 'lap_dir']
        lap_specific_epochs._df[included_column_names] = included_only_laps_dataframe[included_column_names].astype(int)
        ## Get the actual epochs that will be used:
        any_lap_specific_epochs = lap_specific_epochs
        #TODO 2023-06-30 14:34: - [ ] BUG: There is a bug of some sort here because 'lap_dir' and 'lap_id' are np.nan for some of the entries (like 6, 7, etc). Trace this
        is_even_lap = (lap_specific_epochs._df['lap_dir'].to_numpy() == 0)
        is_odd_lap = (lap_specific_epochs._df['lap_dir'].to_numpy() == 1)
        even_lap_specific_epochs = lap_specific_epochs.boolean_indicies_slice(is_even_lap)
        odd_lap_specific_epochs = lap_specific_epochs.boolean_indicies_slice(is_odd_lap)

        # lap_specific_epochs.labels: ['0', '1', ..., '79'] == ['0', ..., f'{len(sess.laps.lap_id)-1}]
        assert even_lap_specific_epochs.n_epochs + odd_lap_specific_epochs.n_epochs <= any_lap_specific_epochs.n_epochs
        if use_direction_dependent_laps:
            desired_computation_epochs = [even_lap_specific_epochs, odd_lap_specific_epochs, any_lap_specific_epochs]
        else:
            desired_computation_epochs = [any_lap_specific_epochs] # no directional laps version


        ## Get specific grid_bin_bounds overrides from the `cls._specific_session_override_dict`
        override_dict = cls.get_specific_session_override_dict().get(sess.get_context(), {})
        if override_dict.get('grid_bin_bounds', None) is not None:
            grid_bin_bounds = override_dict['grid_bin_bounds']
        else:
            # no overrides present
            raise NotImplementedError
            pos_df = sess.position.to_dataframe().copy()
            if not 'lap' in pos_df.columns:
                pos_df = sess.compute_laps_position_df() # compute the lap column as needed.
            laps_pos_df = pos_df[pos_df.lap.notnull()] # get only the positions that belong to a lap
            laps_only_grid_bin_bounds = PlacefieldComputationParameters.compute_grid_bin_bounds(laps_pos_df.x.to_numpy(), laps_pos_df.y.to_numpy()) # compute the grid_bin_bounds for these positions only during the laps. This means any positions outside of this will be excluded!
            print(f'\tlaps_only_grid_bin_bounds: {laps_only_grid_bin_bounds}')
            grid_bin_bounds = laps_only_grid_bin_bounds
            # ## Determine the grid_bin_bounds from the long session:
            # grid_bin_bounds = PlacefieldComputationParameters.compute_grid_bin_bounds(sess.position.x, sess.position.y) # ((22.736279243974774, 261.696733348342), (125.5644705153173, 151.21507349463707))
            # # refined_grid_bin_bounds = ((24.12, 259.80), (130.00, 150.09))
            # DO INTERACTIVE MODE:
            # grid_bin_bounds = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True)
            # print(f'grid_bin_bounds: {grid_bin_bounds}')
            # print(f"Add this to `specific_session_override_dict`:\n\n{curr_active_pipeline.get_session_context().get_initialization_code_string()}:dict(grid_bin_bounds=({(grid_bin_bounds[0], grid_bin_bounds[1]), (grid_bin_bounds[2], grid_bin_bounds[3])})),\n")

        # Lap-restricted computation epochs:
        if debug_print:
            print(f'\tlen(active_session_computation_configs): {len(active_session_computation_configs)}')
        final_active_session_computation_configs = []
        
        # if len(active_session_computation_configs) < len(desired_computation_epochs):
        # Clone the configs for each epoch
        for a_restricted_lap_epoch in desired_computation_epochs:
            # for each lap to be used as a computation epoch:        
            for i in np.arange(len(active_session_computation_configs)):
                curr_config = deepcopy(active_session_computation_configs[i])
                # curr_config.pf_params.time_bin_size = 0.025
                curr_config.pf_params.grid_bin_bounds = grid_bin_bounds # same bounds for all
                curr_config.pf_params.computation_epochs = a_restricted_lap_epoch # add the laps epochs to all of the computation configs.
                final_active_session_computation_configs.append(curr_config)

        if debug_print:    
            print(f'\tlen(final_active_session_computation_configs): {len(final_active_session_computation_configs)}')
        return final_active_session_computation_configs

    
    
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

        # Lap-restricted computation epochs:
        for i in np.arange(len(active_session_computation_configs)):
            active_session_computation_configs[i].pf_params.computation_epochs = any_lap_specific_epochs # add the laps epochs to all of the computation configs.
    
        return active_session_computation_configs
        

    @classmethod
    def build_active_computation_configs(cls, sess, **kwargs):
        return cls.build_lap_only_short_long_bin_aligned_computation_configs(sess, **kwargs)

    # ==================================================================================================================== #
    # Other                                                                                                                #
    # ==================================================================================================================== #
    
    @classmethod
    def get_session_name(cls, basedir):
        """ returns the session_name for this basedir, which determines the files to load. """
        return Path(basedir).parts[-1] # session_name = '2006-6-07_11-26-53'

    @classmethod
    def get_session_spec(cls, session_name) -> SessionFolderSpec:
        return SessionFolderSpec(required=[SessionFileSpec('{}.xml', session_name, 'The primary .xml configuration file', cls._load_xml_file),
                                           SessionFileSpec('{}.spikeII.mat', session_name, 'The MATLAB data file containing information about neural spiking activity.', None),
                                           SessionFileSpec('{}.position_info.mat', session_name, 'The MATLAB data file containing the recorded animal positions (as generated by optitrack) over time.', None),
                                           SessionFileSpec('{}.epochs_info.mat', session_name, 'The MATLAB data file containing the recording epochs. Each epoch is defined as a: (label:str, t_start: float (in seconds), t_end: float (in seconds))', None)]
                                )
        
    @classmethod
    def load_session(cls, session, debug_print=False):
        session, loaded_file_record_list = DataSessionFormatBaseRegisteredClass.load_session(session, debug_print=debug_print) # call the super class load_session(...) to load the common things (.recinfo, .filePrefix, .eegfile, .datfile)
        remaining_required_filespecs = {k: v for k, v in session.config.resolved_required_filespecs_dict.items() if k not in loaded_file_record_list}
        if debug_print:
            print(f'remaining_required_filespecs: {remaining_required_filespecs}')
        
        timestamp_scale_factor = 1.0             
        # active_time_variable_name = 't' # default
        # active_time_variable_name = 't_seconds' # use converted times (into seconds)
        active_time_variable_name = 't_rel_seconds' # use converted times (into seconds)
        
        # Try to load from the FileSpecs:
        for file_path, file_spec in remaining_required_filespecs.items():
            if file_spec.session_load_callback is not None:
                session = file_spec.session_load_callback(file_path, session)
                loaded_file_record_list.append(file_path)

        # IIdata.mat file Position and Epoch:
        session = cls.__default_kdiba_exported_load_mats(session.basepath, session.name, session, time_variable_name=active_time_variable_name)
        
        ## .spikeII.mat file: 
        # provides spikes `spikes_df`, `flat_spikes_out_dict`
        try:
            spikes_df, flat_spikes_out_dict = cls.__default_kdiba_pho_exported_spikeII_load_mat(session, timestamp_scale_factor=timestamp_scale_factor)
        except FileNotFoundError as e:
            print(f'FileNotFoundError: {e}.\n Trying to fall back to original .spikeII.mat file...')
            spikes_df, flat_spikes_out_dict = cls.__default_kdiba_spikeII_load_mat(session, timestamp_scale_factor=timestamp_scale_factor)
            
        except Exception as e:
            import traceback
            # print('e: {}.\n Trying to fall back to original .spikeII.mat file...'.format(e))
            track = traceback.format_exc()
            print(track)
            raise e
        else:
            pass
        
        # Load or compute linear positions if needed:
        session = cls._default_compute_linear_position_if_needed(session) # ISSUE 2023-06-05: `lin_pos` is messed up. 
        
        ## Testing: Fixing spike positions
        if np.isin(['x','y'], spikes_df.columns).all():
            spikes_df['x_loaded'] = spikes_df['x']
            spikes_df['y_loaded'] = spikes_df['y']

        session, spikes_df = cls._default_compute_spike_interpolated_positions_if_needed(session, spikes_df, time_variable_name=active_time_variable_name, force_recompute=True) # TODO: we shouldn't need to force-recomputation, but when we don't pass True we're missing the 'speed' column mid computation
        
        ## Laps:
        try:
            session, laps_df = cls.__default_kdiba_spikeII_load_laps_vars(session, time_variable_name=active_time_variable_name)
        except Exception as e:
            # raise e
            print(f'session.laps could not be loaded from .spikes.mat due to error {e}. Computing.')
            session, spikes_df = cls.__default_kdiba_spikeII_compute_laps_vars(session, spikes_df, active_time_variable_name)
        else:
            # Successful!
            print('session.laps loaded successfully!')
            pass
        

        ## Replays:
        try:
            session, replays_df = cls.__default_kdiba_spikeII_load_replays_vars(session, time_variable_name=active_time_variable_name)
        except Exception as e:
            print(f'session.replays could not be loaded from .replay_info.mat due to error {e}. Skipping (will be unavailable)')
        else:
            # Successful!
            print('session.replays loaded successfully!')
            pass
        
        ## Neurons (by Cell):
        # the `session.neurons` Neurons object which it builds from the `spikes_df` and `flat_spikes_out_dict` 
        session = cls.__default_kdiba_spikeII_compute_neurons(session, spikes_df, flat_spikes_out_dict, active_time_variable_name)
        session.probegroup = ProbeGroup.from_file(session.filePrefix.with_suffix(".probegroup.npy"))
        
        # add the flat spikes to the session so they don't have to be recomputed:
        session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=active_time_variable_name)
        
        # Common Extended properties:
        session = cls._default_extended_postload(session.filePrefix, session, force_recompute=True)
        session.is_loaded = True # indicate the session is loaded

        
        return session, loaded_file_record_list
 

    @classmethod
    def build_session_basedirs_dict(cls, global_data_root_parent_path, debug_print=False) -> Dict[IdentifyingContext, Path]:
        """ generates a dict of session_ctx:basedir. Hardcoded for the KDIBA sessions.
        
        Does not check for existance of the basedirs

        History: 2023-09-21 - Extracted from `pyphoplacecellanalysis.General.Batch.runBatch.run_diba_batch`
        
        """
        if not isinstance(global_data_root_parent_path, Path):
            global_data_root_parent_path = Path(global_data_root_parent_path).resolve()
            
        active_data_mode_name = cls._session_class_name

        

        local_session_root_parent_context = IdentifyingContext(format_name=active_data_mode_name) # , animal_name='', configuration_name='one', session_name=self.session_name
        local_session_root_parent_path = global_data_root_parent_path.joinpath('KDIBA')

        animal_names = ['gor01', 'vvp01', 'pin01']
        experiment_names_lists = [['one', 'two'], ['one', 'two'], ['one']] # there is no 'two' for animal 'pin01'
        exclude_lists = [['PhoHelpers', 'Spike3D-Minimal-Test', 'Unused'], [], [], [], ['redundant','showclus','sleep','tmaze']]

        output_session_basedir_dict = {}
        for animal_name, an_experiment_names_list, exclude_list in zip(animal_names, experiment_names_lists, exclude_lists):
            for an_experiment_name in an_experiment_names_list:
                local_session_parent_context = local_session_root_parent_context.adding_context(collision_prefix='animal', animal=animal_name, exper_name=an_experiment_name)
                local_session_parent_path = local_session_root_parent_path.joinpath(local_session_parent_context.animal, local_session_parent_context.exper_name)
                local_session_paths_list, local_session_names_list =  find_local_session_paths(local_session_parent_path, exclude_list=exclude_list)

                if debug_print:
                    print(f'local_session_paths_list: {local_session_paths_list}')
                    print(f'local_session_names_list: {local_session_names_list}')

                ## Build session contexts list:
                local_session_contexts_list = [local_session_parent_context.adding_context(collision_prefix='sess', session_name=a_name) for a_name in local_session_names_list] # [IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>, ..., IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-13_14-42-6')>]

                # {a_ctx:a_path for a_ctx, a_path in zip(local_session_contexts_list, local_session_paths_list if a_ctx)}
                output_session_basedir_dict.update(dict(zip(local_session_contexts_list, local_session_paths_list)))

                # ## Initialize `session_batch_status` with the NOT_STARTED status if it doesn't already have a different status
                # for curr_session_basedir, curr_session_context in zip(local_session_paths_list, local_session_contexts_list):
                # 	# basedir might be different (e.g. on different platforms), but context should be the same
                    

        ## end for
        return output_session_basedir_dict


    # ---------------------------------------------------------------------------- #
    #                     Extended Computation/Loading Methods                     #
    # ---------------------------------------------------------------------------- #
    
    #######################################################
    ## KDiba Old Format Only Methods:
    ## relies on _load_kamran_spikeII_mat, _default_spikeII_compute_laps_vars, __default_spikeII_compute_neurons, __default_load_kamran_exported_mats, _default_compute_linear_position_if_needed
    
    @staticmethod
    def _default_compute_linear_position_if_needed(session, force_recompute=True):
        """
        # -[ ] TODO ISSUE 2023-06-05: `lin_pos` is apparently messed up when loaded from old files. 
        """
        

        # this is not general, this is only used for this particular flat kind of file:
        from neuropy.utils.position_util import RegularizationApproach # for `_default_compute_linear_position_if_needed`

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
                session.position.compute_linearized_position(regularization_approach=RegularizationApproach.RESTORE_X_RANGE, method="isomap", sigma=0)

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
    
    
    
    @classmethod
    def __default_kdiba_exported_load_mats(cls, basepath, session_name, session, time_variable_name='t_seconds'):
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
    
    @classmethod
    def _spikes_df_post_process(cls, spikes_df):
        """ Converts the ['theta', 'ripple', 'ph'] columns into the correct type and renames them to ["is_theta", "is_ripple", "theta_phase_radians"].
            factors out reused code from __default_kdiba_pho_exported_spikeII_load_mat and __default_kdiba_spikeII_load_mat"""
        # Convert and rename the 'theta' and 'ripple' variables which contain a zero or one indicating whether that activity (theta-activity or ripple-activity) is present for each spikes.
        spikes_df[['theta', 'ripple']] = spikes_df[['theta', 'ripple']].astype('bool') # convert boolean calumns to correct datatype
        spikes_df = spikes_df.rename(columns={"theta": "is_theta", "ripple": "is_ripple"})
        # Extract the theta phase in radians:
        spikes_df = spikes_df.rename(columns={"ph": "theta_phase_radians"})
        spikes_df[['theta_phase_radians']] = spikes_df[['theta_phase_radians']].astype('float') 
        return spikes_df

    @classmethod
    def __default_kdiba_pho_exported_spikeII_load_mat(cls, sess, timestamp_scale_factor=1):
        """ loads the spikes from the .mat exported by the script: `IIDataMat_Export_ToPython_2022_08_01.m` """
        spike_mat_file = Path(sess.basepath).joinpath('{}.spikes.mat'.format(sess.session_name))
        if not spike_mat_file.is_file():
            print('ERROR: file {} does not exist!'.format(spike_mat_file))
            raise FileNotFoundError
        flat_spikes_mat_file = import_mat_file(mat_import_file=spike_mat_file)
        flat_spikes_data = flat_spikes_mat_file['spike']
        mat_variables_to_extract = ['t','t_seconds','t_rel_seconds', 'shank', 'cluster', 'aclu', 'qclu','x','y','speed','traj','lap','maze_relative_lap', 'maze_id', 'theta', 'ripple', 'ph']
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

        spikes_df = cls._spikes_df_post_process(spikes_df)

        spikes_df['neuron_type'] = NeuronType.from_qclu_series(qclu_Series=spikes_df['qclu'])
        # add times in seconds both to the dict and the spikes_df under a new key:
        # flat_spikes_out_dict['t_seconds'] = flat_spikes_out_dict['t'] * timestamp_scale_factor
        # spikes_df['t_seconds'] = spikes_df['t'] * timestamp_scale_factor
        # spikes_df['qclu']
        spikes_df['flat_spike_idx'] = np.array(spikes_df.index)
        spikes_df[['flat_spike_idx']] = spikes_df[['flat_spike_idx']].astype('int') # convert integer calumns to correct datatype
        return spikes_df, flat_spikes_out_dict 
    
    @classmethod
    def __default_kdiba_spikeII_load_laps_vars(cls, session, time_variable_name='t_seconds'):
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
            flat_var_out_dict[curr_var_name] = laps_mat_file[curr_var_name].flatten()
            
        laps_df = Laps.build_dataframe(flat_var_out_dict, time_variable_name=time_variable_name, absolute_start_timestamp=session.config.absolute_start_timestamp)  # 1014937 rows × 11 columns
        session.laps = Laps(laps_df) # new DataFrame-based approach
        return session, laps_df
    
    
    
    @classmethod
    def __default_kdiba_spikeII_load_replays_vars(cls, session, time_variable_name='t_seconds'):
        """ Loads the replays exported from the 'IIDataMat_Export_ToPython_2022_08_01.m' matlab script that produces a '*.replay_info.mat' file.
            Adds session.replay to the session.
        
            WARNING: currently ignores time_variable_name, as the replays are always exported in relative seconds.
            
            ERROR: the 'epoch_id' that's imported has NOTHING to do with the two epochs in the maze for some reason. I checked the MATLAB code and it's unclear in the original data why this is or what they stand for. Probably best to ignore this variable entirely, or only import one, or something similar.
            
            time_variable_name = 't_seconds'
            sess, laps_df = __default_kdiba_spikeII_load_laps_vars(sess, time_variable_name=time_variable_name)
            laps_df

            2023-04-26 - Added: ('file_version'), Removed: ('replay_epoch_ids', 'epoch_rel_replay_ids', 'replay_start_stop_rel_sec')
                Translates to removing: ('epoch_id','rel_id') from replay dataframe
        """
        ## Get Replay Events
        session_replay_mat_file_path = Path(session.basepath).joinpath('{}.replay_info.mat'.format(session.name))
        replay_mat_file = import_mat_file(mat_import_file=session_replay_mat_file_path)


        # mat_variables_to_extract = ['nreplayepochs', 'replay_epoch_ids', 'epoch_rel_replay_ids', 'start_t_seconds', 'end_t_seconds', 'replay_r', 'replay_p', 'replay_template_id']
        mat_variables_to_extract = ['nreplayepochs', 'start_t_seconds', 'end_t_seconds', 'replay_r', 'replay_p', 'replay_template_id']
        optional_mat_variables_to_extract = ['file_version']

        num_mat_variables = len(mat_variables_to_extract)
        flat_var_out_dict = dict()
        for i in np.arange(num_mat_variables):
            curr_var_name = mat_variables_to_extract[i]
            flat_var_out_dict[curr_var_name] = replay_mat_file[curr_var_name].flatten()

        replay_df = pd.DataFrame({
                                # 'epoch_id': flat_var_out_dict['replay_epoch_ids'],
                                # 'rel_id': flat_var_out_dict['epoch_rel_replay_ids'],
                                'start': flat_var_out_dict['start_t_seconds'],
                                'end': flat_var_out_dict['end_t_seconds'],
                                'replay_r': flat_var_out_dict['replay_r'],
                                'replay_p': flat_var_out_dict['replay_p'],
                                'template_id': flat_var_out_dict['replay_template_id'],
                                })

        replay_df['flat_replay_idx'] = np.array(replay_df.index) # Add the flat index column
        replay_df[['flat_replay_idx', 'template_id']] = replay_df[['flat_replay_idx', 'template_id']].astype('int') # convert integer calumns to correct datatype
        replay_df['duration'] = replay_df['end'] - replay_df['start']
        session.replay = replay_df # Assign the replay to the session's .replay object
        return session, replay_df
    
    

    @classmethod
    def __default_kdiba_spikeII_load_mat(cls, sess, timestamp_scale_factor=(1/1E4)):
        spike_mat_file = Path(sess.basepath).joinpath('{}.spikeII.mat'.format(sess.session_name))
        if not spike_mat_file.is_file():
            print('ERROR: file {} does not exist!'.format(spike_mat_file))
            raise FileNotFoundError
        flat_spikes_mat_file = import_mat_file(mat_import_file=spike_mat_file)
        # print('flat_spikes_mat_file.keys(): {}'.format(flat_spikes_mat_file.keys())) # flat_spikes_mat_file.keys(): dict_keys(['__header__', '__version__', '__globals__', 'spike'])
        flat_spikes_data = flat_spikes_mat_file['spike']
        # print("type is: ",type(flat_spikes_data)) # type is:  <class 'numpy.ndarray'>
        # print("dtype is: ", flat_spikes_data.dtype) # dtype is:  [('t', 'O'), ('shank', 'O'), ('cluster', 'O'), ('aclu', 'O'), ('qclu', 'O'), ('cluinfo', 'O'), ('x', 'O'), ('y', 'O'), ('speed', 'O'), ('traj', 'O'), ('lap', 'O'), ('gamma2', 'O'), ('amp2', 'O'), ('ph', 'O'), ('amp', 'O'), ('gamma', 'O'), ('gammaS', 'O'), ('gammaM', 'O'), ('gammaE', 'O'), ('gamma2S', 'O'), ('gamma2M', 'O'), ('gamma2E', 'O'), ('theta', 'O'), ('ripple', 'O')]
        mat_variables_to_extract = ['t', 'shank', 'cluster', 'aclu', 'qclu', 'cluinfo','x','y','speed','traj','lap', 'theta', 'ripple', 'ph']
        num_mat_variables = len(mat_variables_to_extract)
        flat_spikes_out_dict = dict()
        for i in np.arange(num_mat_variables):
            curr_var_name = mat_variables_to_extract[i]
            if curr_var_name == 'cluinfo':
                temp = flat_spikes_data[curr_var_name][0,0] # a Nx4 array
                temp = [tuple(temp[j,:]) for j in np.arange(np.shape(temp)[0])]
                flat_spikes_out_dict[curr_var_name] = temp
            else:
                flat_spikes_out_dict[curr_var_name] = flat_spikes_data[curr_var_name][0,0].flatten()
        spikes_df = pd.DataFrame(flat_spikes_out_dict) # 1014937 rows × 11 columns
        spikes_df[['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']] = spikes_df[['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']].astype('int') # convert integer calumns to correct datatype
        spikes_df['neuron_type'] = NeuronType.from_qclu_series(qclu_Series=spikes_df['qclu'])
        # add times in seconds both to the dict and the spikes_df under a new key:
        flat_spikes_out_dict['t_seconds'] = flat_spikes_out_dict['t'] * timestamp_scale_factor
        spikes_df['t_seconds'] = spikes_df['t'] * timestamp_scale_factor
        # spikes_df['qclu']
        spikes_df['flat_spike_idx'] = np.array(spikes_df.index)
        spikes_df[['flat_spike_idx']] = spikes_df[['flat_spike_idx']].astype('int') # convert integer calumns to correct datatype

        spikes_df = cls._spikes_df_post_process(spikes_df)

        return spikes_df, flat_spikes_out_dict 

    @classmethod
    def __default_kdiba_spikeII_compute_laps_vars(cls, session, spikes_df, time_variable_name='t_seconds'):
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
                
    @classmethod
    def __default_kdiba_spikeII_compute_neurons(cls, session, spikes_df, flat_spikes_out_dict, time_variable_name='t_seconds'):
        """ adds the `session.neurons` Neurons object which it builds from the `spikes_df` and `flat_spikes_out_dict` """
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
        neuron_type = list() # (108,) Array of float64

        for i in np.arange(num_unique_cell_ids):
            curr_cell_id = flat_cell_ids[i] # actual cell ID
            #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
            curr_cell_dataframe = cell_grouped_spikes_df.get_group(curr_cell_id)
            spiketrains.append(curr_cell_dataframe[time_variable_name].to_numpy())
            shank_ids[i] = curr_cell_dataframe['shank'].to_numpy()[0] # get the first shank identifier, which should be the same for all of this curr_cell_id
            cell_quality[i] = curr_cell_dataframe['qclu'].mean() # should be the same for all instances of curr_cell_id, but use mean just to make sure
            neuron_type.append(curr_cell_dataframe['neuron_type'].to_numpy()[0])

        spiketrains = np.array(spiketrains, dtype='object')
        t_stop = np.max(flat_spikes_out_dict[time_variable_name])
        flat_cell_ids = np.array(flat_cell_ids)
        neuron_type = np.array(neuron_type)
        session.neurons = Neurons(spiketrains, t_stop, t_start=0,
            sampling_rate=session.recinfo.dat_sampling_rate,
            neuron_ids=flat_cell_ids,
            neuron_type=neuron_type,
            shank_ids=shank_ids
        )
        ## Ensure we have the 'fragile_linear_neuron_IDX' field, and if not, compute it        
        try:
            test = spikes_df['fragile_linear_neuron_IDX']
        except KeyError as e:
            # build the valid key for fragile_linear_neuron_IDX:
            spikes_df['fragile_linear_neuron_IDX'] = np.array([int(session.neurons.reverse_cellID_index_map[original_cellID]) for original_cellID in spikes_df['aclu'].values])

        return session

    @classmethod
    def __default_kdiba_RippleDatabase_load_mat(cls, session):
        """ UNUSED """
        ## Get laps in/out
        session_ripple_mat_file_path = Path(session.basepath).joinpath('{}.RippleDatabase.mat'.format(session.name))
        ripple_mat_file = import_mat_file(mat_import_file=session_ripple_mat_file_path)
        mat_variables_to_extract = ['database_re'] # it's a 993x3 array of timestamps
        num_mat_variables = len(mat_variables_to_extract)
        flat_var_out_dict = dict()
        for i in np.arange(num_mat_variables):
            curr_var_name = mat_variables_to_extract[i]
            flat_var_out_dict[curr_var_name] = ripple_mat_file[curr_var_name].flatten() # TODO: do we want .squeeze() instead of .flatten()??
            
        ripples = np.array(flat_var_out_dict['database_re'])
        print(f'ripples: {np.shape(ripples)}')
        
        ripples_df = pd.DataFrame({'start':ripples[:,0],'peak':ripples[:,1],'stop':ripples[:,2]})
        session.pbe = Epoch(ripples_df)
        
        # session.laps = Laps(laps_df['lap_id'].to_numpy(), laps_df['num_spikes'].to_numpy(), laps_df[['start_spike_index', 'end_spike_index']].to_numpy(), t_variable)
        
        return session, ripples_df