import re # used in try_extract_date_from_session_name
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from neuropy.core.flattened_spiketrains import FlattenedSpiketrains
from neuropy.core.position import Position
from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec, SessionConfig, ParametersContainer

# For specific load functions:
from neuropy.core import Mua, Epoch
from neuropy.core.epoch import NamedTimerange # required for DataSessionFormatBaseRegisteredClass.build_global_filter_config_function(.)
from neuropy.io import NeuroscopeIO, BinarysignalIO 
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.mixins.dict_representable import override_dict
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter
from neuropy.utils.position_util import compute_position_grid_size
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.mixins.gettable_mixin import KeypathsAccessibleMixin

# ==================================================================================================================== #
# 2022-12-07 - Finding Local Session Paths
# ==================================================================================================================== #

def find_local_session_paths(local_session_parent_path, exclude_list=[], debug_print=False):
	"""Finds the local session paths

	History: From PendingNotebookCode's 2022-12-07 section - "Finding Local Session Paths"

	Args:
		local_session_parent_path (_type_): _description_
		exclude_list (list, optional): _description_. Defaults to [].
		debug_print (bool, optional): _description_. Defaults to True.

	Returns:
		_type_: _description_

	History: extracted from PendingNotebookCode on 2022-12-13 from section "2022-12-07 - Finding Local Session Paths"

	Usage:

		from neuropy.core.session.Formats.BaseDataSessionFormats import find_local_session_paths

		active_data_mode_name = 'kdiba'
		local_session_root_parent_context = IdentifyingContext(format_name=active_data_mode_name)
		local_session_root_parent_path = global_data_root_parent_path.joinpath('KDIBA')

		## Animal `gor01`:
		local_session_parent_context = local_session_root_parent_context.adding_context(collision_prefix='animal', animal='gor01', exper_name='one')
		local_session_parent_path = local_session_root_parent_path.joinpath(local_session_parent_context.animal, local_session_parent_context.exper_name)
		local_session_paths_list, local_session_names_list =  find_local_session_paths(local_session_parent_path, exclude_list=['PhoHelpers', 'Spike3D-Minimal-Test', 'Unused'])

	>>> local_session_names_list: ['2006-6-07_11-26-53', '2006-6-08_14-26-15', '2006-6-09_1-22-43', '2006-6-09_3-23-37', '2006-6-12_15-55-31', '2006-6-13_14-42-6']
		local_session_paths_list: {WindowsPath('W:/Data/KDIBA/gor01/one/2006-6-07_11-26-53'): <SessionBatchProgress.NOT_STARTED: 'NOT_STARTED'>,
			WindowsPath('W:/Data/KDIBA/gor01/one/2006-6-08_14-26-15'): <SessionBatchProgress.NOT_STARTED: 'NOT_STARTED'>,
			WindowsPath('W:/Data/KDIBA/gor01/one/2006-6-09_1-22-43'): <SessionBatchProgress.NOT_STARTED: 'NOT_STARTED'>,
			WindowsPath('W:/Data/KDIBA/gor01/one/2006-6-09_3-23-37'): <SessionBatchProgress.NOT_STARTED: 'NOT_STARTED'>,
			WindowsPath('W:/Data/KDIBA/gor01/one/2006-6-12_15-55-31'): <SessionBatchProgress.NOT_STARTED: 'NOT_STARTED'>,
			WindowsPath('W:/Data/KDIBA/gor01/one/2006-6-13_14-42-6'): <SessionBatchProgress.NOT_STARTED: 'NOT_STARTED'>}
	"""
	try:
		found_local_session_paths_list = [x for x in local_session_parent_path.iterdir() if x.is_dir()]
		local_session_names_list = [a_path.name for a_path in found_local_session_paths_list if a_path.name not in exclude_list]
		local_session_names_list = sorted(local_session_names_list)
		if debug_print:
			print(f'local_session_names_list: {local_session_names_list}')
		local_session_paths_list = [local_session_parent_path.joinpath(a_name).resolve() for a_name in local_session_names_list]
		
	except Exception as e:
		print(f"Error processing path: '{local_session_parent_path}' due to exception: {e}. Skipping...")
		local_session_paths_list = None
		local_session_names_list = None
		
	return local_session_paths_list, local_session_names_list




class DataSessionFormatRegistryHolder(type): # inheriting from type? Is this right?
	""" a metaclass that automatically registers its conformers as a known loadable data session format.
		
	Usage:
		from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
		from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
		from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
		from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat
		from neuropy.core.session.Formats.Specific.HiroDataSessionFormat import HiroDataSessionFormatRegisteredClass

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
	def get_registry_known_data_session_type_dict(cls, override_data_basepath=None, override_parameters_flat_keypaths_dict=None):
		""" returns a dict<str, KnownDataSessionTypeProperties> with keys corresponding to the registered short-names of the data_session_type (like 'kdiba', or 'bapun') and values of KnownDataSessionTypeProperties. """
		return {a_class._session_class_name:a_class.get_known_data_session_type_properties(override_basepath=override_data_basepath, override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict) for a_class_name, a_class in cls.get_registry().items() if a_class_name != 'DataSessionFormatBaseRegisteredClass'}



class DataSessionFormatBaseRegisteredClass(metaclass=DataSessionFormatRegistryHolder):
	"""
	Any class that will inherit from DataSessionFormatBaseRegisteredClass will be included
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
	
	_time_variable_name = None # It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format data

	@classmethod
	def get_session_format_name(cls):
		""" The name of the specific format (e.g. 'bapun', 'kdiba', etc) """
		return cls._session_class_name
	
	@classmethod
	def get_session_name(cls, basedir):
		""" MUST be overriden by implementor to return the session_name for this basedir, which determines the files to load. """
		raise NotImplementedError # innheritor must override

	@classmethod
	def get_session_spec(cls, session_name):
		""" MUST be overriden by implementor to return the a session_spec """
		raise NotImplementedError # innheritor must override
		
	@classmethod
	def build_global_epoch_filter_config_dict(cls, sess, global_epoch_name='maze', first_included_epoch_name=None, last_included_epoch_name=None, filter_name_suffix=None, debug_print=False):
		""" builds the 'global' filter for the entire session that includes by default the times from all other epochs in sess. 
		e.g. builds the 'maze' epoch from ['maze1', 'maze2'] epochs

		Usage:
			global_epoch_filter_fn_dict, global_named_timerange = build_global_epoch_filter_config_dict(sess, global_epoch_name='maze', first_included_epoch_name=None, last_included_epoch_name=None, debug_print=True)
			global_epoch_filter_fn_dict

		"""
		all_epoch_names = list(sess.epochs.get_unique_labels()) # all_epoch_names # ['maze1', 'maze2']
		if global_epoch_name in all_epoch_names:
			global_epoch_name = f"{global_epoch_name}_GLOBAL"
			print(f'WARNING: name collision "{global_epoch_name}" already exists in all_epoch_names: {all_epoch_names}! Using {global_epoch_name} instead.')
		
		if first_included_epoch_name is not None:
			# global_start_end_times[0] = sess.epochs[first_included_epoch_name][0] # 'maze1'
			pass
		else:
			first_included_epoch_name = sess.epochs.get_unique_labels()[0]
			
		if last_included_epoch_name is not None:
			# global_start_end_times[1] = sess.epochs[last_included_epoch_name][1] # 'maze2'
			pass
		else:
			last_included_epoch_name = sess.epochs.get_unique_labels()[-1]
	
		# global_start_end_times = [epochs.t_start, epochs.t_stop]
		global_start_end_times = [sess.epochs[first_included_epoch_name][0], sess.epochs[last_included_epoch_name][1]]
		# global_start_end_times_fn = lambda x: [sess.epochs[first_included_epoch_name][0], sess.epochs[last_included_epoch_name][1]]
		
		global_named_timerange = NamedTimerange(name=global_epoch_name, start_end_times=global_start_end_times)
		# global_epoch_filter_fn = (lambda x: (x.filtered_by_epoch(NamedTimerange(name=global_epoch_name, start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]])), NamedTimerange(name=global_epoch_name, start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]])))
		if debug_print:
			print(f'global_named_timerange: {global_named_timerange}, first_included_epoch_name: {first_included_epoch_name}, last_included_epoch_name: {last_included_epoch_name}')
		global_epoch_filter_fn = (lambda x: (x.filtered_by_epoch(NamedTimerange(name=global_epoch_name, start_end_times=[x.epochs[(first_included_epoch_name or x.epochs.get_unique_labels()[0])][0], x.epochs[(last_included_epoch_name or x.epochs.get_unique_labels()[-1])][1]])), NamedTimerange(name=global_epoch_name, start_end_times=[x.epochs[(first_included_epoch_name or x.epochs.get_unique_labels()[0])][0], x.epochs[(last_included_epoch_name or x.epochs.get_unique_labels()[-1])][1]]), sess.get_context().adding_context('filter', filter_name=f'{global_epoch_name}{filter_name_suffix or ""}')))
		return {global_epoch_name: global_epoch_filter_fn}, global_named_timerange
	
	@classmethod
	def build_default_filter_functions(cls, sess, epoch_name_includelist=None, filter_name_suffix=None, include_global_epoch=True):
		""" OPTIONALLY can be overriden by implementors to provide specific filter functions
		Inputs:
			epoch_name_includelist: an optional list of names to restrict to, must already be valid epochs to filter by. e.g. ['maze1']
			filter_name_suffix: an optional string suffix to be added to the end of each filter_name. An example would be '_PYR'
			include_global_epoch: bool - If True, uses cls.build_global_epoch_filter_config_dict(...) to generate a global epoch that will be included in the filters
		"""
		if epoch_name_includelist is None:
			all_epoch_names = list(sess.epochs.get_unique_labels()) # all_epoch_names # ['maze1', 'maze2']
			epoch_name_includelist = all_epoch_names
			
		if filter_name_suffix is None:
			filter_name_suffix = ''

		if include_global_epoch:
			global_epoch_filter_fn_dict, global_named_timerange = cls.build_global_epoch_filter_config_dict(sess, global_epoch_name='maze', first_included_epoch_name=None, last_included_epoch_name=None, filter_name_suffix=filter_name_suffix, debug_print=False)
		else:
			global_epoch_filter_fn_dict = {} # empty dict

		# returns a session. 
		# epoch_filter_configs_dict = {f'{an_epoch_name}{filter_name_suffix}':lambda a_sess, epoch_name=an_epoch_name: (a_sess.filtered_by_epoch(a_sess.epochs.get_named_timerange(epoch_name)), a_sess.epochs.get_named_timerange(epoch_name), a_sess.get_context().adding_context('filter', filter_name=f'{an_epoch_name}{filter_name_suffix}')) for an_epoch_name in epoch_name_includelist} # epoch_name_includelist: ['maze1', 'maze2']

		# epoch_filter_configs_dict = {}
		# for an_epoch_name in epoch_name_includelist:
		#     new_epoch_name: str = f'{an_epoch_name}{filter_name_suffix}'
		#     a_fn = lambda a_sess, epoch_name=an_epoch_name.copy(): (a_sess.filtered_by_epoch(a_sess.epochs.get_named_timerange(epoch_name)), a_sess.epochs.get_named_timerange(epoch_name), a_sess.get_context().adding_context('filter', filter_name=f'{an_epoch_name}{filter_name_suffix}'))

		epoch_filter_configs_dict = {f'{an_epoch_name}{filter_name_suffix}': lambda a_sess, epoch_name=an_epoch_name: (
			a_sess.filtered_by_epoch(a_sess.epochs.get_named_timerange(epoch_name)),
			a_sess.epochs.get_named_timerange(epoch_name),
			a_sess.get_context().adding_context('filter', filter_name=f'{an_epoch_name}{filter_name_suffix}')
		) for an_epoch_name in epoch_name_includelist}

		# epoch_filter_configs_dict = {f'{an_epoch_name}{filter_name_suffix}':lambda a_sess, epoch_name=an_epoch_name: (a_sess.filtered_by_epoch(a_sess.epochs.get_named_timerange(epoch_name)), a_sess.epochs.get_named_timerange(epoch_name), a_sess.get_context().adding_context('filter', filter_name=f'{an_epoch_name}{filter_name_suffix}')) for an_epoch_name in epoch_name_includelist} # ['maze1', 'maze2']
		final_configs_dict = dict(epoch_filter_configs_dict, **global_epoch_filter_fn_dict)
		return  final_configs_dict


	@classmethod
	def build_default_preprocessing_parameters(cls, **kwargs) -> ParametersContainer:
		""" builds the pre-processing parameters. Could get session_spec, basedir, or other info from the caller but usually not a session itself because this is used to build the config prior to the session loading. 
		
		Used in: ['cls.build_session']
		"""
		override_parameters_flat_keypaths_dict = kwargs.pop('override_parameters_flat_keypaths_dict', {}) or {} # ` or {}` part handles None values
		override_parameters_nested_dicts = KeypathsAccessibleMixin.keypath_dict_to_nested_dict(override_parameters_flat_keypaths_dict)
		override_preprocessing = override_parameters_nested_dicts.get('preprocessing', {})
		
		default_lap_estimation_parameters = DynamicContainer(N=20, should_backup_extant_laps_obj=True, use_direction_dependent_laps=True).override(override_preprocessing.get('laps', {}))  # Passed as arguments to `sess.replace_session_laps_with_estimates(...)`
		default_PBE_estimation_parameters = DynamicContainer(sigma=0.030, thresh=(0, 1.5), min_dur=0.030, merge_dur=0.100, max_dur=0.6).override(override_preprocessing.get('PBEs', {})) # 2023-10-05 Kamran's imposed Parameters, wants to remove the effect of the max_dur which was previously at 0.300  
		default_replay_estimation_parameters = DynamicContainer(require_intersecting_epoch=None, min_epoch_included_duration=0.06, max_epoch_included_duration=0.6, maximum_speed_thresh=None, min_inclusion_fr_active_thresh=0.01, min_num_unique_aclu_inclusions=3).override(override_preprocessing.get('replays', {}))

		preprocessing_parameters = ParametersContainer(epoch_estimation_parameters=DynamicContainer.init_from_dict({
					'laps': default_lap_estimation_parameters,
					'PBEs': default_PBE_estimation_parameters,
					'replays': default_replay_estimation_parameters
				}))
		return preprocessing_parameters


	@classmethod
	def build_default_computation_configs(cls, sess, **kwargs) -> List[DynamicContainer]:
		""" OPTIONALLY can be overriden by implementors to provide specific filter functions """
		override_parameters_flat_keypaths_dict = kwargs.pop('override_parameters_flat_keypaths_dict', {}) or {} # ` or {}` part handles None values
		override_parameters_nested_dicts = KeypathsAccessibleMixin.keypath_dict_to_nested_dict(override_parameters_flat_keypaths_dict)
		
		#TODO 2024-10-30 10:20: - [ ] Should it be `.get('preprocessing', {})`? Or these more top-level?
		kwargs.setdefault('pf_params', PlacefieldComputationParameters(**override_dict({'speed_thresh': 10.0, 'grid_bin': cls.compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), 'grid_bin_bounds': None, 'smooth': (2.0, 2.0), 'frate_thresh': 1.0, 'time_bin_size': 0.1, 'computation_epochs': None}, ## NOTE: 2025-01-15 06:31 on at least one call (but before the main pf computation) this is hit without using the grid_bin_bounds in the computation, seems to be inferring it from the recorded position info and the desired num_bins in each direction
																						(override_parameters_nested_dicts.get('preprocessing', {}).get('pf_params', {}) | kwargs))))
		kwargs.setdefault('spike_analysis', DynamicContainer(**{'max_num_spikes_per_neuron': 20000,
																 'kleinberg_parameters': DynamicContainer(**{'s': 2, 'gamma': 0.2}).override(kwargs),
																 'use_progress_bar': False,
																 'debug_print': False}).override((override_parameters_nested_dicts.get('preprocessing', {}).get('spike_analysis', {}) | kwargs)))
		# return [DynamicContainer(pf_params=kwargs['pf_params'], spike_analysis=kwargs['spike_analysis'])]
		
		return [DynamicContainer(pf_params=kwargs['pf_params'].override(override_parameters_nested_dicts.get('preprocessing', {}).get('pf_params', {})),
								 spike_analysis=kwargs['spike_analysis'].override(override_parameters_nested_dicts.get('preprocessing', {}).get('spike_analysis', {}))
								 )]
		# return [DynamicContainer(pf_params=PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=cls.compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=1.0, computation_epochs = None),
		#                   spike_analysis=DynamicContainer(max_num_spikes_per_neuron=20000, kleinberg_parameters=DynamicContainer(s=2, gamma=0.2), use_progress_bar=False, debug_print=False))]
		# active_grid_bin = compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64))
		# active_session_computation_config.computation_epochs = None # set the placefield computation epochs to None, using all epochs.
		# return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
		# return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(128, 128)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
		# return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(3.777, 1.043), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
		# return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(32, 32)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
		#         PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
		#         PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(128, 128)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
		#        ]
	
	@classmethod
	def build_active_computation_configs(cls, sess, **kwargs):
		""" defines the main computation configs for each class. This is provided as an alternative to `build_default_computation_configs` because some classes use cls.build_default_computation_configs(...) to get the plain configs, which they then update with different properties. """
		return cls.build_default_computation_configs(sess, **kwargs)



	@classmethod
	def get_session(cls, basedir, override_parameters_flat_keypaths_dict=None):
		_test_session = cls.build_session(Path(basedir), override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)
		_test_session, loaded_file_record_list = cls.load_session(_test_session)
		return _test_session    
	
	@classmethod
	def find_session_name_from_sole_xml_file(cls, basedir, debug_print=False):
		""" By default it attempts to find the single *.xml file in the root of this basedir, from which it determines the `session_name` as the stem (the part before the extension) of this file
		Example:
			basedir: Path(r'R:\data\Bapun\Day5TwoNovel')
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
	def get_session_basepath_to_context_parsing_keys(cls):
		""" Just a wrapper to access the cls._session_basepath_to_context_parsing_keys property
		Used only by `parse_session_basepath_to_context(.)`
		"""
		return cls._session_basepath_to_context_parsing_keys



	@classmethod
	def parse_session_basepath_to_context(cls, basedir) -> IdentifyingContext:
		""" parses the session's path to determine its proper context. Depends on the data_type.
		finds global_data_root

		USED: Called only in `cls.build_session(.)`

		KDIBA: 'W:/Data/KDIBA/gor01/one/2006-6-09_3-23-37' | context_keys = ['format_name','animal','exper_name', 'session_name']
		HIRO: 'W:/Data/Hiro/RoyMaze1' | context_keys = ['format_name', 'session_name']
			# Additional parsing needed: W:\Data\Hiro\RoyMaze2

		BAPUN: 'W:/Data/Bapun/RatS/Day5TwoNovel' | context_keys = ['format_name','animal', 'session_name']
		RACHEL: 'W:/Data/Rachel/merged_M1_20211123_raw_phy' | context_keys = ['format_name', 'session_name']

		"""
		# IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
		basedir = Path(basedir) # basedir WindowsPath('W:/Data/KDIBA/gor01/one/2006-6-07_11-26-53')
		dir_parts = basedir.parts # ('W:\\', 'Data', 'KDIBA', 'gor01', 'one', '2006-6-07_11-26-53')
		# Finds the index of the 'Data' or 'data' (global_data_root) part of the path to include only what's after that.

		# 'umms-kdiba' in '/nfs/turbo/umms-kdiba/Data' is basically a Data folder
		if basedir.resolve().is_relative_to(Path('/nfs/turbo/umms-kdiba').resolve()):
			_parent_path = Path('/nfs/turbo/umms-kdiba').resolve()
			# relative_basedir = basedir.resolve().relative_to(_parent_path)
			# dir_parts = basedir.parts
			data_index = len(_parent_path.parts) - 1 # -1 for index
		else:
			try:        
				data_index = tuple(map(str.casefold, dir_parts)).index('DATA'.casefold()) # .casefold is equivalent to .lower, but works for unicode characters
			except ValueError:
				# Enables looking for 'FASTDATA' in the path when DATA is not found
				data_index = tuple(map(str.casefold, dir_parts)).index('FASTDATA'.casefold()) # .casefold is equivalent to .lower, but works for unicode characters
			except Exception:
				raise # unhandled exception

		post_data_root_dir_parts = dir_parts[data_index+1:] # ('KDIBA', 'gor01', 'one', '2006-6-07_11-26-53')

		num_parts = len(post_data_root_dir_parts)
		context_keys = cls.get_session_basepath_to_context_parsing_keys()
		assert len(context_keys) == num_parts
		context_kwargs_dict = dict(zip(context_keys, post_data_root_dir_parts))
		curr_sess_ctx = IdentifyingContext(**context_kwargs_dict)
		# want to replace the 'format_name' with the one known for this session (e.g. 'KDIBA' vs. 'kdiba')
		format_name = cls.get_session_format_name() 
		curr_sess_ctx.format_name = format_name
		return curr_sess_ctx # IdentifyingContext<('KDIBA', 'gor01', 'one', '2006-6-07_11-26-53')>



	# @classmethod
	# def try_extract_date_from_session_name(cls, session_name: str): # Optional[Union[pd.Timestamp, NaTType]]
	#     """ 2023-08-24 - Attempts to determine at least the relative recording date for a given session from the session's name alone.
	#     From the 'session_name' column in the provided data, we can observe two different formats used to specify the date:

	#     Format 1: Dates with the pattern YYYY-M-D_H-M-S (e.g., "2006-6-07_11-26-53").
	#     Format 2: Dates with the pattern MM-DD_H-M-S (e.g., "11-02_17-46-44").
		
	#     """
	#     # Remove any non-digit prefixes or suffixes before parsing. Handles 'fet11-01_12-58-54'

	#     # Check for any non-digit prefix
	#     if re.match(r'^\D+', session_name):
	#         print(f"WARN: Removed prefix from session_name: {session_name}")
	#         session_name = re.sub(r'^\D*', '', session_name)

	#     # Check for any non-digit suffix
	#     if re.search(r'\D+$', session_name):
	#         print(f"WARN: Removed suffix from session_name: {session_name}")
	#         session_name = re.sub(r'\D*$', '', session_name)


	#     # Try Format 1 (YYYY-M-D_H-M-S)
	#     date_match1 = re.search(r'\d{4}-\d{1,2}-\d{1,2}_\d{1,2}-\d{1,2}-\d{1,2}', session_name)
	#     if date_match1:
	#         date_str1 = date_match1.group().replace('_', ' ')
	#         return pd.to_datetime(date_str1, format='%Y-%m-%d %H-%M-%S', errors='coerce')

	#     # Try Format 2 (MM-DD_H-M-S)
	#     date_match2 = re.search(r'\d{1,2}-\d{1,2}_\d{1,2}-\d{1,2}-\d{1,2}', session_name)
	#     if date_match2:
	#         date_str2 = "2000-" + session_name.split('_')[0] # Assuming year 2000
	#         time_str2 = session_name.split('_')[1].replace('-', ':')
	#         full_str2 = date_str2 + ' ' + time_str2
	#         return pd.to_datetime(full_str2, format='%Y-%m-%d %H:%M:%S', errors='coerce')

	#     print(f"WARN: Could not parse date from session_name: {session_name} for any known format.")
	#     return None



	@classmethod
	def get_known_data_session_type_properties(cls, override_basepath=None, override_parameters_flat_keypaths_dict=None):
		""" returns the KnownDataSessionTypeProperties for this class, which contains information about the process of loading the session."""
		if override_basepath is not None:
			basepath = override_basepath
		else:
			basepath = Path(cls._session_default_basedir)
		return KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: cls.get_session(basedir=a_base_dir, override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)), basedir=basepath)
		
	@classmethod
	def build_session(cls, basedir, override_parameters_flat_keypaths_dict=None):
		basedir = Path(basedir)
		session_name = cls.get_session_name(basedir) # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
		session_context = cls.parse_session_basepath_to_context(basedir) 
		session_spec = cls.get_session_spec(session_name)
		format_name = cls.get_session_format_name()
			
		# get the default preprocessing parameters:
		preprocessing_parameters = cls.build_default_preprocessing_parameters(override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)

		session_config = SessionConfig(basedir, format_name=format_name, session_spec=session_spec, session_name=session_name, session_context=session_context, preprocessing_parameters=preprocessing_parameters)
		assert session_config.is_resolved, "active_sess_config could not be resolved!"
		session_obj = DataSession(session_config)
		return session_obj
	
	@classmethod
	def load_session(cls, session, debug_print=False):
		# .recinfo, .filePrefix, .eegfile, .datfile
		loaded_file_record_list = [] # Handled files list
		
		try:
			_test_xml_file_path, _test_xml_file_spec = list(session.config.resolved_required_filespecs_dict.items())[0]
			session = _test_xml_file_spec.session_load_callback(_test_xml_file_path, session)
			loaded_file_record_list.append(_test_xml_file_path)
		except IndexError as e:
			# No XML file can be found, so instead check for a dynamically provided rec_info
			session = cls._fallback_recinfo(None, session)
			# raise e
				
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
			if force_recompute:
				print(f'\t force_recompute is True! Forcing recomputation of {active_file_suffix}\n')
			else:
				print(f'\t Failure loading "{session.filePrefix.with_suffix(active_file_suffix)}". Must recompute.\n')
			with ProgressMessagePrinter('spikes_df', action='Computing', contents_description='interpolate_spike_positions columns'):
				if session.position.has_linear_pos:
					lin_pos = session.position.linear_pos
				else:
					lin_pos = None
				spikes_df = FlattenedSpiketrains.interpolate_spike_positions(spikes_df, session.position.time, session.position.x, session.position.y, position_linear_pos=lin_pos, position_speeds=session.position.speed, spike_timestamp_column_name=time_variable_name)
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
		with ProgressMessagePrinter('spikes_df', action='Computing', contents_description='spikes_df PBEs column'):
			updated_spk_df = session.compute_spikes_PBEs()
		return session
	
	@classmethod
	def _default_add_spike_scISIs_if_needed(cls, session):
		with ProgressMessagePrinter('spikes_df', action='Computing', contents_description='added spike scISI column'):
			updated_spk_df = session.spikes_df.spikes.add_same_cell_ISI_column()
		return session
	
	
	@classmethod
	def _default_compute_flattened_spikes(cls, session, timestamp_scale_factor=(1/1E4), spike_timestamp_column_name='t_seconds', progress_tracing=True):
		""" builds the session.flattened_spiketrains (and therefore spikes_df) from the session.neurons object. """
		spikes_df = FlattenedSpiketrains.build_spike_dataframe(session, timestamp_scale_factor=timestamp_scale_factor, spike_timestamp_column_name=spike_timestamp_column_name, progress_tracing=progress_tracing)
		print(f'spikes_df.columns: {spikes_df.columns}')
		session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=spike_timestamp_column_name, t_start=session.neurons.t_start) # FlattenedSpiketrains(spikes_df)
		print('\t Done!')
		return session
	
	@classmethod
	def _add_missing_spikes_df_columns(cls, spikes_df, neurons_obj):
		""" adds the 'fragile_linear_neuron_IDX' column to the spikes_df and updates the neurons_obj with a new reverse_cellID_index_map """
		spikes_df, neurons_obj._reverse_cellID_index_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
		spikes_df['t'] = spikes_df[cls._time_variable_name] # add the 't' column required for visualization
 
	@classmethod
	def _default_extended_postload(cls, fp, session, **kwargs):
		# Computes Common Extended properties:
		force_recompute = kwargs.pop('force_recompute', False)
		
		## Ripples:
		try:
			# Externally Computed Ripples (from 'ripple_df.pkl') file:
			# Load `ripple_df.pkl` previously saved:
			external_computed_ripple_filepath = session.basepath.joinpath('ripple_df.pkl')
			external_computed_ripple_df = pd.read_pickle(external_computed_ripple_filepath)
			# Add the required columns for Epoch(...):
			external_computed_ripple_df['label'] = [str(an_idx) for an_idx in external_computed_ripple_df.index]
			external_computed_ripple_df = external_computed_ripple_df.reset_index(drop=True)
			found_datafile = Epoch(external_computed_ripple_df) # Epoch from dataframe
		except FileNotFoundError:
			print(f'externally computed ripple_df.pkl not found. Falling back to .ripple.npy...')
			found_datafile = None

		if found_datafile is not None:
			print('Loading success: {}.'.format(external_computed_ripple_filepath))
			session.ripple = found_datafile
			found_datafile.filename = external_computed_ripple_filepath
		else:
			## try the '.ripple.npy' ripples:
			active_file_suffix = '.ripple.npy'
			external_computed_ripple_filepath = fp.with_suffix(active_file_suffix)
			found_datafile = Epoch.from_file(external_computed_ripple_filepath)
			if found_datafile is not None:
				print('Loading success: {}.'.format(active_file_suffix))
				session.ripple = found_datafile
				# ## TODO: overwrite the '.ripple.npy' version?
				# session.ripple.filename = session.filePrefix.with_suffix('.ripple.npy')
				# # print_file_progress_message(ripple_epochs.filename, action='Saving', contents_description='ripple epochs')
				# with ProgressMessagePrinter(session.ripple.filename, action='Saving', contents_description='ripple epochs'):
				#     session.ripple.save()
			else:
				# Otherwise both loads failed, perform the fallback computation:
				if not force_recompute:
					print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
				else:
					print(f'force_recompute is True, recomputing...')
				try:
					session.ripple = DataSession.compute_neurons_ripples(session, save_on_compute=True)
				except (ValueError, AttributeError) as e:
					print(f'Computation failed with error {e}. Skipping .ripple')
					session.ripple = None

		## MUA:
		active_file_suffix = '.mua.npy'
		found_datafile = Mua.from_file(fp.with_suffix(active_file_suffix))
		if (not force_recompute) and (found_datafile is not None):
			print('Loading success: {}.'.format(active_file_suffix))
			session.mua = found_datafile
		else:
			# Otherwise load failed, perform the fallback computation
			if not force_recompute:
				print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
			else:
				print(f'force_recompute is True, recomputing...')
			try:
				session.mua = DataSession.compute_neurons_mua(session, save_on_compute=True)
			except (ValueError, AttributeError) as e:
				print(f'Computation failed with error {e}. Skipping .mua')
				session.mua = None
				
		## PBE Epochs:
		active_file_suffix = '.pbe.npy'
		found_datafile = Epoch.from_file(fp.with_suffix(active_file_suffix))
		if (not force_recompute) and (found_datafile is not None):
			print('Loading success: {}.'.format(active_file_suffix))
			session.pbe = found_datafile
		else:
			# Otherwise load failed, perform the fallback computation
			if not force_recompute:
				print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
			else:
				print(f'force_recompute is True, recomputing...')
			try:
				# active_pbe_parameters = kwargs.pop('pbe_epoch_detection_params', session.config.preprocessing_parameters.epoch_estimation_parameters.PBEs)
				active_pbe_parameters = session.config.preprocessing_parameters.epoch_estimation_parameters.PBEs
				session.pbe = DataSession.compute_pbe_epochs(session, active_parameters=active_pbe_parameters, save_on_compute=True)
			except (ValueError, AttributeError) as e:
				print(f'Computation failed with error {e}. Skipping .pbe')
				session.pbe = None
		
		# add PBE information to spikes_df from session.pbe
		cls._default_add_spike_PBEs_if_needed(session)
		cls._default_add_spike_scISIs_if_needed(session)
		# return the session with the upadated member variables
		return session
	
	
	@classmethod
	def _fallback_recinfo(cls, filepath, session):
		""" called when the .xml-method fails. Implementor can override to provide a valid .recinfo and .filePrefix anyway. """
		raise NotImplementedError # innheritor MAY override
		session.filePrefix = filepath.with_suffix("") # gets the session name (basically) without the .xml extension.
		session.recinfo = DynamicContainer(**{
			"source_file": self.source_file,
			"channel_groups": self.channel_groups,
			"skipped_channels": self.skipped_channels,
			"discarded_channels": self.discarded_channels,
			"n_channels": self.n_channels,
			"dat_sampling_rate": self.dat_sampling_rate,
			"eeg_sampling_rate": self.eeg_sampling_rate,
		})
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
		except (ValueError, FileNotFoundError):
			print('session.recinfo.eeg_filename exists ({}) but file cannot be loaded in the appropriate format. Skipping. \n'.format(filepath))
			session.eegfile = None
		except (FileNotFoundError):
			print('session.recinfo.eeg_filename does not exist or is not accessible ({}). Skipping. \n'.format(filepath))
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
