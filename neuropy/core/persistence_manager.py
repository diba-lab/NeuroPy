from typing import Callable, List, Dict, Optional, Union
from attrs import define, field, Factory
from pathlib import Path

from neuropy.utils.indexing_helpers import flatten

@define(slots=False)
class ContextToPathOption:
    """ Controls how hierarchical contexts (IdentityContext) are mapped to relative output paths.
    In HIERARCHY_UNIQUE mode the folder hierarchy partially specifies the context (mainly the session part, e.g. './kdiba/gor01/two/2006-6-08_21-16-25/') so the filenames don't need to be completely unique (they can drop the 'kdiba_gor01_two_2006-6-08_21-16-25_' portion)
        'output/kdiba/gor01/two/2006-6-08_21-16-25/batch_pho_jonathan_replay_firing_rate_comparison.png

    In GLOBAL_UNIQUE mode the outputs are placed in a flat folder structure ('output/'), meaning the filenames need to be completely unique and specify all parts of the context:
        'output/kdiba_gor01_two_2006-6-08_21-16-25_batch_pho_jonathan_replay_firing_rate_comparison.png'


    Usage: 

        from neuropy.core.persistence_manager import ContextToPathOption

        
    """
    filepath_context_keys = field(default=Factory(list)) # keys in this array will be combined into a directory hierarchy.
    
    # ['format_name', ('animal','exper_name', 'session_name')] # e.g. 	
    # [	'format_name','animal','exper_name', 'session_name']


    filename_base_keys = field(default=Factory(list)) # keys in this array will get flattened into the filename
    filename_base_separation_str: str = field(default='_')


    
    def session_context_to_relative_path(self, parent_path: Union[Path,str], session_ctx, create_directories=False, override_context_tuple_join_character=None, debug_print=False) -> Path:
        """Only uses the keys that define session: ['format_name','animal','exper_name', 'session_name'] to build the relative path

        Args:
            parent_path (Path): _description_
            session_ctx (IdentifyingContext): _description_

        Returns:
            _type_: _description_

        Usage:
            from pyphoplacecellanalysis.General.Mixins.ExportHelpers import session_context_to_relative_path
            
            curr_sess_ctx = local_session_contexts_list[0]
            # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
            figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
            session_context_to_relative_path(figures_parent_out_path, curr_sess_ctx)

        """
        if isinstance(parent_path, str):
            parent_path = Path(parent_path)


        context_tuple_join_character: str = (override_context_tuple_join_character or self.filename_base_separation_str)

        filepath_context_keys = self.filepath_context_keys # ['format_name','animal','exper_name', 'session_name']
        flattened_context_keys = list(flatten(filepath_context_keys))

        all_keys_found, found_keys, missing_keys = session_ctx.check_keys(flattened_context_keys, debug_print=debug_print)
        if not all_keys_found:
            print(f'WARNING: missing {len(missing_keys)} keys from context: {missing_keys}. Building path anyway.')

        # Get the filepath parts:
        curr_path = parent_path.resolve()
        if create_directories:
            curr_path.mkdir(exist_ok=True)

        for subset_includelist in filepath_context_keys:
            is_single_element = (not isinstance(subset_includelist, (list, tuple)))                
            curr_sess_ctx_tuple = session_ctx.as_tuple(subset_includelist=subset_includelist, drop_missing=True) # ('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')
            if not is_single_element:
                # make into a flat string first:
                folder_basename = context_tuple_join_character.join(curr_sess_ctx_tuple) # joins the elements of the context_tuple with '_'
                curr_sess_ctx_tuple = (folder_basename, ) # add as a single element tuple

            if debug_print:
                print(f'is_single_element: {is_single_element}, subset_includelist: {subset_includelist}, subset_includelist: {subset_includelist}, curr_sess_ctx_tuple: {curr_sess_ctx_tuple}, curr_path: {curr_path}')
            curr_path = curr_path.joinpath(*curr_sess_ctx_tuple).resolve() # updated path:
            ## Create if needed
            if create_directories:
                curr_path.mkdir(exist_ok=True)
                print(f'making dir: {curr_path}')

        return curr_path

    def build_basename_from_context(self, active_identifying_ctx, override_subset_includelist=None, override_subset_excludelist=None, override_context_tuple_join_character=None, debug_print=False) -> str:
        """ 
        Usage:
            from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_figure_basename_from_display_context
            curr_fig_save_basename = build_figure_basename_from_display_context(active_identifying_ctx, context_tuple_join_character='_')
            >>> 'kdiba_2006-6-09_1-22-43_batch_plot_test_long_only'
        """
        # subset_excludelist = (override_subset_excludelist or [])
        subset_excludelist = override_subset_excludelist # subset_excludelist must be None if passed into the .as_tuple(...) fcn, so we'd need to remove any entries from `subset_includelist`
        subset_includelist = (override_subset_includelist or self.filename_base_keys)
        context_tuple_join_character: str = (override_context_tuple_join_character or self.filename_base_separation_str)

        if subset_excludelist is not None:
            subset_includelist = [v for v in subset_includelist if v not in subset_excludelist]

        # flattened_context_keys = list(flatten(self.filepath_context_keys))
        # subset_excludelist = subset_excludelist + flattened_context_keys # add the session keys to the subset_excludelist
                
        context_tuple = [str(v) for v in list(active_identifying_ctx.as_tuple(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist, drop_missing=True))]
        file_basename = context_tuple_join_character.join(context_tuple) # joins the elements of the context_tuple with '_'
        if debug_print:
            print(f'file_basename: "{file_basename}"')
        return file_basename


    @classmethod
    def init_HIERARCHY_UNIQUE(cls) -> "ContextToPathOption":
        return cls(['format_name','animal','exper_name', 'session_name'], filename_base_keys=[])

    @classmethod
    def init_GLOBAL_UNIQUE(cls) -> "ContextToPathOption":
        return cls([], filename_base_keys=['format_name','animal','exper_name', 'session_name'])


@define(slots=False)
class PersistenceManager:
    """ handles persistance to/from files

    Usage:
        from neuropy.core.persistance_manager import PersistenceManager


    """
    global_data_root_parent_path: Path


