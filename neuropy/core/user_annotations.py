import sys
from copy import deepcopy
from typing import List, Any, Tuple, Optional, Callable
from attrs import define, field, Factory, asdict
import numpy as np
import pandas as pd
from datetime import datetime
from neuropy.utils.result_context import IdentifyingContext

# ==================================================================================================================== #
# 2023-06-21 User Annotations                                                                      #
# ==================================================================================================================== #

""" 
Migrated from 
from pyphoplacecellanalysis.General.Model.user_annotations import UserAnnotationsManager
to
from neuropy.core.user_annotations import UserAnnotationsManager

"""


@define(slots=False)
class UserAnnotationRecord:
    """ an annotation made by the user at a specific date/time covering a specific context """
    created: datetime
    modified: datetime
    context: IdentifyingContext
    content: dict

@define(slots=False)
class UserAnnotationsManager:
    """ class for holding User Annotations of the data. Performed interactive by the user, and then saved to disk for later use. An example are the selected replays to be used as examples. 
    
    Usage:
        from neuropy.core.user_annotations import UserAnnotationsManager
        
    """
    @staticmethod
    def get_user_annotations():
        """ hardcoded user annotations
        

        New Entries can be generated like:
            from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import SelectionsObject
            from neuropy.core.user_annotations import UserAnnotationsManager
            from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import DecodedEpochSlicesPaginatedFigureController

            ## Stacked Epoch Plot
            example_stacked_epoch_graphics = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', defer_render=False, save_figure=True)
            pagination_controller_L, pagination_controller_S = example_stacked_epoch_graphics.plot_data['controllers']
            ax_L, ax_S = example_stacked_epoch_graphics.axes
            final_figure_context_L, final_context_S = example_stacked_epoch_graphics.context

            user_annotations = UserAnnotationsManager.get_user_annotations()

            ## Capture current user selection
            saved_selection_L: SelectionsObject = pagination_controller_L.save_selection()
            saved_selection_S: SelectionsObject = pagination_controller_S.save_selection()
            final_L_context = saved_selection_L.figure_ctx.adding_context_if_missing(user_annotation='selections')
            final_S_context = saved_selection_S.figure_ctx.adding_context_if_missing(user_annotation='selections')
            user_annotations[final_L_context] = saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected]
            user_annotations[final_S_context] = saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected]
            # Updates the context. Needs to generate the code.

            ## Generate code to insert int user_annotations:
            print('Add the following code to UserAnnotationsManager.get_user_annotations() function body:')
            print(f"user_annotations[{final_L_context.get_initialization_code_string()}] = np.array({list(saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected])})")
            print(f"user_annotations[{final_S_context.get_initialization_code_string()}] = np.array({list(saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected])})")


        Usage:
            user_anootations = get_user_annotations()
            user_anootations

        """
        user_annotations = {}

        ## IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([13,  14,  15,  25,  27,  28,  31,  37,  42,  45,  48,  57,  61,  62,  63,  76,  79,  82,  89,  90, 111, 112, 113, 115])
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([  9,  11,  13,  14,  15,  20,  22,  25,  37,  40,  45,  48,  61, 62,  76,  79,  84,  89,  90,  93,  94, 111, 112, 113, 115, 121])

        # 2023-07-19 - New Annotations, lots more than before.
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([1, 3, 11, 13, 14, 15, 17, 21, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 36, 37, 39, 42, 43, 44, 45, 46, 48, 51, 52, 53, 55, 57, 58, 60, 61, 62, 68, 69, 70, 72, 74, 76, 81, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 100, 101, 105, 106, 109, 112, 113, 114, 115, 118, 119, 120, 121, 123, 124, 125, 126, 127, 128, 130, 131, 132])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([2, 3, 4, 8, 9, 10, 11, 13, 14, 15, 16, 17, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 51, 53, 55, 63, 64, 66, 67, 69, 70, 72, 75, 77, 78, 80, 81, 83, 84, 85, 86, 87, 88, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 110, 111, 112, 113, 114, 115, 116, 118, 119, 120, 121, 122, 123, 124, 126, 127, 131, 132])



        ## IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([5,  13,  15,  17,  20,  21,  24,  31,  33,  43,  44,  49,  63, 64,  66,  68,  70,  71,  74,  76,  77,  78,  84,  90,  94,  95, 104, 105, 122, 123])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([ 12,  13,  15,  17,  20,  24,  30,  31,  32,  33,  41,  43,  49, 54,  55,  68,  70,  71,  73,  76,  77,  78,  84,  89,  94, 100, 104, 105, 111, 114, 115, 117, 118, 122, 123, 131])

        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([10, 11, 12, 17, 18, 22])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([10, 11, 12, 16, 18, 19, 23])


        # IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([13, 23, 41, 46])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([4, 7, 10, 15, 21, 23, 41])

        # ## TODO:-
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([2, 6])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([2, 5, 9, 10])
        
        # ## TODO:
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([3, 4, 5])
        
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([19, 23, 26, 29, 44, 57, 64, 83, 90, 92, 110, 123, 125, 126, 131])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([5, 10, 19, 23, 24, 26, 31, 35, 36, 39, 44, 48, 57, 61, 64, 65, 71, 73, 77, 83, 89, 92, 93, 94, 96, 97, 98, 100, 102, 108, 111, 113, 116, 117, 118, 123, 124, 125, 126, 131])

        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43')
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([3, 13, 16, 18, 19, 20, 23, 24, 27, 28, 36, 38, 40, 43, 44, 47, 48, 52, 55, 64, 65])
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([3, 10, 13, 16, 18, 19, 24, 27, 28, 36, 40, 43, 44, 47, 48, 50, 55, 60, 64, 65])
        # 2023-07-19 Annotations:
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([3, 5, 12, 18, 23, 26, 28, 30, 32, 33, 35, 37, 44, 59, 61, 64, 66, 70, 71, 74, 76, 79, 84, 85, 97, 99])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([5, 6, 8, 18, 23, 28, 29, 30, 32, 40, 44, 59, 61, 64, 66, 70, 71, 73, 74, 79, 80, 81, 84, 85, 88, 97, 99])

        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([2, 13, 18, 23, 25, 27, 32])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([2, 8, 9, 13, 16, 18, 25, 27, 28, 32, 33])

        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([4, 22, 24, 28, 30, 38, 42, 50, 55, 60, 67, 70, 76, 83, 85, 100, 103, 107, 108, 113, 118, 121, 122, 131, 140, 142, 149, 153, 170, 171])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([2, 7, 11, 17, 20, 22, 30, 34, 38, 39, 41, 43, 47, 49, 55, 59, 60, 69, 70, 75, 77, 80, 83, 85, 86, 100, 107, 110, 113, 114, 115, 118, 120, 121, 122, 126, 130, 131, 138, 140, 142, 149, 157, 160, 168, 170])

        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([3, 6, 15, 16, 18])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([3, 4, 6, 9, 13, 15, 16, 18])

        # IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([5, 6, 7, 14, 17, 19, 20, 24])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([5, 6, 19, 20])




        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([0, 3, 4, 5, 9, 12])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([0, 3, 4, 6, 9, 12, 13])

        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([3])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([3])


        return user_annotations
    
    @classmethod
    def has_user_annotation(cls, test_context):
        user_anootations = cls.get_user_annotations()
        was_annotation_found: bool = False
        # try to find a matching user_annotation for the final_context_L
        for a_ctx, selections_array in user_anootations.items():
            an_item_diff = a_ctx.diff(test_context)
            if an_item_diff == {('user_annotation', 'selections')}:
                was_annotation_found = True
                break # done looking
        return was_annotation_found

    @classmethod
    def get_hardcoded_specific_session_override_dict(cls) -> dict:
        """ Extracted from `neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat` 
            ## Create a dictionary of overrides that have been specified manually for a given session:
            # Used in `build_lap_only_short_long_bin_aligned_computation_configs`

            Usage:
                ## Get specific grid_bin_bounds overrides from the `cls._specific_session_override_dict`
                override_dict = cls.get_specific_session_override_dict().get(sess.get_context(), {})
                if override_dict.get('grid_bin_bounds', None) is not None:
                    grid_bin_bounds = override_dict['grid_bin_bounds']
                else:
                    # no overrides present
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


        """
        return { 
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'):{'grid_bin_bounds':((29.16, 261.70), (130.23, 150.99))},
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'):{'grid_bin_bounds':((22.397021260868584, 245.6584673739576), (133.66465594522782, 155.97244934208123))},
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'):dict(grid_bin_bounds=(((17.01858788173554, 250.2171441367766), (135.66814125966783, 154.75073313142283)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'):{'grid_bin_bounds':(((29.088604852961407, 251.70402561515647), (138.496638485457, 154.30675703402517)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'):{'grid_bin_bounds':(((29.16, 261.7), (133.87292045454544, 150.19888636363635)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'):{'grid_bin_bounds':((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'):{'grid_bin_bounds':(((19.639345624112345, 248.63934562411234), (134.21607306829767, 154.57926689187622)))},
            # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'):{'grid_bin_bounds':((28.54313873072426, 255.54313873072425), (80.0, 151.0))}, # (-56.2405385510412, -12.237798967230454)
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'):dict(grid_bin_bounds=(((36.58620390950715, 248.91627658974846), (132.81136363636367, 149.2840909090909)))),
            # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'):{'grid_bin_bounds':((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915))},
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_3-23-37'):{'grid_bin_bounds':(((29.64642522460817, 257.8732552112081), (106.68603845428224, 146.71219371189815)))},
            # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'):{'grid_bin_bounds':(((36.47611374385336, 246.658598426423), (134.75608863422366, 149.10512838805013)))},
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-13_14-42-6'):{'grid_bin_bounds':(((34.889907585004366, 250.88049171752402), (131.38802948402946, 148.80548955773958)))},
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_15-46-47'):{'grid_bin_bounds':(((37.58127153781621, 248.7032779553949), (133.5550653393467, 147.88514770982718)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-17_12-33-47'):{'grid_bin_bounds':(((26.23480758754316, 249.30607830191923), (130.58181353748455, 153.36300919999059)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_13-6-1'):{'grid_bin_bounds':(((31.470464455344967, 252.05028043482017), (128.05945067500747, 150.3229156741395)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-19_13-34-40'):{'grid_bin_bounds':(((29.637787747400818, 244.6377877474008), (138.47834488369824, 155.0993015545914)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-27_14-43-12'):{'grid_bin_bounds':(((27.16098236570231, 249.70986567911666), (106.81005068995495, 118.74413456592755)))},
            # IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58  -3'):{'grid_bin_bounds':(((28.84138997640293, 259.56043988873074), (101.90256273413083, 118.33845994931318)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_12-48-38'):{'grid_bin_bounds':(((21.01014932647431, 250.0101493264743), (92.34934413366932, 128.1552287735411)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_16-2-46'):{'grid_bin_bounds':(((17.270839996578303, 259.97986762679335), (94.26725170377283, 131.3621243061284)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-12_15-25-59'):{'grid_bin_bounds':(((30.511181558838498, 247.5111815588389), (106.97411662767412, 146.12444016982818)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_14-49-24'):{'grid_bin_bounds':(((30.473731136762368, 250.59478046470133), (105.10585244511995, 149.36442051808177)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_18-47-52'):{'grid_bin_bounds':(((27.439671363238585, 252.43967136323857), (106.37372678405141, 149.37372678405143)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-17_12-52-15'):{'grid_bin_bounds':(((25.118453388111003, 253.3770388211908), (106.67602982073078, 145.67602982073078)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-19_13-50-7'):{'grid_bin_bounds':(((22.47237613669028, 247.4723761366903), (109.8597911774777, 148.96242871522395)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-19_16-37-40'):{'grid_bin_bounds':(((27.10059856429566, 249.16997904433555), (104.99819196992492, 148.0743732909197)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-21_11-19-2'):{'grid_bin_bounds':(((19.0172498755827, 255.42277198494864), (110.04725120825609, 146.9523233129975)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-25_13-20-55'):{'grid_bin_bounds':(((12.844282158261015, 249.81408485606906), (107.18107171696062, 147.5733884981106)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-26_13-51-50'):{'grid_bin_bounds':(((29.04362374788327, 248.04362374788326), (104.87398380095135, 145.87398380095135)))},
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-28_12-38-13'):{'grid_bin_bounds':(((14.219834349211556, 256.8892365192059), (104.62582591329034, 144.76901436952045)))},
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44'):dict(grid_bin_bounds=(((26.927879930920472, 253.7869451377655), (129.2279041328145, 152.59317191760715)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0'):dict(grid_bin_bounds=(((20.551685242617875, 249.52142297024744), (136.6282885482392, 154.9308054334688)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25'):dict(grid_bin_bounds=(((22.2851382680749, 246.39985985110218), (133.85711719213543, 152.81579979839964)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54'):dict(grid_bin_bounds=(((24.27436551166163, 254.60064907635376), (136.60434348821698, 150.5038133052293)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'):dict(grid_bin_bounds=(((28.300282316379977, 259.7976187852487), (128.30369397123394, 154.72988093974095)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46'):dict(grid_bin_bounds=(((24.481516142738176, 255.4815161427382), (132.49260896751392, 155.30747604466447)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-07_11-26-53'):dict(grid_bin_bounds=(((38.73738238974907, 252.7760510005677), (132.90403388034895, 148.5092041989809)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-13_15-22-3'):dict(grid_bin_bounds=(((30.65617684737977, 242.10210639375669), (135.26433229328896, 154.7431209356581)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_15-23-32'):dict(grid_bin_bounds=(((22.21801384517548, 249.21801384517548), (96.86784196156162, 123.98778194291367)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_21-26-8'):dict(grid_bin_bounds=(((20.72906822314644, 251.8461197472293), (135.47600437022294, 153.41070963308343)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-05_19-26-43'):dict(grid_bin_bounds=(((22.368808752432248, 251.37564395184629), (137.03177602287607, 158.05294971766054)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_11-43-50'):dict(grid_bin_bounds=(((24.69593655030406, 253.64858157619915), (136.41037216273168, 154.5621182921704)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_12-15-3'):dict(grid_bin_bounds=(((23.604005859617253, 246.60400585961725), (140.24525850461768, 156.87513105457236)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_21-17-16'):dict(grid_bin_bounds=(((26.531441449778306, 250.85301772871418), (138.8068762398286, 154.96868817726707)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_22-4-5'):dict(grid_bin_bounds=(((23.443356646345123, 251.83482088135696), (134.31041516724372, 155.28217042078484)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-03_20-28-3'):dict(grid_bin_bounds=(((22.32225341225395, 251.16261749302362), (134.7141935930816, 152.90682184511172)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-04_21-20-3'):dict(grid_bin_bounds=(((23.682015380447947, 250.68201538044795), (132.83596766730062, 151.01445456436113)))),
            ## 2023-07-20 New, will replace existing:
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-07_11-26-53'):dict(grid_bin_bounds=(((39.69907686853041, 254.50216929581626), (132.04374884996278, 148.989363285708)))),
            # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'):dict(grid_bin_bounds=(((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'):dict(grid_bin_bounds=(((36.58620390950715, 248.91627658974846), (132.81136363636367, 149.2840909090909)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_3-23-37'):dict(grid_bin_bounds=(((31.189597143056417, 248.34843718632368), (129.87370603299934, 152.79894671566146)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'):dict(grid_bin_bounds=(((28.300282316379977, 259.30028231638), (128.30369397123394, 154.72988093974095)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-13_14-42-6'):dict(grid_bin_bounds=(((34.889907585004366, 249.88990758500438), (131.38802948402946, 148.38802948402946)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'):dict(grid_bin_bounds=(((22.397021260868584, 245.3970212608686), (133.66465594522782, 155.97244934208123)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_15-46-47'):dict(grid_bin_bounds=(((24.87832507963516, 243.19318180072503), (138.17144977828153, 154.70022850038026)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'):dict(grid_bin_bounds=(((24.71824744583462, 248.6393456241123), (136.77104473778593, 152.85274652666337)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'):dict(grid_bin_bounds=(((29.088604852961407, 251.70402561515647), (138.496638485457, 153.496638485457)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46'):dict(grid_bin_bounds=(((24.481516142738176, 255.4815161427382), (132.49260896751392, 155.30747604466447)))),
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-13_15-22-3'):dict(grid_bin_bounds=(((30.65617684737977, 241.65617684737975), (135.26433229328896, 154.26433229328896)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'):dict(grid_bin_bounds=(((28.54313873072426, 255.54313873072425), (-55.2405385510412, -12.237798967230454)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'):dict(grid_bin_bounds=(((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-17_12-33-47'):dict(grid_bin_bounds=(((28.818747438744666, 252.93642303882393), (105.90899758151346, 119.13817286828353)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_13-6-1'):dict(grid_bin_bounds=(((29.717076500273222, 259.56043988873074), (95.7227315733896, 133.44723972697744)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_15-23-32'):dict(grid_bin_bounds=(((26.23603700440421, 249.21801384517548), (92.8248259903438, 133.94007118631126)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-19_13-34-40'):dict(grid_bin_bounds=(((24.30309074447524, 252.62022717482705), (88.84795233953739, 129.8246799924719)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-27_14-43-12'):dict(grid_bin_bounds=(((18.92582426040771, 259.0914978964906), (92.73825146448141, 127.71789984751534)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'):dict(grid_bin_bounds=(((29.64642522460817, 257.8732552112081), (106.68603845428224, 146.71219371189815)))),
            # IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3'):dict(grid_bin_bounds=(((28.84138997640293, 259.56043988873074), (106.30424414303856, 118.90256273413083)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_12-48-38'):dict(grid_bin_bounds=(((31.074573205168065, 250.46915764230738), (104.49699700791726, 143.3410531370672)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_16-2-46'):dict(grid_bin_bounds=(((24.297550846655597, 254.08440958433795), (107.3563450759142, 150.37117694134784)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-12_15-25-59'):dict(grid_bin_bounds=(((30.511181558838498, 247.5111815588389), (106.97411662767412, 145.9741166276741)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_14-49-24'):dict(grid_bin_bounds=(((24.981275039670876, 249.50933410068018), (107.26469870056884, 148.83960860385804)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_18-47-52'):dict(grid_bin_bounds=(((27.905254233199088, 250.7946946337514), (105.45717926004222, 146.30022614559135)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-17_12-52-15'):dict(grid_bin_bounds=(((18.204825719140114, 251.30717868551005), (102.22431800161995, 148.39669517071601)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44'):dict(grid_bin_bounds=(((26.927879930920472, 253.7869451377655), (129.2279041328145, 152.2279041328145)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0'):dict(grid_bin_bounds=(((20.551685242617875, 249.52142297024744), (136.6282885482392, 154.9308054334688)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25'):dict(grid_bin_bounds=(((22.2851382680749, 246.39985985110218), (133.85711719213543, 152.81579979839964)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_21-26-8'):dict(grid_bin_bounds=(((25.037045888933548, 251.8461197472293), (134.08933682327196, 153.70077784443535)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-05_19-26-43'):dict(grid_bin_bounds=(((25.586078838864836, 251.7891334064322), (135.5158259968098, 161.196019438371)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_11-43-50'):dict(grid_bin_bounds=(((24.69593655030406, 252.54415392264917), (135.0071025581097, 155.43479839454722)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_12-15-3'):dict(grid_bin_bounds=(((22.348488031027813, 246.60400585961725), (139.61986158820912, 157.67819754950602)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_21-17-16'):dict(grid_bin_bounds=(((27.11122010574816, 251.94824027772256), (141.2080597276766, 157.07149975242388)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_22-4-5'):dict(grid_bin_bounds=(((21.369069920503218, 250.59459989762465), (135.63605454614645, 156.15891604175224)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54'):dict(grid_bin_bounds=(((22.403791476255435, 255.28121598502332), (135.43617904962073, 153.6679723832235)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-03_20-28-3'):dict(grid_bin_bounds=(((22.32225341225395, 251.16261749302362), (129.90833607359593, 156.190369383283)))),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-04_21-20-3'):dict(grid_bin_bounds=(((23.682015380447947, 250.68201538044795), (132.9587491923089, 150.81663988518108)))),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3'):dict(grid_bin_bounds=(((30.511181558838498, 247.5111815588389), (106.97411662767412, 147.52430924258078)))),
        }




