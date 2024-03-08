from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

from copy import deepcopy
from typing import Optional, Union
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from neuropy.core.epoch import Epoch, ensure_dataframe

if TYPE_CHECKING:
    from neuropy.core import Position
    from neuropy.core.session.dataSession import DataSession

from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable
from neuropy.utils.mixins.print_helpers import SimplePrintable

from .datawriter import DataWriter
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs, deduplicate_epochs # for EpochsAccessor's .get_non_overlapping_df()

## Import:
# from neuropy.core.laps import Laps


# TODO: implement: NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, TimeSlicableObjectProtocol
class Laps(Epoch):
    # epoch column labels: ["start", "stop", "label"]
    df_all_fieldnames = ['lap_id','maze_id','start_spike_index', 'end_spike_index', 'start_t', 'end_t', 'start_t_seconds', 'end_t_seconds', 'duration_seconds']
    
    def __init__(self, laps: pd.DataFrame, metadata=None) -> None:
        """[summary]
        Args:
            laps (pd.DataFrame): Each column is a pd.Series(["start", "stop", "label"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        super().__init__(laps, metadata=metadata)
        # self._data = laps # set to the laps dataframe
        self._df = Laps._update_dataframe_computed_vars(self._df)
        self._df = self._df.sort_values(by=['start']) # sorts all values in ascending order

    @property
    def _data(self):
        """ 2023-10-27 - a passthrough property for backwards compatibility. After adapting to a subclass of Epoch, the internal property is known as `self._df` not `self._data` """
        return self._df
    @_data.setter
    def _data(self, value):
        self._df = value


    @property
    def lap_id(self):
        return self._data['lap_id'].to_numpy()
    
    @property
    def maze_id(self):
        return self._data['maze_id'].to_numpy()
            
    @property
    def n_laps(self):
        return len(self.lap_id)
        
    @property
    def starts(self):
        return self._data['start'].to_numpy()

    @property
    def stops(self):
        return self._data['stop'].to_numpy()
    
    def filtered_by_lap_flat_index(self, lap_indicies):
        return self.filtered_by_lap_id(self.lap_id[lap_indicies])
       
    def filtered_by_lap_id(self, lap_ids):
        sliced_copy = deepcopy(self) # get copy of the dataframe
        sliced_copy._data = sliced_copy._data[np.isin(sliced_copy.lap_id, lap_ids)]
        return sliced_copy

    def update_maze_id_if_needed(self, t_start:float, t_delta:float, t_end:float) -> None:
        """ adds the 'maze_id' column to the internal dataframe if needed.
        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        laps_obj: Laps = curr_active_pipeline.sess.laps
        laps_obj.update_maze_id_if_needed(t_start, t_delta, t_end)
        laps_df = laps_obj.to_dataframe()
        laps_df
                
        """
        self._df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end, replace_existing=True, labels_column_name='lap_id')

    def update_lap_dir_from_smoothed_velocity(self, pos_input: Union[Position, DataSession]) -> None:
        # compute_lap_dir_from_smoothed_velocity
        # global_session = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name])
        self._df = self._compute_lap_dir_from_smoothed_velocity(self._df, pos_input, replace_existing=True)


    def adding_true_decoder_identifier(self, t_start:float, t_delta:float, t_end:float, labels_column_name:str='lap_id') -> pd.DataFrame:
        """ adds the 'maze_id' column to the internal dataframe if needed.
        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        laps_obj: Laps = curr_active_pipeline.sess.laps
        laps_obj.update_maze_id_if_needed(t_start, t_delta, t_end)
        laps_df = laps_obj.to_dataframe()
        laps_df
                
        """
        filter_epochs: pd.DataFrame = self._df.epochs.get_valid_df() # ensure_dataframe(filter_epochs).epochs.adding_maze_id_if_needed(t_start, t_delta, t_end, replace_existing=True, labels_column_name=labels_column_name)
        filter_epochs = filter_epochs.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end, replace_existing=True, labels_column_name=labels_column_name)

        assert 'maze_id' in filter_epochs
        assert 'lap_dir' in filter_epochs
        # Creates Columns: 'truth_decoder_name':
        lap_dir_keys = ['LR', 'RL']
        maze_id_keys = ['long', 'short']
        filter_epochs['truth_decoder_name'] = filter_epochs['maze_id'].map(dict(zip(np.arange(len(maze_id_keys)), maze_id_keys))) + '_' + filter_epochs['lap_dir'].map(dict(zip(np.arange(len(lap_dir_keys)), lap_dir_keys)))

        self._df[['maze_id', 'truth_decoder_name']] = filter_epochs[['maze_id', 'truth_decoder_name']] ## modify in-place and return?

        return filter_epochs
        

    

    def filter_to_valid(self) -> "Laps":
        laps_epoch_obj: Epoch = deepcopy(self).as_epoch_obj()
        original_laps_epoch_df = laps_epoch_obj.to_dataframe()        
        filtered_laps_epoch_df = Laps.ensure_valid_laps_epochs_df(original_laps_epoch_df)
        return Laps(filtered_laps_epoch_df, metadata=self.metadata)

    def trimmed_to_non_overlapping(self) -> "Laps":
        return Laps.trim_overlapping_laps(self)[0]

    def to_dataframe(self):
        return self._data

    def get_lap_flat_indicies(self, lap_id):
        return self._data.loc[lap_id, ['start_spike_index', 'end_spike_index']].to_numpy()

    def get_lap_times(self, lap_id):
        return self._data.loc[lap_id, ['start', 'stop']].to_numpy()
    
    def as_epoch_obj(self):
        """ Converts into a core.Epoch object containing the time periods """
        return Epoch(self.to_dataframe())

    @staticmethod
    def from_dict(d: dict):
        return Laps((d.get('_df', None) or d.get('_data', None)), metadata = d.get('metadata', None))
        # return Laps(d['_data'], metadata = d.get('metadata', None))
        
    def to_dict(self):
        return self.__dict__
    
    #TODO: #WM: Test this, it's not done! It should filter out the laps that occur outside of the start/end times that 
    def time_slice(self, t_start=None, t_stop=None):
        # raise NotImplementedError
        laps_obj = deepcopy(self)
        included_indicies = (((laps_obj._data.start >= t_start) & (laps_obj._data.start <= t_stop)) & ((laps_obj._data.stop >= t_start) & (laps_obj._data.stop <= t_stop)))
        included_df = laps_obj._data[included_indicies]
        # included_df = laps_obj._data[((laps_obj._data.start >= t_start) & (laps_obj._data.start <= t_stop))]
        return Laps(included_df, metadata=laps_obj.metadata)
        

     ## For serialization/pickling:
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        # del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # for backwards compatibility with pre-Epoch baseclass versions of Laps loaded from pickle
        if '_df' not in state:
            assert '_data' in state
            state['_df'] = state.pop('_data', None)

        self.__dict__.update(state)
            

    # ==================================================================================================================== #
    # Class Methods                                                                                                        #
    # ==================================================================================================================== #

    @classmethod
    def trim_overlapping_laps(cls, global_laps: "Laps", debug_print=False) -> "Laps":
        """ 2023-10-27 9pm - trims overlaps by removing the overlap from the even_global_laps (assuming that even first... hmmm that might be a problem? No, because even is always first because it refers to the 0 index lap_id.

        Avoids major issues introduced by Portion library by first splitting into odd/even (adjacent epochs) and only considering the overlap between the adjacent ones.
        Then it gets the indicies of the ones that changed so it can manually update the stop times for those epochs only on the even epochs so the other column's data isn't lost like it is in the portion/Epoch methods.

        ## SHOOT: changing this will change the other computed numbers!! 

        Modifies: ['end_t_rel_seconds', 'stop', 'duration']

        Invalidates: ['start_position_index', 'end_position_index', 'start_spike_index', 'end_spike_index', 'num_spikes']


        TODO 2023-10-27 - could refactor to parent Epochs class?

        """
        safe_trim_delta: float = 10.0 * (1.0/30.0) # 10 samples assuming 30Hz sampling of position data. This doesn't matter but just not to introduce artefacts.

        even_global_laps = Laps(global_laps._df[global_laps._df.lap_dir == 0])
        odd_global_laps = Laps(global_laps._df[global_laps._df.lap_dir == 1])

        even_global_laps_epochs = even_global_laps.as_epoch_obj()
        odd_global_laps_epochs = odd_global_laps.as_epoch_obj()

        even_laps_portion = even_global_laps.to_PortionInterval()
        odd_laps_portion = odd_global_laps.to_PortionInterval()

        even_global_laps_df = even_global_laps_epochs.to_dataframe()
        odd_global_laps_df = odd_global_laps_epochs.to_dataframe()
        
        ## Find the stop times in the even_laps
        # Get whichever starts earlier
        # sets: global_laps_end_change_indicies, desired_stops
        if (even_global_laps_df.epochs.t_start < odd_global_laps_df.epochs.t_start):
            assert even_global_laps_df.epochs.t_start < odd_global_laps_df.epochs.t_start, f"even laps should start before odd laps. even_global_laps_df.epochs.t_start: {even_global_laps_df.epochs.t_start}, odd_global_laps_df.epochs.t_start: {odd_global_laps_df.epochs.t_start}"
            intersecting_portion = even_laps_portion.intersection(odd_laps_portion)
            intersecting_epochs = Epoch.from_PortionInterval(intersecting_portion)
            n_epochs_with_changes = intersecting_epochs.n_epochs
            even_epochs_with_changes = np.where(np.isin(even_global_laps_df.stop, intersecting_epochs.stops)) # recovers the indicies of the epochs that changed, index into the even array, e.g. (array([33, 37], dtype=int64),)
            global_laps_end_change_indicies = even_global_laps._df.index[even_epochs_with_changes] # get the original indicies for use in the non even/odd split laps object, e.g. Int64Index([66, 74], dtype='int64')

            ## Replace the lap's current stop with the start of the intersection less a little wiggle room:
            desired_stops = intersecting_epochs.starts - safe_trim_delta
        else:
            # even_global_laps_df.epochs.t_start < odd_global_laps_df.epochs.t_start
            # opposite: (odd first)
            intersecting_portion = odd_laps_portion.intersection(even_laps_portion)
            intersecting_epochs = Epoch.from_PortionInterval(intersecting_portion)
            n_epochs_with_changes = intersecting_epochs.n_epochs

            odd_epochs_with_changes = np.where(np.isin(odd_global_laps_df.stop, intersecting_epochs.stops)) # recovers the indicies of the epochs that changed, index into the even array, e.g. (array([33, 37], dtype=int64),)
            global_laps_end_change_indicies = odd_global_laps._df.index[odd_epochs_with_changes] # get the original indicies for use in the non even/odd split laps object, e.g. Int64Index([66, 74], dtype='int64')

            ## Replace the lap's current stop with the start of the intersection less a little wiggle room:
            desired_stops = intersecting_epochs.starts - safe_trim_delta

        if debug_print:
            print(f'intersecting_epochs: {intersecting_epochs}')            

        # prev-values:
        # 66    1901.413540
        # 74    2002.917111
        # Name: stop, dtype: float64
        backup_values = global_laps._df.loc[global_laps_end_change_indicies, 'stop']
        if debug_print:
            print(f'backup_values: {backup_values}')
            print(f'desired_stops: {desired_stops}')
        global_laps._df.loc[global_laps_end_change_indicies, 'stop'] = desired_stops
        # Recompute duration
        global_laps._df.loc[global_laps_end_change_indicies, 'duration'] = global_laps._df.loc[global_laps_end_change_indicies, 'stop'] - global_laps._df.loc[global_laps_end_change_indicies, 'start']
        global_laps._df.loc[global_laps_end_change_indicies, 'end_t_rel_seconds'] = global_laps._df.loc[global_laps_end_change_indicies, 'stop'] # copy the new time to 'end_t_rel_seconds'

        print("WARN: .trim_overlapping_laps(...): need to recompute  ['start_position_index', 'end_position_index', 'start_spike_index', 'end_spike_index', 'num_spikes'] for the laps after calling self.trim_overlapping_laps()!")
        return global_laps, global_laps_end_change_indicies


    @classmethod
    def ensure_valid_laps_epochs_df(cls, original_laps_epoch_df: pd.DataFrame, rebuild_lap_id_columns=True) -> pd.DataFrame:
        """ De-duplicates, sorts, and filters by duration any potential laps
        
        laps_epoch_obj: Epoch = deepcopy(global_laps).as_epoch_obj()
        original_laps_epoch_df = laps_epoch_obj.to_dataframe()        
        filtered_laps_epoch_df = cls.ensure_valid_laps_epochs_df(original_laps_epoch_df)

        """
        # Drop duplicate rows in columns: 'start', 'stop'
        laps_epoch_df: pd.DataFrame = deepcopy(original_laps_epoch_df) #laps_epoch_obj.to_dataframe()

        # laps_epoch_df = deduplicate_epochs(laps_epoch_df) # this breaks things!
        # Filter rows based on column: 'duration'
        laps_epoch_df = laps_epoch_df[laps_epoch_df['duration'] > 1]
        # Filter rows based on column: 'duration'
        laps_epoch_df = laps_epoch_df[laps_epoch_df['duration'] <= 30]
        # # Drop duplicate rows in columns: 'start', 'stop'
        laps_epoch_df = laps_epoch_df.drop_duplicates(subset=['start', 'stop'])
        # Sort by column: 'start' (ascending)
        laps_epoch_df = laps_epoch_df.sort_values(['start']).reset_index(drop=True)
        ## Rebuild lap_id and label column:
        if rebuild_lap_id_columns:
            laps_epoch_df['lap_id'] = (laps_epoch_df.index + 1)
            laps_epoch_df['label'] = laps_epoch_df['lap_id']

        return laps_epoch_df # Epoch(laps_epoch_df, metadata=laps_epoch_obj.metadata)


    @classmethod
    def _compute_lap_dir_from_smoothed_velocity(cls, laps_df: pd.DataFrame, global_session: Union[Position, DataSession], replace_existing:bool=True) -> pd.DataFrame:
        """ 2024-01-17 - uses the smoothed velocity to determine the proper lap direction

        Adds Columns to laps_df: ['is_LR_dir']
        
        for LR_dir, values become more positive with time

        global_session = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name])
        global_laps = compute_lap_dir_from_smoothed_velocity(global_session)
        global_laps

        """
        from neuropy.core.position import Position
        
        n_laps = np.shape(laps_df)[0]
        if isinstance(global_session, Position):
            global_pos = global_session # passed variable is already a Position object
        else:
            # passed variable is hopefully a DataSession: Extract the position from the passed in session.
            global_pos = global_session.position
            
        global_pos.compute_higher_order_derivatives()
        global_pos.compute_smoothed_position_info()
        
        pos_df: pd.DataFrame = global_pos.to_dataframe()
        pos_df['lap'].unique()

        # Filter rows based on column: 'lap'
        pos_df = pos_df[pos_df['lap'].notna()]
        # Performed 1 aggregation grouped on column: 'lap'
        is_LR_dir = ((pos_df.groupby(['lap']).agg(speed_mean=('velocity_x_smooth', 'mean'))).reset_index()['speed_mean'] > 0.0).to_numpy() # increasing values => LR_dir
        laps_df['is_LR_dir'] = is_LR_dir # ValueError: Length of values (80) does not match length of index (82)
        # global_laps._df['direction_consistency'] = 0.0
        # assert np.all(laps_df[(laps_df['is_LR_dir'].astype(int) == np.logical_not(laps_df['lap_dir'].astype(int)))])
        is_new_dir_substantially_different: bool = not np.all(laps_df[(laps_df['is_LR_dir'].astype(int) == np.logical_not(laps_df['lap_dir'].astype(int)))])
        if (is_new_dir_substantially_different):
            print(f'WARN: Laps._compute_lap_dir_from_smoothed_velocity(...): the velocity-determined lap direction ("is_LR_dir") substantially differs from the previous ("lap_dir") column. This might be because it initially used simple ODD/EVEN determination for the direction.')
            if replace_existing:
                ## Overwrite the "lap_dir" column with the new value
                print(f'\tWARN: overwriting the "lap_dir" column of Laps with the "is_LR_dir" column. Do things need to be recomputed after this?')
                laps_df['lap_dir'] = np.logical_not(laps_df['is_LR_dir'].astype(int) > 0) # I think this should be the proper lap_dir format
            else:
                print(f'\tlap_dir substantially differs but replace_existing=False, so not updating.')

        return laps_df
    

    @classmethod
    def _update_dataframe_computed_vars(cls, laps_df: pd.DataFrame,
                         t_start:Optional[float]=None, t_delta:Optional[float]=None, t_end:Optional[float]=None, # for adding_maze_id_if_needed
                         global_session: Optional[Union[Position, DataSession]]=None, # for _compute_lap_dir_from_smoothed_velocity
                         replace_existing:bool=True):
        # laps_df[['lap_id','maze_id','start_spike_index', 'end_spike_index']] = laps_df[['lap_id','maze_id','start_spike_index', 'end_spike_index']].astype('int')
        laps_df[['lap_id']] = laps_df[['lap_id']].astype('int')

        if ((t_start is not None) and (t_delta is not None) and (t_end is not None)):
            # computes 'track_id' from t_start, t_delta, and t_end where t_delta corresponds to the transition point (track change).
            laps_df = ensure_dataframe(laps_df).epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end, replace_existing=True, labels_column_name='lap_id')

        if 'maze_id' in laps_df.columns:
            laps_df[['maze_id']] = laps_df[['maze_id']].astype('int')
        if set(['start_spike_index','end_spike_index']).issubset(laps_df.columns):
            laps_df[['start_spike_index', 'end_spike_index']] = laps_df[['start_spike_index', 'end_spike_index']].astype('int')
            laps_df['num_spikes'] = laps_df['end_spike_index'] - laps_df['start_spike_index'] # builds 'num_spikes'
    
        if (('lap_dir' not in laps_df.columns) or ('is_LR_dir' not in laps_df.columns)):
            # compute the lap_dir if that field doesn't exist:
            if global_session is not None:
                ## computes proper 'is_LR_dir' and 'lap_dir' columns:
                laps_df = cls._compute_lap_dir_from_smoothed_velocity(laps_df=laps_df, global_session=global_session, replace_existing=True) # adds 'is_LR_dir'
                # if 'lap_dir' not in laps_df.columns:
                #     laps_df['lap_dir'] = laps_df['is_LR_dir']
            else:
                # No global_session or position passed, using old even/odd 'lap_dir' determination.            
                print(f"WARNING: No global_session or position passed, using old even/odd 'lap_dir' determination.")
                laps_df['lap_dir'] = np.full_like(laps_df['lap_id'].to_numpy(), -1)
                laps_df.loc[np.logical_not(np.isnan(laps_df.lap_id.to_numpy())), 'lap_dir'] = np.mod(laps_df.loc[np.logical_not(np.isnan(laps_df.lap_id.to_numpy())), 'lap_id'], 2)
                
        elif (replace_existing and (global_session is not None)):
            laps_df = cls._compute_lap_dir_from_smoothed_velocity(laps_df=laps_df, global_session=global_session, replace_existing=True) # adds 'is_LR_dir'
        else:
            pass

        # Either way, ensure that the lap_dir is an 'int' column.
        laps_df['lap_dir'] = laps_df['lap_dir'].astype('int')
        
        laps_df['label'] = laps_df['lap_id'].astype('str') # add the string "label" column
        return laps_df


    @classmethod
    def build_dataframe(cls, mat_file_loaded_dict: dict, time_variable_name='t_rel_seconds', absolute_start_timestamp=None,
                         t_start:Optional[float]=None, t_delta:Optional[float]=None, t_end:Optional[float]=None, # for _update_dataframe_maze_id_if_needed
                         global_session: Optional[Union[Position, DataSession]]=None, # for _compute_lap_dir_from_smoothed_velocity
                         ):
        laps_df = pd.DataFrame(mat_file_loaded_dict) # 1014937 rows × 11 columns
        laps_df = Laps._update_dataframe_computed_vars(laps_df, t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=global_session, replace_existing=True)
        
        # Build output Laps object to add to session
        print('setting laps object.')
        if time_variable_name == 't_seconds':
            t_variable_column_names = ['start_t_seconds', 'end_t_seconds']
            t_variable = laps_df[t_variable_column_names].to_numpy()
        elif time_variable_name == 't_rel_seconds':
            t_variable_column_names = ['start_t_seconds', 'end_t_seconds']
            t_variable = laps_df[t_variable_column_names].to_numpy()
            # need to subtract off the absolute start timestamp
            t_variable = t_variable - absolute_start_timestamp
            laps_df[['start_t_rel_seconds', 'end_t_rel_seconds']] = t_variable # assign the new variable
        else:
            t_variable_column_names = ['start_t', 'end_t']
            t_variable = laps_df[t_variable_column_names].to_numpy()
            
        # finally assign the 'start' and 'stop' time columns to the appropriate variable
        laps_df[['start','stop']] = t_variable # assign the active t_variable to the start & end columns of the DataFrame
        return laps_df


    @classmethod
    def from_estimated_laps(cls, pos_t_rel_seconds, desc_crossing_begining_idxs, desc_crossing_ending_idxs, asc_crossing_begining_idxs, asc_crossing_ending_idxs, debug_print=True):
        """Builds a Laps object from the output of the neuropy.analyses.laps.estimate_laps function.
        Args:
            pos_t_rel_seconds ([type]): [description]
            desc_crossing_beginings ([type]): [description]
            desc_crossing_endings ([type]): [description]
            asc_crossing_beginings ([type]): [description]
            asc_crossing_endings ([type]): [description]

        Usage:
        
            position_obj = sess.position.linear_pos_obj
            position_obj.compute_higher_order_derivatives()
            pos_df = position_obj.compute_smoothed_position_info(N=N) ## Smooth the velocity curve to apply meaningful logic to it
            pos_df: pd.DataFrame = position_obj.to_dataframe()
            # If the index doesn't start at zero, it will need to for compatibility with the lap splitting logic because it uses the labels via "df.loc"
            if 'index_backup' not in pos_df.columns:
                pos_df['index_backup'] = pos_df.index  # Backup the current index to a new column
            # Drop rows with missing data in columns: 't', 'velocity_x_smooth' and 2 other columns. This occurs from smoothing
            pos_df = pos_df.dropna(subset=['t', 'x', 'x_smooth', 'velocity_x_smooth', 'acceleration_x_smooth'])    
            pos_df.reset_index(drop=True, inplace=True) # Either way, reset the index
            lap_change_indicies = _subfn_perform_estimate_lap_splits_1D(pos_df, hardcoded_track_midpoint_x=None, debug_print=debug_print) # allow smart midpoint determiniation
            (desc_crossing_begining_idxs, desc_crossing_midpoint_idxs, desc_crossing_ending_idxs), (asc_crossing_begining_idxs, asc_crossing_midpoint_idxs, asc_crossing_ending_idxs) = lap_change_indicies    
            custom_test_laps_obj = Laps.from_estimated_laps(pos_df['t'].to_numpy(), desc_crossing_begining_idxs, desc_crossing_ending_idxs, asc_crossing_begining_idxs, asc_crossing_ending_idxs) ## Get the timestamps corresponding to the indicies
            assert custom_test_laps_obj.n_laps > 0, f"estimation for {sess} produced no laps!"
            
        Returns:
            [type]: [description]
        """
        ## Build a custom Laps dataframe from the found points:
        ### Note that these crossing_* indicies are for the position dataframe, not the spikes_df (which is what the previous Laps object was computed from).
            # This means we don't have 'start_spike_index' or 'end_spike_index', and we'd have to compute them if we want them.
        custom_test_laps_df = pd.DataFrame({
            'start_position_index': np.concatenate([desc_crossing_begining_idxs, asc_crossing_begining_idxs]),
            'end_position_index': np.concatenate([desc_crossing_ending_idxs, asc_crossing_ending_idxs]),
            'lap_dir': np.concatenate([np.zeros_like(desc_crossing_begining_idxs), np.ones_like(asc_crossing_begining_idxs)])
        })


        # IMPORTANT 2023-10-27 - iterate through the pairs and insure no overlap between indicies, to prevent needing to fix the times manually later:
        custom_test_laps_df = custom_test_laps_df.sort_values(by=['start_position_index']).reset_index(drop=True) # sorts all values in ascending order

        prev_index = None
        prev_end_value = None
        indicies_to_change = {}

        for index, row in custom_test_laps_df.iterrows():
            # print(row['start_position_index'], row['end_position_index'])
            if prev_end_value is not None:
                if prev_end_value > row['start_position_index']:
                    # print(f'overlap at {index}, row: {row}')
                    indicies_to_change[prev_index] = (row['start_position_index'] - 1) # subtract one from this epoch's the start index to use the correct end index for the last epoch
            prev_end_value = row['end_position_index']
            prev_index = index

        ## Make changes:
        if len(indicies_to_change) > 0:
            if debug_print:
                print(f'WARN: Laps.from_estimated_laps(...) found {len(indicies_to_change)} indicies to change:\n\tindicies_to_change: {indicies_to_change}')
        for an_index, a_new_end_pos_index in indicies_to_change.items():
            custom_test_laps_df.loc[an_index, 'end_position_index'] = a_new_end_pos_index
            # print(f'changed row[{an_index}]')


        # Get start/end times from the indicies
        # custom_test_laps_df['start_t_rel_seconds'] = np.concatenate([pos_t_rel_seconds[desc_crossing_begining_idxs], pos_t_rel_seconds[asc_crossing_begining_idxs]])
        # custom_test_laps_df['end_t_rel_seconds'] = np.concatenate([pos_t_rel_seconds[desc_crossing_ending_idxs], pos_t_rel_seconds[asc_crossing_ending_idxs]])
        custom_test_laps_df['start_t_rel_seconds'] = np.array([pos_t_rel_seconds[an_idx] for an_idx in custom_test_laps_df['start_position_index'].to_numpy()])
        custom_test_laps_df['end_t_rel_seconds'] = np.array([pos_t_rel_seconds[an_idx] for an_idx in custom_test_laps_df['end_position_index'].to_numpy()])
        custom_test_laps_df['start'] = custom_test_laps_df['start_t_rel_seconds']
        custom_test_laps_df['stop'] = custom_test_laps_df['end_t_rel_seconds']
        # Sort the laps based on the start time, reset the index, and finally assign lap_id's from the sorted laps
        custom_test_laps_df = custom_test_laps_df.sort_values(by=['start']).reset_index(drop=True) # sorts all values in ascending order
        custom_test_laps_df['lap_id'] = (custom_test_laps_df.index + 1) # set the lap_id column to the index starting at 1

        custom_test_laps_df['label'] = custom_test_laps_df['lap_id'] # add the label column required by Epoch
        return Laps(custom_test_laps_df).filter_to_valid() ## TODO: instead of `.filter_to_valid().trimmed_to_non_overlapping()` couldn't we just fix the indicies above (or better yet whatever is causing them to be wrong)?

    @classmethod
    def from_dataframe(cls, df):
        return cls(df)
