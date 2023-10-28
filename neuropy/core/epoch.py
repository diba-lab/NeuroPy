from importlib import metadata
import warnings
from warnings import warn
import numpy as np
import pandas as pd
import portion as P # Required for interval search: portion~=2.3.0

from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable, DataFrameInitializable
from .datawriter import DataWriter
from neuropy.utils.mixins.print_helpers import SimplePrintable, OrderedMeta
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicedMixin, TimeColumnAliasesProtocol
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs, deduplicate_epochs # for EpochsAccessor's .get_non_overlapping_df()
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin


class NamedTimerange(SimplePrintable, metaclass=OrderedMeta):
    """ A simple named period of time with a known start and end time """
    def __init__(self, name, start_end_times):
        self.name = name
        self.start_end_times = start_end_times
        
    @property
    def t_start(self):
        return self.start_end_times[0]
    
    @t_start.setter
    def t_start(self, t):
        self.start_end_times[0] = t

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def t_stop(self):
        return self.start_end_times[1]
    
    @t_stop.setter
    def t_stop(self, t):
        self.start_end_times[1] = t
    
    
    def to_Epoch(self):
        return Epoch(pd.DataFrame({'start': [self.t_start], 'stop': [self.t_stop], 'label':[self.name]}))
        

@pd.api.extensions.register_dataframe_accessor("epochs")
class EpochsAccessor(TimeColumnAliasesProtocol, TimeSlicedMixin, StartStopTimesMixin, TimeSlicableObjectProtocol):
    """ A Pandas pd.DataFrame representation of [start, stop, label] epoch intervals """
    
    _time_column_name_synonyms = {"start":{'begin','start_t'},
            "stop":['end','stop_t'],
            "label":['name', 'id', 'flat_replay_idx','lap_id']
        }
    

    _required_column_names = ['start', 'stop', 'label', 'duration']

    def __init__(self, pandas_obj):
        pandas_obj = self.renaming_synonym_columns_if_needed(pandas_obj, required_columns_synonym_dict=self._time_column_name_synonyms) 
        pandas_obj = self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj = self._obj.sort_values(by=["start"]) # sorts all values in ascending order
        # Optional: If the 'label' column of the dataframe is empty, should populate it with the index (after sorting) as a string.
        # self._obj['label'] = self._obj.index
        self._obj["label"] = self._obj["label"].astype("str")
        # Optional: Add 'duration' column:
        self._obj["duration"] = self._obj["stop"] - self._obj["start"]
        # Optional: check for and remove overlaps

    @classmethod
    def _validate(cls, obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('neuron_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """       
        return obj # important! Must return the modified obj to be assigned (since its columns were altered by renaming

    @property
    def is_valid(self):
        """ The dataframe is valid (because it passed _validate(...) in __init__(...) so just return True."""
        return True

    @property
    def starts(self):
        return self._obj.start.values

    @property
    def stops(self):
        return self._obj.stop.values
    
    @property
    def t_start(self):
        return self.starts[0]
    @t_start.setter
    def t_start(self, t):
        include_indicies = np.argwhere(t < self.stops)
        if (np.size(include_indicies) == 0):
            # this proposed t_start is after any contained epochs, so the returned object would be empty
            print('Error: this proposed t_start ({}) is after any contained epochs, so the returned object would be empty'.format(t))
            raise ValueError
        first_include_index = include_indicies[0]
        
        if (first_include_index > 0):
            # drop the epochs preceeding the first_include_index:
            drop_indicies = np.arange(first_include_index)
            print('drop_indicies: {}'.format(drop_indicies))
            raise NotImplementedError # doesn't yet drop the indicies before the first_include_index
        self._obj.loc[first_include_index, ('start')] = t # exclude the first short period where the animal isn't on the maze yet

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def t_stop(self):
        return self.stops[-1]

    @property
    def durations(self):
        return self.stops - self.starts

    @property
    def n_epochs(self):
        return len(self.starts)

    @property
    def labels(self):
        return self._obj.label.values


    @property
    def extra_data_column_names(self):
        """Any additional columns in the dataframe beyond those that exist by default. """
        return list(set(self._obj.columns) - set(self._required_column_names))
        
    @property
    def extra_data_dataframe(self) -> pd.DataFrame:
        """The subset of the dataframe containing additional information in its columns beyond that what is required. """
        return self._obj[self.extra_data_column_names]

    def as_array(self):
        return self._obj[["start", "stop"]].to_numpy()

    def get_unique_labels(self):
        return np.unique(self.labels)

    def get_start_stop_tuples_list(self):
        """ returns a list of (start, stop) tuples. """
        return list(zip(self.starts, self.stops))

    def get_valid_df(self):
        """ gets a validated copy of the dataframe. Looks better than doing `epochs_df.epochs._obj` """
        return self._obj.copy()

    ## Handling overlapping
    def get_non_overlapping_df(self, debug_print=False) -> pd.DataFrame:
        """ Returns a dataframe with overlapping epochs removed. """
        ## 2023-02-23 - PortionInterval approach to ensuring uniqueness:
        from neuropy.utils.efficient_interval_search import convert_PortionInterval_to_epochs_df, _convert_start_end_tuples_list_to_PortionInterval
        ## Capture dataframe properties beyond just the start/stop times:
        
        # _intermedia_start_end_tuples_list = self.get_start_stop_tuples_list()
        _intermediate_portions_interval: P.Interval = _convert_start_end_tuples_list_to_PortionInterval(zip(self.starts, self.stops))
        filtered_epochs_df = convert_PortionInterval_to_epochs_df(_intermediate_portions_interval)
        # is_epoch_included = np.array([(a_tuple.start, a_tuple.stop) in _intermedia_start_end_tuples_list for a_tuple in list(filtered_epochs_df.itertuples(index=False))])

        
        if debug_print:
            before_num_rows = self.n_epochs
            filtered_epochs_df = convert_PortionInterval_to_epochs_df(_intermediate_portions_interval)
            after_num_rows = np.shape(filtered_epochs_df)[0]
            changed_num_rows = after_num_rows - before_num_rows
            print(f'Dataframe Changed from {before_num_rows} -> {after_num_rows} ({changed_num_rows = })')
            return filtered_epochs_df
        else:
            
            
            return filtered_epochs_df



    def get_epochs_longer_than(self, minimum_duration, debug_print=False):
        """ returns a copy of the dataframe contining only epochs longer than the specified minimum_duration. """
        active_filter_epochs = self.get_valid_df()
        if debug_print:
            before_num_rows = np.shape(active_filter_epochs)[0]
        if 'duration' not in active_filter_epochs.columns:
            active_filter_epochs['duration'] = active_filter_epochs['stop'] - active_filter_epochs['start']
        if debug_print:
            filtered_epochs = active_filter_epochs[active_filter_epochs['duration'] >= minimum_duration]
            after_num_rows = np.shape(filtered_epochs)[0]
            changed_num_rows = after_num_rows - before_num_rows
            print(f'Dataframe Changed from {before_num_rows} -> {after_num_rows} ({changed_num_rows = })')
            return filtered_epochs
        else:
            return active_filter_epochs[active_filter_epochs['duration'] >= minimum_duration]

    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop):
        # TODO time_slice should also include partial epochs falling in between the timepoints
        df = self._obj.copy() 
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        df = df[(df["start"] >= t_start) & (df["start"] <= t_stop)].reset_index(drop=True)
        return df
        
    def label_slice(self, label):
        if isinstance(label, (list, np.ndarray)):
            df = self._obj[np.isin(self._obj["label"], label)].reset_index(drop=True)
        else:
            assert isinstance(label, str), "label must be string"
            df = self._obj[self._obj["label"] == label].reset_index(drop=True)
        return df

    def filtered_by_duration(self, min_duration=None, max_duration=None):
        return self._obj[(self.durations >= (min_duration or 0.0)) & (self.durations <= (max_duration or np.inf))].reset_index(drop=True)
        
    # Requires Optional `portion` library
    # import portion as P # Required for interval search: portion~=2.3.0
    @classmethod
    def from_PortionInterval(cls, portion_interval):
        from neuropy.utils.efficient_interval_search import convert_PortionInterval_to_epochs_df
        return convert_PortionInterval_to_epochs_df(portion_interval)

    def to_PortionInterval(self):
        from neuropy.utils.efficient_interval_search import _convert_start_end_tuples_list_to_PortionInterval
        return _convert_start_end_tuples_list_to_PortionInterval(zip(self.starts, self.stops))


class Epoch(HDFMixin, StartStopTimesMixin, TimeSlicableObjectProtocol, DataFrameRepresentable, DataFrameInitializable, DataWriter):
    def __init__(self, epochs: pd.DataFrame, metadata=None) -> None:
        """[summary]
        Args:
            epochs (pd.DataFrame): Each column is a pd.Series(["start", "stop", "label"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        super().__init__(metadata=metadata)
        self._df = epochs.epochs.get_valid_df() # gets already sorted appropriately and everything
        self._check_epochs(epochs) # check anyway

    @property
    def starts(self):
        return self._df.epochs.starts

    @property
    def stops(self):
        return self._df.epochs.stops
    
    @property
    def t_start(self):
        return self.starts[0]
    @t_start.setter
    def t_start(self, t):
        self._df.epochs.t_start = t

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def t_stop(self):
        return self.stops[-1]

    @property
    def durations(self):
        return self.stops - self.starts

    @property
    def n_epochs(self):
        return self._df.epochs.n_epochs
    @property
    def labels(self):
        return self._df.epochs.labels

    def get_unique_labels(self):
        return np.unique(self.labels)
    
    def get_named_timerange(self, epoch_name):
        return NamedTimerange(name=epoch_name, start_end_times=self[epoch_name])


    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """metadata compatibility"""
        self._metadata = metadata

    def _check_epochs(self, epochs):
        assert isinstance(epochs, pd.DataFrame)
        # epochs.epochs.
        assert (
            pd.Series(["start", "stop", "label"]).isin(epochs.columns).all()
        ), "Epoch dataframe should at least have columns with names: start, stop, label"

    def __repr__(self) -> str:
        # return f"{len(self.starts)} epochs"
        return f"{len(self.starts)} epochs\n{self.as_array().__repr__()}\n"

    def _repr_pretty_(self, p, cycle=False):
        """ The cycle parameter will be true if the representation recurses - e.g. if you put a container inside itself. """
        # p.text(self.__repr__() if not cycle else '...')
        p.text(self.to_dataframe().__repr__() if not cycle else '...')

    def __str__(self) -> str:
        return f"{len(self.starts)} epochs\n{self.as_array().__repr__()}\n"
    
    def __len__(self):
        """ allows using `len(epochs_obj)` and getting the number of epochs. """
        return len(self.starts)

    def str_for_concise_display(self) -> str:
        """ returns a minimally descriptive string like: '60 epochs in (17.9, 524.1)' that doesn't print all the array elements only the number of epochs and the first and last. """
        return f"{len(self.starts)} epochs in ({self.starts[0]:.1f}, {self.stops[-1]:.1f})" # "60 epochs in (17.9, 524.1)"

    def str_for_filename(self) -> str:
        return f"Epoch[{len(self.starts)}]({self.starts[0]:.1f}-{self.stops[-1]:.1f})" #


    def __getitem__(self, slice_):
        if isinstance(slice_, str):
            indices = np.where(self.labels == slice_)[0]
            if len(indices) > 1:
                return np.vstack((self.starts[indices], self.stops[indices])).T
            else:
                return np.array([self.starts[indices], self.stops[indices]]).squeeze()
        else:
            return np.vstack((self.starts[slice_], self.stops[slice_])).T

    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop):
        return Epoch(epochs=self._df.epochs.time_slice(t_start, t_stop), metadata=self.metadata) # NOTE: drops metadata
        
    def label_slice(self, label):
        return Epoch(epochs=self._df.epochs.label_slice(label), metadata=self.metadata)

    def boolean_indicies_slice(self, boolean_indicies):
        return Epoch(epochs=self._df[boolean_indicies], metadata=self.metadata)

    def filtered_by_duration(self, min_duration=None, max_duration=None):
        return Epoch(epochs=self._df.epochs.filtered_by_duration(min_duration, max_duration), metadata=self.metadata)

    @classmethod
    def filter_epochs(cls, curr_epochs, pos_df:pd.DataFrame=None, spikes_df:pd.DataFrame=None, require_intersecting_epoch:"Epoch"=None, min_epoch_included_duration=0.06, max_epoch_included_duration=0.6,
        maximum_speed_thresh=2.0, min_inclusion_fr_active_thresh=2.0, min_num_unique_aclu_inclusions=3, debug_print=False) -> "Epoch":
        """filters the provided replay epochs by specified constraints.

        Args:
            curr_epochs (Epoch): the epochs to filter on
            min_epoch_included_duration (float, optional): all epochs shorter than min_epoch_included_duration will be excluded from analysis. Defaults to 0.06.
            max_epoch_included_duration (float, optional): all epochs longer than max_epoch_included_duration will be excluded from analysis. Defaults to 0.6.
            maximum_speed_thresh (float, optional): epochs are only included if the animal's interpolated speed (as determined from the session's position dataframe) is below the speed. Defaults to 2.0 [cm/sec].
            min_inclusion_fr_active_thresh: minimum firing rate (in Hz) for a unit to be considered "active" for inclusion.
            min_num_unique_aclu_inclusions: minimum number of unique active cells that must be included in an epoch to have it included.

            save_on_compute (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            Epoch: the filtered epochs as an Epoch object

        NOTE: this really is a general method that works for any Epoch object or Epoch-type dataframe to filter it.

        TODO 2023-04-11 - This really belongs in the Epoch class or the epoch dataframe accessor. 

        """
        from neuropy.utils.efficient_interval_search import filter_epochs_by_speed
        from neuropy.utils.efficient_interval_search import filter_epochs_by_num_active_units

        if not isinstance(curr_epochs, pd.DataFrame):
            curr_epochs = curr_epochs.to_dataframe() # .get_valid_df() # convert to pd.DataFrame to start
    
        assert isinstance(curr_epochs, pd.DataFrame), f'curr_replays must be a pd.DataFrame or Epoch object, but is {type(curr_epochs)}'
        # Ensure the dataframe representation has the required columns. TODO: is this needed?
        if not 'stop' in curr_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            curr_epochs['stop'] = curr_epochs['end'].copy()
        if not 'label' in curr_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            curr_epochs['label'] = curr_epochs['flat_replay_idx'].copy()
        # must convert back from pd.DataFrame to Epoch object to use the Epoch methods
        curr_epochs = cls(curr_epochs)

        ## Use the existing replay epochs from the session but ensure they look valid:

        ## Filter based on required overlap with Ripples:
        if require_intersecting_epoch is not None:
            curr_epochs = cls.from_PortionInterval(require_intersecting_epoch.to_PortionInterval().intersection(curr_epochs.to_PortionInterval()))
        else:
            curr_epochs = cls.from_PortionInterval(curr_epochs.to_PortionInterval()) # just do this to ensure non-overlapping

        if curr_epochs.n_epochs == 0:
            warn(f'curr_epochs already empty prior to any filtering')

        # Filter by duration bounds:
        curr_epochs = curr_epochs.filtered_by_duration(min_duration=min_epoch_included_duration, max_duration=max_epoch_included_duration)

        # Filter *_replays_Interval by requiring them to be below the speed:
        if maximum_speed_thresh is not None:
            assert pos_df is not None, "must provide pos_df if filtering by speed"
            if curr_epochs.n_epochs > 0:
                curr_epochs, above_speed_threshold_intervals, below_speed_threshold_intervals = filter_epochs_by_speed(pos_df, curr_epochs, speed_thresh=maximum_speed_thresh, debug_print=debug_print)
            else:
                warn(f'curr_epochs already empty prior to filtering by speed')

        # 2023-02-10 - Trimming and Filtering Estimated Replay Epochs based on cell activity and pyramidal cell start/end times:
        if (min_inclusion_fr_active_thresh is not None) or (min_num_unique_aclu_inclusions is not None):
            assert spikes_df is not None, "must provide spikes_df if filtering by active units"
            active_spikes_df = spikes_df.spikes.sliced_by_neuron_type('pyr') # trim based on pyramidal cell activity only
            if curr_epochs.n_epochs > 0:
                curr_epochs, _extra_outputs = filter_epochs_by_num_active_units(active_spikes_df, curr_epochs, min_inclusion_fr_active_thresh=min_inclusion_fr_active_thresh, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions, include_intermediate_computations=False) # TODO: seems wasteful considering we compute all these spikes_df metrics and refinements and then don't return them.
            else:
                warn(f'curr_epochs already empty prior to filtering by firing rate or minimum active units')
                
        return curr_epochs

    def to_dict(self, recurrsively=False):
        d = {"epochs": self._df, "metadata": self._metadata}
        return d
    
    @staticmethod
    def from_dict(d: dict):
        return Epoch(d["epochs"], metadata=d["metadata"])

    ## TODO: refactor these methods into the 'epochs' pd.DataFrame accessor above and then wrap them:
    def fill_blank(self, method="from_left"):
        ep_starts = self.epochs["start"].values
        ep_stops = self.epochs["stop"].values
        ep_durations = self.epochs["duration"].values
        ep_labels = self.epochs["label"].values

        mask = (ep_starts[:-1] + ep_durations[:-1]) < ep_starts[1:]
        (inds,) = np.nonzero(mask)

        if method == "from_left":
            for ind in inds:
                ep_durations[ind] = ep_starts[ind + 1] - ep_starts[ind]

        elif method == "from_right":
            for ind in inds:
                gap = ep_starts[ind + 1] - (ep_starts[ind] + ep_durations[ind])
                ep_starts[ind + 1] -= gap
                ep_durations[ind + 1] += gap

        elif method == "from_nearest":
            for ind in inds:
                gap = ep_starts[ind + 1] - (ep_starts[ind] + ep_durations[ind])
                ep_durations[ind] += gap / 2.0
                ep_starts[ind + 1] -= gap / 2.0
                ep_durations[ind + 1] += gap / 2.0

        self.epochs["start"] = ep_starts
        self.epochs["stop"] = ep_starts + ep_durations
        self.epochs["duration"] = ep_durations

    def delete_in_between(self, t1, t2):
        epochs_df = self.to_dataframe()[["start", "stop", "label"]]
        # delete epochs if they are within t1, t2
        epochs_df = epochs_df[~((epochs_df["start"] >= t1) & (epochs_df["stop"] <= t2))]

        # truncate stop if start is less than t1 but stop is within t1,t2
        epochs_df.loc[
            (epochs_df["start"] < t1)
            & (t1 < epochs_df["stop"])
            & (epochs_df["stop"] <= t2),
            "stop",
        ] = t1

        # truncate start if stop is greater than t2 but start is within t1,t2
        epochs_df.loc[
            (epochs_df["start"] > t1)
            & (epochs_df["start"] <= t2)
            & (epochs_df["stop"] > t2),
            "start",
        ] = t2

        # if epoch starts before and ends after range,
        flank_start = epochs_df[
            (epochs_df["start"] < t1) & (epochs_df["stop"] > t2)
        ].copy()
        flank_start["stop"] = t1
        flank_stop = epochs_df[
            (epochs_df["start"] < t1) & (epochs_df["stop"] > t2)
        ].copy()
        flank_stop["start"] = t2
        epochs_df = epochs_df[~((epochs_df["start"] < t1) & (epochs_df["stop"] > t2))]
        epochs_df = epochs_df.append(flank_start)
        epochs_df = epochs_df.append(flank_stop)
        epochs_df = epochs_df.reset_index(drop=True)

        return Epoch(epochs_df)

    def get_proportion_by_label(self, t_start=None, t_stop=None):
        if t_start is None:
            t_start = self.starts[0]
        if t_stop is None:
            t_stop = self.stops[-1]

        duration = t_stop - t_start

        ep = self._df.copy()
        ep = ep[(ep.stop > t_start) & (ep.start < t_stop)].reset_index(drop=True)

        if ep["start"].iloc[0] < t_start:
            ep.at[0, "start"] = t_start

        if ep["stop"].iloc[-1] > t_stop:
            ep.at[ep.index[-1], "stop"] = t_stop

        ep["duration"] = ep.stop - ep.start

        ep_group = ep.groupby("label").sum().duration / duration

        label_proportion = {}
        for label in self.get_unique_labels():
            label_proportion[label] = 0.0

        for state in ep_group.index.values:
            label_proportion[state] = ep_group[state]

        return label_proportion

    def count(self, t_start=None, t_stop=None, binsize=300):
        if t_start is None:
            t_start = 0

        if t_stop is None:
            t_stop = np.max(self.stops)

        mid_times = self.starts + self.durations / 2
        bins = np.arange(t_start, t_stop + binsize, binsize)
        return np.histogram(mid_times, bins=bins)[0]

    def to_neuroscope(self, ext="PHO"):
        assert self.filename is not None
        out_filepath = self.filename.with_suffix(f".{ext}.evt")
        with out_filepath.open("w") as a:
            for event in self._df.itertuples():
                a.write(f"{event.start*1000} start\n{event.stop*1000} end\n")
        return out_filepath

    def as_array(self):
        return self.to_dataframe()[["start", "stop"]].to_numpy()

    # Requires Optional `portion` library
    @classmethod
    def from_PortionInterval(cls, portion_interval, metadata=None):
        return Epoch(epochs=EpochsAccessor.from_PortionInterval(portion_interval), metadata=metadata) 

    def to_PortionInterval(self):
        return self._df.epochs.to_PortionInterval()

    def get_non_overlapping(self, debug_print=False):
        """ Returns a copy with overlapping epochs removed. """
        return Epoch(epochs=self._df.epochs.get_non_overlapping_df(debug_print=debug_print), metadata=self.metadata)
    

    # HDF5 Serialization _________________________________________________________________________________________________ #
    # HDFMixin Conformances ______________________________________________________________________________________________ #

    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pos_obj: Position = long_one_step_decoder_1D.pf.position
            _pos_obj.to_hdf(hdf5_output_path, key='pos')
        """
        _df = self.to_dataframe()
        _df.to_hdf(path_or_buf=file_path, key=key, format=kwargs.pop('format', 'table'), data_columns=kwargs.pop('data_columns',True), **kwargs)
        return
    
        # # create_group
        # a_key = Path(key)
        # with tb.open_file(file_path, mode='r+') as f:
        #     # group = f.create_group(str(a_key.parent), a_key.name, title='epochs.', createparents=True)
        #     group = f.get_node(str(a_key.parent))
        #     # group = f[key]
        #     table = f.create_table(group, a_key.name, EpochTable, "Epochs")
        #     # Serialization
        #     for i, t_start, t_stop, a_label in zip(np.arange(self.n_epochs), self.starts, self.stops, self.labels):
        #         row = table.row
        #         row['t_start'] = t_start
        #         row['t_end'] = t_stop  # Provide an appropriate session identifier here
        #         row['label'] = str(a_label)
        #         row.append()
                
        #     table.flush()
        #     # Metadata:
        #     group.attrs['t_start'] = self.t_start
        #     group.attrs['t_stop'] = self.t_stop
        #     group.attrs['n_epochs'] = self.n_epochs

    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> "Epoch":
        """  Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pos_obj = Epoch.read_hdf(hdf5_output_path, key='pos')
            _reread_pos_obj
        """
        _df = pd.read_hdf(file_path, key=key, **kwargs)
        return cls(_df, metadata=None) # TODO: recover metadata


    # DataFrameInitializable Conformances ________________________________________________________________________________ #
    
    def to_dataframe(self):
        df = self._df.copy()
        return df
    
    @classmethod
    def from_dataframe(cls, df):
        return cls(df)
