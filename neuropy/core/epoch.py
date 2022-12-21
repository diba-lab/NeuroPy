from importlib import metadata
import numpy as np
import pandas as pd
from .datawriter import DataWriter
from neuropy.utils.mixins.print_helpers import SimplePrintable, OrderedMeta
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicedMixin


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
class EpochsAccessor(TimeSlicedMixin, StartStopTimesMixin, TimeSlicableObjectProtocol):
    """ A Pandas pd.DataFrame representation of [start, stop, label] epoch intervals """
    
    _column_name_synonyms = {"start":{'begin','start_t'},
            "stop":['end','stop_t'],
            "label":['name', 'id', 'flat_replay_idx']
        }

    def __init__(self, pandas_obj):
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
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('cell_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """       
        if "start" not in obj.columns:
            # try to rename based on synonyms
            for a_synonym in cls._column_name_synonyms["start"]:
                if a_synonym in obj.columns:
                    obj = obj.rename({a_synonym: "start"}, axis="columns") # rename the synonym column to "start"
                    # obj["start"] = obj[a_synonym].copy()
            ## must be in there by the time that you're done.
            if "start" not in obj.columns:
                raise AttributeError("Must have unit id column 'start' column.")
        if "stop" not in obj.columns:
            # try to rename based on synonyms
            for a_synonym in cls._column_name_synonyms["stop"]:
                if a_synonym in obj.columns:
                    obj = obj.rename({a_synonym: "stop"}, axis="columns") # rename the synonym column to "stop"
            ## must be in there by the time that you're done.
            if "stop" not in obj.columns:
                raise AttributeError("Must have unit id column 'stop' column.")
        if "label" not in obj.columns:
            # try to rename based on synonyms
            for a_synonym in cls._column_name_synonyms["label"]:
                if a_synonym in obj.columns:
                    obj = obj.rename({a_synonym: "label"}, axis="columns") # rename the synonym column to "label"
            ## must be in there by the time that you're done.
            if "label" not in obj.columns:
                raise AttributeError("Must have unit id column 'label' column.")
        
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

    def as_array(self):
        return self._obj[["start", "stop"]].to_numpy()

    def get_unique_labels(self):
        return np.unique(self.labels)

    def get_valid_df(self):
        """ gets a validated copy of the dataframe. Looks better than doing `epochs_df.epochs._obj` """
        return self._obj.copy()

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


class Epoch(StartStopTimesMixin, TimeSlicableObjectProtocol, DataWriter):
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

    def to_dict(self, recurrsively=False):
        d = {"epochs": self._df, "metadata": self.metadata}
        return d

    def to_dataframe(self):
        df = self._df.copy()
        return df

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
        return Epoch(epochs=self._df.epochs.time_slice(t_start, t_stop)) # NOTE: drops metadata
        
    def label_slice(self, label):
        return Epoch(epochs=self._df.epochs.label_slice(label))

    def to_dict(self, recurrsively=False):
        return {
            "epochs": self._df,
            "metadata": self._metadata,
        }

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

    def to_neuroscope(self, ext="evt"):
        with self.filename.with_suffix(f".evt.{ext}").open("w") as a:
            for event in self.epochs.itertuples():
                a.write(f"{event.start*1000} start\n{event.stop*1000} end\n")

    def as_array(self):
        return self.to_dataframe()[["start", "stop"]].to_numpy()
