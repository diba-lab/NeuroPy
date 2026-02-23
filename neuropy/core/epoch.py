from enum import unique
import numpy as np
import pandas as pd
from .datawriter import DataWriter
from pathlib import Path
import scipy.signal as sg
import typing
from copy import deepcopy, copy


def _unpack_args(values, fs=1):
    """Parsing argument for thresh_epochs"""
    try:
        val_min, val_max = values
    except (TypeError, ValueError):
        val_min, val_max = (values, None)

    val_min = val_min * fs
    val_max = val_max * fs if val_max is not None else None

    return val_min, val_max


class Epoch(DataWriter):
    def __init__(
        self, epochs: pd.DataFrame or dict or None, metadata=None, file=None
    ) -> None:
        super().__init__(metadata=metadata)

        if epochs is None:
            assert (
                file is not None
            ), "Must specify file to load if no epochs dataframe entered"
            epochs = np.load(file, allow_pickle=True).item()["epochs"]
            self.metadata = np.load(file, allow_pickle=True).item()["metadata"]

        self._epochs = self._validate(epochs)

    def union(self, other_epoch, res):
        """Find union with other epoch at 'res' time resolution"""
        t_start = np.min((self.starts.min(), other_epoch.starts.min()))
        t_stop = np.max((self.stops.max(), other_epoch.stops.max()))
        times, bool1 = self.to_point_process(t_start, t_stop, bin_size=res)
        _, bool2 = other_epoch.to_point_process(t_start, t_stop, bin_size=res)

        return self.from_boolean_array(np.bitwise_or(bool1, bool2), times)

    def intersection(self, other_epoch, res):
        """Find intersection with other epoch at 'res' time resolution"""
        t_start = np.min((self.starts.min(), other_epoch.starts.min()))
        t_stop = np.max((self.stops.max(), other_epoch.stops.max()))
        times, bool1, _ = self.to_point_process(t_start, t_stop, bin_size=res)
        _, bool2, _ = other_epoch.to_point_process(t_start, t_stop, bin_size=res)

        return self.from_boolean_array(np.bitwise_and(bool1, bool2), times)


    def replace_start_with_t_start_eeg(self):
        if hasattr(self, 'data'):
            self.data['start'] = self.data['t_start_eeg']

    def _validate(self, epochs):
        if isinstance(epochs, dict):
            try:
                epochs = pd.DataFrame(epochs)
            except:
                print("Error converting dictionary to pandas DataFrame")

        assert isinstance(epochs, pd.DataFrame)
        assert (
            pd.Series(["start", "stop", "label"]).isin(epochs.columns).all()
        ), "epochs should at least have columns/keys with names: start, stop, label"

        ## Make sure labels are formatted correctly as strings.
        # Note that the following would be MUCH simpler but throws a "SettingWithCopyWarning"
        # so we have to add the convoluted code below to avoid it
        # epochs.loc[:, "label"] = epochs.loc[:, "label"].astype("str")
        epochs_labels_str = copy(epochs["label"].astype("str"))
        epochs = epochs.drop(columns="label", inplace=False)  # this also throws a warning if used with inplace=True
        epochs.loc[:, "label"] = epochs_labels_str

        # Sort
        epochs = epochs.sort_values(by=["start"]).reset_index(drop=True)

        return epochs.copy()

    @property
    def starts(self):
        return self._epochs.start.values

    @property
    def stops(self):
        return self._epochs.stop.values
    
    @property
    def peak_times(self):
        return self._epochs.peak_time.values

    @property
    def durations(self):
        return self.stops - self.starts

    @property
    def n_epochs(self):
        return len(self.starts)

    @property
    def labels(self):
        return self._epochs.label.values

    def set_labels(self, labels):
        self._epochs["label"] = labels
        return Epoch(epochs=self._epochs)

    @property
    def has_labels(self):
        return np.all(self._epochs["label"] != "")

    def __add__(self, epochs):
        assert isinstance(epochs, Epoch), "Can only add two core.Epoch objects"
        my_columns = self._epochs.columns
        other_columns = epochs._epochs.columns
        if np.array_equal(my_columns, other_columns):
            df_new = pd.concat([self._epochs, epochs._epochs], ignore_index=True)
        else:
            my_df = self._epochs[["start", "stop", "label"]]
            other_df = epochs._epochs[["start", "stop", "label"]]
            df_new = pd.concat([my_df, other_df]).reset_index(drop=True)

        return Epoch(epochs=df_new)

    @property
    def _df(self):
        return self.to_dataframe()

    def add_epoch_manually(self, start, stop, label="", merge_dt: float or None = 0):
        comb_df = pd.DataFrame(
            {
                "start": np.array(start).reshape(-1),
                "stop": np.array(stop).reshape(-1),
                "label": label,
            }
        )

        if merge_dt is not None:
            return self.__add__(Epoch(comb_df)).merge(merge_dt)
        else:
            return self.__add__(Epoch(comb_df))

    def add_epoch_by_index(self, index, start, stop, label=""):
        assert np.mod(index, 1) > 0, "index must be a non-integer, e.g. -0.5 or 11.5"
        epochs_df = deepcopy(self._epochs)
        line = pd.DataFrame(
            {"start": start, "stop": stop, "label": label}, index=[index]
        )
        epochs_df = pd.concat((epochs_df, line), ignore_index=False)
        self._epochs = epochs_df.sort_index().reset_index(drop=True)

    def shift(self, dt, other_fields: None or str or list = None):
        """other_fields: provide other fields to shift aside from 'start' and 'stop'"""
        epochs = self._epochs.copy()
        epochs[["start", "stop"]] += dt

        assert isinstance(other_fields, (str, list)) or (other_fields is None)
        if other_fields is not None:
            other_fields = [other_fields] if isinstance(other_fields, str) else other_fields
            for field in other_fields:
                epochs[field] += dt

        return Epoch(epochs=epochs, metadata=self.metadata)

    def scale(self, sf):
        epochs = self._epochs.copy()
        epochs[["start", "stop"]] = epochs[["start", "stop"]] * sf
        return Epoch(epochs=epochs, metadata=self.metadata)

    def get_unique_labels(self):
        return np.unique(self.labels)

    def is_labels_unique(self):
        return len(np.unique(self.labels)) == len(self)

    def to_dataframe(self):
        df = self._epochs.copy()
        df["duration"] = self.durations
        return df

    def add_column(self, name: str, arr: np.ndarray):
        data = self.to_dataframe()
        data[name] = arr
        return Epoch(epochs=data, metadata=self.metadata)

    def add_dataframe(self, df: pd.DataFrame):
        assert isinstance(df, pd.DataFrame), "df should be a pandas dataframe"
        data = self.to_dataframe()
        data_new = pd.concat([data, df], axis=1)
        return Epoch(epochs=data_new, metadata=self.metadata)

    def __repr__(self) -> str:
        return f"{len(self.starts)} epochs\nSnippet: \n {self._epochs.head(5)}"

    def __str__(self) -> str:
        pass

    def __getitem__(self, i):
        if isinstance(i, str):
            data = self._epochs[self._epochs["label"] == i].copy()
        # elif all(isinstance(_, str) for _ in i):
        #     data = self._epochs[self._epochs["label"].isin(i)].copy()
        elif isinstance(i, list):
            assert all(isinstance(_, str) for _ in i), "All entries in epochs slicing list must be str"
            data = self._epochs[[label in i for label in self._epochs["label"]]]
        elif isinstance(i, (int, np.integer)):
            data = self._epochs.iloc[[i]].copy()
        else:
            data = self._epochs.iloc[i].copy()

        return Epoch(epochs=data.reset_index(drop=True))

    def __len__(self):
        return self.n_epochs

    def time_slice(self, t_start, t_stop, strict=True):
        """Return epochs which are within the provided time limits

        Parameters
        ----------
        t_start : float, seconds
            start time
        t_stop : float, seconds
            stop time
        strict : bool, optional
            whether to return epochs that strictly begin and end within the time limits, if False --> trim epochs which span partially or completely outside of time limits, by default True

        Returns
        -------
        Epoch
            _description_
        """
        t_start, t_stop = super()._time_slice_params(t_start, t_stop)
        starts = self.starts
        stops = self.stops

        if strict:
            keep = (starts >= t_start) & (stops <= t_stop)  # strictly inside
            epoch_df = self.to_dataframe()[keep].reset_index(drop=True)
            epoch_df = epoch_df.drop(["duration"], axis=1)
        else:
            # also include and trim epochs: that span the entire range, epochs that start before but end inside, epochs that start inside but end outside
            keep = (starts <= t_stop) & (stops >= t_start)
            epoch_df = self.to_dataframe()[keep].reset_index(drop=True)
            epoch_df = epoch_df.drop(["duration"], axis=1)
            epoch_df.loc[epoch_df["start"] < t_start, "start"] = t_start
            epoch_df.loc[epoch_df["stop"] > t_stop, "stop"] = t_stop

        return Epoch(epoch_df, metadata=self.metadata)

    def duration_slice(self, min_dur=None, max_dur=None):
        """return epochs that have durations between given thresholds

        Parameters
        ----------
        min_dur : float, optional
            minimum duration in seconds, by default None
        max_dur : float, optional
            maximum duration in seconds, by default None,

        Returns
        -------
        epoch
            epochs with durations between min_dur and max_dur
        """
        durations = self.durations
        if min_dur is None:
            min_dur = np.min(durations)
        if max_dur is None:
            max_dur = np.max(durations)

        return self[(durations >= min_dur) & (durations <= max_dur)]

    def label_slice(self, labels: typing.Union[list[str], str]):
        """Returns Epoch for input labels

        Parameters
        ----------
        labels : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if isinstance(labels, str):
            labels = [labels]

        assert np.all([isinstance(_, str) for _ in labels])
        df = self._epochs[np.isin(self.labels, labels)].reset_index(drop=True)
        return Epoch(epochs=df)

    @staticmethod
    def from_array(starts, stops, labels=None):
        df = pd.DataFrame({"start": starts, "stop": stops, "label": labels})
        return Epoch(epochs=df)

    @staticmethod
    def from_string_array(arr, dt: float = 1.0, t: np.array = None):
        """Convert a string array of type ['A','A','B','C','C'] to epochs
        Parameters
        ----------
        arr : np.array
            array of strings
        dt : float
            sampling time of arr, by default 1 second
        t : np.array
            time array of same length as arr giving corresponding time in seconds, if provided it overrides dt
        """
        unique_labels = np.unique(arr)
        pad = lambda x: np.pad(x, (1, 1), "constant", constant_values=(0, 0))

        starts, stops, labels = [], [], []
        for l in unique_labels:
            l_transition = np.diff(pad(np.where(arr == l, 1, 0)))
            l_start = np.where(l_transition == 1)[0]
            l_stop = np.where(l_transition == -1)[0]

            starts.append(l_start)
            stops.append(l_stop)
            labels.extend([l] * len(l_start))

        starts = np.concatenate(starts)
        stops = np.concatenate(stops)

        # padding correction
        stops[stops == len(arr)] = len(arr) - 1

        if t is not None:
            assert len(t) == len(arr), "time length should be same as input array"
            starts = t[starts]
            stops = t[stops]
        else:
            starts = starts * dt
            stops = stops * dt

        return Epoch.from_array(starts, stops, labels)

    @staticmethod
    def from_file(f):
        d = DataWriter.from_file(f)
        if d is not None:
            return Epoch.from_dict(d)
        else:
            return None

    @staticmethod
    def from_csv(f):
        """Load properly formatted csv file"""
        return Epoch(pd.read_csv(f))

    @property
    def is_overlapping(self):
        if self.n_epochs > 1:
            starts = self.starts
            stops = self.stops
            return np.all((starts[1:] - stops[:-1]) < 0)
        else:
            return False

    def itertuples(self):
        return self.to_dataframe().itertuples()

    def fill_blank(
        self,
        method: typing.Literal["from_left", "from_right", "from_nearest"] = "from_left",
    ):
        """Gaps in the epochs will be filled based on given criteria.
        Visualization:

        from_left:    |epoch1| gap |epoch2| --> |epoch1  ->|epoch2|
        from_right:   |epoch1| gap |epoch2| --> |epoch1|<-  epoch2|
        from_nearest: |epoch1| gap |epoch2| --> |epoch1->|<-epoch2|

        Parameters
        ----------
        method : str, optional
            how will the gaps be filled, by default "from_left"
            from_left = epoch preceding the gap is extended to fill
            from_right = epoch succeeding the gap is extended to fill
            from_nearest = first half of gap filled by extending preceding epoch and    second half is filled by extending succeeding epoch

        Returns
        -------
        core.Epoch
            epochs after filling the blank timepoints
        """
        ep_starts = self.starts
        ep_stops = self.stops
        ep_durations = self.durations
        ep_labels = self.labels

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

        # self.epochs["start"] = ep_starts
        # self.epochs["stop"] = ep_starts + ep_durations
        # self.epochs["duration"] = ep_durations

        return self.from_array(
            starts=ep_starts, stops=ep_starts + ep_durations, labels=ep_labels
        )

    def merge(self, dt):
        """Merge epochs that are within some temporal distance

        Parameters
        ----------
        dt : float
            temporal distance in seconds

        Returns
        -------
        Epoch
        """
        n_epochs = self.n_epochs
        starts, stops = self.starts, self.stops
        ind_delete = []
        for i in range(n_epochs - 1):
            if (starts[i + 1] - stops[i]) < dt:
                # stretch the second epoch to cover the range of both epochs
                starts[i + 1] = min(starts[i], starts[i + 1])
                stops[i + 1] = max(stops[i], stops[i + 1])

                ind_delete.append(i)

        epochs_arr = np.vstack((starts, stops)).T
        epochs_arr = np.delete(epochs_arr, ind_delete, axis=0)

        return Epoch.from_array(epochs_arr[:, 0], epochs_arr[:, 1])

    def merge_neighbors(self, max_epoch_sep=1e-6):
        """Epochs of same label and common boundary will be merged. For example,
        [1,2] and [2,3] --> [1,3]

        Returns
        -------
        core.Epoch
            epochs after merging neighbours sharing same label and boundary
        """
        ep_times, ep_stops, ep_labels = (deepcopy(self.starts), deepcopy(self.stops), deepcopy(self.labels))
        ep_durations = self.durations

        ind_delete = []
        for label in ep_labels:
            (inds,) = np.nonzero(ep_labels == label)
            for i in range(len(inds) - 1):
                # if two sequentially adjacent epochs with the same label
                # overlap or have less than 1 microsecond separation, merge them
                if ((inds[i+1] - inds[i]) == 1) & ((ep_times[inds[i + 1]] - ep_stops[inds[i]]) < max_epoch_sep):
                    # stretch the second epoch to cover the range of both epochs
                    ep_times[inds[i + 1]] = min(
                        ep_times[inds[i]], ep_times[inds[i + 1]]
                    )
                    ep_stops[inds[i + 1]] = max(
                        ep_stops[inds[i]], ep_stops[inds[i + 1]]
                    )
                    ep_durations[inds[i + 1]] = (
                        ep_stops[inds[i + 1]] - ep_times[inds[i + 1]]
                    )

                    ind_delete.append(inds[i])

        epochs_arr = np.vstack((ep_times, ep_stops)).T
        epochs_arr = np.delete(epochs_arr, ind_delete, axis=0)
        labels_arr = np.delete(ep_labels, ind_delete)

        return Epoch.from_array(epochs_arr[:, 0], epochs_arr[:, 1], labels_arr)

    def contains(self, t, return_closest: bool = False):
        """Check if timepoints lie within epochs, must be non-overlapping epochs

        Parameters
        ----------
        t : array
            timepoints in seconds
        return_closest: bool
            True = return closest epoch before to all points in t even if t is outside epoch

        Returns
        -------
        _type_
            _description_
        """

        assert self.is_overlapping == False, "Epochs must be non overlapping"
        assert isinstance(t, np.ndarray), "t must be a numpy.ndarray"

        labels = self.labels
        bin_loc = np.digitize(t, self.flatten())
        indx_bool = bin_loc % 2 == 1

        if not return_closest:
            return (
                indx_bool,
                t[indx_bool],
                labels[((bin_loc[indx_bool] - 1) / 2).astype("int")],
            )
        else:
            return indx_bool, t, labels[bin_loc], bin_loc

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
        epochs_df = pd.concat([epochs_df, flank_start, flank_stop], ignore_index=True)
        return Epoch(epochs_df)

    def proportion_by_label(self, t_start=None, t_stop=None, ignore_gaps=False):
        """Get proportion of time for each label type

        Parameters
        ----------
        t_start : float, optional
            start time in seconds, by default None
        t_stop : float, optional
            stop time in seconds, by default None
        ignore_gaps: will return None if set and there is no epoch in the time period selected.

        Returns
        -------
        dict
            dictionary containing proportion for each unique label between t_start and t_stop
        """
        if t_start is None:
            t_start = self.starts[0]
        if t_stop is None:
            t_stop = self.stops[-1]

        duration = t_stop - t_start

        ep = self._epochs.copy()
        ep = ep[(ep.stop > t_start) & (ep.start < t_stop)].reset_index(drop=True)
        if not ignore_gaps:
            assert ep.shape[0] > 0, "cannot have empty time gaps between epoch labels with ignore_gaps=False"
        elif ignore_gaps and (ep.shape[0] > 0):
            if ep["start"].iloc[0] < t_start:
                ep.at[0, "start"] = t_start

            if ep["stop"].iloc[-1] > t_stop:
                ep.at[ep.index[-1], "stop"] = t_stop

            ep["duration"] = ep.stop - ep.start

            ep_group = ep.groupby("label").sum(numeric_only=True).duration / duration

            label_proportion = {}
            for label in self.get_unique_labels():
                label_proportion[label] = 0.0

            for state in ep_group.index.values:
                label_proportion[state] = ep_group[state]

            return label_proportion
        else:
            return None

    def durations_by_label(self):
        """Return total duration for each unique label

        Returns
        -------
        dict
            dictionary containing duration of each unique label
        """
        labels = self.labels
        durations = self.durations
        unique_labels = self.get_unique_labels()
        label_durations = {}
        for label in unique_labels:
            label_durations[label] = durations[labels == label].sum()

        return label_durations

    def resample_labeled_epochs(self, res, t_start=None, t_stop=None, merge_neighbors=True):
        """Resample epochs to different size blocks using a winner take all method to assign
        a label name. e.g. if the first 100-second epoch is 40% quiet wake, 50% REM, and 10% NREM
        it would get labeled as REM.  Pretty slow, even slower with merge_neighbors=True

        :param: res: block size in seconds
        :param: t_start: start time in seconds, default = start of first epoch
        :param: t_stop : stop time in seconds, default = stop of last epoch
        :param merge_neighbors: combine adjacent epochs of the same label, default=True"""

        if t_start is None:
            t_start = self.starts[0]
        elif t_start < self.starts[0]:
            t_start = self.starts[0]
            print('t_start < start time of first epoch, reassigned to match first epoch start time')

        if t_stop is None:
            t_stop = self.stops[-1]
        if t_stop > self.stops[-1]:
            t_stop = self.stops[-1]
            print('t_stop > stop time of first epoch, reassigned to match last epoch stop time')
        bins = np.arange(t_start, t_stop + res, res)
        start_rs = bins[:-1]
        stop_rs = bins[1:]
        label_rs = []
        for start, stop in zip(start_rs, stop_rs):
            props = self.proportion_by_label(start, stop, ignore_gaps=True)
            label_add = list(props.keys())[np.argmax(list(props.values()))] if props is not None else ""
            label_rs.append(label_add)
        # except AssertionError:  # Append nothing if gap found in epochs
        #     label_rs.append("")

        epoch_rs = Epoch(pd.DataFrame({"start": start_rs, "stop": stop_rs, "label": label_rs}))
        epoch_rs = epoch_rs.merge_neighbors() if merge_neighbors else epoch_rs

        return epoch_rs

    def count(self, t_start=None, t_stop=None, binsize=300):
        if t_start is None:
            t_start = 0

        if t_stop is None:
            t_stop = np.max(self.stops)

        mid_times = self.starts + self.durations / 2
        bins = np.arange(t_start, t_stop + binsize, binsize)
        return np.histogram(mid_times, bins=bins)[0]

    def as_array(self):
        """Returns starts and stops as 2d numpy array"""
        return self.to_dataframe()[["start", "stop"]].to_numpy()

    def flatten(self):
        """Returns 1d numpy array of alternating starts and stops
        NOTE: returned array is monotonically increasing only if epochs are non-overlapping
        """
        return self.as_array().flatten("C")

    def to_point_process(self, t_start=None, t_stop=None, bin_size=(1 / 1250), return_labels=False):
        """Returns 1d numpy boolean where True = epochs"""
        if t_start is None:
            t_start = 0

        if t_stop is None:
            t_stop = np.max(self.stops)

        times = np.arange(t_start, t_stop + bin_size, bin_size)

        # Super slow
        time_bool = np.zeros_like(times).astype(bool)
        labels = np.zeros_like(times).astype('str')

        for start_ind, end_ind, label in zip(
                ((self.starts - t_start) / bin_size).astype(int),
                ((self.stops - t_start) / bin_size).astype(int),
                self.labels,
        ):
            time_bool[start_ind:end_ind] = True
            labels[start_ind:end_ind] = label

        if return_labels:
            return times, time_bool, labels
        else:
            return times, time_bool

    def add_epoch_buffer(self, buffer_sec: float or int or tuple or list):
        df = self._epochs.copy()
        self._epochs = add_epoch_buffer(df, buffer_sec)

        # Run below to update start and stop properties
        self.starts
        self.stops
        print(f"Buffer of {buffer_sec} added before/after each epoch")
    @staticmethod
    def from_peaks(arr: np.ndarray, thresh, length, sep=0, boundary=0, fs=1):
        hmin, hmax = _unpack_args(thresh)  # does not need fs
        lmin, lmax = _unpack_args(length, fs=fs)
        sep = sep * fs + 1e-6

        assert hmin >= boundary, "boundary must be smaller than min thresh"

        arr_thresh = np.where(arr >= boundary, arr, 0)
        peaks, props = sg.find_peaks(arr_thresh, height=[hmin, hmax], prominence=0)

        starts, stops = props["left_bases"], props["right_bases"]
        peaks_values = arr_thresh[peaks]

        # ----- merge overlapping epochs ------
        n_epochs = len(starts)
        ind_delete = []
        for i in range(n_epochs - 1):
            if (starts[i + 1] - stops[i]) < sep:
                # stretch the second epoch to cover the range of both epochs
                starts[i + 1] = min(starts[i], starts[i + 1])
                stops[i + 1] = max(stops[i], stops[i + 1])

                peaks_values[i + 1] = max(peaks_values[i], peaks_values[i + 1])
                peaks[i + 1] = [peaks[i], peaks[i + 1]][
                    np.argmax([peaks_values[i], peaks_values[i + 1]])
                ]
                ind_delete.append(i)

        epochs_arr = np.vstack((starts, stops, peaks, peaks_values)).T
        epochs_arr = np.delete(epochs_arr, ind_delete, axis=0)

        # ----- duration thresholds ------
        epochs_length = epochs_arr[:, 1] - epochs_arr[:, 0]
        if lmax is None:
            lmax = epochs_length.max()
        ind_keep = (epochs_length >= lmin) & (epochs_length <= lmax)

        starts, stops, peaks, peaks_values = epochs_arr[ind_keep, :].T

        # return starts / fs, stops / fs, peaks / fs, peaks_values

        return Epoch.from_array(starts / fs, stops / fs), peaks / fs, peaks_values

    @staticmethod
    def from_boolean_array(arr, t=None):
        """Create epochs from a boolean array

        Parameters
        ----------
        arr : np.array
            timeseries of boolean values
        t : np.array, optional
            corresponding time in seconds, by default None

        Returns
        -------
        core.Epoch
            epochs where the arr is high
        """
        if isinstance(t, pd.Series):  # grab values only
            t = t.values
        assert np.array_equal(arr, arr.astype(bool)), "Only boolean array accepted"
        int_arr = arr.astype("int")
        pad_arr = np.pad(int_arr, 1)
        diff_arr = np.diff(pad_arr)
        starts, stops = np.where(diff_arr == 1)[0], np.where(diff_arr == -1)[0]
        stops[stops == len(arr)] = len(arr) - 1

        if t is not None:
            assert len(t) == len(arr), "time length should be same as input array"

            starts, stops = t[starts], t[stops]

        return Epoch.from_array(starts, stops, "high")

    def get_indices_for_time(self, t: np.array):
        """Mark indices of a time array that lie within the epochs.

        Parameters
        ----------
        t : np.array
            array of time points in seconds

        Returns
        -------
        Boolean array

        """
        time_bool = np.zeros_like(t)

        for e in self.as_array():
            time_bool[np.where((t >= e[0]) & (t <= e[1]))[0]] = 1

        return time_bool.astype("bool")

    @property
    def epochs(self):
        return self._epochs

def add_epoch_buffer(epoch_df: pd.DataFrame, buffer_sec: float or int or tuple or list):
    """Extend each epoch by buffer_sec before/after start/stop of each epoch"""
    if type(buffer_sec) in [int, float]:
        buffer_sec = (buffer_sec, buffer_sec)
    else:
        assert len(buffer_sec) == 2

    epoch_df["start"] -= buffer_sec[0]
    epoch_df["stop"] += buffer_sec[1]

    return epoch_df


def get_epoch_overlap_duration(epochs1: Epoch, epochs2: Epoch):
    """Calculate time of overlapping epochs"""
    e_array1 = epochs1.to_dataframe().loc[:, ["start", "stop"]].values
    e_array2 = epochs2.to_dataframe().loc[:, ["start", "stop"]].values
    overlaps = []
    for e1 in e_array1:
        for e2 in e_array2:
            overlaps.append(getOverlap(e1, e2))

    return np.array(overlaps).sum()


def getOverlap(a, b):
    """From https://stackoverflow.com/questions/2953967/built-in-function-for-computing-overlap-in-python"""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def combine_epochs(epochs_df: pd.DataFrame, inplace: bool = True):
    """Combine epochs so that there are no starts or stops contained
    entirely within another epoch. Epoch.union might be better"""

    all([col in epochs_df.columns for col in ["start", "stop"]])

    # First find epochs that overlap and get id to replace with
    start_overlaps, stop_overlaps = [], []
    for ide, epoch in epochs_df.iterrows():
        overlap_start = np.bitwise_and(epoch['start'] > epochs_df['start'],
                                       epoch['start'] < epochs_df['stop'])
        overlap_stop = np.bitwise_and(epoch['stop'] > epochs_df['start'],
                                      epoch['stop'] < epochs_df['stop'])
        if overlap_start.sum() == 1:
            start_overlap_id = np.where(overlap_start)[0][0]
            #             print('epoch ' + str(ide) + ' overlap start w epoch '
            #                   + str(start_overlap_id))
            start_overlaps.append([ide, start_overlap_id])

        if overlap_stop.sum() == 1:
            stop_overlap_id = np.where(overlap_stop)[0][0]
            #             print('epoch ' + str(ide) + ' overlap stop w epoch '
            #                  + str(stop_overlap_id))
            stop_overlaps.append([ide, stop_overlap_id])
    # Now walk through and replace
    for start in start_overlaps:
        epochs_df.loc[start[0], "start"] = epochs_df.loc[start[1], "start"]

    for stop in stop_overlaps:
        epochs_df.loc[stop[0], "stop"] = epochs_df.loc[stop[1], "stop"]

    #     overlap_ids = np.hstack((np.asarray(start_overlaps)[:, 1],
    #                   np.asarray(stop_overlaps)[:, 1]))

    #     print('Dropping duplicate epochs: ' + str(overlap_ids))

    if inplace:
        epochs_df.drop_duplicates(inplace=inplace, ignore_index=True)
        epochs_df.sort_values(by='start', inplace=inplace, ignore_index=True)

        return None
    else:
        epochs_out = epochs_df.drop_duplicates(inplace=inplace, ignore_index=True)
        epochs_out = epochs_out.sort_values(by='start', inplace=inplace, ignore_index=True)

        return epochs_out


if __name__ == "__main__":
    base_dir = Path("/data/Clustering/sessions/RatU/RatUNSD_sleep_score_test")
    sleep_file = sorted(base_dir.glob("*.brainstates.npy"))[0]
    sleep_epochs = Epoch(epochs=None, file=sleep_file)

    # Resample data to match epoch length specfied above and in AccuSleePy
    sleep_df_rs = sleep_epochs.to_point_process(bin_size=2.5)