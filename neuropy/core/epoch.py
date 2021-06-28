import numpy as np
import pandas as pd
from .datawriter import DataWriter


class Epoch(DataWriter):
    def __init__(self, epochs: pd.DataFrame, metadata=None, filename=None) -> None:
        super().__init__(filename=filename)

        self._check_epochs(epochs)
        self._data = epochs.sort_values(by=["start"])
        self._metadata = metadata

    @property
    def starts(self):
        return self._data.start.values

    @property
    def stops(self):
        return self._data.stop.values

    @property
    def durations(self):
        return self.stops - self.starts

    @property
    def labels(self):
        return self._data.label.values

    @property
    def to_dict(self):
        d = {"epochs": self._data, "metadata": self.metadata}
        return d

    def to_dataframe(self):
        df = self._data.copy()
        df["duration"] = self.durations
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
        assert (
            pd.Series(["start", "stop", "label"]).isin(epochs.columns).all()
        ), "Epoch dataframe should at least have columns with names: start, stop, label"

    def __repr__(self) -> str:
        return f"{len(self.starts)} epochs"

    def __str__(self) -> str:
        pass

    def __getitem__(self, slice_):

        if isinstance(slice_, str):
            indices = np.where(self.labels == slice_)[0]
            if len(indices) > 1:
                return np.vstack((self.starts[indices], self.stops[indices])).T
            else:
                return np.array([self.starts[indices], self.stops[indices]]).squeeze()
        else:
            return np.vstack((self.starts[slice_], self.stops[slice_])).T

    def time_slice(self, t_start, t_stop):
        pass

    def to_dict(self):
        return {
            "epochs": self._epochs,
            "metadata": self._metadata,
            "filename": self.filename,
        }

    @staticmethod
    def from_dict(d: dict):
        if "filename" not in d.keys():
            filename = None
        else:
            filename = d["filename"]

        return Epoch(d["epochs"], d["metadata"], filename)

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

        ep_starts = self.epochs["start"].values
        ep_stops = self.epochs["stop"].values
        ep_durations = self.epochs["duration"].values
        ep_labels = self.epochs["label"].values

        for i in range(len(ep_starts)):

            # if epoch starts and ends inside range, delete it
            if ep_starts[i] >= t1 and ep_stops[i] <= t2:
                ep_durations[
                    i
                ] = -1  # non-positive duration flags this epoch for clean up

            # if epoch starts before and ends inside range, truncate it
            elif ep_starts[i] < t1 and (t1 < ep_stops[i] <= t2):
                ep_durations[i] = t1 - ep_starts[i]

            # if epoch starts inside and ends after range, truncate it
            elif (t1 <= ep_starts[i] < t2) and ep_stops[i] > t2:
                ep_durations[i] = ep_stops[i] - t2
                ep_starts[i] = t2

            # if epoch starts before and ends after range,
            # truncate the first part and add a new epoch for the end part
            elif ep_starts[i] <= t1 and ep_stops[i] >= t2:
                ep_durations[i] = t1 - ep_starts[i]
                ep_starts = np.append(ep_starts, t2)
                ep_durations = np.append(ep_durations, ep_stops[i] - t2)
                ep_labels = np.append(ep_labels, ep_labels[i])
                ep_ids = np.append(ep_ids, self._next_id)
                self._next_id += 1

    def proportion(self, t_start=None, t_stop=None):

        if t_start is None:
            t_start = self.starts[0]
        if t_stop is None:
            t_stop = self.stops[-1]

        duration = t_stop - t_start

        ep = self.epochs.copy()
        ep = ep[(ep.stop > t_start) & (ep.start < t_stop)].reset_index(drop=True)

        if ep["start"].iloc[0] < t_start:
            ep.at[0, "start"] = t_start

        if ep["stop"].iloc[-1] > t_stop:
            ep.at[ep.index[-1], "stop"] = t_stop

        ep["duration"] = ep.stop - ep.start

        ep_group = ep.groupby("label").sum().duration / duration

        states_proportion = {"rem": 0.0, "nrem": 0.0, "quiet": 0.0, "active": 0.0}

        for state in ep_group.index.values:
            states_proportion[state] = ep_group[state]

        return states_proportion

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
        return self.epochs[["start", "stop"]].to_numpy()
