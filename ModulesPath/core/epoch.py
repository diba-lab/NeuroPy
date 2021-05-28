import numpy as np
import pandas as pd
from pathlib import Path


class Epoch:
    def __init__(self, epochs: pd.DataFrame = None, metadata=None) -> None:
        self.epochs = epochs
        self.metadata = metadata

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epochs):
        """check epochs compatibility

        Parameters
        ----------
        df : pd.Datarame
            pandas dataframe should have at least 3 columns with names 'start', 'stop', 'label'
        """

        if epochs is not None:

            assert isinstance(epochs, pd.DataFrame)
            assert (
                pd.Series(["start", "stop", "label"]).isin(epochs.columns).all()
            ), "Epoch dataframe should have columns with names: start, stop, label"

            if "duration" not in epochs.columns:
                epochs["duration"] = epochs["stop"] - epochs["start"]

        self._epochs = epochs

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """metadata compatibility"""

        self._metadata = metadata

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        pass

    def delete_file(self):
        self.filename.unlink()

        print("file removed")

    def create_backup(self):
        pass

    def time_slice(self, period):
        pass

    def add_epochs(self):
        pass

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

    def to_neuroscope(self, ext="evt"):
        with self.filename.with_suffix(f".evt.{ext}").open("w") as a:
            for event in self.epochs.itertuples():
                a.write(f"{event.start*1000} start\n{event.stop*1000} end\n")

    def as_array(self):
        return self.epochs[["start", "stop"]].to_numpy()

    def plot(self):
        pass


class WritableEpoch(Epoch):
    def __init__(self, filename, epochs=None, metadata=None) -> None:
        super().__init__(epochs=epochs, metadata=metadata)
        self.filename = filename

    def load(self):
        if self.filename.is_file():
            self.data = np.load(self.filename, allow_pickle=True).item()

            if "metadata" in self.data:
                self.metadata = self.data["metadata"]

            if "epochs" in self.data:
                self.epochs = self.data["epochs"]
            if "events" in self.data:
                self.epochs = self.data["events"]

    def save(self):
        data = {"epochs": self._epochs, "metadata": self._metadata}

        np.save(self.filename, data)
        print(f"{self.filename.name} created")
