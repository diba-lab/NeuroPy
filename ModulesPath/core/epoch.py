from curses import meta
import numpy as np
import pandas as pd


class Epoch:
    def __init__(
        self, epochs: pd.DataFrame = None, filename=None, metadata=None
    ) -> None:
        self.filename = filename
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

    def delete_file(self):
        self.filename.unlink()

        print("file removed")

    def create_backup(self):
        pass

    def time_slice(self, period):
        pass

    def add_epochs(self):
        pass

    def to_neuroscope(self, ext="evt"):
        with self.filename.with_suffix(f".evt.{ext}").open("w") as a:
            for event in self.epochs.itertuples():
                a.write(f"{event.start*1000} start\n{event.stop*1000} end\n")

    def as_array(self):
        return self.epochs[["start", "stop"]].to_numpy()

    def plot(self):
        pass
