import numpy as np
import pandas as pd


class Epoch:
    def __init__(self, filename) -> None:
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

        else:
            self.data = {"epochs": {}, "metadata": {}}

    def save(self, df, metadata=None):
        """Save epoch data

        Parameters
        ----------
        df : pd.Datarame
            pandas dataframe should have at least 3 columns with names 'start', 'stop', 'label'
        metadata : dict, optional
            dictionary for any relevant information, by default None
        """

        assert isinstance(df, pd.DataFrame)
        assert (
            pd.Series(["start", "stop", "label"]).isin(df.columns).all()
        ), "Epoch dataframe should have columns with names: start, stop, label"

        if "duration" not in df.columns:
            df["duration"] = df["stop"] - df["start"]

        data = {"epochs": df, "metadata": metadata}

        np.save(self.filename, data)

    def delete_file(self):
        self.filename.unlink()

        print("file removed")

    def create_backup(self):
        pass

    def time_slice(self, period):
        pass

    def to_neuroscope(self, ext=".evt"):
        with self.files.neuroscope.open("w") as a:
            for event in self.epochs.itertuples():
                a.write(f"{event.start*1000} start\n{event.stop*1000} end\n")

    def plot(self):
        pass
