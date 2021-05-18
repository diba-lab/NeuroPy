import numpy as np
import pandas as pd


class Epoch:
    def __init__(self, filename) -> None:
        self.filename = filename

    def load(self):
        if self.filename.is_file():
            self.data = np.load(self.filename, allow_pickle=True).item()
            self.metadata = self.data["metadata"]

            if "epochs" in self.data:
                self.epochs = self.data["epochs"]
            if "events" in self.data:
                self.epochs = self.data["events"]

        else:
            self.data = {"epochs": {}, "metadata": {}}

    def save(self, start, stop, label=None, metadata=None, **kwargs):

        assert len(start) == len(stop)
        df = pd.DataFrame({"start": start, "stop": stop})
        df["duration"] = df["stop"] - df["start"]

        if label is not None:
            assert len(start) == len(label)
            df["label"] = label

        for key in kwargs:
            assert len(kwargs[key]) == len(start)
            df[key] = kwargs[key]

        data = {"epochs": df, "metadata": metadata}
        np.save(self.filename, data)

    def delete_file(self):
        self.filename.unlink()

        print("file removed")

    def time_slice(self, period):
        pass

    def to_neuroscope(self, ext=".evt"):
        with self.files.neuroscope.open("w") as a:
            for event in self.epochs.itertuples():
                a.write(f"{event.start*1000} start\n{event.end*1000} end\n")

    def plot(self):
        pass
