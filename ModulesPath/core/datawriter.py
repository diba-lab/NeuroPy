import numpy as np
from pathlib import Path


class DataWriter:
    def __init__(self, filename=None) -> None:

        if filename is not None:
            self.filename = Path(filename)
        else:
            self.filename = None

    def load(self):
        if self.filename.is_file():
            return np.load(self.filename, allow_pickle=True).item()
        else:
            return None

    def save(self, data):
        np.save(self.filename, data)
        print("data saved")

    def delete_file(self):
        self.filename.unlink()

        print("file removed")

    def create_backup(self):
        pass
