import numpy as np
from pathlib import Path


class DataWriter:
    def __init__(self, filename=None) -> None:

        if filename is not None:
            self.filename = Path(filename)
        else:
            self.filename = None

    @staticmethod
    def from_dict(d):
        return NotImplementedError

    @staticmethod
    def from_file(f):
        if f.is_file():
            d = np.load(f, allow_pickle=True).item()
            return d
        else:
            return None

    def to_dict(self):
        return NotImplementedError

    def save(self):

        data = self.to_dict()
        if self.filename is not None:
            assert isinstance(self.filename, Path)
            np.save(self.filename, data)
            print("data saved")
        else:
            print("filename not understood")

    def delete_file(self):
        self.filename.unlink()

        print("file removed")

    def create_backup(self):
        pass
