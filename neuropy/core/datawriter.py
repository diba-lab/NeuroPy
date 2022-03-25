import numpy as np
from pathlib import Path


class DataWriter:
    def __init__(self, metadata: dict = None) -> None:

        if metadata is not None:
            assert isinstance(metadata, dict), "Only dictionary accepted as metadata"
            self._metadata: dict = metadata
        else:
            self._metadata: dict = {}

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, d):
        """metadata compatibility"""
        if d is not None:
            assert isinstance(d, dict), "Only dictionary accepted"
            self._metadata = self._metadata | d

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @staticmethod
    def from_file(f):
        f = Path(f) if isinstance(f, str) else f
        if f.is_file():
            d = np.load(f, allow_pickle=True).item()
            return d
        else:
            return None

    def to_dict(self):
        d = dict()
        attrs = self.__dict__.keys()
        for k in attrs:
            if k.startswith("_"):
                d[k[1:]] = getattr(self, k)
            else:
                d[k] = getattr(self, k)
        return d

    def save(self, fp):

        assert isinstance(fp, (str, Path)), "filename is invalid"
        data = self.to_dict()
        np.save(fp, data)
        print(f"{fp} saved")
