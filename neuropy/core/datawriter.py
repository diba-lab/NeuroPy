import numpy as np
from pathlib import Path
import datetime


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

    def save_with_date(self, fp):

        assert isinstance(fp, (str, Path)), "filename is invalid"
        date_suffix = "." + datetime.date.today().strftime("%d-%m-%y")
        fname = fp.name + date_suffix
        data = self.to_dict()
        fp = fp.with_name(fname)
        np.save(fp, data)
        print(f"{fp} saved")

    def _time_slice_params(self, t1=None, t2=None):

        if t1 is None:
            t1 = self.t_start

        if t2 is None:
            t2 = self.t_stop

        assert t2 > t1, "t2 must be greater than t1"

        if hasattr(self, "time"):
            return (self.time >= t1) & (self.time <= t2)
        else:
            return t1, t2
