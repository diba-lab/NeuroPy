import numpy as np
import pathlib
from pathlib import Path

from neuropy.utils.mixins.dict_representable import DictRepresentable
from neuropy.utils.mixins.file_representable import FileRepresentable
from neuropy.utils.mixins.print_helpers import SimplePrintable


class LegacyDataLoadingMixin:
    @classmethod
    def legacy_from_dict(cls, dict_rep: dict):
        """ Tries to load the dict using previous versions of this code. """
        raise NotImplementedError
    
        

class DataWriter(FileRepresentable, DictRepresentable, LegacyDataLoadingMixin, SimplePrintable):
    def __init__(self, metadata=None) -> None:

        self._filename = None

        if metadata is not None:
            assert isinstance(metadata, dict), "Only dictionary accepted as metadata"

        self._metadata: dict = metadata

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, f):
        assert isinstance(f, (str, Path))
        self._filename = f

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, d):
        """metadata compatibility"""
        if d is not None:
            assert isinstance(d, dict), "Only dictionary accepted"
            if self._metadata is not None:
                self._metadata = self._metadata | d # if we already have valid metadata, merge the dicts
            else:
                self._metadata = d # otherwise we can just set it directly
                
    ## DictRepresentable protocol:
    @staticmethod
    def from_dict(d):
        return NotImplementedError

    def to_dict(self, recurrsively=False):
        return NotImplementedError

    ## FileRepresentable protocol:
    @classmethod
    def from_file(cls, f):
        if f.is_file():
            dict_rep = None
            try:
                dict_rep = np.load(f, allow_pickle=True).item()
                # return dict_rep
            except NotImplementedError:
                print("Issue with pickled POSIX_PATH on windows for path {}, falling back to non-pickled version...".format(f))
                temp = pathlib.PosixPath
                # pathlib.PosixPath = pathlib.WindowsPath # Bad hack
                pathlib.PosixPath = pathlib.PurePosixPath # Bad hack
                dict_rep = np.load(f, allow_pickle=True).item()
            
            if dict_rep is not None:
                # Convert to object
                try:
                    obj = cls.from_dict(dict_rep)
                except KeyError as e:
                    # print(f'f: {f}, dict_rep: {dict_rep}')
                    # Tries to load using any legacy methods defined in the class
                    obj = cls.legacy_from_dict(dict_rep)
                    # raise e
                
                obj.filename = f
                return obj
            return dict_rep
            
        else:
            return None
        
    @classmethod
    def to_file(cls, data: dict, f):
        if f is not None:
            assert isinstance(f, Path)
            np.save(f, data)
            print(f"{f.name} saved")
        else:
            print("filename can not be None")


    def save(self):
        data = self.to_dict()
        DataWriter.to_file(data, self.filename)

    def delete_file(self):
        self.filename.unlink()

        print("file removed")

    def create_backup(self):
        pass
