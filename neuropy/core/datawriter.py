import numpy as np
import pathlib
from pathlib import Path

class DataWriter:
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
            self._metadata = self._metadata | d

    @staticmethod
    def from_dict(d):
        return NotImplementedError

    # @staticmethod
    # def from_file(f):
    #     if f.is_file():
    #         try:
    #             d = np.load(f, allow_pickle=True).item()
    #             return d
    #         except NotImplementedError:
    #             print("Issue with pickled POSIX_PATH on windows for path {}, falling back to non-pickled version...".format(f))
    #             temp = pathlib.PosixPath
    #             # pathlib.PosixPath = pathlib.WindowsPath # Bad hack
    #             pathlib.PosixPath = pathlib.PurePosixPath # Bad hack
    #             d = np.load(f, allow_pickle=True).item()
    #             # d['filename']
    #             # print("Post hack decode: {}\n".format(d))
    #             return d
            
    #         return None
    #     else:
    #         return None
    
    # @staticmethod
    # def from_file(f):
    #     d = DataWriter.from_file(f)
    #     if d is not None:
    #         return Position.from_dict(d)
    #     else:
    #         return None


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
                # d['filename']
                # print("Post hack decode: {}\n".format(d))
                # return dict_rep
            
            if dict_rep is not None:
                # Convert to object
                obj = cls.from_dict(dict_rep)
                obj.filename = f
                return obj
            
            
            return dict_rep
            
        else:
            return None
        
        



    def to_dict(self, recurrsively=False):
        return NotImplementedError

    def save(self):
        data = self.to_dict()
        if self.filename is not None:
            assert isinstance(self.filename, Path)
            np.save(self.filename, data)
            print(f"{self.filename.name} saved")
        else:
            print("filename can not be None")

    def delete_file(self):
        self.filename.unlink()

        print("file removed")

    def create_backup(self):
        pass
