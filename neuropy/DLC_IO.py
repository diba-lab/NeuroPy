"""Class to deal with DeepLabCut data"""


class DLC:
    """Import and parse DeepLabCut tracking data"""

    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ------- defining file names ---------
        filePrefix = self._obj.files.filePrefix

        @dataclass
        class files:
            placemap: str = filePrefix.with_suffix(".DLC.npy")

        self.files = files()

        self.tracking_files = sorted(base_dir.glob("*" + session + "/*.h5"))
        self.movie_files = Path(str(file)[0 : str(file).find("DLC")] + ".avi")

    def scorername(pos_data):
        """Get DLC scorername - assumes only 1 used."""
        scorername = pos_data.columns.levels[
            np.where([name == "scorer" for name in pos_data.columns.names])[0][0]
        ][0]

        return scorername

    def bodyparts(pos_data):
        """Get names of bodyparts"""
        bodyparts = pos_data.columns.levels[
            np.where([name == "bodyparts" for name in pos_data.columns.names])[0][0]
        ]

        return bodyparts
