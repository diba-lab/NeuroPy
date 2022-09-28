from pathlib import Path
import numpy as np
from pickle import dump, load
import xarray as xr
import pandas as pd

from neuropy.core.ca_neurons import CaNeurons
from neuropy.io.miniscopeio import MiniscopeIO


class MinianIO:
    def __init__(
        self, dirname: str or None = None, basedir: str or None = None
    ) -> None:
        self.basedir = basedir
        # Try to autodetect minian folder - must specify basedir
        if dirname is None:
            assert (
                len(dirname := sorted(self.basedir.glob("**/minian"))) == 1
            ), "More than one minian folder found, fill in directory manually"
            dirname = Path(dirname[0])
        self.minian_dir = Path(dirname)

        # Infer base directory name from minian folder
        if self.basedir is None:
            basedir_id = (
                len(self.minian_dir.parts)
                - np.where(
                    [part == "Miniscope_combined" for part in self.minian_dir.parts]
                )[0][0]
                - 1
            )
            self.basedir = self.minian_dir.parents[basedir_id]

        # Load in relevant variables
        import_vars = ["A", "C", "S", "YrA"]
        for var in import_vars:
            setattr(self, var, np.load(self.minian_dir / (var + ".npy")))

        # Load in hand-curated neurons
        with open(self.minian_dir / "curated_neurons.pkl", "rb") as f:
            self.curated_neurons = load(f)

        # Load in good frames
        try:
            self.good_frames = np.load(self.minian_dir / "frames.npy")
        except FileNotFoundError:
            print(
                "frames.npy not found in minian directory, trying to load in zarr files to get frames"
            )
            motion = xr.open_zarr(self.minian_dir / "motion.zarr")
            self.good_frames = motion.frame.values

        # Import timestamps
        try:
            msio = MiniscopeIO(self.basedir)
            self.times = msio.load_all_timestamps()
        except ValueError:
            print(
                "Error importing miniscope timestamps. Check .basedir and look for miniscope folders"
            )
            self.times = None
        assert (
            self.times.shape[0] == self.C.shape[1]
        ), "Different # frames in C and times vars. Check to make sure corrupted videos are properly accounted for"

        # Remove any timestamps corresponding to frames you've removed.
        if self.times is not None:
            try:
                good_frames_bool = np.load(self.minian_dir / "good_frames_bool.npy")
                print(
                    "Keeping "
                    + str(good_frames_bool.sum())
                    + ' good frames found in "good_frames_bool.npy" file'
                )
                self.times = self.times[good_frames_bool]
            except:
                print(
                    "no "
                    + str(self.minian_dir / "good_frames_bool.npy")
                    + " file found"
                )
                print("Check and make sure frames in data and timestamps match!")
                pass

    def to_caneurons(self):
        """Send to CaNeurons class"""

        return CaNeurons(A=self.A, C=self.C, S=self.S, YrA=self.YrA, t=self.time)


if __name__ == "__main__":
    from session_directory import get_session_dir

    sesh_dir = get_session_dir("Finn", "Recall1")
    min_test = MinianIO(basedir=sesh_dir)
