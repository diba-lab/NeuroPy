from pathlib import Path
import numpy as np
from pickle import dump, load
import xarray as xr
import pandas as pd

from neuropy.core.ca_neurons import CaNeurons
from neuropy.io.miniscopeio import MiniscopeIO


class MinianIO:
    def __init__(
        self,
        dirname: str or None = None,
        basedir: str or None = None,
        ignore_time_mismatch: bool = False,
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
        for var_name in import_vars:
            setattr(self, var_name, np.load(self.minian_dir / (var_name + ".npy")))

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
            try:
                motion = xr.open_zarr(self.minian_dir / "motion.zarr")
                self.good_frames = motion.frame.values
            except:
                print(
                    "No motion.zarr file found, unable to load in frames used during analysis"
                )

        # Import timestamps
        try:
            msio = MiniscopeIO(self.basedir)
            self.times = msio.load_all_timestamps()
        except ValueError:
            print(
                "Error importing miniscope timestamps. Check .basedir and look for miniscope folders"
            )
            self.times = None

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
            except FileNotFoundError:
                try:
                    good_frames = np.load(self.minian_dir / "frames.npy")
                    self.times = self.times.iloc[good_frames]
                    print(
                        "Keeping "
                        + str(good_frames.shape[0])
                        + ' good frames found in "frames.npy" file'
                    )
                except FileNotFoundError:
                    print(
                        "no "
                        + str(self.minian_dir / "good_frames_bool.npy")
                        + " file found"
                    )
                    print("Check and make sure frames in data and timestamps match!")
                pass

        if not ignore_time_mismatch:
            assert (
                self.times.shape[0] == self.C.shape[1]
            ), "Different # frames in C and times vars. Check to make sure corrupted videos are properly accounted for"

    def trim_neurons(self, keep: str or list or None, trim: str or list or None = None):
        """
        Keep or trim out certain neurons from A, C, S, and YrA variables.
        :param keep: str or list with names of neuron types to keep from self.curated_neurons
        :param trim: str or list with names of neuron types to remove from self.curated_neurons.
        'keep' must be set to None for this to work
        :return:
        """

        # First send everything to CaNeurons class
        caneurons = self.to_caneurons(trim={"keep": keep, "trim": trim})

        # Next check to see if there is a field for unit_id present in some data where minian
        # prunes neurons during one of the last steps
        if (
            "unit_id_bool" in self.curated_neurons.keys()
            or "unit_id" in self.curated_neurons.keys()
        ):
            try:
                unit_ids = self.curated_neurons["unit_id_bool"]
                # Make sure that (despite its name, which is incorrect) that the array is of type int
                assert (
                    unit_ids.dtype == int
                ), "curated_neurons['unit_id'] is not the correct dtype, must be int"
            except KeyError:
                unit_ids = self.curated_neurons["unit_id"]
        else:
            unit_ids = np.arange(self.C.shape[0])

        if keep is not None:
            # Set up and check variable input
            keep = [keep] if isinstance(keep, str) else keep  # send to list

            keep_bool = np.zeros(self.C.shape[0], dtype=bool)
            keep_uid = []  # unit ids to keep from unit_ids var
            for keep_type in keep:
                assert (
                    keep_type in self.curated_neurons.keys()
                ), '"keep" input must be a key in "curated_neurons" field'
                # keep_bool[self.curated_neurons[keep_type]] = True
                keep_uid.extend(self.curated_neurons[keep_type])
            keep_ind = np.sort(
                [np.where(nid == unit_ids)[0][0] for nid in keep_uid]
            )  # inds in keep_bool corresponding to uids
            keep_bool[keep_ind] = True  # add in neurons to keep

        else:
            assert trim is not None, '"trim" must be specified with keep=None'
            trim = [trim] if isinstance(trim, str) else trim

            keep_bool = np.ones(self.C.shape[0], dtype=bool)
            trim_uid = []  # unit ids to trim from unit_ids var
            for trim_type in trim:
                assert (
                    trim_type in self.curated_neurons.keys()
                ), '"trim" input must be a key in "curated_neurons"'
                trim_uid.extend(self.curated_neurons[trim_type])
                trim_ind = np.sort(
                    [np.where(nid == unit_ids)[0][0] for nid in trim_uid]
                )
                keep_bool[trim_ind] = False  # cut out neurons

        # Now re-assign everything
        for var_name in ["A", "C", "S", "YrA"]:
            setattr(caneurons, var_name, getattr(caneurons, var_name)[keep_bool])

        return caneurons

    def to_caneurons(self, trim=None):
        """Send to CaNeurons class"""

        return CaNeurons(
            A=self.A, C=self.C, S=self.S, YrA=self.YrA, t=self.times, trim=trim
        )


if __name__ == "__main__":
    from session_directory import get_session_dir

    animal = "Rey"
    session = "Recall1"

    # Get session directory
    sesh_dir = get_session_dir(animal, session)

    # Load in ca imaging data from minian
    minian = MinianIO(basedir=sesh_dir)

    # Keep only good neurons
    caneurons = minian.trim_neurons(keep="good")
