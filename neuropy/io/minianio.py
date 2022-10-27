import pathlib
from pathlib import Path
import numpy as np
from pickle import dump, load
import xarray as xr
import zarr
import pandas as pd

from neuropy.core.ca_neurons import CaNeurons
from neuropy.io.miniscopeio import MiniscopeIO
from neuropy.utils.misc import flatten
from neuropy.utils.minian_util import load_subset


class MinianIO:
    def __init__(
        self,
        dirname: str or None = None,
        basedir: str or None = None,
        ignore_time_mismatch: bool = False,
    ) -> None:
        self.basedir = Path(basedir)
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
        # Lots of error catching here to track different ways of saving good frames
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
                    try:
                        motion = xr.open_zarr(self.minian_dir / "motion.zarr")
                        self.good_frames = motion.frame.values
                        self.times = self.times.iloc[self.good_frames]
                        print(
                            f"Keeping {len(self.good_frames)} good frames found in motion.zarr file"
                        )
                    except zarr.errors.GroupNotFoundError:
                        print(
                            "No .zarr or frames.npy or good_frames.npy file found in directory. Check and make sure frames in data and timestamps match!"
                        )
                pass

        if not ignore_time_mismatch:
            assert (
                self.times.shape[0] == self.C.shape[1]
            ), "Different # frames in C and times vars. Check to make sure corrupted videos are properly accounted for"

    def save_curated_neurons(self):
        # Save updated hand-curated neurons to pickle file
        with open(self.minian_dir / "curated_neurons.pkl", "wb") as f:
            dump(self.curated_neurons, f)

    def trim_neurons(self, keep: str or list or None, trim: str or list or None = None):
        """
        Keep or trim out certain neurons from A, C, S, and YrA variables.
        :param keep: str or list with names of neuron types to keep from self.curated_neurons
        :param trim: str or list with names of neuron types to remove from self.curated_neurons.
        'keep' must be set to None for this to work
        :return:
        """

        # First see if there is a field for unit_id present in some data where minian
        # prunes neurons during one of the last steps
        if (
            "unit_id_bool" in self.curated_neurons.keys()
            or "unit_id" in self.curated_neurons.keys()
        ):
            try:
                unit_ids = self.curated_neurons["unit_id_bool"]
                # Make sure that (despite its name, which is incorrect) that the array is of type int
                if isinstance(unit_ids, xr.DataArray):
                    unit_ids = unit_ids["unit_id"].values
                assert (
                    unit_ids.dtype == int
                ), "curated_neurons['unit_id'] is not the correct dtype, must be int"
            except KeyError:
                unit_ids = self.curated_neurons["unit_id"]
        else:
            try:
                unit_ids = np.load(self.minian_dir / "unit_id.npy")
            except FileNotFoundError:
                unit_ids = np.arange(self.C.shape[0])

        # Next send everything to CaNeurons class
        caneurons = self.to_caneurons(
            trim={"keep": keep, "trim": trim}, neuron_ids=unit_ids
        )

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

                # Flatten list in case you have nested lists
                keep_uid_flat = [_ for _ in flatten(keep_uid)]
                if keep_uid_flat != keep_uid:
                    print(
                        "Had to flatten list of units to keep - check to make sure you aren't un-merging units!"
                    )
                    keep_uid = keep_uid_flat
            try:
                keep_ind = np.sort(
                    [np.where(nid == unit_ids)[0][0] for nid in keep_uid]
                )  # inds in keep_bool corresponding to uids
            except IndexError:
                raise IndexError(
                    'unit_id mismatch: check data folder for unit_id.npy or curated_neurons.pkl with "unit_id" field',
                )
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
                try:
                    trim_ind = np.sort(
                        [np.where(nid == unit_ids)[0][0] for nid in trim_uid]
                    )
                except IndexError:
                    raise IndexError(
                        'unit_id mismatch: check data folder for unit_id.npy or curated_neurons.pkl with "unit_id" field',
                    )
                keep_bool[trim_ind] = False  # cut out neurons

        # Now re-assign everything
        for var_name in ["A", "C", "S", "YrA", "neuron_ids"]:
            try:
                setattr(caneurons, var_name, getattr(caneurons, var_name)[keep_bool])
            except IndexError:
                print(
                    f"Error trimming variable {var_name}: using un-trimmed version. Be sure to cleanup before processing!"
                )
                setattr(caneurons, var_name, getattr(caneurons, var_name))

        return caneurons

    def to_caneurons(self, trim=None, **kwargs):
        """Send to CaNeurons class"""

        return CaNeurons(
            A=self.A,
            C=self.C,
            S=self.S,
            YrA=self.YrA,
            t=self.times,
            trim=trim,
            basedir=self.basedir,
            **kwargs,
        )


def fix_dim_mismatch(
    bad_zarr_path, good_zarr_path, mismatch_dim="unit_id", save_numpy=True
):
    """Fixes 'bad_zarr' which has too many items (typically too many units)
    by inferring those times from good_zarr.  Mostly happens with YrA."""
    bad_var = xr.open_zarr(bad_zarr_path)
    good_var = xr.open_zarr(good_zarr_path)

    fixed_var = bad_var.sel({mismatch_dim: good_var[mismatch_dim].values})
    var_name = list(bad_var.values())[0].name
    fixed_var_path = Path(bad_zarr_path).parent / (var_name + "_fixed.zarr")

    assert (
        not fixed_var_path.is_dir()
    ), f'Existing {var_name + "_fixed.zarr"} group exists. Delete and re-try.'
    fixed_var.to_zarr(fixed_var_path)

    if save_numpy:
        fixed_var_np_path = Path(bad_zarr_path).parent / (var_name + "_fixed.npy")
        save_zarr_to_numpy(fixed_var, fixed_var_np_path)

    return None


def save_zarr_to_numpy(
    zarr_var: str or Path or xr.Dataset or xr.DataArray, save_dir=None
):
    """Saves a zarr/xarray variable as a numpy file"""
    if isinstance(zarr_var, str) or isinstance(zarr_var, Path):
        zarr = list(xr.open_zarr(zarr_var).values())[0]
        save_dir = Path(zarr_var).parent
    elif isinstance(zarr_var, xr.Dataset):
        zarr = list(zarr_var.values())[0]
        assert (
            save_dir is not None
        ), "Must specify save_dir to save to if inputting xr.Dataset"
        save_dir = Path(save_dir).parent
    elif isinstance(zarr_var, xr.DataArray):
        zarr = zarr_var
        assert (
            save_dir is not None
        ), "Must specify save_dir to save to if inputting xr.DataArray"
        save_dir = Path(save_dir).parent
    elif not isinstance(zarr_var, xr.DataArray):
        assert False, "zarr_var is not appropriate input type"

    var_name = zarr.name
    np.save(save_dir / (var_name + ".npy"), zarr.values)
    pass


def load_var(var_name, allow_numpy=False):
    """Loads a variable in .zarr format OR in numpy if allowed"""
    pass


def check_integrity(var_list, fix: str or bool in [False, "min_size"] = "min_size"):
    """Checks integrity of variables to make sure they match across frames, width, and height depending on variable type.
    e.g. checks if A and min_proj and max_proj all have the same width/height.
    Optionally can fix variables to match the min # frames and min width/height if specified"""
    pass


if __name__ == "__main__":
    from session_directory import get_session_dir

    animal = "Jyn"
    session = "Training"

    # Get session directory
    sesh_dir = get_session_dir(animal, session)

    # Load in ca imaging data from minian
    minian = MinianIO(basedir=sesh_dir)

    # Keep only good neurons
    caneurons = minian.trim_neurons(keep=["good", "maybe_interneurons"], trim=None)
    # caneurons.plot_rois_with_min_proj()
