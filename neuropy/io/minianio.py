from pathlib import Path
import numpy as np
from pickle import dump, load
import xarray as xr
import pandas as pd

from neuropy.core.ca_neurons import CaNeurons
from neuropy.io.miniscopeio import MiniscopeIO


class MinianIO:
    def __init__(self, dirname: str or None = None) -> None:

        # Try to autodetect minian folder
        if dirname is None:
            assert (
                len(dirname := sorted(self.basedir.glob("**/minian"))) == 1
            ), "More than one minian folder found, fill in directory manually"
            dirname = Path(dirname[0])
        self.dir = Path(dirname)

        # Load in relevant variables
        import_vars = ["A", "C", "S", "YrA"]
        for var in import_vars:
            setattr(self, var, np.load(self.dir / (var + ".npy")))

        # Load in hand-curated neurons
        with open(self.dir / "curated_neurons.pkl", "rb") as f:
            self.curated_neurons = load(f)

        # Load in good frames
        try:
            self.good_frames = np.load(self.dir / "frames.npy")
        except FileNotFoundError:
            print(
                "frames.npy not found in minian directory, trying to load in zarr files to get frames"
            )
            motion = xr.open_zarr(self.dir / "motion.zarr")
            self.good_frames = motion.frame.values

        # Import timestamps
        msio = MiniscopeIO(self.dir)
        self.times = msio.load_all_timestamps()

    def to_caneurons(self):
        """Send to CaNeurons class"""

        return CaNeurons(A=self.A, C=self.C, S=self.S, YrA=self.YrA, t=self.time)
