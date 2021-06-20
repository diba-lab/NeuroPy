from .io import NeuroscopeIO, BinarySignalIO
import pickle
from . import core
from pathlib import Path
import numpy as np


class ProcessData:
    def __init__(self, basepath):
        basepath = Path(basepath)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        self.filePrefix = xml_files[0].with_suffix("")
        self.recinfo = NeuroscopeIO(xml_files[0])
        self.eegfile = BinarySignalIO(
            self.recinfo.eeg_filename,
            n_chans=self.recinfo.n_channels,
            sampling_rate=self.recinfo.eeg_sampling_rate,
        )
        self.datfile = BinarySignalIO(
            self.recinfo.dat_filename,
            n_chans=self.recinfo.n_channels,
            sampling_rate=self.recinfo.dat_sampling_rate,
        )

        self.probegroup = core.ProbeGroup(
            filename=self.filePrefix.with_suffix(".prb.npy")
        )

        if (f := self.filePrefix.with_suffix(".paradim.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.paradigm = core.Epoch.from_dict(d)

        self.position = core.Position(
            filename=self.filePrefix.with_suffix(".position.npy")
        )

        if (f := self.filePrefix.with_suffix(".artifact.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.artifact = core.Epoch.from_dict(d)

        if (f := self.filePrefix.with_suffix(".brainstates.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.brainstates = core.Epoch.from_dict(d)

        if (f := self.filePrefix.with_suffix(".ripple.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.ripple = core.Epoch.from_dict(d)

        if (f := self.filePrefix.with_suffix(".theta.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.theta = core.Epoch.from_dict(d)

        if (f := self.filePrefix.with_suffix(".gamma.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.gamma = core.Epoch.from_dict(d)

        if (f := self.filePrefix.with_suffix(".spindle.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.spindle = core.Epoch.from_dict(d)

        # ---- spiketrains related ------------

        if (f := self.filePrefix.with_suffix(".neurons.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.neurons = core.Neurons.from_dict(d)

        if (f := self.filePrefix.with_suffix(".pbe.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.pbe = core.Epoch.from_dict(d)

        if (f := self.filePrefix.with_suffix(".localsleep.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.pbe = core.Epoch.from_dict(d)

        if (f := self.filePrefix.with_suffix(".lowstates.npy").is_file()) :
            d = np.load(f, allow_pickle=True).item()
            self.pbe = core.Epoch.from_dict(d)

        # self.pf1d = sessobj.PF1d(self.recinfo)
        # self.pf2d = sessobj.PF2d(self.recinfo)
        # self.decode1D = sessobj.Decode1d(self.pf1d)
        # self.decode2D = sessobj.Decode2d(self.pf2d)
        # self.localsleep = sessobj.LocalSleep(self.recinfo)
        # self.pbe = sessobj.Pbe(self.recinfo)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.session.sessionName})"

    def export_rearing_data(self):
        """Export position/ripple/pbe data for rearing analysis"""
        rearing_data = {
            "time": self.position.data["time"],
            "x": self.position.data["x"],
            "y": self.position.data["y"],
            "z": self.position.data["z"],
            "datetime": self.position.data["datetime"],
            "ripple": self.ripple.epochs,
            "pbe": self.pbe.epochs,
            "lfpsRate": self.recinfo.lfpSrate,
            "video_start_time": self.position.video_start_time,
        }

        with open(
            self.recinfo.files.filePrefix.with_suffix(".rearing_data.pkl"), "wb"
        ) as f:
            pickle.dump(rearing_data, f)
