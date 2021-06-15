from .io import NeuroscopeIO, RawBinarySignalIO
import pickle
from . import sessobj
from .core import ProbeGroup
from pathlib import Path


class ProcessData:
    def __init__(self, basepath, data_format="NeuroscopeIO"):
        basepath = Path(basepath)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        self.filePrefix = xml_files[0].with_suffix("")
        self.recinfo = NeuroscopeIO(xml_files[0])
        self.eegfile = RawBinarySignalIO(
            self.recinfo.eeg_filename,
            n_chans=self.recinfo.n_channels,
            sampling_rate=self.recinfo.eeg_sampling_rate,
        )
        self.datfile = RawBinarySignalIO(
            self.recinfo.dat_filename,
            n_chans=self.recinfo.n_channels,
            sampling_rate=self.recinfo.dat_sampling_rate,
        )

        self.probegroup = ProbeGroup(filename=self.filePrefix.with_suffix(".prb.npy"))
        self.artifact = sessobj.Artifact(
            signal=self.eegfile, filename=self.filePrefix.with_suffix(".artifact.npy")
        )
        self.paradigm = sessobj.Paradigm(
            filename=self.filePrefix.with_suffix(".paradigm.npy")
        )
        # self.position = sessobj.SessPosition(self.recinfo)
        # self.track = sessobj.SessTrack(basepath=self.recinfo, position=self.position)

        # self.neurons = sessobj.SessNeurons(self.recinfo)
        # self.brainstates = sessobj.BrainStates(self.recinfo)
        # self.swa = sessobj.Hswa(self.recinfo)
        # self.theta = sessobj.Theta(self.recinfo)
        # self.spindle = sessobj.Spindle(self.recinfo)
        # self.gamma = sessobj.Gamma(self.recinfo)
        self.ripple = sessobj.Ripple(
            self.eegfile,
            self.probegroup,
            filename=self.filePrefix.with_suffix(".ripples.npy"),
        )
        # self.expvar = sessobj.ExplainedVariance(self.recinfo, self.neurons)
        # self.assembly = sessobj.CellAssembly(self.recinfo, self.neurons)
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
