from .parsePath import Recinfo
import pickle
from . import sessobj


class ProcessData:
    def __init__(self, basepath):
        self.recinfo = Recinfo(basepath)

        self.artifact = sessobj.Artifact(self.recinfo)
        self.paradigm = sessobj.Paradigm(self.recinfo)
        self.position = sessobj.SessPosition(self.recinfo)
        self.track = sessobj.SessTrack(basepath=self.recinfo, position=self.position)

        self.neurons = sessobj.SessNeurons(self.recinfo)
        self.brainstates = sessobj.BrainStates(self.recinfo)
        self.swa = sessobj.Hswa(self.recinfo)
        self.theta = sessobj.Theta(self.recinfo)
        self.spindle = sessobj.Spindle(self.recinfo)
        self.gamma = sessobj.Gamma(self.recinfo)
        self.ripple = sessobj.Ripple(self.recinfo)
        self.expvar = sessobj.ExplainedVariance(self.recinfo, self.neurons)
        self.assembly = sessobj.CellAssembly(self.recinfo, self.neurons)
        self.pf1d = sessobj.PF1d(self.recinfo)
        self.pf2d = sessobj.PF2d(self.recinfo)
        self.decode1D = sessobj.Decode1d(self.pf1d)
        self.decode2D = sessobj.Decode2d(self.pf2d)
        self.localsleep = sessobj.LocalSleep(self.recinfo)
        self.pbe = sessobj.Pbe(self.recinfo)

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


# test
if __name__ == "__main__":
    sess = processData("/data/Working/Opto/Jackie671/Jackie_propofol_2020-09-30")
    sess.spikes.load_rough_mua()
    sess.spikes.roughmua2neuroscope([7, 8, 6, 5, 9], [4, 4, 4, 4, 4])
pass
