from .decoders import DecodeBehav
from .parsePath import Recinfo
from .pfPlot import pf
from .replay import Replay
from .sessionUtil import SessionUtil
from .sleepDetect import SleepScore
from .spkEvent import PBE, LocalSleep
from .viewerData import SessView
import pickle
from . import sessobj


class processData:
    def __init__(self, basepath):
        self.recinfo = Recinfo(basepath)

        self.position = sessobj.SessPosition(self.recinfo)
        self.track = sessobj.SessTrack(basepath=self.recinfo, position=self.position)
        self.paradigm = sessobj.Paradigm(self.recinfo)
        self.artifact = sessobj.Artifact(self.recinfo)
        self.utils = SessionUtil(self.recinfo)

        self.neurons = sessobj.SessNeurons(self.recinfo)
        self.brainstates = SleepScore(self.recinfo)
        self.swa = sessobj.Hswa(self.recinfo)
        self.theta = sessobj.Theta(self.recinfo)
        self.spindle = sessobj.Spindle(self.recinfo)
        self.gamma = sessobj.Gamma(self.recinfo)
        self.ripple = sessobj.Ripple(self.recinfo)
        self.placefield = pf(self.recinfo)
        self.replay = Replay(self.recinfo, self.neurons)
        self.decode = DecodeBehav(self.placefield.pf1d, self.placefield.pf2d)
        self.localsleep = LocalSleep(self.recinfo)
        self.viewdata = SessView(self.recinfo)
        self.pbe = PBE(self.recinfo)

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
