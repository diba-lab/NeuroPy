from .artifactDetect import findartifact
from .behavior import behavior_epochs
from .decoders import DecodeBehav
from .getPosition import ExtractPosition
from .getSpikes import Spikes
from .lfpEvent import Hswa, Ripple, Spindle, Theta, Gamma
from .parsePath import Recinfo
from .pfPlot import pf
from .replay import Replay
from .sessionUtil import SessionUtil
from .sleepDetect import SleepScore
from .spkEvent import PBE, LocalSleep
from .viewerData import SessView
from .track import Track
from .getPosition import SessPosition, SessTrack
import pickle


class processData:
    def __init__(self, basepath):
        """Make sure to enter in the tracking scale factor if you have used a properly sized wand to optitrack calibration"""
        self.recinfo = Recinfo(basepath)

        self.position = ExtractPosition(self.recinfo)
        self.tracks = Track(self.recinfo)
        self.pos = SessPosition(self.recinfo)
        self.tracku = SessTrack(basepath=self.recinfo, position=self.pos)
        self.paradigm = behavior_epochs(self.recinfo)
        self.artifact = findartifact(self.recinfo)
        self.utils = SessionUtil(self.recinfo)

        self.spikes = Spikes(self.recinfo)
        self.brainstates = SleepScore(self.recinfo)
        self.swa = Hswa(self.recinfo)
        self.theta = Theta(self.recinfo)
        self.spindle = Spindle(self.recinfo)
        self.gamma = Gamma(self.recinfo)
        self.ripple = Ripple(self.recinfo)
        self.placefield = pf(self.recinfo)
        self.replay = Replay(self.recinfo)
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
