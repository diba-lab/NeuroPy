from artifactDetect import findartifact
from behavior import behavior_epochs
from decoders import DecodeBehav
from eventCorr import event_event
from getPosition import ExtractPosition
from getSpikes import Spikes
from lfpEvent import Hswa, Ripple, Spindle, Theta, Gamma
from MakePrmKlusta import makePrmPrb
from parsePath import Recinfo
from pfPlot import pf
from replay import Replay
from sessionUtil import SessionUtil
from sleepDetect import SleepScore
from spkEvent import PBE, LocalSleep
from viewerData import SessView


class processData:
    def __init__(self, basepath):
        """Make sure to enter in the tracking scale factor if you have used a properly sized wand to optitrack calibration"""
        self.recinfo = Recinfo(basepath)

        self.position = ExtractPosition(self.recinfo)
        self.epochs = behavior_epochs(self.recinfo)
        self.artifact = findartifact(self.recinfo)
        self.makePrmPrb = makePrmPrb(self.recinfo)
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
        self.decode = DecodeBehav(self.recinfo)
        self.localsleep = LocalSleep(self.recinfo)
        self.viewdata = SessView(self.recinfo)
        self.pbe = PBE(self.recinfo)

        self.eventpsth = event_event()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.session.sessionName})"


if __name__ == "__main__":
    sess = processData(
        r"C:\Users\Nat\Documents\UM\Working\Opto\Jackie671\Jackie_3well_day4\Jackie_UTRACK_combined"
    )
    sess.position.getPosition(method="from_files", scale=4.0)
pass
