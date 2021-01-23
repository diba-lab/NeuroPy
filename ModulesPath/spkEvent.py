from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg
from pathlib import Path
from mathutil import threshPeriods
from parsePath import Recinfo
from getSpikes import Spikes
from plotUtil import Fig


class LocalSleep:

    """This detects local sleep which are brief periods of silence in ongoing acitvity, originally reported in the cortex.


    Raises:
        ValueError: [description]

    Returns:
        [type] -- [description]

    References
    ------------
    1) Vyazovskiy, V. V., Olcese, U., Hanlon, E. C., Nir, Y., Cirelli, C., & Tononi, G. (2011). Local sleep in awake rats. Nature, 472(7344), 443-447. https://www.nature.com/articles/nature10009

    """

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ------- defining file names ---------
        filePrefix = self._obj.files.filePrefix

        @dataclass
        class files:
            events: Path = filePrefix.with_suffix(".localsleep.npy")

        self.files = files()
        self._load()

    def _load(self):

        if (f := self.files.events).is_file():
            data = np.load(f, allow_pickle=True).item()
            self.events = data["events"]
            self.params = data["params"]

    def detect(self, period):
        """Detects local OFF events in within period

        Parameters
        ----------
        period : list,array-like
            period in seconds

        """

        if period is None:
            raise ValueError("please provide a valid time period")

        assert len(period) == 2, "length of period should be 2"

        instfiring = Spikes(self._obj).instfiring
        instfiring = instfiring[
            (instfiring.time > period[0]) & (instfiring.time < period[1])
        ]
        time = instfiring.time.to_numpy()
        frate = instfiring.frate.to_numpy()

        # off periods
        off = np.diff(np.where(frate < np.median(frate), 1, 0))
        start_off = np.where(off == 1)[0]
        end_off = np.where(off == -1)[0]

        if start_off[0] > end_off[0]:
            end_off = end_off[1:]
        if start_off[-1] > end_off[-1]:
            start_off = start_off[:-1]

        offperiods = np.vstack((start_off, end_off)).T
        duration = np.diff(offperiods, axis=1).squeeze()

        # ---- calculate minimum instantenous frate within intervals ------
        minValue = np.zeros(len(offperiods))
        for i in range(0, len(offperiods)):
            minValue[i] = min(frate[offperiods[i, 0] : offperiods[i, 1]])

        # --- selecting only top 10 percent of lowest peak instfiring -----
        quantiles = pd.qcut(minValue, 10, labels=False)
        top10percent = np.where(quantiles == 0)[0]
        offperiods = offperiods[top10percent, :]
        duration = duration[top10percent]

        events = pd.DataFrame(
            {
                "start": time[offperiods[:, 0]],
                "end": time[offperiods[:, 1]],
                "duration": duration / 1000,
            }
        )
        params = {"period": period}

        data = {"events": events, "params": params}

        np.save(self.files.events, data)
        self._load()

    def plot_examples(self, ax=None, num=3, chan=None, flank_time=0.5):

        spikes = Spikes(self._obj)
        event_id = np.random.choice(len(self.events), num)

        if ax is None:
            figure = Fig()
            _, gs = figure.draw(grid=(1, num))

        events = self.events.iloc[event_id]
        for i, epoch in enumerate(events.itertuples()):
            ax = plt.subplot(gs[i])
            spikes.plot_raster(
                spikes=spikes.times,
                ax=ax,
                period=[epoch.start - flank_time, epoch.end + flank_time],
                sort_by_frate=False,
                color="#14213e",
            )
            ax.axvspan(epoch.start, epoch.end, color="gray", alpha=0.5)
            ax.set_xlim(epoch.start - flank_time, epoch.end + flank_time)
            ax.axis("off")


class PBE:
    """Populations burst events"""

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        filePrefix = self._obj.files.filePrefix
        self.spikes = Spikes(self._obj)

        @dataclass
        class files:
            pbe: str = Path(str(filePrefix) + "_pbe.pkl")
            neuroscope: Path = filePrefix.with_suffix(".evt.pbe")

        self.files = files()

        if self.files.pbe.is_file():
            self.events = pd.read_pickle(self.files.pbe)

    def detect(self):
        instfiring = self.spikes.instfiring
        events = threshPeriods(
            stats.zscore(instfiring.frate), lowthresh=0, highthresh=3, minDuration=100
        )

        time = np.asarray(instfiring.time)
        pbe_times = time[events]

        data = pd.DataFrame(
            {
                "start": pbe_times[:, 0],
                "end": pbe_times[:, 1],
                "duration": np.diff(pbe_times, axis=1).squeeze(),
            }
        )

        self.events = data  # make this available immediately after setting

        data.to_pickle(self.files.pbe)

    def export2Neuroscope(self):
        with self.files.neuroscope.open("w") as a:
            for event in self.events.itertuples():
                a.write(f"{event.start*1000} start\n{event.end*1000} end\n")
