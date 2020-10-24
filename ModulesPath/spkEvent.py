from matplotlib.pyplot import axis
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mathutil import parcorr_mult, getICA_Assembly
import scipy.stats as stats
import scipy.signal as sg
from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mathutil import threshPeriods
from parsePath import Recinfo


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

    binSize = 0.001  # in seconds
    gauss_std = 0.025  # in seconds

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        filePrefix = self._obj.files.filePrefix
        self._filename = Path(str(filePrefix) + "_localsleep.npy")
        if self._filename.is_file():
            data = np.load(self._filename, allow_pickle=True).item()
            self.events = pd.DataFrame(
                {key: data[key] for key in ("start", "end", "duration")}
            )
            self.instfiringbefore = data["instfiringbefore"]
            self.instfiringafter = data["instfiringafter"]
            self.instfiring = data["instfiring"]
            self.avglfp = data["avglfp"]
            self.period = data["period"]

    def _gaussian(self):
        """Gaussian function for generating instantenous firing rate

        Returns:
            [array] -- [gaussian kernel centered at zero and spans from -1 to 1 seconds]
        """
        sigma = self.gauss_std
        t_gauss = np.arange(-1, 1, self.binSize)
        A = 1 / np.sqrt(2 * np.pi * sigma ** 2)
        gaussian = A * np.exp(-(t_gauss ** 2) / (2 * sigma ** 2))

        return gaussian

    def detect(self, period=None):

        if period is None:
            raise ValueError("please provide a valid time period")

        assert len(period) == 2, "length of period should be 2"

        tstart = period[0]
        tend = period[1]
        spikes = self._obj.spikes.times

        tbin = np.arange(tstart, tend, self.binSize)
        mua = np.concatenate(spikes)
        spikecount = np.histogram(mua, tbin)[0]
        gaussKernel = self._gaussian()
        instfiring = sg.convolve(spikecount, gaussKernel, mode="same", method="direct")

        # off periods
        off = np.diff(np.where(instfiring < np.median(instfiring), 1, 0))
        start_off = np.where(off == 1)[0]
        end_off = np.where(off == -1)[0]

        if start_off[0] > end_off[0]:
            end_off = end_off[1:]
        if start_off[-1] > end_off[-1]:
            start_off = start_off[:-1]

        offperiods = np.vstack((start_off, end_off)).T
        duration = np.diff(offperiods, axis=1).squeeze()

        # ===== calculate the minimum instantenous firing rate within the intervals
        minValue = np.zeros(len(offperiods))
        for i in range(0, len(offperiods)):
            minValue[i] = min(instfiring[offperiods[i, 0] : offperiods[i, 1]])

        # selecting only top 10 percent of lowest peak instfiring , refer Vyazovskiy et al.
        quantiles = pd.qcut(minValue, 10, labels=False)
        top10percent = np.where(quantiles == 0)[0]
        offperiods = offperiods[top10percent, :]
        duration = duration[top10percent]

        lfpSrate = self._obj.recinfo.lfpSrate
        lfp, _, _ = self._obj.spindle.best_chan_lfp()
        t = np.linspace(0, len(lfp) / lfpSrate, len(lfp))
        fratebefore, frateafter = [], []
        avglfp = np.zeros(lfpSrate * 2)
        for (start, stop) in offperiods:
            fratebefore.append(instfiring[start - 1000 : start])
            frateafter.append(instfiring[stop : stop + 1000])
            tlfp_start = int(tbin[start] * lfpSrate)
            avglfp = avglfp + lfp[tlfp_start - lfpSrate : tlfp_start + lfpSrate]

        avglfp = stats.zscore(avglfp / len(offperiods))

        locsleep = {
            "start": tbin[offperiods[:, 0]],
            "end": tbin[offperiods[:, 1]],
            "duration": duration / 1000,
            "instfiringbefore": np.asarray(fratebefore),
            "instfiringafter": np.asarray(frateafter),
            "instfiring": instfiring,
            "avglfp": avglfp,
            "period": period,
        }

        np.save(self._filename, locsleep)

    def plot(self, fig=None, ax=None):
        lfpSrate = self._obj.recinfo.lfpSrate
        lfp, _, _ = self._obj.spindle.best_chan_lfp()
        t = np.linspace(0, len(lfp) / lfpSrate, len(lfp))
        spikes = self._obj.spikes.times

        post = self._obj.epochs.post
        period = post
        period_duration = np.diff(period)
        spikes_sd = [
            cell[np.where((cell > period[0]) & (cell < period[1]))[0]]
            for cell in spikes
        ]
        frate = np.asarray(
            [len(cell) / period_duration for cell in spikes_sd]
        ).squeeze()
        sort_frate_indices = np.argsort(frate)
        spikes = [spikes[indx] for indx in sort_frate_indices]

        selectedEvents = self.events.sample(n=5)
        instfiring = self.instfiring
        t_instfiring = np.linspace(self.period[0], self.period[1], len(instfiring))

        if ax is None:
            fig = plt.figure(num=None, figsize=(20, 7))
            gs = gridspec.GridSpec(3, 5, figure=fig)
            fig.subplots_adjust(hspace=0.5)

        else:
            gs = gridspec.GridSpecFromSubplotSpec(
                2, 5, subplot_spec=ax, wspace=0.1, hspace=0.1
            )

        taround = 2
        for ind, period in enumerate(selectedEvents.itertuples()):

            ax = fig.add_subplot(gs[0, ind])
            lfp_period = lfp[(t > period.start - taround) & (t < period.end + taround)]
            t_period = np.linspace(
                period.start - taround, period.end + taround, len(lfp_period)
            )
            instfiring_period = instfiring[
                (t_instfiring > period.start - taround)
                & (t_instfiring < period.end + taround)
            ]
            inst_tperiod = t_instfiring[
                (t_instfiring > period.start - taround)
                & (t_instfiring < period.end + taround)
            ]

            # ax.plot([period.start, period.start], [0, 100], "r")
            # ax.plot([period.end, period.end], [0, 100], "k")
            ax.fill_between(
                [period.start, period.end], [0, 0], [90, 90], alpha=0.3, color="#BDBDBD"
            )
            ax.fill_between(
                inst_tperiod, instfiring_period / 50, alpha=0.3, color="#212121",
            )
            ax.plot(
                t_period,
                stats.zscore(lfp_period) * 4 + len(spikes) + 15,
                "k",
                linewidth=0.8,
            )

            cmap = mpl.cm.get_cmap("inferno_r")

            for cell, spk in enumerate(spikes):
                color = cmap(cell / len(spikes))

                spk = spk[(spk > period.start - taround) & (spk < period.end + taround)]
                ax.plot(spk, cell * np.ones(len(spk)), "|", color=color, markersize=2)

            ax.set_title(f"{round(period.duration,2)} s")
            ax.axis("off")

        ax = fig.add_subplot(gs[1, 0])
        self.events["duration"].plot.kde(ax=ax, color="#616161")
        ax.set_xlim([0, max(self.events.duration)])
        ax.set_xlabel("Duration (s)")

        ax = fig.add_subplot(gs[1, 1])
        fbefore = self.instfiringbefore[:-1].mean(axis=0)
        fbeforestd = self.instfiringbefore[:-1].std(axis=0) / np.sqrt(len(self.events))
        fafter = self.instfiringafter[:-1].mean(axis=0)
        fafterstd = self.instfiringafter[:-1].std(axis=0) / np.sqrt(len(self.events))
        tbefore = np.linspace(-1, 0, len(fbefore))
        tafter = np.linspace(0.2, 1.2, len(fafter))

        ax.fill_between(
            [0, 0.2],
            [min(fbefore), min(fbefore)],
            [max(fbefore), max(fbefore)],
            color="#BDBDBD",
            alpha=0.3,
        )
        ax.fill_between(
            tbefore, fbefore + fbeforestd, fbefore - fbeforestd, color="#BDBDBD"
        )
        ax.plot(tbefore, fbefore, color="#616161")
        ax.fill_between(tafter, fafter + fafterstd, fafter - fafterstd, color="#BDBDBD")
        ax.plot(tafter, fafter, color="#616161")

        # self.events["duration"].plot.kde(ax=ax, color="k")
        # ax.set_xlim([0, max(self.events.duration)])
        ax.set_xlabel("Time from local sleep (s)")
        ax.set_ylabel("Instantneous firing")
        ax.set_xticks([-1, -0.5, 0, 0.2, 0.7, 1.2])
        ax.set_xticklabels(["-1", "-0.5", "start", "end", "0.5", "1"], rotation=45)
        # ax.set_ylim([10, 50])

        # ax = fig.add_subplot(gs[1, 2])
        # ax.plot(self.avglfp)
        # ax.set_xlim([0, max(self.events.duration)])
        # ax.set_xlabel("Duration (s)")

        subname = self._obj.sessinfo.session.subname
        fig.suptitle(f"Local sleep during sleep deprivation in {subname}")

    def plotAll(self):
        spikes = self._obj.spikes.times
        tstart = self._obj.epochs.post[0]
        tend = self._obj.epochs.post[0] + 5 * 3600
        lfp, _, _ = self._obj.spindle.best_chan_lfp()
        t = np.linspace(0, len(lfp) / 1250, len(lfp))
        lfpsd = stats.zscore(lfp[(t > tstart) & (t < tend)]) + 50

        for period in self.events.itertuples():
            plt.plot([period.start, period.start], [0, 70], "r")
            plt.plot([period.end, period.end], [0, 70], "k")

        plt.plot(np.linspace(tstart, tend, len(lfpsd)), lfpsd, "k")
        for cell, spk in enumerate(spikes):
            spk = spk[(spk > tstart) & (spk < tend)]
            plt.plot(spk, cell * np.ones(len(spk)), "|")


class PBE:
    """Populations burst events 
    """

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        filePrefix = self._obj.files.filePrefix

        @dataclass
        class files:
            pbe: str = Path(str(filePrefix) + "_pbe.pkl")

        self.files = files()

        if self.files.pbe.is_file():
            self.events = pd.read_pickle(self.files.pbe)

    def detect(self):
        instfiring = self._obj.spikes.instfiring
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

        data.to_pickle(self.files.pbe)

