from matplotlib.pyplot import plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib as mpl
import random
import ipywidgets as widgets
from parsePath import Recinfo


def make_boxes(
    ax, xdata, ydata, xerror, yerror, facecolor="r", edgecolor="None", alpha=0.5
):

    # Loop over data points; create box from errors at each point
    errorboxes = [
        Rectangle((x, y), xe, ye) for x, y, xe, ye in zip(xdata, ydata, xerror, yerror)
    ]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(
        errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor
    )

    # Add collection to axes
    ax.add_collection(pc)

    return 1


class SessView:
    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def specgram(self, chan=None, period=None, window=4, overlap=2, ax=None):
        """Generating spectrogram plot for given channel

        Parameters
        ----------
        chan : [int], optional
            channel to plot, by default None and chooses a channel randomly
        period : [type], optional
            plot only for this duration in the session, by default None
        window : [float, seconds], optional
            window binning size for spectrogram, by default 4
        overlap : [float, seconds], optional
            overlap between windows, by default 2
        ax : [obj], optional
            if none generates a new figure, by default None
        """

        if chan is None:
            goodchans = self._obj.goodchans
            chan = random.choice(goodchans)

        spec = self._obj.utils.spectrogram(
            chan=chan, period=period, window=window, overlap=overlap
        )
        sxx = spec.sxx / np.max(spec.sxx)
        sxx = gaussian_filter(sxx, sigma=2)
        vmax = np.max(sxx) / 4

        if ax is None:
            _, ax = plt.subplots(1, 1)

        # ---------- plotting ----------------
        def plotspec(val, cmap, freq):
            ax.pcolorfast(spec.time, spec.freq, sxx, cmap=cmap, vmax=val)
            ax.set_ylim(freq)

        ax.text(
            np.max(spec.time) / 2,
            25,
            f"Spectrogram for channel {chan}",
            ha="center",
            color="w",
        )
        ax.set_xlim([np.min(spec.time), np.max(spec.time)])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        axins = ax.inset_axes([0, 0.6, 0.1, 0.25])
        self._obj.utils.plotChanPos(chans=[chan], ax=axins)
        axins.axis("off")

        # ---- updating plotting values for interaction ------------
        widgets.interact(
            plotspec,
            val=widgets.FloatSlider(
                value=vmax,
                min=vmax / 10,
                max=4 * vmax,
                step=0.001,
                description="Spec. Clim:",
            ),
            cmap=widgets.Dropdown(
                options=["Spectral_r", "copper", "hot_r"],
                value="Spectral_r",
                description="Colormap:",
            ),
            freq=widgets.IntRangeSlider(
                value=[0, 30], min=0, max=625, step=1, description="Freq. range:"
            ),
        )

    def epoch(self, ax=None):
        epochs = self._obj.epochs.times

        if ax is None:
            ax = plt.subplots(1, 1)

        ind = -1
        for col in epochs.columns:
            period = epochs[col].values
            ax.fill_between(
                period,
                (ind + 1) * np.ones(len(period)),
                ind * np.ones(len(period)),
                color="#1DE9B6",
            )
            ind = ind - 1
        ax.set_yticks([-0.5, -1.5, -2.5])
        ax.set_yticklabels(["pre", "maze", "post"])
        ax.spines["left"].set_visible(False)
        # ax.set_xticklabels([""])

    def position(self):
        pass

    def raster(self, ax=None, period=None):
        spikes = self._obj.spikes.times
        totalduration = self._obj.epochs.totalduration
        frate = [len(cell) / totalduration for cell in spikes]

        if ax is None:
            fig = plt.figure(1, figsize=(6, 10))
            gs = GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.4)
            ax = fig.add_subplot(gs[0])

        if period is not None:
            period_duration = np.diff(period)
            spikes = [
                cell[np.where((cell > period[0]) & (cell < period[1]))[0]]
                for cell in spikes
            ]
            frate = np.asarray(
                [len(cell) / period_duration for cell in spikes]
            ).squeeze()
            print(frate.shape)

        sort_frate_indices = np.argsort(frate)
        spikes = [spikes[indx] for indx in sort_frate_indices]

        cmap = mpl.cm.get_cmap("inferno_r")
        for cell, spk in enumerate(spikes):
            color = cmap(cell / len(spikes))
            plt.plot(
                spk, (cell + 1) * np.ones(len(spk)), "|", markersize=0.75, color=color
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neurons")

    def hypnogram(self, ax1=None, tstart=0.0, unit="s"):
        """Plots hypnogram in the given axis

        Args:
            ax1 (axis, optional): axis for plotting. Defaults to None.
            tstart (float, optional): Start time of hypnogram. Defaults to 0.
            unit (str, optional): Unit of time in seconds or hour. Defaults to "s".
        """
        states = self._obj.brainstates.states

        if ax1 is None:
            fig = plt.figure(1, figsize=(6, 10))
            gs = GridSpec(9, 1, figure=fig)
            fig.subplots_adjust(hspace=0.4)
            ax1 = fig.add_subplot(gs[0, 0])

        x = np.asarray(states.start) - tstart
        y = np.zeros(len(x)) + np.asarray(states.state)
        width = np.asarray(states.duration)
        height = np.ones(len(x))

        colors = ["#6b90d1", "#eb9494", "#b6afaf", "#474343"]
        col = [colors[int(state) - 1] for state in states.state]

        if unit == "s":
            make_boxes(ax1, x, y, width, height, facecolor=col)
        if unit == "h":
            make_boxes(ax1, x / 3600, y, width / 3600, height, facecolor=col)
        ax1.set_ylim(1, 5)
        ax1.axis("off")

    def lfpevents(self, ax=None):
        ripples = self._obj.ripple.time
        spindles = self._obj.spindle.time

        if ax is None:
            fig = plt.figure(1, figsize=(6, 10))
            gs = GridSpec(9, 1, figure=fig)
            fig.subplots_adjust(hspace=0.4)
            ax = fig.add_subplot(gs[0, 0])

        # colors = ["#6b90d1", "#eb9494", "#b6afaf", "#474343"]
        # col = [colors[int(state) - 1] for state in states.state]
        width = np.diff(ripples, axis=1).squeeze()
        height = 0.2 * np.ones(len(ripples))
        # ax.plot(ripples[:, 0], np.ones(len(ripples)), ".", markersize=0.5)

        make_boxes(
            ax,
            ripples[:, 0],
            np.ones(len(ripples)),
            width,
            height,
            facecolor="#eb9494",
        )
        ax.set_ylim(1, 1.2)

    def summary(self):

        fig = plt.figure(num=None, figsize=(20, 7))
        gs = GridSpec(10, 5, figure=fig)
        fig.subplots_adjust(hspace=0.5)

        ax = fig.add_subplot(gs[1:3, :])
        self.specgram(ax=ax)

        ax = fig.add_subplot(gs[0, :], sharex=ax)
        self.hypnogram(ax1=ax)

        ax = fig.add_subplot(gs[3, :], sharex=ax)
        self.epoch(ax=ax)

        ax = fig.add_subplot(gs[4:6, :], sharex=ax)
        self.raster(ax=ax)

        ax = fig.add_subplot(gs[6, :], sharex=ax)
        self.lfpevents(ax=ax)

    def testsleep(self):
        lfp = self._obj.spindle.best_chan_lfp()[0]
        lfp = np.c_[lfp, lfp].T
        lfpt = np.linspace(0, len(lfp) / 1250, len(lfp))
        channels = [1, 2]

        hypno = None
        # Sleep(data=lfp, hypno=hypno, channels=channels, sf=1250).show()
