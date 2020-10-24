from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.ndimage as filtSig
import scipy.signal as sg
import scipy.stats as stats
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.ndimage import gaussian_filter

from parsePath import Recinfo
from signal_process import spectrogramBands


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

    # Plot errorbars
    # artists = ax.errorbar(
    #     xdata, ydata, xerr=xerror, yerr=yerror, fmt="None", ecolor="k"
    # )
    return 1


def genepoch(start, end):

    if start[0] > end[0]:
        end = end[1:]

    if start[-1] > end[-1]:
        start = start[:-1]

    firstPass = np.vstack((start, end)).T

    # ===== merging close ripples
    minInterSamples = 20
    secondPass = []
    state = firstPass[0]
    for i in range(1, len(firstPass)):
        if firstPass[i, 0] - state[1] < minInterSamples:
            # Merging states
            state = [state[0], firstPass[i, 1]]
        else:
            secondPass.append(state)
            state = firstPass[i]

    secondPass.append(state)
    secondPass = np.asarray(secondPass)

    state_duration = np.diff(secondPass, axis=1)

    # delete very short ripples
    minstateDuration = 90
    shortRipples = np.where(state_duration < minstateDuration)[0]
    thirdPass = np.delete(secondPass, shortRipples, 0)

    return thirdPass


def hmmfit1d(Data):
    # hmm states on 1d data and returns labels with highest mean = highest label
    flag = None
    if np.isnan(Data).any():
        nan_indices = np.where(np.isnan(Data) == 1)[0]
        non_nan_indices = np.where(np.isnan(Data) == 0)[0]
        Data_og = Data
        Data = np.delete(Data, nan_indices)
        hmmlabels = np.nan * np.ones(len(Data_og))
        flag = 1

    Data = (np.asarray(Data)).reshape(-1, 1)
    model = GaussianHMM(n_components=2, n_iter=100).fit(Data)
    hidden_states = model.predict(Data)
    mus = np.squeeze(model.means_)
    sigmas = np.squeeze(np.sqrt(model.covars_))
    transmat = np.array(model.transmat_)

    idx = np.argsort(mus)
    mus = mus[idx]
    sigmas = sigmas[idx]
    transmat = transmat[idx, :][:, idx]

    state_dict = {}
    states = [i for i in range(4)]
    for i in idx:
        state_dict[idx[i]] = states[i]

    relabeled_states = np.asarray([state_dict[h] for h in hidden_states])
    relabeled_states[:2] = [0, 0]
    relabeled_states[-2:] = [0, 0]

    if flag:

        hmmlabels[non_nan_indices] = relabeled_states

    else:
        hmmlabels = relabeled_states

    # state_diff = np.diff(relabeled_states)
    # start = np.where(state_diff == 1)[0]
    # stop = np.where(state_diff == -1)[0]

    # for s, e in zip(start, stop):
    #     if e - s < 50:
    #         relabeled_states[s + 1 : e] = 0
    # print(start_ripple.shape, stop_ripple.shape)
    # states = np.concatenate((start_ripple, stop_ripple), axis=1)

    # relabeled_states = hidden_states
    return hmmlabels


class SleepScore:
    # TODO add support for bad time points

    window = 1  # seconds
    overlap = 0.2  # seconds

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        if Path(self._obj.files.stateparams).is_file():
            self.params = pd.read_pickle(self._obj.files.stateparams)

        if Path(self._obj.files.states).is_file():
            self.states = pd.read_pickle(self._obj.files.states)
            # Adding name convention to the states
            state_number_dict = {
                1: "nrem",
                2: "rem",
                3: "quiet",
                4: "active",
            }
            self.states["name"] = self.states["state"].map(state_number_dict)

    def detect(self):

        self.params, self.sxx, self.states = self._getparams()
        self.params.to_pickle(self._obj.files.stateparams)
        self.states.to_pickle(self._obj.files.states)

    @staticmethod
    def _label2states(theta_delta, delta_l, emg_l):

        state = np.zeros(len(theta_delta))
        for i, (ratio, delta, emg) in enumerate(zip(theta_delta, delta_l, emg_l)):

            if ratio == 1 and emg == 1:
                state[i] = 4
            elif ratio == 0 and emg == 1:
                state[i] = 3
            elif ratio == 1 and emg == 0:
                state[i] = 2
            elif ratio == 0 and emg == 0:
                state[i] = 1

        return state

    @staticmethod
    def _states2time(label):

        states = np.unique(label)

        all_states = []
        for state in states:

            binary = np.where(label == state, 1, 0)
            binary = np.concatenate(([0], binary, [0]))
            binary_change = np.diff(binary)

            start = np.where(binary_change == 1)[0]
            end = np.where(binary_change == -1)[0]
            start = start[:-1]
            end = end[:-1]
            # duration = end - start
            stateid = state * np.ones(len(start))
            firstPass = np.vstack((start, end, stateid)).T

            all_states.append(firstPass)

        all_states = np.concatenate(all_states)

        return all_states

    @staticmethod
    def _removetransient(statetime):

        duration = statetime.duration
        start = statetime.start
        end = statetime.end
        state = statetime.state

        arr = np.zeros((len(start), 4))
        arr[:, 0] = start
        arr[:, 1] = end
        arr[:, 2] = duration
        arr[:, 3] = state

        srt_ind = np.argsort(arr[:, 0])
        arr = arr[srt_ind, :]

        ind = 1
        while ind < len(arr) - 1:
            if (arr[ind, 2] < 50) and (arr[ind - 1, 3] == arr[ind + 1, 3]):
                arr[ind - 1, :] = np.array(
                    [
                        arr[ind - 1, 0],
                        arr[ind + 1, 1],
                        arr[ind + 1, 1] - arr[ind - 1, 0],
                        arr[ind - 1, 3],
                    ]
                )
                arr = np.delete(arr, [ind, ind + 1], 0)
            else:
                ind += 1

        statetime = pd.DataFrame(
            {
                "start": arr[:, 0],
                "end": arr[:, 1],
                "duration": arr[:, 2],
                "state": arr[:, 3],
            }
        )

        return statetime

    def _getparams(self):
        sRate = self._obj.recinfo.lfpSrate
        # lfp = np.load(self._obj.sessinfo.files.thetalfp)
        lfp = self._obj.spindle.best_chan_lfp()[0]
        # ripplelfp = np.load(self._obj.files.ripplelfp).item()["BestChan"]

        lfp = stats.zscore(lfp)
        bands = spectrogramBands(
            lfp, window=self.window, overlap=self.overlap, smooth=10
        )
        time = bands.time
        delta = bands.delta
        deltaplus = bands.deltaplus
        theta = bands.theta
        spindle = bands.spindle
        gamma = bands.gamma
        ripple = bands.ripple
        theta_deltaplus_ratio = theta / deltaplus
        sxx = stats.zscore(bands.sxx, axis=None)  # zscored only for visualization

        emg = self._emgfromlfp(fromfile=1)
        print(emg.shape, theta_deltaplus_ratio.shape)

        deadfile = (self._obj.sessinfo.files.filePrefix).with_suffix(".dead")
        if deadfile.is_file():
            with deadfile.open("r") as f:
                noisy = []
                for line in f:
                    epc = line.split(" ")
                    epc = [float(_) for _ in epc]
                    noisy.append(epc)
                noisy = np.asarray(noisy) / 1000  # seconds

            noisy_timepoints = []
            for noisy_ind in range(noisy.shape[0]):
                st = noisy[noisy_ind, 0]
                en = noisy[noisy_ind, 1]
                # numnoisy = en - st
                noisy_indices = np.where((time > st) & (time < en))[0]
                noisy_timepoints.extend(noisy_indices)

            # noisy_boolean = np.zeros(len(deltaplus))
            # noisy_boolean[noisy_timepoints] = np.ones(len(noisy_timepoints))
            theta_deltaplus_ratio[noisy_timepoints] = np.nan
            emg[noisy_timepoints] = np.nan
            deltaplus[noisy_timepoints] = np.nan

            # emg = np.asarray(pd.Series.fillna(pd.Series(emg), method="bfill"))
            # theta_deltaplus_ratio = np.asarray(
            #     pd.Series.fillna(pd.Series(theta_deltaplus_ratio), method="bfill")
            # )

        deltaplus_label = hmmfit1d(deltaplus)
        theta_deltaplus_label = hmmfit1d(theta_deltaplus_ratio)
        emg_label = hmmfit1d(emg)

        states = self._label2states(theta_deltaplus_label, deltaplus_label, emg_label)

        print(
            states.shape, emg.shape, deltaplus_label.shape, theta_deltaplus_label.shape
        )
        statetime = (self._states2time(states)).astype(int)

        data = pd.DataFrame(
            {
                "time": time,
                "delta": delta,
                "deltaplus": deltaplus,
                "theta": theta,
                "spindle": spindle,
                "gamma": gamma,
                "ripple": ripple,
                "theta_deltaplus_ratio": theta_deltaplus_ratio,
                "emg": emg,
                "state": states,
            }
        )

        statetime = pd.DataFrame(
            {
                "start": time[statetime[:, 0]],
                "end": time[statetime[:, 1]],
                "duration": time[statetime[:, 1]] - time[statetime[:, 0]],
                "state": statetime[:, 2],
            }
        )

        statetime_new = self._removetransient(statetime)

        # data_label = pd.DataFrame({"theta_delta": theta_delta_label, "emg": emg_label})

        return data, sxx, statetime_new

    def _emgfromlfp(self, fromfile=0):

        emgfilename = self._obj.sessinfo.files.corr_emg

        if fromfile:
            emg_lfp = np.load(emgfilename)

        else:

            highfreq = 600
            lowfreq = 300
            sRate = self._obj.recinfo.lfpSrate
            nChans = self._obj.recinfo.nChans
            nyq = 0.5 * sRate
            window = self.window * sRate
            overlap = self.overlap * sRate
            channels = self._obj.recinfo.channels
            badchans = self._obj.recinfo.badchans

            emgChans = np.setdiff1d(channels, badchans, assume_unique=True)
            nemgChans = len(emgChans)

            # filtering for high frequency band
            eegdata = np.memmap(
                self._obj.sessinfo.recfiles.eegfile, dtype="int16", mode="r"
            )
            eegdata = np.memmap.reshape(eegdata, (int(len(eegdata) / nChans), nChans))
            b, a = sg.butter(3, [lowfreq / nyq, highfreq / nyq], btype="bandpass")
            nframes = len(eegdata)

            # windowing signal
            frames = np.arange(0, nframes - window, window - overlap)

            def corrchan(start):
                start_frame = int(start)
                end_frame = start_frame + window
                lfp_req = eegdata[start_frame:end_frame, emgChans]
                yf = sg.filtfilt(b, a, lfp_req, axis=0).T
                ltriang = np.tril_indices(nemgChans, k=-1)
                return np.corrcoef(yf)[ltriang].mean()

            corr_per_frame = Parallel(n_jobs=8, require="sharedmem")(
                delayed(corrchan)(start) for start in frames
            )
            emg_lfp = np.asarray(corr_per_frame)
            np.save(emgfilename, emg_lfp)

        emg_lfp = filtSig.gaussian_filter1d(emg_lfp, 10)
        return emg_lfp

    def addBackgroundtoPlots(self, tstart=0, ax=None):
        states = self.states
        x = (np.asarray(states.start) - tstart) / 3600

        y = -1 * np.ones(len(x))  # + np.asarray(states.state)
        width = np.asarray(states.duration) / 3600
        height = np.ones(len(x)) * 1.3
        qual = states.state

        colors = ["#6b90d1", "#eb9494", "#b6afaf", "#474343"]
        col = [colors[int(state) - 1] for state in states.state]

        make_boxes(ax, x, y, width, height, facecolor=col)

    def plot(self):
        states = self.states
        params = self.params
        post = self._obj.epochs.post
        lfpSrate = self._obj.recinfo.lfpSrate
        lfp = self._obj.spindle.best_chan_lfp()[0]
        spec = spectrogramBands(lfp, window=5)

        fig = plt.figure(num=None, figsize=(6, 10))
        gs = GridSpec(4, 4, figure=fig)
        fig.subplots_adjust(hspace=0.4)

        axhypno = fig.add_subplot(gs[0, :])
        x = np.asarray(states.start)
        y = np.zeros(len(x)) + np.asarray(states.state)
        width = np.asarray(states.duration)
        height = np.ones(len(x))
        colors = ["#6b90d1", "#eb9494", "#b6afaf", "#474343"]
        col = [colors[int(state) - 1] for state in states.state]
        make_boxes(axhypno, x, y, width, height, facecolor=col)
        axhypno.set_xlim(0, post[1])
        axhypno.set_ylim(1, 5)

        axspec = fig.add_subplot(gs[1, :], sharex=axhypno)
        sxx = spec.sxx / np.max(spec.sxx)
        sxx = gaussian_filter(sxx, sigma=1)
        vmax = np.max(sxx) / 60
        specplot = axspec.pcolorfast(
            spec.time, spec.freq, sxx, cmap="Spectral_r", vmax=vmax
        )
        axspec.set_ylim([0, 30])
        axspec.set_xlim([np.min(spec.time), np.max(spec.time)])
        axspec.set_xlabel("Time (s)")
        axspec.set_ylabel("Frequency (s)")

        axemg = fig.add_subplot(gs[2, :], sharex=axhypno)
        axemg.plot(params.time, params.emg)

        axthdel = fig.add_subplot(gs[3, :], sharex=axhypno)
        axthdel.plot(params.time, params.theta_deltaplus_ratio)

        # ==== change spectrogram visual properties ========

        # @widgets.interact(time=(maze[0], maze[1], 10))
        # def update(time=0.5):
        #     # tnow = timetoPlot.val
        #     allplts(time - 5, time + 5)
        #     specplot.set_clim([0, freq])

        @widgets.interact(norm=(np.min(sxx), np.max(sxx), 0.001))
        def update(norm=0.5):
            # tnow = timetoPlot.val
            # allplts(time - 5, time + 5)
            specplot.set_clim([0, norm])

    def hypnogram(self, ax1=None, tstart=0.0, unit="s"):
        """Plots hypnogram in the given axis

        Args:
            ax1 (axis, optional): axis for plotting. Defaults to None.
            tstart (float, optional): Start time of hypnogram. Defaults to 0.
            unit (str, optional): Unit of time in seconds or hour. Defaults to "s".
        """
        states = self.states

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
