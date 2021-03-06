from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as filtSig
import scipy.signal as sg
import scipy.stats as stats
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from parsePath import Recinfo
import signal_process
from artifactDetect import findartifact


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
    # colors = ["#6b90d1", "#eb9494", "#b6afaf", "#474343"]
    colors = {
        "nrem": "#6b90d1",
        "rem": "#eb9494",
        "quiet": "#b6afaf",
        "active": "#474343",
    }

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ------- defining file names ---------
        filePrefix = self._obj.files.filePrefix

        @dataclass
        class files:
            stateparams: str = filePrefix.with_suffix(".stateparams.pkl")
            states: str = filePrefix.with_suffix(".states.pkl")
            emg: Path = filePrefix.with_suffix(".emg.npy")

        self.files = files()
        self._load()

    def _load(self):
        if (f := self.files.stateparams).is_file():
            self.params = pd.read_pickle(f)

        if (f := self.files.states).is_file():
            self.states = pd.read_pickle(f)
            # Adding name convention to the states
            state_number_dict = {
                1: "nrem",
                2: "rem",
                3: "quiet",
                4: "active",
            }
            self.states["name"] = self.states["state"].map(state_number_dict)

    @staticmethod
    def _label2states(theta_delta, delta_l, emg_l):

        state = np.zeros(len(theta_delta))
        for i, (ratio, delta, emg) in enumerate(zip(theta_delta, delta_l, emg_l)):

            if ratio == 1 and emg == 1:  # active wake
                state[i] = 4
            elif ratio == 0 and emg == 1:  # quiet wake
                state[i] = 3
            elif ratio == 1 and emg == 0:  # REM
                state[i] = 2
            elif ratio == 0 and emg == 0:  # NREM
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

    def detect(self, chans=None, window=1, overlap=0.2, emgfile=False):
        """detects sleep states for the recording

        Parameters
        ----------
        chans : int, optional
            channel you want to use for sleep detection, by default None
        window : int, optional
            bin size, by default 1
        overlap : float, optional
            seconds of overlap between adjacent window , by default 0.2
        emgfile : bool, optional
            if True load the emg file in the basepath, by default False

        """
        artifact = findartifact(self._obj)
        sRate = self._obj.lfpSrate

        if emgfile:
            print("emg loaded")
            emg = np.load(self.files.emg)
        else:
            emg = self._emgfromlfp(window=window, overlap=overlap)

        emg = filtSig.gaussian_filter1d(emg, 10)

        if chans is None:
            changroup = self._obj.goodchangrp
            bottom_chans = [shank[-1] for shank in changroup if shank]
            chans = np.random.choice(bottom_chans)

        print(f"channel for sleep detection: {chans}")

        lfp = self._obj.geteeg(chans=chans)
        lfp = stats.zscore(lfp)
        bands = signal_process.spectrogramBands(
            lfp, sampfreq=sRate, window=window, overlap=overlap, smooth=10
        )
        time = bands.time
        delta = bands.delta
        deltaplus = bands.deltaplus
        theta = bands.theta
        spindle = bands.spindle
        gamma = bands.gamma
        ripple = bands.ripple
        theta_deltaplus_ratio = theta / deltaplus
        print(f"spectral properties calculated")

        if (noisy := artifact.time).any():
            noisy_timepoints = []
            for noisy_ind in range(noisy.shape[0]):
                st = noisy[noisy_ind, 0]
                en = noisy[noisy_ind, 1]
                noisy_indices = np.where((time > st) & (time < en))[0]
                noisy_timepoints.extend(noisy_indices)

            theta_deltaplus_ratio[noisy_timepoints] = np.nan
            emg[noisy_timepoints] = np.nan
            deltaplus[noisy_timepoints] = np.nan

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

        data.to_pickle(self.files.stateparams)
        statetime_new.to_pickle(self.files.states)

    def _emgfromlfp(self, window, overlap, n_jobs=8):
        """Calculating emg

        Parameters
        ----------
        window : int
            window size in seconds
        overlap : float
            overlap between windows in seconds
        n_jobs: int,
            number of cpu/processes to use

        Returns
        -------
        array
            emg calculated at each time window
        """
        print("starting emg calculation")
        highfreq = 600
        lowfreq = 300
        sRate = self._obj.lfpSrate
        nProbes = self._obj.nProbes
        changrp = self._obj.goodchangrp
        nShanksProbe = self._obj.nShanksProbe
        probesid = np.concatenate([[_] * nShanksProbe[_] for _ in range(nProbes)])

        # ----selecting a fixed number of shanks from each probe-----
        # max_shanks_probe = [np.min([3, _]) for _ in nShanksProbe]
        # selected_shanks = []
        # for probe in range(nProbes):
        #     shanks_in_probe = [
        #         changrp[_] for _ in np.where(probesid == probe)[0] if changrp[_]
        #     ]
        #     selected_shanks.append(
        #         np.concatenate(
        #             np.random.choice(
        #                 shanks_in_probe, max_shanks_probe[probe], replace=False
        #             )
        #         )
        #     )

        # ---- selecting probe with most number of shanks -------
        which_probe = np.argmax(nShanksProbe)
        selected_shanks = np.where(probesid == which_probe)[0]
        # making sure shanks are not empty
        selected_shanks = [changrp[_] for _ in selected_shanks if changrp[_]]

        emgChans = np.concatenate(selected_shanks)
        nemgChans = len(emgChans)
        eegdata = self._obj.geteeg(chans=0)
        total_duration = len(eegdata) / sRate

        timepoints = np.arange(0, total_duration - window, window - overlap)

        # ---- Mean correlation across all selected channels calculated in parallel --
        def corrchan(start):
            lfp_req = np.asarray(
                self._obj.geteeg(chans=emgChans, timeRange=[start, start + window])
            )
            yf = signal_process.filter_sig.bandpass(
                lfp_req, lf=lowfreq, hf=highfreq, fs=sRate
            )
            ltriang = np.tril_indices(nemgChans, k=-1)
            return np.corrcoef(yf)[ltriang].mean()

        corr_per_frame = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(corrchan)(start) for start in timepoints
        )
        emg_lfp = np.asarray(corr_per_frame)

        np.save(self.files.emg, emg_lfp)
        print("emg calculation done")
        return emg_lfp

    def proportion(self, period=None):

        if period is None:
            period = [self.states.iloc[0].start, self.states.iloc[-1].end]

        assert len(period) == 2
        period_duration = np.diff(period)[0]
        states = self.states[
            (self.states.start > period[0]) & (self.states.start < period[1])
        ][["duration", "name"]]
        states_duration = states.groupby("name").agg("sum")

        return states_duration / period_duration

    def plot(self):
        states = self.states
        params = self.params
        post = self._obj.epochs.post
        lfp = self._obj.spindle.best_chan_lfp()[0]
        spec = signal_process.spectrogramBands(lfp, window=5)

        fig = plt.figure(num=None, figsize=(6, 10))
        gs = GridSpec(4, 4, figure=fig)
        fig.subplots_adjust(hspace=0.4)

        axhypno = fig.add_subplot(gs[0, :])
        x = np.asarray(states.start)
        y = np.zeros(len(x)) + np.asarray(states.state)
        width = np.asarray(states.duration)
        height = np.ones(len(x))
        col = [self.colors[state] for state in states.name]
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

    def hypnogram(self, ax=None, tstart=0.0, unit="s", collapsed=False):
        """Plot hypnogram

        Parameters
        ----------
        ax : [type], optional
            axis to plot onto, by default None
        tstart : float, optional
            start of hypnogram, by default 0.0, helps in positioning of hypnogram
        unit : str, optional
            unit of timepoints, 's'=seconds or 'h'=hour, by default "s"
        collapsed : bool, optional
            if true then all states have same vertical spans, by default False and has classic look

        Returns
        -------
        [type]
            [description]
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        unit_norm = None
        if unit == "s":
            unit_norm = 1
        elif unit == "h":
            unit_norm = 3600

        span_ = {
            "nrem": [0, 0.25],
            "rem": [0.25, 0.5],
            "quiet": [0.5, 0.75],
            "active": [0.75, 1],
        }

        for state in span_:
            ax.annotate(
                state,
                (1, span_[state][1] - 0.15),
                xycoords="axes fraction",
                fontsize=7,
                color=self.colors[state],
            )
        if collapsed:
            span_ = {
                "nrem": [0, 1],
                "rem": [0, 1],
                "quiet": [0, 1],
                "active": [0, 1],
            }

        for state in self.states.itertuples():
            if state.name in self.colors.keys():
                ax.axvspan(
                    (state.start - tstart) / unit_norm,
                    (state.end - tstart) / unit_norm,
                    ymin=span_[state.name][0],
                    ymax=span_[state.name][1],
                    facecolor=self.colors[state.name],
                    alpha=0.7,
                )
        ax.axis("off")

        return ax
