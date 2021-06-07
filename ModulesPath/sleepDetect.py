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
from scipy.ndimage import gaussian_filter

from . import signal_process
from .parsePath import Recinfo
from .core import Epoch

try:
    import ephyviewer
except:
    "Ephyviewer is not installed, will need it if you want sleep state editor"


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


class SleepScore(Epoch):
    # TODO add support for bad time points
    colors = {
        "nrem": "#6b90d1",
        "rem": "#eb9494",
        "quiet": "#b6afaf",
        "active": "#474343",
    }
    labels = ["nrem", "rem", "quiet", "active"]

    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

        # ------- defining file names ---------
        filePrefix = self._obj.files.filePrefix
        filename = filePrefix.with_suffix(".brainstates.npy")
        super().__init__(filename=filename)
        self.load()

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
        end = statetime.stop
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
                "stop": arr[:, 1],
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
            emg = self.metadata["emg"]
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
        deltaplus = bands.deltaplus
        theta = bands.theta
        theta_deltaplus_ratio = theta / deltaplus
        print(f"spectral properties calculated")

        if (noisy := artifact.time) is not None:
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
        statetime = (self._states2time(states)).astype(int)

        statetime = pd.DataFrame(
            {
                "start": time[statetime[:, 0]],
                "stop": time[statetime[:, 1]],
                "duration": time[statetime[:, 1]] - time[statetime[:, 0]],
                "state": statetime[:, 2],
            }
        )

        epochs = self._removetransient(statetime)

        state_to_label = {1: "nrem", 2: "rem", 3: "quiet", 4: "active"}
        epochs["label"] = epochs["state"].map(state_to_label)
        epochs.drop("state", axis=1, inplace=True)

        self.epochs = epochs
        self.metadata = {"window": window, "overlap": overlap, "emg": emg}
        self.save()
        self.load()

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

        print("emg calculation done")
        return emg_lfp

    def proportion(self, period=None):

        if period is None:
            period = [self.states.iloc[0].start, self.states.iloc[-1].end]

        period_duration = np.diff(period)

        ep = self.epochs.copy()
        ep = ep[(ep.stop > period[0]) & (ep.start < period[1])].reset_index(drop=True)

        if ep["start"].iloc[0] < period[0]:
            ep.at[0, "start"] = period[0]

        if ep["stop"].iloc[-1] > period[1]:
            ep.at[ep.index[-1], "stop"] = period[1]

        ep["duration"] = ep.stop - ep.start

        ep_group = ep.groupby("label").sum().duration / period_duration

        states_proportion = {"rem": 0.0, "nrem": 0.0, "quiet": 0.0, "active": 0.0}

        for state in ep_group.index.values:
            states_proportion[state] = ep_group[state]

        return states_proportion

    def plot_hypnogram(self, ax=None, tstart=0.0, unit="s", collapsed=False):
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

        for state in self.epochs.itertuples():
            if state.label in self.colors.keys():
                ax.axvspan(
                    (state.start - tstart) / unit_norm,
                    (state.stop - tstart) / unit_norm,
                    ymin=span_[state.label][0],
                    ymax=span_[state.label][1],
                    facecolor=self.colors[state.label],
                    alpha=0.7,
                )
        ax.axis("off")

        return ax

    def editor(self, chan, spikes=None):
        class StatesSource(ephyviewer.WritableEpochSource):
            def __init__(
                self,
                filename,
                possible_labels,
                color_labels=None,
                channel_name="",
                restrict_to_possible_labels=False,
            ):

                self.filename = filename

                ephyviewer.WritableEpochSource.__init__(
                    self,
                    epoch=None,
                    possible_labels=possible_labels,
                    color_labels=color_labels,
                    channel_name=channel_name,
                    restrict_to_possible_labels=restrict_to_possible_labels,
                )

            def load(self):
                """
                Returns a dictionary containing the data for an epoch.
                Data is loaded from the CSV file if it exists; otherwise the superclass
                implementation in WritableEpochSource.load() is called to create an
                empty dictionary with the correct keys and types.
                The method returns a dictionary containing the loaded data in this form:
                { 'time': np.array, 'duration': np.array, 'label': np.array, 'name': string }
                """

                if self.filename.is_file():
                    # if file already exists, load previous epoch
                    data = pd.read_pickle(self.filename)
                    state_number_dict = {1: "nrem", 2: "rem", 3: "quiet", 4: "active"}
                    data["name"] = data["state"].map(state_number_dict)

                    epoch_labels = np.array([f" State{_}" for _ in data["state"]])
                    epoch = {
                        "time": data["start"].values,
                        "duration": data["end"].values - data["start"].values,
                        "label": epoch_labels,
                    }
                else:
                    # if file does NOT already exist, use superclass method for creating
                    # an empty dictionary
                    epoch = super().load()

                return epoch

            def save(self):
                df = pd.DataFrame()
                df["start"] = np.round(self.ep_times, 6)  # round to nearest microsecond
                df["end"] = np.round(self.ep_times, 6) + np.round(
                    self.ep_durations
                )  # round to nearest microsecond
                df["duration"] = np.round(
                    self.ep_durations, 6
                )  # round to nearest microsecond
                state_number_dict = {"nrem": 1, "rem": 2, "quiet": 3, "active": 4}
                df["name"] = self.ep_labels
                df["state"] = df["name"].map(state_number_dict)
                df.sort_values(["time", "duration", "name"], inplace=True)
                df.to_pickle(self.filename)

        states_source = StatesSource(self.files.states, self.labels)
        # you must first create a main Qt application (for event loop)
        # app = ephyviewer.mkQApp()

        sigs = np.asarray(self._obj.geteeg(chans=chan)).reshape(-1, 1)
        filtered_sig = signal_process.filter_sig.bandpass(
            sigs, lf=120, hf=150, ax=0, fs=1250
        )
        sample_rate = self._obj.lfpSrate
        t_start = 0.0

        # Create the main window that can contain several viewers
        win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

        # create a viewer for signal
        view1 = ephyviewer.TraceViewer.from_numpy(
            np.hstack((sigs, filtered_sig)), sample_rate, t_start, "Signals"
        )
        view1.params["scale_mode"] = "same_for_all"
        view1.auto_scale()
        win.add_view(view1)

        source_sig = ephyviewer.InMemoryAnalogSignalSource(sigs, sample_rate, t_start)
        # create a viewer for the encoder itself
        view2 = ephyviewer.EpochEncoder(
            source=states_source, name="Dev mood states along day"
        )
        win.add_view(view2)

        view3 = ephyviewer.TimeFreqViewer(source=source_sig, name="tfr")
        view3.params["show_axis"] = False
        view3.params["timefreq", "deltafreq"] = 1
        win.add_view(view3)

        # ----- spikes --------
        if spikes is not None:
            spk_id = np.arange(len(spikes))

            all_spikes = []
            for i, (t, id_) in enumerate(zip(spikes, spk_id)):
                all_spikes.append({"time": t, "name": f"Unit {i}"})

            spike_source = ephyviewer.InMemorySpikeSource(all_spikes=all_spikes)
            view4 = ephyviewer.SpikeTrainViewer(source=spike_source)
            win.add_view(view4)
            # show main window and run Qapp
        # win.show()
        # return win, app

        # app.exec_()

        return win
