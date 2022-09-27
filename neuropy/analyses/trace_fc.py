import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import neuropy.io.dlcio as dio
from pathlib import Path
import scipy.ndimage as simage
import pandas as pd
import re
import csv
from matplotlib.backends.backend_pdf import PdfPages

# list parameters used in each experiment here. Should move outside this class for later maybe?
# trace_params = {
#     "Pilot1": {
#         "Shock": {"pix2cm": 0.13, "camera": "Camera 1"},
#         "New": {"pix2cm": 0.26, "camera": "Camera 3"},
#     },
#     "Pilot2": {
#         "Shock": {"pix2cm": 0.13, "camera": "Camera 1"},
#         "New": {"pix2cm": 0.26, "camera": "Camera 3"},
#     },
# }

trace_params = {
    "Pilot1": {
        "Shock": {"pix2cm": 0.13, "camera": "_Shockbox"},
        "New": {"pix2cm": 0.26, "camera": "_newarena"},
    },
    "Pilot2": {
        "Shock": {"pix2cm": 0.13, "camera": "_Shockbox"},
        "New": {"pix2cm": 0.26, "camera": "_newarena"},
    },
    "Recording_Rats": {
        "Shock": {"pix2cm": 0.13, "camera": "_Shockbox"},
        "New": {"pix2cm": 0.26, "camera": "_Newarena"},
    },
}


class trace_behavior:
    """Class to analyze trace_conditioning behavioral data"""

    def __init__(self, base_path, search_str=None, movie_type=".mp4", pix2cm=1):

        # Make base directory into a pathlib object for easy manipulation
        self.base_dir = Path(base_path)

        # Infer animal name and session from base_path directory structure.
        self.animal_name = self.base_dir.parts[-2]
        self.session = self.base_dir.parts[-1]
        self.session_date = fix_date(self.base_dir.parts[-1][0:10])
        self.session_type = self.base_dir.parts[-1][11:]

        # Import DeepLabCut file and pre-process the data
        self.dlc = dio.DLC(
            base_path, search_str=search_str, movie_type=movie_type, pix2cm=pix2cm
        )
        print(
            "Automatically smoothing position and calculating speed with defaults - re-run to tweak parameters"
        )
        self.dlc.smooth_pos()
        self.dlc.calculate_speed()

        # make session type more specific if recall session
        if "recall" in self.session_type:
            if (
                re.search("shock", str(self.dlc.tracking_file).lower()) is not None
            ):  # check if in shock arena
                self.session_type = "ctx_" + self.session_type
            elif re.search("newarena", str(self.dlc.tracking_file).lower()) is not None:
                self.session_type = "tone_" + self.session_type
            else:
                raise NameError("Error in input file names")

        # load events
        self.load_events()

    def get_freezing_epochs(
        self,
        bodyparts=["neck_base", "mid_back"],
        speed_threshold=0.25,
        min_freeze_length=1,
        plot=True,
    ):
        """
        Identify freezing epochs based on a speed threshold on given bodyparts. If one bodypart is obscured or below
        the likelihood threshold, it will look at the remaining body parts. If all are obscured, sends to NaN.
        :param bodyparts:
        :param speed_threshold:
        :param min_freeze_length:
        :param plot: True = plot into new axes OR provide axes to plot into
        :return: long_freezing_times: list of pandas dataframes with freezing times.
        :return: freeze_bool: boolean array the same size as pos_smooth with freezing indices.
        """

        assert hasattr(
            self.dlc, "pos_smooth"
        ), "You must smooth data first with DLC.smooth_pos()"

        # calculate speed if not done already
        if "speed" not in self.dlc.pos_smooth[self.dlc.bodyparts[0]].keys():
            self.dlc.calculate_speed()

        # Identify all the places where the animal's bodypart is below the speed threshold,
        # putting nans where you don't have good tracking
        no_move_bool = []
        for bodypart in bodyparts:
            nan_bool = np.isnan(self.dlc.pos_smooth[bodypart]["speed"])
            no_move = self.dlc.pos_smooth[bodypart]["speed"] < speed_threshold
            no_move[nan_bool] = np.nan
            no_move_bool.append(no_move)

        # Now count potential freezing points as any time all well-tracked body-parts are below the speed threshold
        freeze_candidate_array = np.bitwise_and(
            np.nansum(np.asarray(no_move_bool), axis=0) > 0,
            np.nansum(np.asarray(no_move_bool), axis=0)
            == np.nansum(~np.isnan(np.asarray(no_move_bool)), axis=0),
        )

        # NRK todo: debug everything below here! All the stuff above looks good.
        # Now identify contiguous regions of freezing.
        regions = simage.find_objects(simage.label(freeze_candidate_array > 0)[0])

        # Calculate the length (in frames) of each potential freezing epoch
        immobile_length = np.asarray(
            [
                len(self.dlc.pos_smooth[bodypart]["speed"][region])
                / self.dlc.SampleRate
                for region in regions
            ]
        )

        # Identify freezing bouts longer than minimum freeze length
        long_freezing_bouts = np.where(immobile_length > min_freeze_length)[0]
        long_freezing_times = [
            self.dlc.pos_smooth[bodypart]["time"].iloc[regions[bout]]
            for bout in long_freezing_bouts
        ]
        # Now keep only bouts that are longer than the minimum freeze length
        freeze_bool = np.zeros_like(freeze_candidate_array)
        for bout in long_freezing_bouts:
            freeze_bool[regions[bout]] = 1

        if plot:
            legend_labels = bodyparts.copy()

            if plot is True:  # set up new axes if none specified
                fig, ax = plt.subplots()
                fig.set_size_inches([24, 7])
            else:  # otherwise use input axes
                ax = plot

            for bodypart in bodyparts:
                self.dlc.pos_smooth[bodypart].plot(
                    x="time", y="speed", legend=False, ax=ax
                )

            # Plot final freezing events.
            self.dlc.pos_smooth[bodyparts[-1]].iloc[freeze_bool].plot(
                x="time", y="speed", legend=False, ax=ax, style="r."
            )
            legend_labels.extend(["final freezing events"])

            # Make sure y-axis doesn't extend beyond 100cm/sec in case of edge effects
            ax.set_ylim([-2, np.min([ax.get_ylim()[-1], 100])])

            # Label things nicely
            ax.set_title(self.dlc.animal_name + ": " + self.dlc.session)
            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Speed (cm/sec)")
            ax.legend(legend_labels)

            # Now plot events if you have them!
            alpha_use = 0.3
            legend_handles = ax.get_legend_handles_labels()[0]
            if re.match("tone", self.session_type) is not None or re.match(
                "training", self.session_type
            ):
                tone_starts = self.events["Time (s)"][
                    (
                        np.asarray(["CS" in a for a in self.events["Event"]])
                        | np.asarray(["tone" in a for a in self.events["Event"]])
                    )
                    & np.asarray(["_start" in a for a in self.events["Event"]])
                ]
                tone_ends = self.events["Time (s)"][
                    (
                        np.asarray(["CS" in a for a in self.events["Event"]])
                        | np.asarray(["tone" in a for a in self.events["Event"]])
                    )
                    & np.asarray(["_end" in a for a in self.events["Event"]])
                ]
                for start, end in zip(tone_starts, tone_ends):
                    htone = ax.axvspan(start, end, color=[0, 1, 0, alpha_use])
                legend_handles.append(htone)
                legend_labels.append("Tone")
                ax.legend(legend_handles, legend_labels)

            if "shock_start" in self.events["Event"].values:
                shock_starts = self.events["Time (s)"][
                    self.events["Event"] == "shock_start"
                ]
                shock_ends = self.events["Time (s)"][
                    self.events["Event"] == "shock_end"
                ]
                for start, end in zip(shock_starts, shock_ends):
                    hshock = ax.axvspan(start, end, color=[1, 0, 0, alpha_use])
                legend_handles.append(hshock)
                legend_labels.append("Shock")
                ax.legend(legend_handles, legend_labels)

            sns.despine(ax=ax)

        return long_freezing_times, freeze_bool

    def load_events(self):
        """Load events (start/end times, CS/US times, etc.) from .CSV file."""
        # Grab tracking files and corresponding movie file.
        event_files = sorted(self.base_dir.glob("**/*.csv"))
        self.event_file = dio.get_matching_files(
            event_files,
            match_str=[
                self.session_type.replace("LTM", "").replace("1", "").replace("2", ""),
                self.session_date,
            ],
            exclude_str=["test"],
        )[0]

        # Load in event start time metadata

        # Load in all events
        with open(str(self.event_file)) as f:
            reader = csv.reader(f)
            self.event_start = next(reader)

        self.events = pd.read_csv(self.event_file, header=1)

        # Grab start time
        header = pd.read_csv(self.event_file, header=None, nrows=1)
        self.events_start_time = pd.Timestamp(header[1][0]) + pd.Timedelta(
            header[3][0] / 10**6, unit="sec"
        )

        # Finally, add in absolute timestamps to events pandas array
        self.events["Timestamps"] = self.events_start_time + pd.to_timedelta(
            self.events["Time (s)"], unit="sec"
        )

    def get_event_times(self, event_type: str = "CS"):
        """
        Get starts and ends of CS, US, sync tone, etc.
        :param event_type:
        :return: pandas arrays containing start and end times
        """
        start_bool = [
            event.find(event_type) >= 0 and event.find("start") >= 0
            for event in self.events["Event"]
        ]
        end_bool = [
            event.find(event_type) >= 0 and event.find("end") >= 0
            for event in self.events["Event"]
        ]

        start_times = self.events[start_bool]
        end_times = self.events[end_bool]

        return start_times, end_times

    def get_mean_speed(
        self,
        bodyparts=["neck_base", "mid_back"],
        by_trial=False,
        trial_buffer_sec=10,
        paradigm=None,
        include_US=True,
    ):
        """
        Gets the mean speed of bodyparts indicated across whole session or broken up by trials
        :param bodyparts: list
        :param by_trial: boolean
        :param trial_buffer_sec: time before/after trial start (CSon) or end (CSoff or USoff) for applicable sessions.
        Can be float or list of floats.
        :param paradigm: 'Pilot1' had a different CS schedule for tone_recall session so be sure to specify if plotting
        a session from this paradigm.  Otherwise should not matter.
        :param include_US: True (default): training trials start with CS and end with US.
                           False: trial ends at CS for plotting and calculation purposes.
        :return:
        """

        if isinstance(trial_buffer_sec, int):
            trial_buffer_sec = [trial_buffer_sec, trial_buffer_sec]
        SampleRate = self.dlc.SampleRate
        nframes_buffer = np.asarray(
            [int(SampleRate * buffer) for buffer in trial_buffer_sec]
        )
        bodypart_speed = []
        for bodypart in bodyparts:
            bodypart_speed.append(self.dlc.pos_smooth[bodypart]["speed"])

        mean_speed = np.nanmean(bodypart_speed, axis=0)
        nframes = len(mean_speed)
        times = np.arange(0, mean_speed.shape[0]) / SampleRate

        # Break-down by trial now
        stimuli = get_stimuli_times(
            self.events, self.session_type, paradigm=paradigm
        )  # Get US/CS times if applicable
        if by_trial and self.session_type in [
            "habituation",
            "training",
            "tone_recall",
            "tone_LTMrecall",
        ]:

            # Get start/end times of trial and CS/US
            trial_starts = stimuli["absolute"]["CS_starts"]
            if self.session_type == "training" and include_US:
                trial_ends = stimuli["absolute"]["US_ends"]
            else:
                trial_ends = stimuli["absolute"]["CS_ends"]

            assert len(trial_starts) == len(trial_ends)
            ntrials = len(trial_starts)

            # Get max # frames across all events (should be generally the same but might be off by 1-2 between
            # trials due to hardware lags)
            nframes_event_max = (
                np.asarray((trial_ends - trial_starts) * SampleRate).astype("int").max()
            )

            # Now build up the speed_by_trial array
            speed_by_trial = (
                np.ones((ntrials, nframes_event_max + nframes_buffer.sum())) * np.nan
            )
            for idt, start in enumerate(trial_starts):
                start_ind = np.where(times >= start)[0][0]
                start_frame = start_ind - nframes_buffer[0]
                end_frame = start_ind + nframes_event_max + nframes_buffer[1]
                if end_frame < nframes:
                    speed_by_trial[idt] = mean_speed[start_frame:end_frame]
                else:
                    speed_by_trial[idt][0 : (nframes - start_frame)] = mean_speed[
                        start_frame:nframes
                    ]

            mean_speed = speed_by_trial
            times = (
                np.arange(-nframes_buffer[0], nframes_event_max + nframes_buffer[1])
                / SampleRate
            )

            stimuli_out = stimuli["by_trial"]
        if not by_trial and self.session_type in [
            "habituation",
            "training",
            "tone_recall",
            "tone_LTMrecall",
        ]:
            stimuli_out = stimuli["absolute"]

        return mean_speed, times, stimuli_out

    def get_baseline_speed(self, baseline_time=[120, 180]):
        """Pulls out speed during baseline period whose limits are specified"""
        mean_speed_abs, time_abs, _ = self.get_mean_speed(by_trial=False)
        baseline_bool = np.bitwise_and(
            time_abs > baseline_time[0], time_abs < baseline_time[1]
        )
        bl_time = time_abs[baseline_bool]
        bl_speed = mean_speed_abs[baseline_bool]

        return bl_speed, bl_time

    def plot_trial_speed_traces(
        self, trial_buffer_sec=10, ax=None, plot_baseline=True, paradigm=None, **kwargs
    ):
        """Plot trial speeds (and in future, freezing) for a given training or tone recall session."""

        assert self.session_type in ["training", "tone_recall", "tone_LTMrecall"], (
            "No trials in " + self.session_type + " session"
        )

        #  Get mean speed across all trials
        mean_speed, times, stimuli = self.get_mean_speed(
            by_trial=True, trial_buffer_sec=trial_buffer_sec, paradigm=paradigm
        )

        if ax is None and not plot_baseline:
            fig, ax_trial = plt.subplots()
            fig.set_size_inches([14, 5.75])
            ax_out = ax_trial
        elif ax is None and plot_baseline:
            fig = plt.figure(figsize=[14, 5.75])
            gs = GridSpec(1, 5, figure=fig)
            ax_bl = fig.add_subplot(gs[0, 0:2])
            ax_trial = fig.add_subplot(gs[0, 2:], sharey=ax_bl)
            ax_out = [ax_bl, ax_trial]
        elif ax is not None:
            ax_out = ax
            if isinstance(ax, list):
                ax_bl, ax_trial = ax
            else:
                ax_trial = ax

        # Get baseline speed if plotting
        if plot_baseline:
            bl_speed, bl_time = self.get_baseline_speed(**kwargs)
            bl_mean = np.nanmean(bl_speed)

        # Plot velocity
        htrial = ax_trial.plot(times, mean_speed.T, color=[0, 0, 0, 0.2])
        hmean = ax_trial.plot(
            times, np.nanmean(mean_speed, axis=0), color="b", linewidth=2
        )

        # Label events
        CS_start = 0
        CS_end = stimuli["CS_ends"].mean()
        hCS = ax_trial.axvspan(xmin=CS_start, xmax=CS_end, color=[0, 1, 0, 0.3])
        plt.legend([htrial[0], hmean[0], hCS], ["trial", "mean", "CS"])

        if isinstance(stimuli["US_starts"], np.ndarray):
            US_start = stimuli["US_starts"].mean()
            US_end = stimuli["US_ends"].mean()
            hUS = ax_trial.axvspan(xmin=US_start, xmax=US_end, color=[1, 0, 0, 0.3])
            plt.legend([htrial[0], hmean[0], hCS, hUS], ["trial", "mean", "CS", "US"])

        # Pretty it up and label axes/title
        sns.despine(ax=ax_trial)
        ax_trial.set_xlabel("Time from CS onset (s)")
        ax_trial.set_ylabel("Speed (cm/s)")
        ax_trial.set_title(self.animal_name + ": " + self.session_type.title())

        # Plot baseline
        if plot_baseline:
            ax_bl.plot(bl_time, bl_speed, "k", linewidth=1)
            ax_bl.plot(bl_time, bl_mean * np.ones_like(bl_time), "r--")
            ax_trial.plot(times, bl_mean * np.ones_like(times), "r--")
            ax_bl.set_xlabel("Time from session start (s)")
            ax_bl.set_ylabel("Speed (cm/s)")
            ax_bl.set_title("Baseline")
            ax_bl.set_ylim()
            sns.despine(ax=ax_bl)

        return ax_out


class trace_animal:
    """Class to analyze and plot trace fear conditioning data for a given animal across all experiments.
    **kwargs can be anything from trace_behavior class."""

    def __init__(self, animal_dir, paradigm, **kwargs):

        names_dict = generate_session_names(paradigm)

        arenas = names_dict["arenas"]
        sessions = names_dict["sessions"]
        session_names = names_dict["session_names"]
        self.titles = names_dict["titles"]
        self.session_names = session_names

        animal_path = Path(animal_dir)
        self.data = {}
        for session, name, arena in zip(sessions, session_names, arenas):
            print(session + " " + name + " " + arena)
            base_dir = sorted(animal_path.glob("*" + session))[0]
            params_use = trace_params[paradigm][arena]

            try:
                self.data[name] = trace_behavior(
                    base_dir,
                    search_str=params_use["camera"],
                    pix2cm=params_use["pix2cm"],
                    **kwargs,
                )
                self.animal_name = self.data[name].animal_name
            except FileNotFoundError:
                self.data[name] = False

    def plot_all_sessions(self, sessions=False, **kwargs):
        "Plots velocity, freezing, etc. info for all sessions in the same figure"
        if not sessions:
            sessions_plot = self.data.keys()
        else:
            sessions_plot = sessions
        fig, ax = plt.subplots(len(sessions_plot), 1)
        fig.set_size_inches([29, 17])
        fig.suptitle(self.animal_name, fontsize=20, fontweight="bold")

        for a, session, title in zip(ax, sessions_plot, self.titles):
            if self.data[session]:
                self.data[session].get_freezing_epochs(**kwargs, plot=a)
                a.set_ylim([-3, 90])
                a.set_title(title, fontsize=14, fontweight="bold")
            else:
                a.text(0.1, 0.5, "No tracking data found for this session")

        # Don't show xlabel except on bottom!
        [a.set_xlabel("") for a in ax[:-1]]

    def plot_trials(
        self,
        session_type,
        bodyparts=["neck_base", "mid_back"],
        by_trial=False,
        trial_buffer_sec=10,
        event_start_name="CS[0-6]?_start",
        event_end_name="CS[0-6]?_start",
    ):

        SampleRate = self.data[session_type].dlc.SampleRate
        nframes_buffer = int(SampleRate * trial_buffer_sec)
        bodypart_speed = []
        for bodypart in bodyparts:
            bodypart_speed.append(
                self.data[session_type].dlc.pos_smooth[bodypart]["speed"]
            )

        mean_speed = np.nanmean(bodypart_speed, axis=0)
        times = np.arange(0, mean_speed.shape[0]) / SampleRate

        # Break-down by trial now
        if by_trial and session_type in [
            "habituation",
            "training",
            "tone_recall",
            "tone_LTMrecall",
        ]:
            events_table = self.data[session_type].events
            trial_starts = events_table["Time (s)"][
                events_table["Event"].str.match(event_start_name)
            ]
            trial_ends = events_table["Time (s)"][
                events_table["Event"].str.match(event_end_name)
            ]

            assert len(trial_starts) == len(trial_ends)
            ntrials = len(trial_starts)

            # Get max # frames across all events (should be generally the same but might be off by 1-2 between trials due to hardware lags
            nframes_event_max = (
                np.asarray((trial_ends.values - trial_starts.values) * SampleRate)
                .astype("int")
                .max()
            )
            speed_by_trial = (
                np.ones((ntrials, nframes_event_max + 2 * nframes_buffer)) * np.nan
            )

            time_lookup = (
                np.arange(self.data[session_type].dlc.nframes)
                / self.data[session_type].dlc.SampleRate
            )
            for idt, start in enumerate(trial_starts):
                start_ind = np.where(time_lookup >= start)[0][0]
                start_frame = start_ind - nframes_buffer
                end_frame = start_ind + nframes_event_max + nframes_buffer
                speed_by_trial[idt] = mean_speed[start_frame:end_frame]

            mean_speed = speed_by_trial
            times = (
                np.arange(-nframes_buffer, nframes_event_max + nframes_buffer)
                / SampleRate
            )

        return mean_speed, times


class trace_group:
    """Class to analyze and plot trace fear conditioning data for a given animal across all experiments in a given paradigm."""

    def __init__(self, paradigm, base_dir):

        # Set up base directory for paradigm
        self.paradigm_path = Path(base_dir) / paradigm
        self.animal_dirs = sorted(self.paradigm_path.glob("Rat*"))
        self.animal_dirs = self.animal_dirs[
            np.where([d.is_dir() for d in self.animal_dirs])[0][0]
        ]  # Make sure you grab only directories
        if type(self.animal_dirs) is not list:
            self.animal_dirs = [self.animal_dirs]
        self.animal_names = [adir.parts[-1] for adir in self.animal_dirs]
        self.paradigm = paradigm

        self.animal = {}
        for animal_dir, name in zip(self.animal_dirs, self.animal_names):
            self.animal[name] = trace_animal(animal_dir, paradigm)

    def plot_all_animals(self, speed_threshold=0.5, save_to_pdf=False, **kwargs):
        """Plot speed/freezing/CS/US vs. time for all animals and save if specified"""
        if not save_to_pdf:
            for animal in self.animal.values():
                animal.plot_all_sessions(speed_threshold=0.25, **kwargs)
        elif save_to_pdf:
            save_name = self.paradigm_path / (
                self.paradigm
                + "_all_sessions_thresh"
                + "_".join(str(speed_threshold).split("."))
                + ".pdf"
            )
            with PdfPages(save_name) as pdf:
                for animal in self.animal.values():
                    animal.plot_all_sessions(speed_threshold=0.25, **kwargs)
                    pdf.savefig()

    def plot_mean_speed(
        self,
        session_type,
        bodyparts=["neck_base", "mid_back"],
        plot_by_trial=False,
        trial_buffer=10,
    ):
        """THIS IS CURRENTLY BROKEN !!!
        Plots mean speed for ALL animals with average (5 sec rolling mean) overlaid.  Can also plot by trial.
        :param session_type: str
        :param bodyparts: list
        :param plot_by_trial: boolean
        :param trial_buffer: time in seconds to include before/after CS (and US for training session)
        :return: ax
        """

        assert session_type in [
            "habituation",
            "training",
            "ctx_recall",
            "ctx_LTMrecall",
            "tone_recall",
            "tone_LTMrecall",
        ], "Invalid session type specified"

        # Get mean speed
        speed_all, time_plot, event_times = self.get_mean_speed(
            session_type, bodyparts=bodyparts, by_trial=plot_by_trial
        )

        # Set up plots
        fig, ax = plt.subplots()
        fig.set_size_inches([21.5, 4.9])
        SampleRate = self.animal[self.animal_names[0]].data[session_type].dlc.SampleRate
        if not plot_by_trial:

            time_plot = np.arange(0, speed_all.shape[1]) / SampleRate

            # Plot individual animals
            ax.plot(time_plot, speed_all.T, color=[0, 0, 0, 0.1])

            # Now plot average speed smoothed in 5 sec rolling window
            speed_all_mean = (
                pd.Series(np.nanmean(speed_all, axis=0))
                .rolling(int(SampleRate * 5))
                .mean()
            )
            ax.plot(time_plot, speed_all_mean, color="b", linewidth=2)  # mean

            # Tidy up and label everything
            ax.set_ylim([0, 40])
            ax.set_ylabel("Speed (cm/s)")
            ax.set_xlabel("Time (s)")
            sns.despine()
            ax.set_title(self.paradigm + ": " + session_type.title())
        elif plot_by_trial:
            pass

        return ax

    def plot_trial_speed_traces(
        self, session_type, trial_buffer_sec=20, plot_baseline=True, **kwargs
    ):

        if not plot_baseline:
            fig, ax = plt.subplots(len(self.animal_names))
            fig.set_size_inches([21, 17.4])
        else:
            fig = plt.figure(figsize=[21, 17.4])
            gs = GridSpec(len(self.animal_names), 5)
            ax = []
            for ida, animal_name in enumerate(self.animal_names):
                ax_bl = fig.add_subplot(gs[ida, 0:2])
                ax_trial = fig.add_subplot(gs[ida, 2:])
                ax.append([ax_bl, ax_trial])

        for ida, (a, animal_name) in enumerate(zip(ax, self.animal_names)):
            self.animal[animal_name].data[session_type].plot_trial_speed_traces(
                trial_buffer_sec=trial_buffer_sec,
                ax=a,
                plot_baseline=True,
                paradigm=self.paradigm,
                **kwargs,
            )

            # Clean up axes and set limits to be the same across all plots
            if ida < len(self.animal_names) - 1:
                [_.set_xlabel("") for _ in a]

            if session_type == "training":
                ylims = [-5, 70]
            else:
                ylims = [-5, 70]
            [_.set_ylim(ylims) for _ in a]

        return fig

    def plot_speed_summary(
        self, session_type, time_use=60, baseline_end=180, average_trials=True, ax=None
    ):
        """Plots mean speed during baseline period versus mean speed across all trials. Baseline goes from baseline_end
        - time_use to baseline_end"""

        assert session_type in [
            "habituation",
            "training",
            "tone_recall",
            "tone_LTMrecall",
        ]

        trial_mean_speed_all = []
        bl_mean_speed_all = []
        paradigm = self.paradigm
        for animal_name in self.animal_names:
            session = self.animal[animal_name].data[session_type]
            trial_speed, _, _ = session.get_mean_speed(
                by_trial=True,
                trial_buffer_sec=[0, time_use],
                paradigm=paradigm,
            )
            bl_speed, _ = session.get_baseline_speed(
                baseline_time=[baseline_end - time_use, baseline_end]
            )
            bl_mean_speed_all.append(np.nanmean(bl_speed))
            if average_trials:
                trial_mean_speed_all.append(np.nanmean(trial_speed))
            else:
                trial_mean_speed_all.append(np.nanmean(trial_speed, axis=1))

        fig, ax = plt.subplots()
        if average_trials:
            sns.stripplot(data=[bl_mean_speed_all, trial_mean_speed_all], ax=ax)

            ax.set_xticklabels(
                [
                    "Baseline "
                    + str(baseline_end - time_use)
                    + "-"
                    + str(baseline_end)
                    + " sec",
                    "CS onset + " + str(time_use) + " sec",
                ]
            )

        else:
            sns.stripplot(
                data=np.hstack(
                    (
                        np.asarray(bl_mean_speed_all)[:, None],
                        np.asarray(trial_mean_speed_all),
                    )
                ),
                ax=ax,
            )
            ntrials = len(trial_mean_speed_all[0])
            xticks = [
                "Baseline \n"
                + str(baseline_end - time_use)
                + "-"
                + str(baseline_end)
                + " sec"
            ]
            [
                xticks.append("Trial " + str(trial) + "\n CS + " + str(time_use))
                for trial in range(1, ntrials + 1)
            ]
            ax.set_xticklabels(xticks)
        ax.set_title(self.paradigm + ": " + session_type.title())
        sns.despine()
        ax.set_ylabel("Speed (cm/s)")

    def get_mean_speed(
        self,
        bodyparts=["neck_base", "mid_back"],
        by_trial=False,
        trial_buffer=10,
    ):
        """
        Get mean speed of the animal. For trials, assumes a consistent relationship between start (CS on) and end
        (CS off or shock off) of trial. Will need to revise if this is not true.
        :param bodyparts:
        :param by_trial: boolean. True = get time relative to CS (trial start), False (default): return absolute times
        :param trial_buffer: time in seconds (float) before/after trial beginning/end to include
        :return: speed_all: size nanimals x nframes ndarray if by_trial == False,
        len(animals) list of nCS x nframes ndarrays if by_trial == True
        """

        # Get mean speed across the WHOLE session

        # First get # frames in each video
        nframes_all = []
        for animal_name in self.animal_names:
            if self.animal[animal_name].data[
                session_type
            ]:  # Skip if no data for that session!
                nframes_all.append(
                    self.animal[animal_name]
                    .data[session_type]
                    .dlc.pos_smooth[bodyparts[0]]["speed"]
                    .size
                )

        # Now assemble an array with all the speed data together
        speed_all = np.ones((len(self.animal_names), np.max(nframes_all))) * np.nan
        for ida, (animal_name, nframes) in enumerate(
            zip(self.animal_names, nframes_all)
        ):

            speed_all[ida][0:nframes] = self.animal[animal_name].get_mean_speed(
                session_type, bodyparts=bodyparts
            )

        SampleRate = self.animal[self.animal_names[0]].data[session_type].dlc.SampleRate
        time_lookup = np.arange(0, np.max(nframes_all)) / SampleRate
        time_plot = time_lookup

        events = False
        # Get speed aligned to each trial if specified
        if by_trial and session_type in [
            "training",
            "tone_recall",
            "tone_LTMrecall",
        ]:

            nframes_buffer = int(SampleRate * trial_buffer)

            if session_type == "training":
                event_start_name, event_end_name = ["tone.*_start", "shock.*_end"]
                CS_end, US_start = ["tone_end", "shock_start"]
            else:
                event_start_name, event_end_name = ["CS.*start", "CS.*end"]

            ntrials = (
                self.animal[self.animal_names[0]]
                .data[session_type]
                .events["Event"]
                .str.match(event_start_name)
                .sum()
            )

            # Now assemble an array with all the speed data together
            speed_all_by_trial = []
            time_plot = []
            events = {}
            for ida, animal in enumerate(self.animal_names):
                events_table = self.animal[animal].data[session_type].events

                trial_starts = get_event_times(events_table, event_start_name)
                trial_ends = get_event_times(events_table, event_end_name)

                # trial_starts = events_table["Time (s)"][
                #     events_table["Event"].str.match(event_start_name)
                # ]
                # trial_ends = events_table["Time (s)"][
                #     events_table["Event"].str.match(event_end_name)
                # ]

                assert len(trial_starts) == len(trial_ends)
                ntrials = len(trial_starts)

                # Get max # frames across all events (should be generally the same but might be off by 1-2 between trials due to hardware lags
                nframes_event_max = (
                    np.asarray((trial_ends.values - trial_starts.values) * 30)
                    .astype("int")
                    .max()
                )
                speed_by_trial = (
                    np.ones((ntrials, nframes_event_max + 2 * nframes_buffer)) * np.nan
                )

                for idt, start in enumerate(trial_starts):
                    start_ind = np.where(time_lookup >= start)[0][0]
                    speed_by_trial[idt] = speed_all[ida][
                        (start_ind - nframes_buffer) : (
                            start_ind + nframes_event_max + nframes_buffer
                        )
                    ]
                speed_all_by_trial.append(speed_by_trial)
                time_plot.append(
                    np.arange(-nframes_buffer, nframes_event_max + nframes_buffer)
                    / SampleRate
                )

                # Get event timing based on first animal only
                if ida == 0:
                    events["CS_start"] = 0
                    events["CS_end"] = (trial_ends.values - trial_starts.values).mean()

                    if session_type == "training":

                        shock_starts = (
                            get_event_times(events_table, "shock.*_start")
                            - trial_starts.values
                        )
                        CS_ends = (
                            get_event_times(events_table, "tone.*_end")
                            - trial_starts.values
                        )
                        # shock_starts = (
                        #     events_table["Time (s)"][
                        #         events_table["Event"].str.match("shock.*_start")
                        #     ].values
                        #     - trial_starts.values
                        # )
                        # CS_ends = (
                        #     events_table["Time (s)"][
                        #         events_table["Event"].str.match("tone.*_end")
                        #     ].values
                        #     - trial_starts.values
                        # )

                        events["CS_end"] = CS_ends.mean()
                        events["US_start"] = shock_starts.mean()
                        events["US_end"] = (
                            trial_ends.values - trial_starts.values
                        ).mean()

            speed_all = speed_all_by_trial

        return speed_all, time_plot, events


def get_stimuli_times(events_table, session_type, paradigm=None):
    """More specific function to grab CS and US times"""
    CS_starts, CS_ends = False, False
    US_starts, US_ends = False, False

    good_session = False
    if session_type in ["habituation", "training", "tone_recall", "tone_LTMrecall"]:
        good_session = True
        if session_type == "training":
            CS_start_name, CS_end_name = ["tone.*_start", "tone.*_end"]
            US_start_name, US_end_name = ["shock.*_start", "shock.*_end"]
        elif session_type in ["habituation", "tone_recall", "tone_LTMrecall"]:
            CS_start_name, CS_end_name = ["CS.*start", "CS.*end"]

        CS_starts = get_event_times(events_table, CS_start_name)
        CS_ends = get_event_times(events_table, CS_end_name)
        if (
            session_type == "habituation"
        ):  # NK todo: document/fix this: Fake trials to match paradigm2 structure
            CS_starts = np.asarray([180, 250, 320, 390, 460, 520])
            CS_ends = CS_starts + 10

        # Exclude first CS from Pilot1 tone recall sessions - just plot the long CS twice
        if paradigm is not None and session_type != "training" and paradigm == "Pilot1":
            CS_starts = np.asarray([CS_starts[-1], CS_starts[-1]])
            CS_ends = np.asarray([CS_ends[-1], CS_ends[-1]])

        if session_type == "training":
            US_starts = get_event_times(events_table, US_start_name)
            US_ends = get_event_times(events_table, US_end_name)

        # Now make CS onset time zero if specifed
        CS_ends_by_trial = CS_ends - CS_starts
        if session_type == "training":
            US_starts_by_trial = US_starts - CS_starts
            US_ends_by_trial = US_ends - CS_starts
        else:
            US_starts_by_trial, US_ends_by_trial = False, False
        CS_starts_by_trial = np.zeros_like(CS_starts)

    if good_session:
        # Now dump everything into a dictionary
        stimuli_dict = {
            "absolute": {
                "CS_starts": CS_starts,
                "CS_ends": CS_ends,
                "US_starts": US_starts,
                "US_ends": US_ends,
            },
            "by_trial": {
                "CS_starts": CS_starts_by_trial,
                "CS_ends": CS_ends_by_trial,
                "US_starts": US_starts_by_trial,
                "US_ends": US_ends_by_trial,
            },
        }
    else:
        stimuli_dict = False

    return stimuli_dict


def get_event_times(events_table, event_name):
    """General function to grab ALL event times from pandas dataframe."""
    event_times = events_table["Time (s)"][
        events_table["Event"].str.match(event_name)
    ].values

    return np.asarray(event_times)


def fix_date(date_str):
    """Sends date from YYYY_MM_DD format to MM_DD_YYYY format"""
    # Find first underscore
    first_under = re.search("_", date_str).span()[0]
    if first_under == 2:
        date_fixed = date_str
    elif first_under == 4:
        date_fixed = date_str[5:7] + "_" + date_str[8:10] + "_" + date_str[0:4]

    return date_fixed


def generate_session_names(paradigm):
    """Generates session names and titles for plots and such based on paradigm.

    :param paradigm: 'Pilot1', 'Pilot2', or 'Recording_Rats'
    :return:
    """
    arenas = ["Shock", "Shock", "Shock", "Shock", "New", "New"]
    sessions = [
        "_habituation",
        "_training",
        "_recall",
        "_LTMrecall",
        "_recall",
        "_LTMrecall",
    ]
    session_names = [
        "habituation",
        "training",
        "ctx_recall",
        "ctx_LTMrecall",
        "tone_recall",
        "tone_LTMrecall",
    ]

    titles = [
        "Habituation",
        "Training",
        "CTX Recall",
        "CTX LTM Recall",
        "Tone Recall",
        "Tone LTM Recall",
    ]
    if paradigm == "Recording_Rats":
        arenas12 = ["Shock", "Shock"]
        sessions12 = ["_habituation1", "_habituation2"]
        session_names12 = ["habituation1", "habituation2"]
        titles12 = ["Habituation 1", "Habituation 2"]

        arenas12.extend(arenas[1:])
        sessions12.extend(sessions[1:])
        session_names12.extend(session_names[1:])
        titles12.extend(titles[1:])

        arenas, sessions = arenas12, sessions12
        session_names, titles = session_names12, titles12

    names_dict = {
        "arenas": arenas,
        "sessions": sessions,
        "session_names": session_names,
        "titles": titles,
    }

    return names_dict


def load_events_from_csv(csvfile: str):
    """Load events into pandas format and get absolute timestamps."""
    event_header = pd.read_csv(csvfile, header=None, nrows=1)
    assert (event_header[0] == "Start time").all(), "csv file not formatted properly"
    assert (event_header[2] == "microseconds").all(), "csv file not formatted properly"

    start_time = pd.Timestamp(event_header[1][0]) + pd.Timedelta(
        event_header[3][0], unit="microseconds"
    )

    event_df = pd.read_csv(csvfile, header=1)
    event_df["Timestamp"] = start_time + pd.to_timedelta(
        event_df["Time (s)"], unit="sec"
    )

    return event_df


def add_TTL_times_to_events(events_df, TTL):
    """To be written function that adds another column to an events dataframe with timestamps from TTLs"""
    pass


if __name__ == "__main__":
    r1 = trace_animal("/data2/Trace_FC/Recording_Rats/Rat698", "Recording1")
