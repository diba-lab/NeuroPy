import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from parsePath import Recinfo
import DLC_IO as dio
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
}


class trace_behavior:
    """Class to analyze trace_conditioning behavioral data"""

    def __init__(self, base_path, search_str=None, movie_type=".mp4", pix2cm=1):
        # if isinstance(basepath, Recinfo):
        #     self._obj = basepath
        # else:
        #     self._obj = Recinfo(basepath)

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
                re.search("shock", self.dlc.tracking_file.lower()) is not None
            ):  # check if in shock arena
                self.session_type = "ctx_" + self.session_type
            elif re.search("newarena", self.dlc.tracking_file.lower()) is not None:
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

            # # this plots candidate freezing events - un-comment below if you want to plot in the future
            # for region in regions:
            #     self.pos_smooth[bodyparts[-1]].iloc[region[0]].plot(
            #         x="time", y="speed", legend=False, ax=ax, style="r--"
            #     )
            # legend_labels.extend(["candidate freezing epochs"])

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
                tone_starts = self.events["Time (s)"][
                    self.events["Event"] == "shock_start"
                ]
                tone_ends = self.events["Time (s)"][self.events["Event"] == "shock_end"]
                for start, end in zip(tone_starts, tone_ends):
                    hshock = ax.axvspan(start, end, color=[1, 0, 0, 0.3])
                legend_handles.append(hshock)
                legend_labels.append("Shock")
                ax.legend(legend_handles, legend_labels)

            sns.despine(ax=ax)

        return long_freezing_times, freeze_bool

    def load_events(self):
        """Load events (start/end times, CS/US times, etc.) from .CSV file."""
        # Grab tracking files and corresponding movie file.
        event_files = sorted(self.base_dir.glob("*.csv"))
        self.event_file = dio.get_matching_files(
            event_files,
            match_str=[self.session_type.replace("LTM", ""), self.session_date],
            exclude_str=["test"],
        )[0]

        # Load in event start time metadata

        # Load in all events
        with open(str(self.event_file)) as f:
            reader = csv.reader(f)
            self.event_start = next(reader)

        self.events = pd.read_csv(self.event_file, header=1)


class trace_animal:
    """Class to analyze and plot trace fear conditioning data for a given animal across all experiments.
    **kwargs can be anything from trace_behavior class."""

    def __init__(self, animal_dir, paradigm, **kwargs):
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

        self.titles = [
            "Habituation",
            "Training",
            "CTX Recall",
            "CTX LTM Recall",
            "Tone Recall",
            "Tone LTM Recall",
        ]

        self.session_names = session_names

        animal_path = Path(animal_dir)
        self.data = {}
        for session, name, arena in zip(sessions, session_names, arenas):
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

    def plot_all_sessions(self, **kwargs):
        "Plots velocity, freezing, etc. info for all sessions in the same figure"
        fig, ax = plt.subplots(6, 1)
        fig.set_size_inches([29, 17])
        fig.suptitle(self.animal_name, fontsize=20, fontweight="bold")
        for a, session, title in zip(ax, self.data.keys(), self.titles):
            if self.data[session]:
                self.data[session].get_freezing_epochs(**kwargs, plot=a)
                a.set_ylim([-3, 90])
                a.set_title(title, fontsize=14, fontweight="bold")
            else:
                a.text(0.1, 0.5, "No tracking data found for this session")

        # Don't show xlabel except on bottom!
        [a.set_xlabel("") for a in ax[:-1]]


class trace_group:
    """Class to analyze and plot trace fear conditioning data for a given animal across all experiments in a given paradigm."""

    def __init__(self, paradigm, base_dir):

        # Set up base directory for paradigm
        self.paradigm_path = Path(base_dir) / paradigm
        self.animal_dirs = sorted(self.paradigm_path.glob("Rat*"))
        self.animal_names = [adir.parts[-1] for adir in self.animal_dirs]
        self.paradigm = paradigm

        self.animal = {}
        for animal_dir, name in zip(self.animal_dirs, self.animal_names):
            self.animal[name] = trace_animal(animal_dir, paradigm)

    def plot_all_animals(self, speed_threshold=0.5, save_to_pdf=False):
        """Plot speed/freezing/CS/US vs. time for all animals and save if specified"""
        if not save_to_pdf:
            for animal in self.animal:
                animal.plot_all_sessions(speed_threshold=0.25)
        elif save_to_pdf:
            save_name = self.paradigm_path / (
                self.paradigm
                + "_all_sessions_thresh"
                + "_".join(str(speed_threshold).split("."))
                + ".pdf"
            )
            with PdfPages(save_name) as pdf:
                for animal in self.animal.values():
                    animal.plot_all_sessions(speed_threshold=0.25)
                    pdf.savefig()

    def plot_mean_speed(self, session_type, bodyparts=["neck_base", "mid_back"]):

        assert session_type.lower() in [
            "habituation",
            "training",
            "ctx_recall",
            "ctx_ltmrecall",
            "tone_recall",
            "tone_ltmrecall",
        ], "Invalid session type specified"

        if session_type in ["habituation", "training", "ctx_recall", "ctx_LTMrecall"]:

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
                bodypart_speed = []
                for bodypart in bodyparts:
                    bodypart_speed.append(
                        self.animal[animal_name]
                        .data[session_type]
                        .dlc.pos_smooth[bodypart]["speed"]
                    )
                speed_all[ida][0:nframes] = np.mean(bodypart_speed, axis=0)

        # Now plot!!!
        fig, ax = plt.subplots()
        fig.set_size_inches([21.5, 4.9])
        SampleRate = self.animal[self.animal_names[0]].data[session_type].dlc.SampleRate
        time = np.arange(1, np.max(nframes_all) + 1) / SampleRate
        ax.plot(time, speed_all.T, color=[0, 0, 0, 0.2])  # individual animals
        # Now plot average speed smoothed in 5 sec rolloing window
        speed_all_mean = (
            pd.Series(np.nanmean(speed_all, axis=0)).rolling(int(SampleRate * 5)).mean()
        )
        ax.plot(time, speed_all_mean, color="b", linewidth=2)  # mean
        ax.set_ylim([0, 40])
        ax.set_ylabel("Speed (cm/s)")
        ax.set_xlabel("Time (s)")
        sns.despine()
        ax.set_title(session_type.title())

        return ax


def fix_date(date_str):
    """Sends date from YYYY_MM_DD format to MM_DD_YYYY format"""
    # Find first underscore
    first_under = re.search("_", date_str).span()[0]
    if first_under == 2:
        date_fixed = date_str
    elif first_under == 4:
        date_fixed = date_str[5:7] + "_" + date_str[8:10] + "_" + date_str[0:4]

    return date_fixed


if __name__ == "__main__":
    tg2 = trace_group("Pilot2", "/data2/Trace_FC")
    tg2.plot_mean_speed("ctx_LTMrecall", ["neck_base", "mid_back"])
