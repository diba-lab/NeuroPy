import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from neuropy.utils.misc import interp_nans, find_nearest, arg_find_nearest
from pathlib import Path
from copy import deepcopy, copy


class Raster:
    def __init__(
        self,
        ca_activity: np.ndarray,
        times: pd.Series,
        event_start_times: pd.Series,
        event_end_times: pd.Series,
        cell_id: int or None,
        start_buffer_sec: float = 10.0,
        end_buffer_sec: float = 10.0,
        auto_generate=True,
    ):
        self.ca_activity = ca_activity
        self.times = times
        self.cell_id = cell_id
        self.event_starts = event_start_times
        self.event_ends = event_end_times
        self.start_buffer_sec = start_buffer_sec
        self.end_buffer_sec = end_buffer_sec

        if ca_activity.ndim == 1:
            assert len(ca_activity) == len(
                times
            ), "calcium activity and times must be the same length"
        elif ca_activity.ndim == 2:
            assert ca_activity.shape[1] == len(
                times
            ), "calcium activity and times must have the same number frames"
            self.ca_activity = ca_activity[cell_id]

        self.raster_time, self.raster, self.remove_nan = None, None, None
        if auto_generate:
            self.generate_raster(self.start_buffer_sec, self.end_buffer_sec)

    @property
    def raster_mean(self):
        raster_mean = (
            np.nanmean(self.raster, axis=0) if self.raster is not None else None
        )
        return raster_mean

    @property
    def raster_baseline(self):
        """Get baseline activity from start to end of raster activity period"""
        event_ends_use = (
            self.event_starts if self.event_ends is None else self.event_ends
        )
        baseline, _ = self.get_baseline(
            self.ca_activity,
            self.times,
            self.event_starts,
            event_ends_use,
            self.start_buffer_sec,
            self.end_buffer_sec,
        )

        return baseline

    def generate_raster(self, start_buffer_sec, end_buffer_sec, remove_nan=True):
        """Generate a peri-event raster - see below staticmethod for more documentation"""

        if (
            self.raster is not None
            and start_buffer_sec == self.start_buffer_sec
            and end_buffer_sec == end_buffer_sec
            and remove_nan == self.remove_nan
        ):
            return self.raster, self.raster_time
        else:  # Update if not already run or if changing buffer secs.
            self.raster, self.raster_time = self.generate_raster_(
                self.ca_activity,
                self.times,
                self.cell_id,
                self.event_starts,
                self.event_ends,
                start_buffer_sec=start_buffer_sec,
                end_buffer_sec=end_buffer_sec,
                remove_nan=remove_nan,
            )

            self.start_buffer_sec = start_buffer_sec
            self.end_buffer_sec = end_buffer_sec
            self.remove_nan = remove_nan

            return self.raster, self.raster_time

    def get_mean_peak(self):
        """Get peak of mean raster and index where it occurs"""
        idx = np.argmax(self.raster_mean)

        return idx, self.raster_mean[idx]

    def get_mean_trough(self):
        """Get trough of mean raster and index where it occurs"""
        idx = np.argmin(self.raster_mean)

        return idx, self.raster_mean[idx]

    def split(
        self, split_type: str in ["odd_even", "random", "half_time", "half_events"]
    ):
        """Split rasters up to assess consistency
        Better would be to make this super generic and let you grab any set of events"""
        assert split_type in [
            "odd_even",
            "random" "half_time",
            "half_events",
        ], "'split_type' entered not supported"
        raster_split = [deepcopy(self), deepcopy(self)]
        if split_type == "odd_even":
            for idr, rast_sp in enumerate(raster_split):
                rast_sp.event_starts = self.event_starts[idr::2]
                if self.event_ends is not None:
                    rast_sp.event_ends = self.event_ends[idr::2]
                if self.raster is not None:
                    rast_sp.raster = self.raster[idr::2]
        elif split_type == "random":  # Split up randomly into halves
            nevents = self.event_starts.shape[0]
            set1 = np.random.choice(nevents, int(nevents / 2), replace=True)
            set2 = [nid if nid not in set1 else -1 for nid in range(nevents)]
            set2 = np.array(set2)[np.array(set2) != -1].astype(int)
            for event_ids, rast_sp in zip([set1, set2], raster_split):
                rast_sp.event_starts = self.event_starts[event_ids]
                if self.event_ends is not None:
                    rast_sp.event_ends = self.event_ends[event_ids]
                if self.raster is not None:
                    rast_sp.raster = self.raster[event_ids]
        else:
            assert False, "'split_type' entered is not yet implemented"
        return raster_split[0], raster_split[1]

    @staticmethod
    def generate_shuffled_raster_(
        activity,
        times,
        cell_id,
        event_starts,
        event_ends,
        start_buffer_sec=10,
        end_buffer_sec=10,
    ):
        # First isolate activity to the event limits

        # Next circularly permute activity

        # Last calculate your raster -
        pass

    @staticmethod
    def generate_raster_(
        activity: np.ndarray,
        times: pd.Series,
        cell_id: int or None,
        event_starts: pd.Series,
        event_ends: pd.Series,
        start_buffer_sec=10.0,
        end_buffer_sec=10.0,
        remove_nan=True,
    ):
        """Generate a peri-event raster for the times in event_starts to event_ends
        (if specified) +/ buffer times indicated"""

        # Grab appropriate cell's activity
        if activity.ndim == 2:
            activity = activity[cell_id]

        # Send end times to start times if not specified
        if event_ends is None:
            event_ends = event_starts

        # Calculate event durations
        event_durs = (event_ends.values - event_starts.values).astype(
            "timedelta64[ns]"
        ).astype(float) / 10 ** 9
        avg_event_sec = np.nanmean(event_durs)

        # Now loop through and chop out peri-event activity
        # start_id, end_id = [], []
        raster, raw_raster, time_list = [], [], []
        # for start, end in zip(event_starts, event_ends):
        #     # first, id neural data time for start, end and account for buffer times before/after
        #     # start_id.append((times - start).dt.total_seconds().abs().argmin())
        #     start_buffer = (
        #         (times - (start - pd.Timedelta(start_buffer_sec, unit="sec")))
        #         .dt.total_seconds()
        #         .abs()
        #         .argmin()
        #     )
        #     # end_id.append((times - end).dt.total_seconds().abs().argmin())
        #     end_buffer = (
        #         (
        #             times
        #             - (end + pd.Timedelta(end_buffer_sec + avg_event_sec, unit="sec"))
        #         )
        #         .dt.total_seconds()
        #         .abs()
        #         .argmin()
        #     )
        #
        #     # Get time from event start for this event and keep only frames within buffer second
        #     # This is necessary to avoid grabbing frames super far away across a disconnect
        #     trial_dt = (times[start_buffer : end_buffer + 1] - start).dt.total_seconds()
        #     good_frame_bool = np.bitwise_and(
        #         trial_dt > -start_buffer_sec,
        #         trial_dt < (end_buffer_sec + avg_event_sec),
        #     )

        # Identify time points for each trial
        start_buffers, end_buffers, good_frames_bool, trial_dts = Raster.get_start_ends(
            event_starts,
            event_ends,
            times,
            start_buffer_sec,
            end_buffer_sec,
            avg_event_sec,
            pd_to_np=True,
        )

        for start_buffer, end_buffer, good_frame_bool, trial_dt in zip(
            start_buffers, end_buffers, good_frames_bool, trial_dts
        ):
            if (
                start_buffer == end_buffer
            ):  # Skip adding anything if no data for that event
                pass
            else:
                # Build up raster(s) of activity around event times
                raster.append(activity[start_buffer : end_buffer + 1][good_frame_bool])
                time_list.append(trial_dt[good_frame_bool])

        ## Get times
        # First infer sampling rate
        dt_good_bool = (
            times.diff().dt.total_seconds() < 0.2
        )  # Assume any frames more than 0.2 sec apart are due to a disconnect
        sr = 1 / times.diff().dt.total_seconds()[dt_good_bool].mean()
        # Calculate trial duration
        dur_sec = start_buffer_sec + end_buffer_sec + avg_event_sec
        # last get times for each bin in the raster array relative to event start
        rast_time = np.linspace(
            -start_buffer_sec,
            -start_buffer_sec + dur_sec,
            np.floor(dur_sec * sr).astype(int),
        )

        # Now build up arrays!
        nevents = len(raster)  # Only keep events with calcium activity during them
        nframes = len(rast_time)
        rast_array = np.ones((nevents, nframes)) * np.nan  # pre-allocate
        for idt, (time, activity) in enumerate(zip(time_list, raster)):
            bins = np.digitize(time, rast_time, right=True)
            rast_array[idt][bins] = activity

        if remove_nan:  # Remove any columns that are all nan
            good_bool = np.bitwise_not(np.all(np.isnan(rast_array), axis=0))
            rast_array = rast_array[:, good_bool]
            rast_time = rast_time[good_bool]

        # Last interpolate any nan values
        rast_array = interp_nans(rast_array)

        return rast_array, rast_time

    @staticmethod
    def get_baseline(
        activity,
        times,
        event_starts,
        event_ends,
        start_buffer_sec=10,
        end_buffer_sec=10,
    ):
        # Get baselines for activity using only activity from the first to last event +/- buffers
        bl_start = event_starts.min() - pd.Timedelta(start_buffer_sec, unit="sec")
        bl_end = event_ends.max() + pd.Timedelta(end_buffer_sec, unit="sec")
        bl_bool = (times > bl_start) & (times < bl_end)
        baseline = np.nanmean(activity[bl_bool])

        return baseline, bl_bool

    @staticmethod
    def get_start_ends(
        event_starts,
        event_ends,
        times,
        start_buffer_sec,
        end_buffer_sec,
        avg_event_sec,
        pd_to_np=True,
    ):
        """This automatically makes things run faster by using pd.Series only when necessary for identfying
        frames with a trial.  Good target for elimination once data alignment is set!"""

        if pd_to_np or isinstance(times, np.ndarray):
            if pd_to_np and isinstance(times, pd.Series):
                start_time = times.iloc[0]
                event_starts = (event_starts - start_time).dt.total_seconds().values
                event_ends = (event_ends - start_time).dt.total_seconds().values
                times = (times - start_time).dt.total_seconds().values

                (
                    start_buffers,
                    end_buffers,
                    good_frames_bool,
                    trial_dts,
                ) = Raster.get_start_ends_np(
                    event_starts,
                    event_ends,
                    times,
                    start_buffer_sec,
                    end_buffer_sec,
                    avg_event_sec,
                )
        else:
            (
                start_buffers,
                end_buffers,
                good_frames_bool,
                trial_dts,
            ) = Raster.get_start_ends_pd(
                event_starts,
                event_ends,
                times,
                start_buffer_sec,
                end_buffer_sec,
                avg_event_sec,
            )

        return start_buffers, end_buffers, good_frames_bool, trial_dts

    @staticmethod
    def get_start_ends_pd(
        event_starts, event_ends, times, start_buffer_sec, end_buffer_sec, avg_event_sec
    ):
        """Code to get start and end times for building rasters for above functions. Much slower using
        pandas than numpy (6x slowdown)."""
        start_buffers, end_buffers, good_frames_bool, trial_dts = [], [], [], []
        for start, end in zip(event_starts, event_ends):
            # first, id neural data time for start, end and account for buffer times before/after
            start_buffer = (
                (times - (start - pd.Timedelta(start_buffer_sec, unit="sec")))
                .dt.total_seconds()
                .abs()
                .argmin()
            )
            start_buffers.append(start_buffer)
            end_buffer = (
                (
                    times
                    - (end + pd.Timedelta(end_buffer_sec + avg_event_sec, unit="sec"))
                )
                .dt.total_seconds()
                .abs()
                .argmin()
            )
            end_buffers.append(end_buffer)

            # Get time from event start for this event and keep only frames within buffer second
            # This is necessary to avoid grabbing frames super far away across a disconnect
            trial_dt = (times[start_buffer : end_buffer + 1] - start).dt.total_seconds()
            trial_dts.append(trial_dt)
            good_frame_bool = np.bitwise_and(
                trial_dt > -start_buffer_sec,
                trial_dt < (end_buffer_sec + avg_event_sec),
            )
            good_frames_bool.append(good_frame_bool)

        return start_buffers, end_buffers, good_frames_bool, trial_dts

    @staticmethod
    def get_start_ends_np(
        event_starts,
        event_ends,
        times,
        start_buffer_sec,
        end_buffer_sec,
        avg_event_sec,
    ):
        """Code for getting start and end times for building rasters above. Much faster than pandas (6x speedup)"""
        start_buffers, end_buffers, good_frames_bool, trial_dts = [], [], [], []
        for start, end in zip(event_starts, event_ends):
            # first, id neural data time for start, end and account for buffer times before/after
            start_buffer = np.argmin(np.abs(times - (start - start_buffer_sec)))
            start_buffers.append(start_buffer)

            end_buffer = np.argmin(
                np.abs(times - (end + end_buffer_sec + avg_event_sec))
            )
            end_buffers.append(end_buffer)

            # Get time from event start for this event and keep only frames within buffer second
            # This is necessary to avoid grabbing frames super far away across a disconnect
            trial_dt = times[start_buffer : end_buffer + 1] - start
            trial_dts.append(trial_dt)
            good_frame_bool = np.bitwise_and(
                trial_dt > -start_buffer_sec,
                trial_dt < (end_buffer_sec + avg_event_sec),
            )
            good_frames_bool.append(good_frame_bool)

        return start_buffers, end_buffers, good_frames_bool, trial_dts


class RasterGroup:
    def __init__(
        self,
        ca_activity: np.ndarray,
        times: pd.Series,
        event_start_times: pd.Series,
        event_end_times: pd.Series,
        cell_ids: list or np.ndarray or None,
        start_buffer_sec: float = 10.0,
        end_buffer_sec: float = 10.0,
        auto_generate: bool = True,
    ):
        self.Raster = []
        self.cell_ids = (
            cell_ids if cell_ids is not None else np.arange(ca_activity.shape[0])
        )
        for cell_id in self.cell_ids:
            self.Raster.append(
                Raster(
                    ca_activity,
                    times,
                    event_start_times,
                    event_end_times,
                    cell_id,
                    start_buffer_sec,
                    end_buffer_sec,
                )
            )

        self.start_buffer_sec = start_buffer_sec
        self.end_buffer_sec = end_buffer_sec
        if auto_generate:
            self.generate_rasters()

    # def split(self, split_type: str in ["odd_even", "random", "half_time", "half_events"]
    # ):
    #
    #     rast_split = [deepcopy(self), deepcopy(self)]
    #     for rast_sp in rast_split:
    #         rast_sp.Raster = []
    #

    def cell_slice(self, cell_ids_to_grab):
        """Grab cell_ids cells from larger set of rasters"""
        # Get indexes of cells to grab, leaving silent/unmatched cells as -1s
        cell_idxs = np.array(
            [
                np.where(cell_to_grab == self.cell_ids)[0][0]
                if cell_to_grab >= 0
                else -1
                for cell_to_grab in cell_ids_to_grab
            ]
        )
        RastGroupSlice = deepcopy(self)
        rasters = []
        for cell_idx in cell_idxs:
            # for cell_to_grab in cell_ids_to_grab:
            #     cell_idx = (
            #         np.where(cell_to_grab == self.cell_ids)[0][0] if cell_idx >= 0 else -1
            #     )
            if cell_idx >= 0:
                rasters.append(self.Raster[cell_idx])
            else:  # Sent raster to all NaNs if not specified
                rasters.append(deepcopy(self.Raster[0]))
                rasters[-1].raster = np.ones_like(rasters[-1].raster) * np.nan

        RastGroupSlice.Raster = rasters
        RastGroupSlice.cell_ids = cell_ids_to_grab

        return RastGroupSlice

    def generate_rasters(self, start_buffer_sec=None, end_buffer_sec=None):
        start_buffer_sec = (
            self.start_buffer_sec if start_buffer_sec is None else start_buffer_sec
        )
        end_buffer_sec = (
            self.end_buffer_sec if end_buffer_sec is None else end_buffer_sec
        )
        for rast_obj in self.Raster:
            rast_obj.generate_raster(start_buffer_sec, end_buffer_sec)

    def snake_plot(
        self,
        sortby: list or np.ndarray or str = "peak_time",
        norm_each_row: None or str in ["max", "z"] = "max",
        ax=None,
        xlabel_increment=10,
        **kwargs,
    ):
        """Make snake plot of sorted cell activity"""

        # Sort mean rasters
        sorted_mean_rast, sort_ids = self.sort_rasters(sortby, norm_each_row)

        # Plot
        assert ax is None or isinstance(ax, plt.Axes)
        if ax is None:
            _, ax = plt.subplots(figsize=(7.75, 10))
        sns.heatmap(
            sorted_mean_rast, ax=ax, xticklabels=self.Raster[0].raster_time, **kwargs
        )

        # Pretty up the plots
        time_plot = self.Raster[0].raster_time
        time_range = [np.floor(time_plot.min()), np.ceil(time_plot.max())]
        xticks = [
            ax.get_xticks()[arg_find_nearest(time_plot, t)]
            for t in np.arange(time_range[0], time_range[1], xlabel_increment - 0.001)
        ]
        xticklabels = [
            f"{t:0.0f}"
            for t in np.arange(time_range[0], time_range[1], xlabel_increment - 0.001)
        ]
        ax.set(xticks=xticks, xticklabels=xticklabels)
        ax.set_xlabel("Time (s)")
        yticklabels = [str(cell) for cell in [0, len(self.cell_ids)]]
        yticks = [ax.get_yticks()[0], ax.get_yticks()[-1]]
        ax.set(yticks=yticks, yticklabels=yticklabels)
        ax.set_ylabel("Cell #")

        return sort_ids

    def sort_rasters(
        self,
        sortby: list or np.ndarray or str = "peak_time",
        norm_each_row: None or str in ["max", "z"] = "max",
    ):

        assert isinstance(sortby, (list, np.ndarray)) or (
            isinstance(sortby, str) and sortby in ["peak_time", "trough_time"]
        )
        if isinstance(sortby, str) and (
            sortby == "peak_time" or sortby == "trough_time"
        ):
            peak_idx = []
            for rast in self.Raster:
                pid, _ = (
                    rast.get_mean_peak()
                    if sortby == "peak_time"
                    else rast.get_mean_trough()
                )
                peak_idx.append(pid)
            sort_ids = np.argsort(peak_idx)
        else:
            sort_ids = sortby

        sorted_mean_rast = np.array([self.Raster[idx].raster_mean for idx in sort_ids])

        # Normalize each row to itself
        if norm_each_row == "max":
            sorted_mean_rast = sorted_mean_rast / sorted_mean_rast.max(axis=1)[:, None]
        elif norm_each_row == "z":
            sorted_mean = np.array(
                [np.nanmean(self.Raster[idx].raster.reshape(-1)) for idx in sort_ids]
            )
            sorted_std = np.array(
                [np.nanstd(self.Raster[idx].raster.reshape(-1)) for idx in sort_ids]
            )
            sorted_mean_rast = (sorted_mean_rast - sorted_mean[:, None]) / sorted_std[
                :, None
            ]
        return sorted_mean_rast, sort_ids


def plot_pe_traces(
    times: pd.Series,
    activity: np.ndarray,
    event_starts: pd.Series,
    cell_id: int or None = None,
    event_ends: pd.Series = None,
    raw_trace: np.ndarray = None,
    start_buffer_sec: float or int = 10,
    end_buffer_sec: float or int = 10,
    ax: plt.axes = None,
    event_type: str = "",
    ylabels: list or tuple = ["Processed Activity", "Raw Activity"],
    event_color: list or tuple = [0, 1, 0, 0.3],
):
    """
    Plot peri-event calcium event traces. If using start and end times, assumes same event duration each time.
    :param times: timestamps for whole session
    :param activity: processed activity (e.g. C or S from CNMF_E) for one cell or all cells, same size as times
    :param event_starts: pandas Series of timestamps for event starts
    :param cell_id: int, required if activity or raw_trace is for all neurons
    :param event_ends: pandas Series of timestamps for event ends
    :param raw_trace: raw(ish) activity for one cell or all cells, e.g. raw traces or C from CNMF_E
    :param start_buffer_sec: # sec before event start to include
    :param end_buffer_sec: # sec after event end to include
    :param ax: axes to plot into, default (None) = create new figure
    :param event_type: str, label for plots
    :param ylabels: ylabels corresponding to activity and raw_trace
    :param
    :return:
    """

    # Grab appropriate cell's activity
    if activity.ndim == 2:
        activity = activity[cell_id]

    if raw_trace is not None and raw_trace.ndim == 2:
        raw_trace = raw_trace[cell_id]

    # Send end times to start times if not specified
    if event_ends is None:
        event_ends = event_starts

    # Calculate event durations
    event_durs = (event_ends.values - event_starts.values).astype(
        "timedelta64[ns]"
    ).astype(float) / 10 ** 9
    avg_event_sec = np.nanmean(event_durs)

    # Get rasters
    rast_array, time_plot = Raster.generate_raster_(
        activity,
        times,
        cell_id,
        event_starts,
        event_ends,
        start_buffer_sec,
        end_buffer_sec,
    )

    baseline, bl_bool = Raster.get_baseline(
        activity,
        times,
        event_starts,
        event_ends,
        start_buffer_sec,
        end_buffer_sec,
    )
    ### NRK todo: Future cleanup - don't even run the raw traces, just make wrapper function to
    # plot each into predefined axes.
    if raw_trace is not None:
        raw_rast_array, time_plot = Raster.generate_raster_(
            raw_trace,
            times,
            cell_id,
            event_starts,
            event_ends,
            start_buffer_sec,
            end_buffer_sec,
        )
        raw_baseline, bl_bool = Raster.get_baseline(
            activity,
            times,
            event_starts,
            event_ends,
            start_buffer_sec,
            end_buffer_sec,
        )
    else:
        raw_baseline = None
        raw_rast_array = rast_array

    # Set up figure and axes
    if ax is None:
        if raw_trace is not None:
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches([12, 4])
        else:
            fig, ax = plt.subplots(squeeze=False)
            ax = ax.reshape(-1)
            fig.set_size_inches([4, 4])
    else:
        if isinstance(ax, plt.Axes):
            fig = ax.figure
            ax = np.array(ax).reshape(-1)  # Make into array for compatibility below
        elif isinstance(ax, np.ndarray):
            fig = ax.reshape(-1)[0].figure

    if raw_trace is not None:
        ax[0].sharex(ax[1])

    # Plot rasters
    for raw_rast, rast in zip(raw_rast_array, rast_array):
        ax[0].plot(time_plot, rast, color=[0, 0, 1, 0.3])
        if raw_trace is not None:
            ax[1].plot(time_plot, raw_rast, color=[0, 0, 1, 0.3])
    good_frame_bool = np.bitwise_not(np.all(np.isnan(rast_array), axis=0))
    ax[0].plot(
        time_plot[good_frame_bool],
        np.nanmean(rast_array[:, good_frame_bool], axis=0),
        "k",
    )
    if raw_trace is not None:
        ax[1].plot(
            time_plot[good_frame_bool],
            np.nanmean(raw_rast_array[:, good_frame_bool], axis=0),
            "k",
        )

    for a, arr, bline in zip(
        ax.reshape(-1), (rast_array, raw_rast_array), (baseline, raw_baseline)
    ):
        a.plot(time_plot, arr.mean(axis=0), "k")
        a.axvspan(0, avg_event_sec, color=event_color)
        a.axhline(bline, color="g", linestyle="-")
        sns.despine(ax=a)
        a.set_xlabel("Time (s) from " + event_type + " start")
        a.set_title("Cell #" + str(cell_id))

    # Label axes
    for a, label in zip(ax, ylabels):
        a.set_ylabel(label)

    return fig, ax, rast_array, raw_rast_array, time_plot


if __name__ == "__main__":
    import session_directory as sd
    from neuropy.io.minianio import MinianIO
    from neuropy.analyses.trace_fc import load_events_from_csv

    # Specify session to plot here
    animal = "Finn"
    session = "Recall1"

    # Get session directory
    sesh_dir = sd.get_session_dir(animal, session)

    # Load in ca imaging data from minian
    minian = MinianIO(basedir=sesh_dir)

    # Load in event data
    event_df = load_events_from_csv(
        sesh_dir / "1_tone_recall" / "tone_recall01_21_2022-12_37_59.csv"
    )
    event_starts = event_df[
        event_df["Event"].str.contains("CS") & event_df["Event"].str.contains("start")
    ]["Timestamp"]
    event_ends = event_df[
        event_df["Event"].str.contains("CS") & event_df["Event"].str.contains("end")
    ]["Timestamp"]

    _, ax, _, _, _ = plot_pe_traces(
        minian.times["Timestamps"],
        minian.S[45],
        event_starts,
        event_ends=event_ends,
        raw_trace=minian.C[45],
        end_buffer_sec=40,
    )
