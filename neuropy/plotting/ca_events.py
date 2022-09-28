import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

## NRK todo: break up this into sub-functions, 1 calculates raster, the other plots.
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

    # Now loop through and chop out peri-event activity
    start_id, end_id = [], []
    start_buffer_list, end_buffer_list = [], []
    delta = []
    raster, raw_raster, time_list = [], [], []
    for start, end in zip(event_starts, event_ends):
        # first, id neural data time for start, end and account for buffer times before/after
        start_id.append((times - start).dt.total_seconds().abs().argmin())
        start_buffer = (
            (times - (start - pd.Timedelta(start_buffer_sec, unit="sec")))
            .dt.total_seconds()
            .abs()
            .argmin()
        )
        end_id.append((times - end).dt.total_seconds().abs().argmin())
        end_buffer = (
            (times - (end + pd.Timedelta(end_buffer_sec, unit="sec")))
            .dt.total_seconds()
            .abs()
            .argmin()
        )

        # Get time from event start for this event and keep only frames within buffer second
        # This is necessary to avoid grabbing frames super far away across a disconnect
        trial_dt = (times[start_buffer : end_buffer + 1] - start).dt.total_seconds()
        good_frame_bool = np.bitwise_and(
            trial_dt > -start_buffer_sec, trial_dt < end_buffer_sec
        )
        if start_buffer == end_buffer:  # Skip adding anything if no data for that event
            pass
        else:
            # Build up raster(s) of activity around event times
            # raster.append(activity[start_buffer : end_buffer + 1])
            raster.append(activity[start_buffer : end_buffer + 1][good_frame_bool])
            if raw_trace is not None:
                # raw_raster.append(raw_trace[start_buffer : end_buffer + 1])
                raw_raster.append(
                    raw_trace[start_buffer : end_buffer + 1][good_frame_bool]
                )
            delta.append(end - start)  # get duration of each event
            # time_list.append(
            #     (times[start_buffer : end_buffer + 1] - start).dt.total_seconds()
            # )
            time_list.append(trial_dt[good_frame_bool])

        # start_buffer_list.append(start_buffer)
        # end_buffer_list.append(end_buffer)

    if (
        raw_trace is None
    ):  # Set raw raster equal to deconvolved raster to make code below work.
        raw_raster = raster

    ## Set up times for plot
    # First infer sampling rate
    dt_good_bool = (
        times.diff().dt.total_seconds() < 0.2
    )  # Assume any frames more than 0.2 sec apart are due to a disconnect
    sr = 1 / times.diff().dt.total_seconds()[dt_good_bool].mean()
    # Next get event duration
    avg_event_dur = np.mean([d.total_seconds() for d in delta])
    dur_sec = start_buffer_sec + end_buffer_sec + avg_event_dur
    # last get times for each bin in the raster array relative to event start
    time_plot = np.linspace(
        -start_buffer_sec,
        -start_buffer_sec + dur_sec,
        np.floor(dur_sec * sr).astype(int),
    )
    # Now build up arrays!
    nevents = len(raster)  # Only keep events with calcium activity during them
    nframes = len(time_plot)
    rast_array = np.ones((nevents, nframes)) * np.nan  # pre-allocate
    raw_rast_array = np.ones((nevents, nframes)) * np.nan  # pre-allocate
    for idt, (time, activity, raw_activity) in enumerate(
        zip(time_list, raster, raw_raster)
    ):
        bins = np.digitize(time, time_plot, right=True)
        rast_array[idt][bins] = activity
        raw_rast_array[idt][bins] = raw_activity

    # # Now chop off the last timestamps(s) in the event you end up with different length rasters due to
    # # timestamp inaccuracies
    # min_raster_length = min(
    #     [len(a) for a in raster]
    # )  # Get the shortest raster you generated - buggy!!! doesn't work for missing data!
    # raster_aligned = [trace[:min_raster_length] for trace in raster]
    # rast_array = np.asarray(raster_aligned)
    # if raw_trace is not None:
    #     raw_raster_aligned = [raw_trace[:min_raster_length] for raw_trace in raw_raster]
    #     raw_rast_array = np.asarray(raw_raster_aligned)
    # else:  # Copy over raw activity raster if not specified
    #     raw_rast_array = rast_array

    # Set up times for plot
    # avg_event_dur = np.mean([d.total_seconds() for d in delta])
    # dur_sec = start_buffer_sec + end_buffer_sec + avg_event_dur
    # t_plot = np.linspace(
    #     -start_buffer_sec, -start_buffer_sec + dur_sec, rast_array.shape[1]
    # )

    # Set up figure and axes
    if ax is None:
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches([12, 4])

    # Plot rasters
    for raw_rast, rast in zip(raw_rast_array, rast_array):
        ax[0].plot(time_plot, rast, color=[0, 0, 1, 0.3])
        ax[1].plot(time_plot, raw_rast, color=[0, 0, 1, 0.3])
    ax[0].plot(time_plot, np.nanmean(rast_array, axis=0), "k")
    ax[1].plot(time_plot, np.nanmean(raw_rast_array, axis=0), "k")

    for a, arr in zip(ax.reshape(-1), (rast_array, raw_rast_array)):
        a.plot(time_plot, arr.mean(axis=0), "k")
        a.axvspan(0, avg_event_dur, color=event_color)
        sns.despine(ax=a)
        a.set_xlabel("Time from " + event_type + " start")
        a.set_title("Cell #" + str(cell_id))

    # Label axes
    for a, label in zip(ax, ylabels):
        a.set_ylabel(label)

    return fig, ax, rast_array, raw_rast_array, time_plot


if __name__ == "__main__":
    from pickle import load

    basepath = Path("/data/Working/Trace_FC/Recording_Rats/Finn/2022_01_20_training")

    file_names = ["ms_times_all_debug.pkl", "minian_debug.pkl", "swr_epochs_debug.pkl"]
    debug_vars = []
    for file in file_names:
        with open(basepath / file, "rb") as f:
            debug_vars.append(load(f))
    times_all, minian, swr_epochs = debug_vars
    cell = 27
    plot_pe_traces(
        times_all["Timestamps"],
        minian["S"],
        swr_epochs["Start Timestamp"],
        cell_id=27,
        raw_trace=minian["C"],
        start_buffer_sec=5,
        end_buffer_sec=5,
        event_type="CS start",
        ylabels=["S (au)", "C (DF/F)"],
    )
