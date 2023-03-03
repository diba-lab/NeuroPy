import matplotlib.pyplot as plt
import seaborn as sns
try:
    import modin.pandas as pd # modin is a drop-in replacement for pandas that uses multiple cores
except ImportError:
    import pandas as pd # fallback to pandas when modin isn't available
import numpy as np


def plot_pe_traces(
    times: pd.Series,
    activity: np.ndarray,
    event_starts: pd.Series,
    cell_id: int = None,
    event_ends: pd.Series = None,
    raw_trace: np.ndarray = None,
    start_buffer_sec: float or int = 10,
    end_buffer_sec: float or int = 10,
    ax: plt.axes = None,
    event_type: str = "",
    ylabels: list or tuple = ["Processed Activity", "Raw Activity"],
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

    if raw_trace.ndim == 2:
        raw_trace = raw_trace[cell_id]

    # Send end times to start times if not specified
    if event_ends is None:
        event_ends = event_starts

    # Now loop through and chop out peri-event activity
    start_id, end_id = [], []
    start_buffer_list, end_buffer_list = [], []
    delta = []
    raster, raw_raster = [], []
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

        # Build up raster(s) of activity around event times
        raster.append(activity[start_buffer : end_buffer + 1])
        if raw_trace is not None:
            raw_raster.append(raw_trace[start_buffer : end_buffer + 1])
        delta.append(end - start)  # get duration of each event

        # start_buffer_list.append(start_buffer)
        # end_buffer_list.append(end_buffer)

    # Now chop off the last timestamps(s) in the event you end up with different length rasters due to
    # timestamp inaccuracies
    min_raster_length = min(
        [len(a) for a in raster]
    )  # Get the shortest raster you generated
    raster_aligned = [trace[:min_raster_length] for trace in raster]
    rast_array = np.asarray(raster_aligned)
    if raw_trace is not None:
        raw_raster_aligned = [raw_trace[:min_raster_length] for raw_trace in raw_raster]
        raw_rast_array = np.asarray(raw_raster_aligned)
    else:  # Copy over raw activity raster if not specified
        raw_rast_array = rast_array

    # Set up times for plot
    avg_event_dur = np.mean([d.total_seconds() for d in delta])
    dur_sec = start_buffer_sec + end_buffer_sec + avg_event_dur
    t_plot = np.linspace(
        -start_buffer_sec, -start_buffer_sec + dur_sec, rast_array.shape[1]
    )

    # Set up figure and axes
    if ax is None:
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches([10, 3])

    # Plot rasters
    for raw_rast, rast in zip(raw_rast_array, rast_array):
        ax[0].plot(t_plot, rast, color=[0, 0, 1, 0.3])
        ax[1].plot(t_plot, raw_rast, color=[0, 0, 1, 0.3])

    for a, arr in zip(ax.reshape(-1), (rast_array, raw_rast_array)):
        a.plot(t_plot, arr.mean(axis=0), "k")
        a.axvspan(0, avg_event_dur, color=[0, 1, 0, 0.3])
        sns.despine(ax=a)
        a.set_xlabel("Time from " + event_type + " start")
        a.set_title("Cell #" + str(cell_id))

    # Label axes
    for a, label in zip(ax, ylabels):
        a.set_ylabel(label)

    return ax
