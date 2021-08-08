import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from ..core import Epoch, Signal
from scipy import stats


def plot_epochs(
    ax, epochs: Epoch, ymin=0.5, ymax=0.55, color="Set3", style="step_blocks"
):
    """Plots epochs on a given axis, with different style of plotting

    Parameters
    ----------
    ax : axis
        [description]
    epochs : [type]
        [description]
    ymin : float, optional
        [description], by default 0.5
    ymax : float, optional
        [description], by default 0.55
    color : str, optional
        [description], by default "gray"

    Returns
    -------
    [type]
        [description]
    """
    delta = 0
    n_epochs = epochs.n_epochs
    cmap = mpl.cm.get_cmap(color)

    for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
        ax.axvspan(
            epoch.start,
            epoch.stop,
            ymin + delta,
            ymax + delta,
            color=cmap(i / n_epochs),
            alpha=0.5,
        )
        # ax.text(
        #     epochs.stops[-1],
        #     ymax + delta,
        #     epoch.label,
        #     transform=ax.get_yaxis_transform(),
        # )
        delta = delta + 0.07

    return ax


def plot_hypnogram(epochs: Epoch, ax=None, tstart=0.0, unit="s", collapsed=False):
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
    colors = {
        "nrem": "#667cfa",
        "rem": "#eb9494",
        "quiet": "#b6afaf",
        "active": "#474343",
    }
    labels = ["nrem", "rem", "quiet", "active"]

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
            color=colors[state],
        )
    if collapsed:
        span_ = {
            "nrem": [0, 1],
            "rem": [0, 1],
            "quiet": [0, 1],
            "active": [0, 1],
        }

    for state in epochs.to_dataframe().itertuples():
        if state.label in colors.keys():
            ax.axvspan(
                (state.start - tstart) / unit_norm,
                (state.stop - tstart) / unit_norm,
                ymin=span_[state.label][0],
                ymax=span_[state.label][1],
                facecolor=colors[state.label],
                alpha=0.7,
            )
    ax.axis("off")

    return ax


def plot_epochs_with_raster(self, ax=None):
    spikes = Spikes(self._obj)
    total_dur = self._obj.getNframesEEG / self._obj.lfpSrate

    if ax is None:
        _, ax = plt.subplots(1, 1)

    def plot(tstart):
        ax.clear()
        events = self.events[(self.events > tstart) & (self.events < tstart + 10)]
        for epoch in events.itertuples():
            ax.axvspan(
                epoch.start,
                epoch.end,
                color="gray",
                alpha=0.4,
                edgecolor=None,
                zorder=1,
            )
        spikes.plot_raster(ax=ax, period=[tstart, tstart + 10])

    widgets.interact(
        plot,
        tstart=widgets.IntSlider(
            value=int(total_dur / 2),
            min=0,
            max=total_dur,
            step=1,
            description="Start Time (s):",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        ),
    )


def plot_artifact_epochs(epochs: Epoch, signal: Signal, downsample_factor: int = 5):
    """Plots artifact epochs against a signal

    Parameters
    ----------
    epochs : Epoch
        [description]
    signal : Signal
        [description]
    downsample_factor : int, optional
        It is much faster to plot downsampled signal, by default 5

    Returns
    -------
    [type]
        [description]
    """

    assert signal.n_channels == 1, "signal should only have one trace"

    threshold = epochs.metadata["threshold"]
    sig = signal.traces.reshape((-1))
    zsc = np.abs(stats.zscore(sig))

    zsc_downsampled = zsc[::downsample_factor]
    artifacts = np.vstack((epochs.starts, epochs.stops)).T
    t = np.linspace(signal.t_start, signal.t_stop, len(zsc_downsampled))

    _, ax = plt.subplots(1, 1)
    ax.axhline(threshold, color="#37474F", ls="--")
    for artifact in artifacts:
        ax.axvspan(
            artifact[0],
            artifact[1],
            facecolor="#ff928a",
            ec="#ff6257",
            alpha=0.7,
        )
    ax.plot(t, zsc_downsampled, "gray")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute zscore")

    return ax
