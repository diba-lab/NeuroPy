import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from ..core import Epoch, Signal
from scipy import stats


def plot_epochs(
    epochs: Epoch, labels_order=None, colors="Set3", alpha=1, collapsed=False, colorby="label", yaxis_label=True,
        ax=None
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
    color : str or dict, optional
        [description], by default "gray", if dict = {"value1": color, "value2": color2}
        where value1, value2, ... are values in the column defined by colorby param
    colorby: str, column in epochs to map colors to
    yaxis_label: bool, True (default) = add epoch labels to y axis
    collapsed:

    Returns
    -------
    [type]
        [description]
    """
    if isinstance(epochs, pd.DataFrame):
        epochs = Epoch(epochs)

    # assert isinstance(epochs, Epoch), "epochs must be neuropy.Epoch object"

    n_epochs = epochs.n_epochs

    if isinstance(colors, str):
        try:
            cmap = mpl.cm.get_cmap(colors)
            colors = [cmap(i / n_epochs) for i in range(n_epochs)]
        except:
            colors = [colors] * n_epochs
    elif isinstance(colors, dict):
        # Define colors, sending those not in the colors dict to white
        colors = [colors[label] if label in colors.keys() else "#ffffff" for label in epochs.to_dataframe()[colorby]]

    if epochs.has_labels or (len(epochs.to_dataframe().label) > 1):
        labels = epochs.labels
        # unique_labels = np.unique(epochs.labels)
        unique_labels = epochs.to_dataframe().label.unique()
        n_labels = len(unique_labels)

        # Update to order labels correctly
        if labels_order is not None:
            # assert np.array_equal(
            #     np.sort(labels_order), np.sort(unique_labels)
            # ), "labels_order does not match with epochs labels"

            # Make sure all labels are in labels_order

            # This code might be necessary, keep for potential debugging
            # if np.array_equal(np.sort(labels_order), np.sort(unique_labels)) or \
            #         np.all([label in labels_order for label in unique_labels]):
            if np.all([label in labels_order for label in unique_labels]):
                unique_labels = labels_order
                n_labels = len(unique_labels)
            else:
                assert False, "labels_order does not match with epochs labels"

        dh = 1 if collapsed else 1 / n_labels
        y_min = np.zeros(len(epochs))
        if not collapsed:
            for i, l in enumerate(unique_labels):
                y_min[labels == l] = i * dh
    else:
        dh = 1
        y_min = np.zeros(len(epochs))

    if ax is None:
        _, ax = plt.subplots()

    for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
        ax.axvspan(
            epoch.start,
            epoch.stop,
            y_min[i],
            y_min[  i] + dh,
            facecolor=colors[i],
            edgecolor=None,
            alpha=alpha,
        )
        ax.set_ylim([0, 1])

    # Label epochs on plot
    if not collapsed and yaxis_label:
        yticks = np.linspace(1 / n_labels / 2, 1 - 1 / n_labels / 2, n_labels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(unique_labels)

    return ax


def plot_hypnogram(epochs: Epoch, ax=None, unit="s", collapsed=False, annotate=False):
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

    if annotate:
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
                state.start / unit_norm,
                state.stop / unit_norm,
                ymin=span_[state.label][0],
                ymax=span_[state.label][1],
                facecolor=colors[state.label],
                # alpha=0.7,
            )
    ax.axis("off")
    ax.set_xlim([epochs.starts[0], epochs.stops[-1]])

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
