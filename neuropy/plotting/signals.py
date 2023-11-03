import matplotlib.pyplot as plt
import ipywidgets
import numpy as np
import matplotlib as mpl
from ..core import Signal
from scipy.ndimage import gaussian_filter1d, gaussian_filter


def plot_spectrogram(
    sxx,
    time_lims,
    freq_lims=(0, 30),
    ax=None,
    cmap="jet",
    sigma=None,
    std_sxx=None,
    widget=True,
):
    """Generating spectrogram plot for given channel

    Parameters
    ----------
    chan : [int], optional
        channel to plot, by default None and chooses a channel randomly
    period : [type], optional
        plot only for this duration in the session, by default None
    window : [float, seconds], optional
        window binning size for spectrogram, by default 10
    overlap : [float, seconds], optional
        overlap between windows, by default 2
    ax : [obj], optional
        if none generates a new figure, by default None
    widget: [bool], optional
        enable use of sliders for adjust clim and zooming in on a frequency range
    """

    # Figure out if using legacy functionality or updated
    # assert isinstance(sxx, (np.ndarray, Spectrogram, WaveletSg, FourierSg))  # this is buggy and doesn't work, omit for now
    if isinstance(sxx, np.ndarray):
        legacy = True
        spec_use = None
    else:
        legacy = False
        spec = sxx
        sxx = spec.traces

    if sigma is not None:
        sxx = gaussian_filter(sxx, sigma=sigma)
    if std_sxx is None:  # Calculate standard deviation if needed for plotting purposes.
        std_sxx = np.std(sxx)
    # time = np.linspace(time[0], time[1], sxx.shape[1])
    # freq = np.linspace(freq[0], freq[1], sxx.shape[0])

    if ax is None:
        _, ax = plt.subplots(1, 1)

    # ---------- plotting ----------------
    if legacy:

        def plotspec(n_std, freq_lim):
            """Plots data fine but doesn't preserve time and frequency info on axes"""
            ax.imshow(
                sxx,
                cmap=cmap,
                vmax=n_std * std_sxx,
                rasterized=True,
                origin="lower",
                extent=[time_lims[0], time_lims[-1], freq_lims[0], freq_lims[-1]],
                aspect="auto",
            )
            ax.set_ylim(freq_lim[0], freq_lim[1])

        ax.set_xlim([time_lims[0], time_lims[-1]])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        # ---- updating plotting values for interaction ------------
        if widget:
            ipywidgets.interact(
                plotspec,
                n_std=ipywidgets.FloatSlider(
                    value=6,
                    min=0.1,
                    max=30,
                    step=0.1,
                    description="Clim :",
                ),
                freq_lim=ipywidgets.IntRangeSlider(
                    value=freq_lims, min=0, max=625, step=1, description="Freq. range:"
                ),
            )
        else:
            plotspec(6, freq_lims)
    else:

        def plotspec(n_std, freq):
            """Plots data from Spectrogram class and preserves time and frequency info on axes"""
            spec_use = spec.time_slice(t_start=time_lims[0], t_stop=time_lims[1])
            ax.pcolorfast(
                spec_use.time,
                spec_use.freqs,
                spec_use.traces,
                cmap=cmap,
                vmax=n_std * std_sxx,
                rasterized=True,
            )
            ax.set_ylim(freq)

        ax.set_xlim([time_lims[0], time_lims[-1]])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        # ---- updating plotting values for interaction ------------
        if widget:
            ipywidgets.interact(
                plotspec,
                n_std=ipywidgets.FloatSlider(
                    value=6,
                    min=0.1,
                    max=30,
                    step=0.1,
                    description="Clim :",
                ),
                freq=ipywidgets.IntRangeSlider(
                    value=freq_lims, min=0, max=625, step=1, description="Freq. range:"
                ),
            )
        else:
            plotspec(6, freq_lims)

    return ax


def plot_signal_traces(
    signal: Signal,
    ax=None,
    pad=0.2,
    color="k",
    lw=1,
    axlabel=False,
    epochs=None,
):

    n_channels = signal.n_channels
    sig = signal.traces
    sig = sig / np.max(sig)  # scaling
    sig = sig - sig[:, 0][:, np.newaxis]  # np.min(sig, axis=1, keepdims=True)
    pad_vals = np.linspace(0, len(sig) * pad, len(sig))[::-1]
    sig = sig + pad_vals[:, np.newaxis]

    if ax is None:
        _, ax = plt.subplots(1, 1)

    try:
        cmap = mpl.cm.get_cmap(color)
        colors = [cmap(_ / n_channels) for _ in range(n_channels)]
    except:
        colors = [color] * n_channels

    for i, trace in enumerate(sig):
        ax.plot(signal.time, trace, color=colors[i], lw=lw)

    channel_id = (
        [signal.channel_id] if isinstance(signal.channel_id, int) else signal.channel_id
    )
    ax.set_yticks(pad_vals)
    ax.set_yticklabels(channel_id)
    if not axlabel:
        ax.set_xticklabels([])
        ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="both", length=0)

    if epochs is not None:
        epochs_plot = epochs.time_slice(t_start=signal.t_start, t_stop=signal.t_stop)
        for start, stop in zip(epochs_plot.starts, epochs_plot.stops):
            ax.axvspan(start, stop, color=[0, 0.3, 0, 0.5])

    return ax


def plot_signal_heatmap(signal: Signal, ax=None, **kwargs):

    if ax is None:
        _, ax = plt.subplots()

    ax.pcolormesh(
        signal.time, signal.channel_id, signal.traces, shading="gouraud", **kwargs
    )


def plot_signal_w_epochs(signal, channel, epochs, ax=None):
    """Plot a trace from a single electrode with epochs overlying it for quick sanity check"""

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(signal.time, signal.traces[channel])

    for start, stop in zip(epochs.starts, epochs.stops):
        ax.axvspan(start, stop, color=[0, 0.3, 0, 0.5])

    return ax
