import matplotlib.pyplot as plt
import ipywidgets
import numpy as np
import matplotlib as mpl
from ..core import Signal
from scipy.ndimage import gaussian_filter1d, gaussian_filter


def plot_spectrogram(
    sxx, time, freq, freq_lim=(0, 30), ax=None, cmap="jet", sigma=None
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
    """

    if sigma is not None:
        sxx = gaussian_filter(sxx, sigma=sigma)
    std_sxx = np.std(sxx)
    # time = np.linspace(time[0], time[1], sxx.shape[1])
    # freq = np.linspace(freq[0], freq[1], sxx.shape[0])

    if ax is None:
        _, ax = plt.subplots(1, 1)

    # ---------- plotting ----------------
    def plotspec(n_std, freq_lim):
        # slow to plot
        # ax.pcolormesh(
        #     spec.time,
        #     spec.freq,
        #     sxx,
        #     cmap=cmap,
        #     vmax=n_std * std_sxx,
        #     rasterized=True,
        # )
        # ax.set_ylim(freq)

        # fast to plot
        ax.imshow(
            sxx,
            cmap=cmap,
            vmax=n_std * std_sxx,
            rasterized=True,
            origin="lower",
            extent=[time[0], time[-1], freq[0], freq[-1]],
            aspect="auto",
        )
        ax.set_ylim(freq_lim[0], freq_lim[1])

    ax.set_xlim([time[0], time[-1]])
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

    # ---- updating plotting values for interaction ------------
    ipywidgets.interact(
        plotspec,
        n_std=ipywidgets.FloatSlider(
            value=20,
            min=0.1,
            max=30,
            step=0.1,
            description="Clim :",
        ),
        # cmap=ipywidgets.Dropdown(
        #     options=["Spectral_r", "copper", "hot_r"],
        #     value=cmap,
        #     description="Colormap:",
        # ),
        freq_lim=ipywidgets.IntRangeSlider(
            value=freq_lim, min=0, max=625, step=1, description="Freq. range:"
        ),
    )
    return ax


def plot_signal_traces(signal: Signal, ax=None, pad=0.2, color="k", lw=1):

    n_channels = signal.n_channels
    sig = signal.traces
    sig = sig / np.max(sig)  # scaling
    sig = sig - sig[:, 0][:, np.newaxis]  # np.min(sig, axis=1, keepdims=True)
    pad_vals = np.linspace(0, len(sig) * pad, len(sig))[::-1]
    sig = sig + pad_vals[:, np.newaxis]

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.clear()

    try:
        cmap = mpl.cm.get_cmap(color)
        colors = [cmap(_ / n_channels) for _ in range(n_channels)]
    except:
        colors = [color] * n_channels

    for i, trace in enumerate(sig):
        ax.plot(signal.time, trace, color=colors[i], lw=lw)

    ax.set_yticks(pad_vals)
    ax.set_yticklabels(signal.channel_id)
    ax.set_xticklabels([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)

    return ax


def plot_signal_heatmap(signal: Signal, ax=None, **kwargs):

    if ax is None:
        _, ax = plt.subplots()

    ax.pcolormesh(
        signal.time, signal.channel_id, signal.traces, shading="gouraud", **kwargs
    )
