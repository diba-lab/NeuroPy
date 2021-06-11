import matplotlib.pyplot as plt
import ipywidgets
import numpy as np


def plot_spectrogram(sxx, time, freq, ax=None):
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

    std_sxx = np.std(sxx)
    time = np.linspace(time[0], time[1], sxx.shape[1])
    freq = np.linspace(freq[0], freq[1], sxx.shape[0])

    if ax is None:
        _, ax = plt.subplots(1, 1)

    # ---------- plotting ----------------
    def plotspec(n_std, cmap, freq):
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
            extent=[time[0], time[1], freq[0], freq[1]],
            aspect="auto",
        )
        ax.set_ylim(freq)

    ax.set_xlim(time)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

    # ---- updating plotting values for interaction ------------
    ipywidgets.interact(
        plotspec,
        n_std=ipywidgets.FloatSlider(
            value=2,
            min=0.1,
            max=30,
            step=0.1,
            description="Clim :",
        ),
        cmap=ipywidgets.Dropdown(
            options=["Spectral_r", "copper", "hot_r"],
            value="Spectral_r",
            description="Colormap:",
        ),
        freq=ipywidgets.IntRangeSlider(
            value=[0, 30], min=0, max=625, step=1, description="Freq. range:"
        ),
    )
    return ax