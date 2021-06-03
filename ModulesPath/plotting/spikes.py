import matplotlib.pyplot as plt
import numpy as np


def plot_raster():
    pass


def plot_placefields():
    pass


def plot_spectrogram():
    pass


def plot_ccg(ccgs, bin_size, ax=None):

    if ax is None:
        fig, ax = plt.subplots(ccgs.shape[0], ccgs.shape[1])

    window_size = 2 * ccgs.shape[1]

    winsize_bins = 2 * int(0.5 * window_size / bin_size) + 1
    bins = np.linspace(0, 1, winsize_bins)
    for a, ccg in zip(ax.reshape(-1), ccgs.reshape(-1, ccgs.shape[2])):
        a.bar(bins, ccg, width=1 / (winsize_bins - 1))
        a.set_xticks([0, 1])
        a.set_xticklabels(np.ones((2,)) * np.round(window_size / 2, 2))
        a.set_xlabel("Time (s)")
        a.set_ylabel("Spike Count")
    return ax