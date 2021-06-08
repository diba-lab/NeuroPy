import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_epochs(ax, epochs, ymin=0.5, ymax=0.55, color="gray"):

    delta = 0
    for epoch in epochs.itertuples():
        ax.axvspan(epoch.start, epoch.stop, ymin + delta, ymax + delta)
        ax.text(epoch.start + epoch.duration / 2, (ymax - ymin) / 2, epoch.label)
        delta = delta + 0.01

    return ax
