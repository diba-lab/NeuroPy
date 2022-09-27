import matplotlib.pyplot as plt
import numpy as np


def match_max_lims(ax, axis: str in ["x", "y", "both"]):
    """Make all axes in the input match their limits"""
    xlims, ylims = [], []
    for a in ax.reshape(-1):
        xlims.append(a.get_xlim())
        ylims.append(a.get_ylim())
    xlims = np.asarray(xlims)
    ylims = np.asarray(ylims)
    xlim_use = [np.min(xlims.reshape(-1)), np.max(xlims.reshape(-1))]
    ylim_use = [np.min(ylims.reshape(-1)), np.max(ylims.reshape(-1))]

    if axis in ["x", "both"]:
        for a in ax.reshape(-1):
            a.set_xlim(xlim_use)

    if axis in ["y", "both"]:
        for a in ax.reshape(-1):
            a.set_ylim(ylim_use)
