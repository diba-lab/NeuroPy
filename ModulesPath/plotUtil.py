import numpy as np
from dataclasses import dataclass
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


class Colormap:
    def dynamicMap(self):
        white = 255 * np.ones(48).reshape(12, 4)
        white = white / 255
        red = mpl.cm.get_cmap("Reds")
        brown = mpl.cm.get_cmap("YlOrBr")
        green = mpl.cm.get_cmap("Greens")
        blue = mpl.cm.get_cmap("Blues")
        purple = mpl.cm.get_cmap("Purples")

        colmap = np.vstack(
            (
                white,
                ListedColormap(red(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(brown(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(green(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(blue(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(purple(np.linspace(0.2, 0.8, 16))).colors,
            )
        )

        colmap = ListedColormap(colmap)

        return colmap

    def dynamic2(self):
        white = 255 * np.ones(80).reshape(20, 4)
        white = white / 255
        red = mpl.cm.get_cmap("Reds")
        brown = mpl.cm.get_cmap("YlOrBr")
        green = mpl.cm.get_cmap("Greens")
        blue = mpl.cm.get_cmap("Blues")
        purple = mpl.cm.get_cmap("Purples")

        colmap = np.vstack(
            (
                white,
                ListedColormap(red(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(brown(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(green(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(blue(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(purple(np.linspace(0.2, 0.8, 16))).colors,
            )
        )

        colmap = ListedColormap(colmap)

        return colmap

    def dynamic3(self):
        white = 255 * np.ones(80).reshape(20, 4)
        white = white / 255
        YlOrRd = mpl.cm.get_cmap("YlOrRd")
        RdPu = mpl.cm.get_cmap("RdPu")
        blue = mpl.cm.get_cmap("Blues_r")

        colmap = np.vstack(
            (
                ListedColormap(blue(np.linspace(0, 0.8, 16))).colors,
                white,
                ListedColormap(YlOrRd(np.linspace(0.2, 0.5, 4))).colors,
                ListedColormap(YlOrRd(np.linspace(0.52, 0.7, 4))).colors,
                ListedColormap(YlOrRd(np.linspace(0.72, 0.85, 4))).colors,
                ListedColormap(RdPu(np.linspace(0.6, 1, 4))).colors,
            )
        )

        colmap = ListedColormap(colmap)

        return colmap


class Fig:
    labelsize = 8

    def draw(self, num=1, grid=(2, 2), size=(8.5, 11), style="figPublish", **kwargs):

        # --- plot settings --------
        if style == "figPublish":
            mpl.rcParams["axes.labelsize"] = 8
            mpl.rcParams["axes.titlesize"] = 8
            mpl.rcParams["xtick.labelsize"] = 8
            mpl.rcParams["ytick.labelsize"] = 8
        if style == "Pres":
            mpl.rcParams["axes.labelsize"] = 10

        fig = plt.figure(num=num, figsize=(8.5, 11), clear=True)
        fig.set_size_inches(size[0], size[1])
        gs = gridspec.GridSpec(grid[0], grid[1], figure=fig)
        fig.subplots_adjust(**kwargs)

        self.fig = fig
        return self.fig, gs

    def panel_label(self, ax, label):
        ax.text(
            x=-0.08,
            y=1.15,
            s=label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="right",
        )

    def savefig(self, fname, scriptname, fig=None, folder=None):

        scriptname = Path(scriptname).name

        if folder is None:
            folder = "/home/bapung/Documents/MATLAB/figures/"
        if fig is None:
            fig = self.fig

        filename = folder + fname + ".pdf"

        today = date.today().strftime("%m/%d/%y")

        fig.text(
            0.95,
            0.01,
            f"{scriptname}\n Date: {today}",
            fontsize=6,
            color="gray",
            ha="right",
            va="bottom",
            alpha=0.5,
        )
        fig.savefig(filename)
