from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


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

    def draw(self, num=None, grid=(2, 2), size=(8.5, 11), style="figPublish", **kwargs):

        # --- plot settings --------
        mpl_rcParams_style_dict = self.get_mpl_style(style=style)
        mpl.rcParams.update(mpl_rcParams_style_dict)

        fig = plt.figure(num=num, figsize=(8.5, 11), clear=True)
        fig.set_size_inches(size[0], size[1])
        gs = gridspec.GridSpec(grid[0], grid[1], figure=fig)
        fig.subplots_adjust(**kwargs)

        self.fig = fig
        return self.fig, gs

    def add_subplot(self, subplot_spec):
        return plt.subplot(subplot_spec)

    def subplot2grid(self, subplot_spec, grid=(1, 3), **kwargs):
        """Subplots within a subplot

        Parameters
        ----------
        subplot_spec : gridspec of figure
            subplot inside which subplots are created
        grid : tuple, optional
            number of rows and columns for subplots, by default (1, 3)

        Returns
        -------
        gridspec
        """
        gs = gridspec.GridSpecFromSubplotSpec(
            grid[0], grid[1], subplot_spec=subplot_spec, **kwargs
        )
        return gs

    def panel_label(self, ax, label, fontsize=12):
        ax.text(
            x=-0.08,
            y=1.15,
            s=label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            va="top",
            ha="right",
        )

    def savefig(self, fname: Path, scriptname=None, fig=None):

        if fig is None:
            fig = self.fig

        filename = fname.with_suffix(".pdf")

        today = date.today().strftime("%m/%d/%y")

        if scriptname is not None:
            scriptname = Path(scriptname).name
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

    @staticmethod
    def pf_1D(ax):
        ax.spines["left"].set_visible(False)
        ax.tick_params("y", length=0)

    @staticmethod
    def remove_spines(ax, sides=("top", "right")):

        for side in sides:
            ax.spines[side].set_visible(False)

    @staticmethod
    def set_spines_width(ax, lw=2, sides=("bottom", "left")):
        for side in sides:
            ax.spines[side].set_linewidth(lw)


    @classmethod
    def get_mpl_style(cls, style:str="figPublish"):
        """ Gets the matplotlib rcParams for various formatted styles
        Usage:
            from neuropy.plotting.figure import Fig
            mpl_rcParams_style_dict = Fig.get_mpl_style(style='figPublish')
            mpl.rcParams.update(mpl_rcParams_style_dict)

        with mpl.rc_context(Fig.get_mpl_style(style='figPublish')):
            plt.plot(data)

        @mpl.rc_context(Fig.get_mpl_style(style='figPublish'))
        def plotting_function():
            plt.plot(data)

        """
        if style == "figPublish":
            return {'axes.linewidth': 2,
                'axes.labelsize': 8,
                'axes.titlesize': 8,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.prop_cycle': cycler("color", [ "#5cc0eb", "#faa49d", "#05d69e", "#253237", "#ef6e4e", "#f0a8e6", "#aaa8f0", "#f0a8af", "#dfe36b", "#825265", "#e8594f", ], )
                }
        elif style == "Pres":
            return {'axes.linewidth': 3,
                'axes.labelsize': 10,
                'axes.titlesize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'axes.spines.right': False,
                'axes.spines.top': False
                }
        else:
            raise NotImplementedError


def pretty_plot(ax, round_ylim=False):
    """Generic function to make plot pretty, bare bones for now, will need updating
    :param round_ylim set to True plots on ticks/labels at 0 and max, rounded to the nearest decimal. default = False
    """

    # set ylims to min/max, rounded to nearest 10
    if round_ylim == True:
        ylims_round = np.round(ax.get_ylim(), decimals=-1)
        ax.set_yticks(ylims_round)
        ax.set_yticklabels([f"{lim:g}" for lim in iter(ylims_round)])

    # turn off top and right axis lines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax



def debug_print_matplotlib_figure_size(F):
    """ Prints the current figure size and DPI for a matplotlib figure F. 
    See https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib 
    Usage:
        SizeInches, DPI = debug_print_matplotlib_figure_size(a_fig)
    """
    DPI = F.get_dpi()
    print(f'DPI: {DPI}')
    SizeInches = F.get_size_inches()
    print(f'Default size in Inches: {SizeInches}')
    print('Which should result in a {} x {} Image'.format(DPI*SizeInches[0], DPI*SizeInches[1]))
    return SizeInches, DPI

def rescale_figure_size(F, scale_multiplier=2.0, debug_print=False):
    """ Scales up the Matplotlib Figure by a factor of scale_multiplier (in both width and height) without distorting the fonts or line sizes. 
    Usage:
        rescale_figure_size(a_fig, scale_multiplier=2.0, debug_print=True)
    """
    CurrentSize = F.get_size_inches()
    F.set_size_inches((CurrentSize[0]*scale_multiplier, CurrentSize[1]*scale_multiplier))
    if debug_print:
        RescaledSize = F.get_size_inches()
        print(f'Size in Inches: {RescaledSize}')
    return F


def compute_figure_size_pixels(figure_size_inches):
    # px_to_inches = 1/plt.rcParams['figure.dpi']  # pixel in inches
    inches_to_px = plt.rcParams['figure.dpi']  # pixel in inches
    return (figure_size_inches[0]*inches_to_px, figure_size_inches[1]*inches_to_px)


def compute_figure_size_inches(figure_size_pixels):
    """ inverse of compute_figure_size_pixels """
    inches_to_px = float(plt.rcParams['figure.dpi'])  # pixel in inches
    return (np.round(float(figure_size_pixels[0])/inches_to_px), np.round(float(figure_size_pixels[1])/inches_to_px))

def neuron_number_title(neurons):
    titles = ["Neuron: " + str(n) for n in neurons]

    return titles


def make_boxes(
    ax, xdata, ydata, xerror, yerror, facecolor="r", edgecolor="None", alpha=0.5
):

    # Loop over data points; create box from errors at each point
    errorboxes = [
        Rectangle((x, y), xe, ye) for x, y, xe, ye in zip(xdata, ydata, xerror, yerror)
    ]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(
        errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor
    )

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    # artists = ax.errorbar(
    #     xdata, ydata, xerr=xerror, yerr=yerror, fmt="None", ecolor="k"
    # )
    return 1
