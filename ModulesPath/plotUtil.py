import numpy as np
from dataclasses import dataclass
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from cycler import cycler
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


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
        if style == "figPublish":
            mpl.rcParams["axes.linewidth"] = 2
            mpl.rcParams["axes.labelsize"] = 8
            mpl.rcParams["axes.titlesize"] = 8
            mpl.rcParams["xtick.labelsize"] = 8
            mpl.rcParams["ytick.labelsize"] = 8
            mpl.rcParams["axes.spines.top"] = False
            mpl.rcParams["axes.spines.right"] = False
            mpl.rcParams["axes.prop_cycle"] = cycler(
                "color",
                [
                    "#5cc0eb",
                    "#faa49d",
                    "#05d69e",
                    "#253237",
                    "#ef6e4e",
                    "#f0a8e6",
                    "#aaa8f0",
                    "#f0a8af",
                    "#dfe36b",
                    "#825265",
                    "#e8594f",
                ],
            )

        if style == "Pres":
            mpl.rcParams["axes.labelsize"] = 10

        fig = plt.figure(num=num, figsize=(8.5, 11), clear=True)
        fig.set_size_inches(size[0], size[1])
        gs = gridspec.GridSpec(grid[0], grid[1], figure=fig)
        fig.subplots_adjust(**kwargs)

        self.fig = fig
        return self.fig, gs

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

    def savefig(self, fname, scriptname=None, fig=None, folder=None):

        if folder is None:
            folder = "/home/bapung/Documents/figures/"
        if fig is None:
            fig = self.fig

        filename = folder + fname + ".pdf"

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


# class ScrollPlot:
#     """
#     Plot stuff then scroll through it! A bit hacked together as of 2/28/2020. Better would be to input a figure and axes
#     along with the appropriate plotting functions?
#     Created on Thu Jan 18 10:53:29 2018
#     @author: William Mau, modified by Nat Kinsky
#     :param
#         plot_func: tuple of plotting functions to plot into the appropriate axes (default) or figure (have to
#         specify config='figure').
#         x: X axis data.
#         y: Y axis data.
#         xlabel = 'x': X axis label.
#         ylabel = 'y': Y axis label.
#         combine_rows = list of subplots rows to combine into one subplot. Currently only supports doing all bottom
#         rows which must match the functions specified in plot_func
#     """

#     # Initialize the class. Gather the data and labels.
#     def __init__(
#         self,
#         plot_func,
#         xlabel="",
#         ylabel="",
#         titles=([" "] * 10000),
#         n_rows=1,
#         n_cols=1,
#         figsize=(8, 6),
#         combine_rows=[],
#         config="axes",
#         **kwargs,
#     ):

#         self.plot_func = plot_func
#         self.xlabel = xlabel
#         self.ylabel = ylabel
#         self.titles = titles
#         self.n_rows = n_rows  # NK can make default = len(plot_func)
#         self.n_cols = n_cols
#         self.share_y = False
#         self.share_x = False
#         self.figsize = figsize
#         self.config = config  # options are 'axes' or 'figure'

#         # Dump all arguments into ScrollPlot.
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#         if config == "axes":
#             self.fig, self.ax, = plt.subplots(
#                 self.n_rows,
#                 self.n_cols,
#                 sharey=self.share_y,
#                 sharex=self.share_x,
#                 figsize=self.figsize,
#             )
#             if n_cols == 1 and n_rows == 1:
#                 self.ax = (self.ax,)

#             # Make rows into one subplot if specified
#             if len(combine_rows) > 0:
#                 for row in combine_rows:
#                     plt.subplot2grid(
#                         (self.n_rows, self.n_cols),
#                         (row, 0),
#                         colspan=self.n_cols,
#                         fig=self.fig,
#                     )
#                 self.ax = self.fig.get_axes()

#             # Flatten into 1d array if necessary and not done already via combining rows
#             if n_cols > 1 and n_rows > 1 and hasattr(self.ax, "flat"):
#                 self.ax = self.ax.flat

#             # Necessary for scrolling.
#             if not hasattr(self, "current_position"):
#                 self.current_position = 0

#             # Plot the first plot of each function and label
#             for ax_ind, plot_f in enumerate(self.plot_func):
#                 plot_f(self, ax_ind)
#                 self.apply_labels()
#                 # print(str(ax_ind))

#             # Connect the figure to keyboard arrow keys.
#             self.fig.canvas.mpl_connect(
#                 "key_press_event", lambda event: self.update_plots(event)
#             )
#         elif config == "figure":
#             print("not yet configured")

#     # Go up or down the list. Left = down, right = up.
#     def scroll(self, event):
#         if event.key == "right" and self.current_position <= self.last_position:
#             if self.current_position <= self.last_position:
#                 if self.current_position == self.last_position:
#                     self.current_position = 0
#                 else:
#                     self.current_position += 1
#         elif event.key == "left" and self.current_position >= 0:
#             if self.current_position == 0:
#                 self.current_position = self.last_position
#             else:
#                 self.current_position -= 1
#         elif event.key == "6":
#             if (self.current_position + 15) < self.last_position:
#                 self.current_position += 15
#             elif (self.current_position + 15) >= self.last_position:
#                 if self.current_position == self.last_position:
#                     self.current_position = 0
#                 else:
#                     self.current_position = self.last_position
#         elif event.key == "4":
#             print("current position before = " + str(self.current_position))
#             if self.current_position > 15:
#                 self.current_position -= 15
#             elif self.current_position <= 15:
#                 if self.current_position == 0:
#                     self.current_position = self.last_position
#                 else:
#                     self.current_position = 0
#             print("current position after = " + str(self.current_position))
#         elif event.key == "9" and (self.current_position + 100) < self.last_position:
#             self.current_position += 100
#         elif event.key == "7" and self.current_position > 100:
#             self.current_position -= 100

#             # Apply axis labels.

#     def apply_labels(self):
#         plt.xlabel(self.xlabel)
#         plt.ylabel(self.ylabel)
#         plt.title(self.titles[self.current_position])

#     # Update the plot based on keyboard inputs.
#     def update_plots(self, event):
#         # Clear axis.
#         try:
#             for ax in self.ax:
#                 ax.cla()
#                 # print('Cleared axes!')
#         except:
#             self.ax.cla()

#         # Scroll then update plot.
#         self.scroll(event)

#         # Run the plotting function.
#         for ax_ind, plot_f in enumerate(self.plot_func):
#             plot_f(self, ax_ind)
#             # self.apply_labels()

#         # Draw.
#         self.fig.canvas.draw()

#         if event.key == "escape":
#             plt.close(self.fig)

#     def update_fig(self, event, **kwargs):
#         self.fig.clf()
#         self.scroll(event)
#         self.plot_func(fig=self.fig, **kwargs)
#         self.fig.canvas.draw()
#         if event.key == "escape":
#             plt.close(self.fig)


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
