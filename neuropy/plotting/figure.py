from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import ListedColormap


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

    def dynamic4(self):
        white = 255 * np.ones(80).reshape(20, 4)
        white = white / 255
        jet = mpl.cm.get_cmap("jet")
        greys = mpl.cm.get_cmap("Greys")

        colmap = np.vstack(
            (
                ListedColormap(greys(np.linspace(0.5, 0.8, 12))).colors,
                ListedColormap(jet(np.linspace(0, 1, 30))).colors,
                ListedColormap(greys(np.linspace(0.5, 0.8, 12)[::-1])).colors,
            )
        )

        colmap = ListedColormap(colmap)

        return colmap


class Fig:
    def __init__(
        self,
        num=None,
        grid=(2, 2),
        size=(8.5, 11),
        fontsize=5,
        axis_color="#545454",
        axis_lw=1.2,
        tick_size=3.5,
        constrained_layout=True,
        fontname="Arial",
        **kwargs,
    ):

        # --- plot settings --------
        mpl.rcParams["font.family"] = fontname
        # mpl.rcParams["font.sans-serif"] = "Arial"
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["axes.linewidth"] = axis_lw
        mpl.rcParams["axes.labelsize"] = fontsize
        mpl.rcParams["axes.titlesize"] = fontsize
        mpl.rcParams["axes.edgecolor"] = axis_color
        mpl.rcParams["xtick.labelsize"] = fontsize
        mpl.rcParams["xtick.major.pad"] = 2
        mpl.rcParams["ytick.labelsize"] = fontsize
        mpl.rcParams["axes.spines.top"] = False
        mpl.rcParams["axes.spines.right"] = False
        mpl.rcParams["xtick.major.width"] = axis_lw
        mpl.rcParams["xtick.major.size"] = tick_size
        mpl.rcParams["ytick.major.size"] = tick_size
        mpl.rcParams["xtick.color"] = axis_color
        mpl.rcParams["xtick.labelcolor"] = "k"
        mpl.rcParams["ytick.major.width"] = axis_lw
        mpl.rcParams["ytick.color"] = axis_color
        mpl.rcParams["ytick.labelcolor"] = "k"
        mpl.rcParams["figure.constrained_layout.use"] = constrained_layout
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

        fig = plt.figure(num=num, figsize=(8.5, 11), clear=True)
        fig.set_size_inches(size[0], size[1])
        gs = gridspec.GridSpec(grid[0], grid[1], figure=fig, **kwargs)

        # fig.subplots_adjust(**kwargs)

        self.fig = fig
        self.gs = gs

    def subplot(self, subplot_spec, sharex=None, sharey=None, **kwargs):
        return plt.subplot(subplot_spec, sharex=sharex, sharey=sharey, **kwargs)

    def add_subfigure(self, *args, **kwargs) -> mpl.figure.SubFigure:
        return self.fig.add_subfigure(*args, **kwargs)

    def subplot2grid(
        self, subplot_spec, grid=(1, 3), return_axes: bool = False, **kwargs
    ):
        """Subplots within a subplot

        Parameters
        ----------
        subplot_spec : gridspec of figure
            subplot inside which subplots are created
        grid : tuple, optional
            number of rows and columns for subplots, by default (1, 3)
        return_axes: returns axes instead of gridspec

        Returns
        -------
        gridspec (or axes if specified)
        """
        gs = gridspec.GridSpecFromSubplotSpec(
            grid[0], grid[1], subplot_spec=subplot_spec, **kwargs
        )

        if not return_axes:
            return gs
        elif return_axes:
            ax = []
            for row in range(grid[0]):
                ax_col = []
                for col in range(grid[1]):
                    ax_col.append(self.fig.add_subplot(gs[row, col]))
                ax.append(ax_col)

            return np.array(ax).squeeze() if np.array(ax).ndim > 1 else np.array(ax)

    def panel_label(self, ax, label, fontsize=12, x=-0.14, y=1.15):
        ax.text(
            x=x,
            y=y,
            s=label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            va="top",
            ha="right",
        )

    def legend(self, ax, text, color, fontsize=8, x=0.65, y=0.9, dy=0.1):
        for i, (s, c) in enumerate(zip(text, color)):
            ax.text(
                x=x,
                y=y - i * dy,
                s=s,
                color=c,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight="bold",
                va="top",
                ha="left",
            )

    def savefig(self, fname: Path, scriptname=None, fig=None, caption=None, dpi=300):

        if fig is None:
            fig = self.fig

        # fig.set_dpi(300)
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

        fig.savefig(filename, dpi=dpi, backend="pdf")

        if caption is not None:
            fig_caption = Fig(grid=(1, 1))
            ax_caption = fig_caption.subplot(fig_caption.gs[0])
            ax_caption.text(0, 0.5, caption, wrap=True)
            ax_caption.axis("off")
            fig_caption.savefig(filename.with_suffix(".caption.pdf"))

            """ Previously caption was combined to create a multi-page pdf with main figure. But this created dpi issue where we can't increase dpi to only saved pdf (pdfpages does not have that functionality yet) without affecting the plot in matplotlib widget which becomes bigger because of dpi-pixels relationsip)
            """
            # with PdfPages(filename) as pdf:
            #     pdf.savefig(self.fig)

            #     fig_caption = Fig(grid=(1, 1))
            #     ax_caption = fig_caption.subplot(fig_caption.gs[0])

            #     ax_caption.text(0, 0.5, caption, wrap=True)
            #     ax_caption.axis("off")
            #     pdf.savefig(fig_caption.fig)

    @staticmethod
    def pf_1D(ax):
        ax.spines["left"].set_visible(False)
        ax.tick_params("y", length=0)

    @staticmethod
    def toggle_spines(ax, sides=("top", "right"), keep=False):

        for side in sides:
            ax.spines[side].set_visible(keep)

    @staticmethod
    def set_spines_width(ax, lw=2, sides=("bottom", "left")):

        for side in sides:
            ax.spines[side].set_linewidth(lw)

    @staticmethod
    def center_spines(ax):
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")


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


def neuron_number_title(neurons):
    titles = ["Neuron: " + str(n) for n in neurons]

    return titles
