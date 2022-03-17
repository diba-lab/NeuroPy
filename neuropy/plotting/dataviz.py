import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

"""Hosts some frequently plotted data visualizations
"""


def lineplot(data: pd.DataFrame, x, y, hue, palette, ci="sem", ax=None):
    """Seaborn type lineplot but with sem errorbars

    Parameters
    ----------
    data : pd.DataFrame
        long form pandas dataframe
    x : str
        column specifying the x variable
    y : str
        column specifying the y variable
    hue : str
        grouping variable
    palette : colormap
        colormap for each nested group
    ci : str, optional
        standard error of mean (sem), mean abs deviation (mad), standard deviation (sd), by default "sem"
    ax : [type], optional
        ax to plot on, by default None
    """
    hues = np.unique(data[hue].values)
    for i, h in enumerate(hues):
        df = data[data[hue] == h]
        group = df.groupby(x)
        mean_y = group.mean()[y]
        x_val = mean_y.index
        err = group.sem(ddof=0)[y]
        ax.fill_between(
            x_val, mean_y - err, mean_y + err, color=palette[i], alpha=0.4, ec="none"
        )
        ax.plot(x_val, mean_y, color=palette[i])

    return ax


def boxplot(
    data: pd.DataFrame,
    x,
    y,
    hue=None,
    hue_palette=None,
    x_palette=None,
    ax=None,
    width=0.8,
):
    """Seaborn type boxplot but all lines associated with a corresponding boxplot will have same colors.

    Parameters
    ----------
    data : pd.DataFrame
        long form pandas dataframe
    x : str
        column specifying the x variable
    y : str
        column specifying the y variable
    hue : str
        grouping variable
    palette : colormap
        colormap for each nested group
    ci : str, optional
        standard error of mean (sem), mean abs deviation (mad), standard deviation (sd), by default "sem"
    ax : [type], optional
        ax to plot on, by default None
    """
    props = lambda c: dict(
        boxprops=dict(facecolor="none", color=c, lw=2),
        medianprops=dict(color=c, lw=2),
        whiskerprops=dict(color=c, lw=2),
        capprops=dict(color=c, lw=2),
        patch_artist=True,
        showcaps=True,
        showfliers=False,
    )

    if ax is None:
        _, ax = plt.subplots()
    x_cat = data[x].unique()

    if hue is not None:
        hue_levels = np.unique(data[hue])
        n_levels = len(hue_levels)
        each_width = width / n_levels
        offsets = np.linspace(0, width - each_width, n_levels)
        offsets -= offsets.mean()

        if (hue_palette is None) and (x_palette is not None):
            colors = np.concatenate(
                [[x_palette[i]] * n_levels for i in range(len(x_cat))]
            )

        i2 = 0
        for i, x_val in enumerate(x_cat):
            for i1, hue_level in enumerate(hue_levels):
                center = i + offsets[i1]
                box_data = data[(data[x] == x_val) & (data[hue] == hue_level)][y]
                ax.boxplot(
                    box_data,
                    positions=[center],
                    widths=each_width - 0.05,
                    **props(colors[i2]),
                )
                i2 += 1

        ax.set_xticks(np.arange(len(x_cat)), x_cat)
    else:

        for i, x_val in enumerate(x_cat):
            box_data = data[data[x] == x_val][y]
            ax.boxplot(box_data, positions=[i], widths=width, **props(x_palette[i]))
            ax.set_xticks(np.arange(len(x_cat)), x_cat)

    return ax


def pointplot(
    data: pd.DataFrame,
    x,
    y,
    hue,
    palette="r",
    ax=None,
    width=0.8,
    **kwargs,
):
    """Seaborn type boxplot but all lines associated with a corresponding boxplot will have same colors.

    Parameters
    ----------
    data : pd.DataFrame
        long form pandas dataframe
    x : str
        column specifying the x variable
    y : str
        column specifying the y variable
    hue : str
        grouping variable
    palette : colormap
        colormap for each nested group
    ci : str, optional
        standard error of mean (sem), mean abs deviation (mad), standard deviation (sd), by default "sem"
    ax : [type], optional
        ax to plot on, by default None
    """
    props = lambda c: dict(
        boxprops=dict(facecolor="none", color=c, lw=2),
        medianprops=dict(color=c, lw=2),
        whiskerprops=dict(color=c, lw=2),
        capprops=dict(color=c, lw=2),
        patch_artist=True,
        showcaps=True,
        showfliers=False,
    )

    if ax is None:
        _, ax = plt.subplots()
    x_cat = data[x].unique()

    if hue is not None:
        hue_levels = np.unique(data[hue])
        n_levels = len(hue_levels)
        each_width = width / n_levels
        offsets = np.linspace(0, width - each_width, n_levels)
        offsets -= offsets.mean()
        for i, x_val in enumerate(x_cat):
            for i1, hue_level in enumerate(hue_levels):
                center = i + offsets[i1]
                box_data = data[(data[x] == x_val) & (data[hue] == hue_level)][y]
                ax.boxplot(
                    box_data,
                    positions=[center],
                    widths=each_width - 0.05,
                    **props(palette[i1]),
                )
        ax.set_xticks(np.arange(len(x_cat)), x_cat)
    else:
        for i, x_val in enumerate(x_cat):
            box_data = data[data[x] == x_val][y]
            ax.boxplot(box_data, positions=[i], widths=width, **props(palette[i]))
            ax.set_xticks(np.arange(len(x_cat)), x_cat)

    return ax
