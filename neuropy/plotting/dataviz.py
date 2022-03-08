import pandas as pd
import numpy as np

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
