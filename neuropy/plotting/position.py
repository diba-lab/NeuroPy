import matplotlib.pyplot as plt
from neuropy import core
from neuropy.plotting.figure import Fig


def _none_axis(ax):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    return ax


def plot_position(position: core.Position, ax=None):

    ndim = position.ndim
    posdata = position.traces
    time = position.time
    dim_names = ["x", "y", "z"]

    figure = Fig()
    fig, gs = figure.draw(grid=(2, ndim), size=(8.5, 5))

    for i, pos in enumerate(posdata):
        ax = plt.subplot(gs[0, i])
        ax.plot(time, pos, "gray", lw=1)
        ax.set_xlabel("Time")
        ax.set_ylabel(dim_names[i])

    if ndim > 1:
        ax = plt.subplot(gs[1, 0])
        ax.plot(posdata[0], posdata[1], "gray", lw=1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(f"{ndim}D position data")
    return ax


def plot_run_epochs():
    pass
