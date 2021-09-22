import matplotlib.pyplot as plt
from .. import core


def _none_axis(ax):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    return ax


def plot_position(position: core.Position, ax=None):

    ax = _none_axis(ax)
    ndim = position.ndim
    if ndim == 1:
        plt.plot(position.time, position.x)
    if ndim == 2:
        ax.plot(position.x, position.y)
    if ndim == 3:
        ax = plt.axes(projection="3d")
        ax.plot3D(position.x, position.y, position.z)
    return ax


def plot_run_epochs():
    pass