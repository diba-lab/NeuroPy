import matplotlib.pyplot as plt


def _none_axis(ax):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    return ax


def plot_position(x, y, ax):

    ax = _none_axis(ax)
    plt.plot(x, y)
    return ax


def plot_run_epochs():
    pass