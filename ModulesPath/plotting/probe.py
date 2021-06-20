import matplotlib.pyplot as plt


def plot_probe(probe, ax=None):

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(probe.x, probe.y, ".")
    ax.plot(probe.get_disconnected.x, probe.get_disconnected.y, "r.")
    ax.set_title(f"Probe {probe.n_contacts}ch")

    return ax
