import matplotlib.pyplot as plt
from ..core import ProbeGroup


def plot_probe(
    probe: ProbeGroup,
    annotate_channels=None,
    channel_id=True,
    disconnected=True,
    ax=None,
):

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(
        probe.x,
        probe.y,
        s=12,
        marker="o",
        color="gray",
        zorder=1,
        linewidths=0.5,
        alpha=0.5,
    )
    if channel_id:
        for x, y, chan_id in zip(probe.x, probe.y, probe.channel_id):
            ax.annotate(chan_id, (x, y), fontsize=8)

    if disconnected:
        ax.scatter(
            probe.get_disconnected.x,
            probe.get_disconnected.y,
            s=18,
            edgecolors="#FF5252",
            facecolors="none",
        )

    if annotate_channels is not None:
        prb_data = probe.to_dataframe().set_index("channel_id")
        for channel in annotate_channels:
            x, y = (
                prb_data.loc[[channel]].x.values[0],
                prb_data.loc[[channel]].y.values[0],
            )
            ax.scatter(x, y, s=30, edgecolors="g", facecolors="none", linewidths=2)

    # ax.axhline(probe.y_max + 10, probe.x_min, probe.x_max, lw=4)
    ax.axis("off")
    ax.set_title(f"Probe {probe.n_contacts}ch")

    return ax


def plot_on_probe(probe: ProbeGroup):
    pass