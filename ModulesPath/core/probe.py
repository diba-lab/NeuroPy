import numpy as np
import pandas as pd


class Shank:
    def __init__(
        self,
        columns=2,
        contacts_per_column=10,
        xpitch=15,
        ypitch=20,
        y_shift_per_column=None,
        channel_id=None,
    ) -> None:

        if isinstance(contacts_per_column, int):
            contacts_per_column = [contacts_per_column] * columns

        if y_shift_per_column is None:
            y_shift_per_column = [0] * columns

        positions = []
        for i in range(columns):
            x = np.ones(contacts_per_column[i]) * xpitch * i
            y = np.arange(contacts_per_column[i]) * ypitch + y_shift_per_column[i]
            positions.append(np.hstack((x[:, None], y[:, None])))
        positions = np.vstack(positions)

        self.x = positions[:, 0]
        self.y = positions[:, 1]

        if channel_id is None:
            self.channel_id = np.arange(np.sum(contacts_per_column))

    @property
    def n_contacts(self):
        return len(self.x)

    def to_dict(self):
        layout = {"x": self.x, "y": self.y, "channel_id": self.channel_id}
        return layout

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict())

    def move(self):
        pass


class Probe:
    def __init__(self, shanks) -> None:

        if isinstance(shanks, Shank):
            shanks = [shanks]

        if isinstance(shanks, list):
            assert np.all([_.__class__.__name__ == "Shank" for _ in shanks])

        self.shanks: list[Shank] = shanks

    @property
    def n_shanks(self):
        return len(self.shanks)

    @property
    def shank_id(self):
        shank_id = []
        for n, shank in enumerate(self.shanks):
            shank_id.append([n] * shank.n_contacts)

        return np.concatenate(shank_id)

    @property
    def x(self):
        x = []
        for shank in self.shanks:
            x.append(shank.x)

        return np.concatenate(x)

    @property
    def y(self):
        y = []
        for shank in self.shanks:
            y.append(shank.y)

        return np.concatenate(y)

    @property
    def channel_id(self):
        y = []
        for shank in self.shanks:
            y.append(shank.y)

        return np.concatenate(y)

    def add_shank(self, shank: Shank, shank_pitch=(150, 0)):
        self.shanks.append(shank)

    def to_dict(self):
        layout = {
            "x": self.x,
            "y": self.y,
            "channel_id": self.channel_id,
            "shank_id": self.shank_id,
        }
        return layout

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict())

    def move(self):
        pass


class ProbeGroup:
    def __init__(self) -> None:
        self.probes: list[Probe] = []

    @property
    def n_probes(self):
        return len(self.probes)

    def n_shanks(self):

        return np.sum([probe.n_shanks for probe in self.probes])

    def add_probe(self, probe_pitch=(500, 0)):
        pass

    def to_dataframe(self):
        pass