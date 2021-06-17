import numpy as np
import pandas as pd
from .datawriter import DataWriter


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
        self.connected = np.ones(np.sum(contacts_per_column), dtype=bool)
        self.contact_id = np.arange(np.sum(contacts_per_column))

        if channel_id is None:
            self.channel_id = np.arange(np.sum(contacts_per_column))
        else:
            self.channel_id = channel_id

    @property
    def channel_id(self):
        return self._channel_id

    @channel_id.setter
    def channel_id(self, chan_ids):
        assert self.n_contacts == len(chan_ids)
        self._channel_id = chan_ids

    @property
    def n_contacts(self):
        return len(self.x)

    def to_dict(self):
        layout = {
            "x": self.x,
            "y": self.y,
            "contact_id": self.contact_id,
            "channel_id": self.channel_id,
            "connected": self.connected,
        }
        return layout

    def set_disconnected_channels(self, channel_ids):
        self.connected[np.isin(self.channel_id, channel_ids)] = False

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict())

    def move(self, translation):
        x, y = translation
        self.x += x
        self.y += y


class Probe:
    def __init__(self, shanks, shank_pitch=(150, 0)) -> None:

        if isinstance(shanks, Shank):
            shanks = [shanks]

        if isinstance(shanks, list):
            assert np.all([_.__class__.__name__ == "Shank" for _ in shanks])

        self._data = pd.DataFrame(
            columns=["x", "y", "contact_id", "channel_id", "connected", "shank_id"]
        )
        x = np.arange(len(shanks)) * shank_pitch[0]
        y = np.arange(len(shanks)) * shank_pitch[1]
        for i, shank in enumerate(shanks):
            shank_df = shank.to_dataframe()
            shank_df["x"] += x[i]
            shank_df["y"] += y[i]
            shank_df["shank_id"] = i * np.ones(shank.n_contacts)
            self._data = self._data.append(shank_df)

    @property
    def n_contacts(self):
        return len(self._data)

    @property
    def n_shanks(self):
        return np.max(self._data["shank_id"]) + 1

    @property
    def shank_id(self):
        return self._data["shank_id"].values

    @property
    def x(self):
        return self._data["x"].values

    @property
    def x_max(self):
        return np.max(self._data["x"].values)

    @property
    def y(self):
        return self._data["y"].values

    @property
    def channel_id(self):
        return self._data["channel_id"].values

    @property
    def connected(self):
        return self._data["connected"].values

    def add_shank(self, shank: Shank):
        shank_df = shank.to_dataframe()
        shank_df["shank_id"] = (self.n_shanks - 1) * np.ones(shank.n_contacts)
        self._data = self._data.append(shank_df)

    def to_dict(self):
        return self._data.to_dict()

    def to_dataframe(self):
        return self._data

    def move(self, translation):
        x, y = translation
        self._data["x"] += x
        self._data["y"] += y


class ProbeGroup(DataWriter):
    def __init__(self, filename=None) -> None:

        super().__init__(filename=filename)
        self._data = pd.DataFrame(
            {
                "x": np.array([]),
                "y": np.array([]),
                "contact_id": np.array([]),
                "channel_id": np.array([]),
                "shank_id": np.array([]),
                "connected": np.array([], dtype=bool),
                "probe_id": np.array([]),
            }
        )
        self.metadata = {}
        self.load()

    def load(self):
        if self.filename is not None:
            data = super().load()
            if data is not None:
                self._data, self.metadata = data["probemap"], data["metadata"]

    @property
    def x(self):
        return self._data["x"].values

    @property
    def y(self):
        return self._data["y"].values

    @property
    def n_contacts(self):
        return len(self._data)

    @property
    def channel_id(self):
        return self._data["channel_id"].values

    @property
    def shank_id(self):
        return self._data["shank_id"].values

    @property
    def probe_id(self):
        return self._data["probe_id"].values

    @property
    def n_probes(self):
        return len(np.unique(self.probe_id))

    def n_shanks(self):
        return np.sum(np.unique(self.shank_id, return_counts=True))

    @property
    def get_disconnected(self):
        return self._data[self._data["connected"] == False]

    def add_probe(self, probe: Probe):
        probe_df = probe.to_dataframe()
        probe_df["probe_id"] = self.n_probes * np.ones(probe.n_contacts)
        self._data = self._data.append(probe_df)

    def to_dict(self):
        return self._data

    def from_dict(self, d: dict):
        if d is not None:
            self._data = d

    def to_dataframe(self):
        return pd.DataFrame(self._data)

    def remove_probes(self, probe_id=None):
        self.probes = []

    def save(self):
        data = self.to_dict()
        return super().save(data)
