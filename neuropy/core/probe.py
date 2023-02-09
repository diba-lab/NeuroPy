import numpy as np
import pandas as pd
from .datawriter import DataWriter


class Shank:
    def __init__(self) -> None:

        self._x = None
        self._y = None
        self._connected = None
        self._contact_id = None
        self._channel_id = None

    @staticmethod
    def auto_generate(
        columns=2,
        contacts_per_column=10,
        xpitch=15,
        ypitch=20,
        y_shift_per_column=None,
        channel_id=None,
    ):
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

        shank = Shank()
        shank._x = positions[:, 0]
        shank._y = positions[:, 1]
        shank._channel_id = channel_id
        shank._connected = np.ones(np.sum(contacts_per_column), dtype=bool)
        shank._contact_id = np.arange(np.sum(contacts_per_column))
        if channel_id is None:
            shank._channel_id = np.arange(np.sum(contacts_per_column))
        else:
            shank._channel_id = channel_id

        return shank

    @staticmethod
    def from_library(probe_name):
        pass

    @staticmethod
    def set_contacts(positions, channel_ids):
        pass

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, arr):
        assert (
            len(arr) == self.n_contacts
        ), "number of x coordinates should match number of contacts"
        self._x = arr

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, arr):
        assert (
            len(arr) == self.n_contacts
        ), "number of y coordinates should match number of contacts"
        self._y = arr

    @property
    def contact_id(self):
        return self._contact_id

    @property
    def channel_id(self):
        return self._channel_id

    @channel_id.setter
    def channel_id(self, chan_ids):
        assert self.n_contacts == len(chan_ids)
        self._channel_id = chan_ids

    @property
    def connected(self):
        return self._connected

    @connected.setter
    def connected(self, arr):
        self._connected = arr

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

    def from_dict(self):
        pass

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
            {
                "x": np.array([]),
                "y": np.array([]),
                "contact_id": np.array([]),
                "channel_id": np.array([]),
                "connected": np.array([], dtype=bool),
                "shank_id": np.array([]),
            }
        )

        x = np.arange(len(shanks)) * shank_pitch[0]
        y = np.arange(len(shanks)) * shank_pitch[1]
        for i, shank in enumerate(shanks):
            shank_df = shank.to_dataframe()
            shank_df["x"] += x[i]
            shank_df["y"] += y[i]
            shank_df["shank_id"] = i * np.ones(shank.n_contacts)
            # self._data = self._data.append(shank_df)
            self._data = pd.concat([self._data, shank_df])
        self._data = self._data.reset_index(drop=True)
        self._data["contact_id"] = np.arange(len(self._data))

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

    def add_shanks(self, shanks: Shank, shank_pitch=(150, 0)):

        if isinstance(shanks, list):
            assert np.all([_.__class__.__name__ == "Shank" for _ in shanks])
        else:
            assert isinstance(shanks, Shank)
            shanks = [shanks]

        for shank in shanks:
            shank_df = shank.to_dataframe()
            shank_df["shank_id"] = (self.n_shanks - 1) * np.ones(shank.n_contacts)
            self._data = pd.concat([self._data, shank_df])

    def to_dict(self):
        return self._data.to_dict()

    def to_dataframe(self):
        return self._data

    def move(self, translation):
        x, y = translation
        self._data["x"] += x
        self._data["y"] += y


class ProbeGroup(DataWriter):
    def __init__(self, metadata=None) -> None:

        super().__init__(metadata=metadata)
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

    @property
    def x(self):
        return self._data["x"].values

    @property
    def x_min(self):
        return np.min(self.x)

    @property
    def x_max(self):
        return np.max(self.x)

    @property
    def y(self):
        return self._data["y"].values

    @property
    def y_min(self):
        return np.min(self.y)

    @property
    def y_max(self):
        return np.max(self.y)

    @property
    def n_contacts(self):
        return len(self._data)

    @property
    def channel_id(self):
        return self._data["channel_id"].values

    @property
    def shank_id(self):
        return self._data["shank_id"].values

    def get_channels(self, groupby="shank"):
        prb = self.to_dataframe()

        if groupby == "shank":
            prb = prb.groupby("shank_id")
            channels = []
            for i in prb.groups.keys():
                channels.append(prb.get_group(i).channel_id.values)

        return np.array(channels, dtype="object")

    def get_shank_id_for_channels(self, channel_id):
        """Get shank ids for the channels.

        Parameters
        ----------
        channel_id : array
            channel_ids, can have repeated values

        Returns
        -------
        array
            shank_ids corresponding to the channels
        """
        shank_ids = self.shank_id
        channel_ids = self.channel_id

        # indx_location = np.concatenate(
        #     list(map(lambda x: channel_ids[channel_ids == x], channel_id))
        # )

        # return shank_ids[indx_location.astype(int)]
        return np.concatenate(
            [shank_ids[np.where(channel_ids == _)[0]] for _ in channel_id]
        )

    def get_probe_id_for_channels(self, channel_id):
        """Get probe ids for the channels.

        Parameters
        ----------
        channel_id : array
            channel_ids, can have repeated values

        Returns
        -------
        array
            probe_ids corresponding to the channels
        """
        probe_ids = self.probe_id
        channel_ids = self.channel_id

        return np.concatenate(
            [probe_ids[np.where(channel_ids == _)[0]] for _ in channel_id]
        ).astype("int")

    def get_probe(self):
        pass

    def get_connected_channels(self, groupby="shank"):
        df = self.to_dataframe()
        df = df[df["connected"] == True]
        chans = []
        probe_grp = df.groupby("probe_id")

        if groupby == "probe":
            for i in range(self.n_probes):
                chans.append(probe_grp.get_group(i).channel_id.values)
        if groupby == "shank":
            probe_grp = df.groupby("probe_id")
            for i in range(self.n_probes):
                shank_grp = probe_grp.get_group(i).groupby("shank_id")
                for i1 in shank_grp.groups.keys():
                    chans.append(shank_grp.get_group(i1).channel_id.values)

        return np.array(chans, dtype=object)

    @property
    def probe_id(self):
        return self._data["probe_id"].values

    @property
    def n_probes(self):
        return len(np.unique(self.probe_id))

    @property
    def n_shanks(self):
        return len(np.unique(self.shank_id))

    @property
    def get_disconnected(self):
        return self._data[self._data["connected"] == False]

    def add_probe(self, probe: Probe):
        probe_df = probe.to_dataframe()
        probe_df["probe_id"] = self.n_probes * np.ones(probe.n_contacts)
        if self.n_probes > 0:
            probe_df["shank_id"] = probe_df["shank_id"] + self.n_shanks

        self._data = pd.concat([self._data, probe_df])

        # _, counts = np.unique(self.get_channel_ids(), return_counts=True)

    def to_dict(self):
        return {
            "data": self._data,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict):
        prbgrp = ProbeGroup(metadata=d["metadata"])
        prbgrp._data = d["data"].sort_values(["shank_id", "y"], ascending=[True, False])
        return prbgrp

    @staticmethod
    def from_file(f):
        d = DataWriter.from_file(f)
        if d is not None:
            return ProbeGroup.from_dict(d)

    def to_dataframe(self):
        return pd.DataFrame(self._data)

    def remove_probes(self, probe_id=None):
        self._data = {}
