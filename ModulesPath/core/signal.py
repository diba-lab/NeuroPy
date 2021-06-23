import numpy as np


class Signal:
    def __init__(
        self,
        traces,
        sampling_rate,
        t_start=0.0,
        channel_id=None,
    ) -> None:
        self.traces = traces
        self.t_start = t_start
        self._sampling_rate = int(sampling_rate)
        if channel_id is None:
            self.channel_id = np.arange(self.n_channels)

    @property
    def t_stop(self):
        return self.t_start + self.duration

    @property
    def duration(self):
        return self.traces.shape[1] / self.sampling_rate

    @property
    def n_channels(self):
        return self.traces.shape[0]

    @property
    def n_frames(self):
        return self.traces.shape[0]

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, srate):
        self._sampling_rate = srate

    def time_slice(self, channel_id, t_start=None, t_stop=None):
        # TODO fix channel_index vs channel_id confusion
        if isinstance(channel_id, int):
            channel_id = [channel_id]

        if t_start is None:
            t_start = 0.0

        if t_stop is None:
            t_stop = t_start + self.duration

        frame_start = int(t_start * self.sampling_rate)
        frame_stop = int(t_stop * self.sampling_rate)

        if channel_id is None:
            traces = self.traces[:, frame_start:frame_stop]
        else:
            traces = self.traces[channel_id, frame_start:frame_stop]

        return traces
