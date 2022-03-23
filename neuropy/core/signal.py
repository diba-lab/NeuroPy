import numpy as np

# from .core import DataWriter


class Signal:
    def __init__(
        self,
        traces,
        sampling_rate,
        t_start=0.0,
        channel_id=None,
        source_file=None,
    ) -> None:
        self.traces = traces
        self.t_start = t_start
        self._sampling_rate = sampling_rate
        if channel_id is None:
            self.channel_id = np.arange(self.n_channels)
        else:
            self.channel_id = channel_id
        self.source_file = source_file

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
        return self.traces.shape[-1]

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, srate):
        self._sampling_rate = srate

    @property
    def time(self):
        return np.linspace(self.t_start, self.t_stop, self.n_frames)

    def time_slice(self, channel_id=None, t_start=None, t_stop=None):
        # TODO fix channel_index vs channel_id confusion
        if isinstance(channel_id, int):
            channel_id = [channel_id]

        if t_start is None:
            t_start = self.t_start

        assert t_start >= self.t_start, "t_start should be greater than signal.t_start"

        if t_stop is None:
            t_stop = t_start + self.duration

        assert t_stop <= self.t_stop, "t_stop should be less than signal.t_stop"
        assert t_stop > t_start, "t_stop should be greater than t_start"

        frame_start = int((t_start - self.t_start) * self.sampling_rate)
        frame_stop = int((t_stop - self.t_start) * self.sampling_rate)

        if channel_id is None:
            traces = self.traces[:, frame_start:frame_stop]
        else:
            traces = self.traces[channel_id, frame_start:frame_stop]

        return Signal(traces, self.sampling_rate, t_start, channel_id)


class Spectrogram(Signal):
    def __init__(self, traces, freqs, sampling_rate=1, t_start=0) -> None:
        super().__init__(traces, sampling_rate, t_start=t_start, channel_id=freqs)

    @property
    def freqs(self):
        return self.channel_id

    def time_slice(self, t_start=None, t_stop=None):
        return super().time_slice(t_start=t_start, t_stop=t_stop)

    def mean_power(self):
        return np.mean(self.traces, axis=0)

    def get_band_power(self, f1=None, f2=None):

        if f1 is None:
            f1 = self.freqs[0]

        if f2 is None:
            f2 = self.freqs[-1]

        assert f1 >= self.freqs[0], "f1 should be greater than lowest frequency"
        assert (
            f2 <= self.freqs[-1]
        ), "f2 should be lower than highest possible frequency"
        assert f2 > f1, "f2 should be greater than f1"

        ind = np.where((self.freqs >= f1) & (self.freqs <= f2))[0]
        band_power = np.mean(self.traces[ind, :], axis=0)
        return band_power

    @property
    def delta(self):
        return self.get_band_power(f1=0.5, f2=4)

    @property
    def deltaplus(self):
        deltaplus_ind = np.where(
            ((self.freqs > 0.5) & (self.freqs < 4))
            | ((self.freqs > 12) & (self.freqs < 15))
        )[0]
        deltaplus_sxx = np.mean(self.traces[deltaplus_ind, :], axis=0)
        return deltaplus_sxx

    @property
    def theta(self):
        return self.get_band_power(f1=5, f2=10)

    @property
    def spindle(self):
        return self.get_band_power(f1=10, f2=20)

    @property
    def gamma(self):
        return self.get_band_power(f1=30, f2=90)

    @property
    def ripple(self):
        return self.get_band_power(f1=140, f2=250)

    @property
    def theta_delta_ratio(self):
        return self.theta / self.delta

    @property
    def theta_deltaplus_ratio(self):
        return self.theta / self.deltaplus
