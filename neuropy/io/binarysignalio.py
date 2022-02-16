import numpy as np
from ..core import Signal, Epoch
from pathlib import Path


class BinarysignalIO:
    def __init__(
        self, filename, dtype="int16", n_channels=2, sampling_rate=30000
    ) -> None:
        pass

        self._raw_traces = (
            np.memmap(filename, dtype=dtype, mode="r").reshape(-1, n_channels).T
        )

        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.dtype = dtype
        self.source_file = filename

    def __str__(self) -> str:
        return (
            f"duration: {self.duration:.2f} seconds \n"
            f"duration: {self.duration/3600:.2f} hours \n"
        )

    @property
    def duration(self):
        return self._raw_traces.shape[1] / self.sampling_rate

    @property
    def n_frames(self):
        return self._raw_traces.shape[1]

    def get_signal(self, channel_indx=None, t_start=None, t_stop=None):

        if isinstance(channel_indx, int):
            channel_indx = [channel_indx]

        if t_start is None:
            t_start = 0.0

        if t_stop is None:
            t_stop = t_start + self.duration

        frame_start = int(t_start * self.sampling_rate)
        frame_stop = int(t_stop * self.sampling_rate)

        if channel_indx is None:
            sig = self._raw_traces[:, frame_start:frame_stop]
        else:
            sig = self._raw_traces[channel_indx, frame_start:frame_stop]

        if sig.ndim == 1:
            sig = sig.reshape(1, -1)
        return Signal(
            traces=sig,
            sampling_rate=self.sampling_rate,
            t_start=t_start,
            channel_id=channel_indx,
        )

    def frame_slice(self, channel_indx=None, frame_start=None, frame_stop=None):

        if isinstance(channel_indx, int):
            channel_indx = [channel_indx]

        if frame_start is None:
            frame_start = 0

        if frame_stop is None:
            frame_stop = self.n_frames

        if channel_indx is None:
            sig = self._raw_traces[:, frame_start:frame_stop]
        else:
            sig = self._raw_traces[channel_indx, frame_start:frame_stop]

        if sig.ndim == 1:
            sig = sig.reshape(1, -1)
        return Signal(
            traces=sig,
            sampling_rate=self.sampling_rate,
            t_start=frame_start / self.sampling_rate,
            channel_id=channel_indx,
        )

    def get_frames_within_epochs(self, epochs: Epoch, channel_indx):
        """Return concatenated frames corresponding to epochs

        Parameters
        ----------
        epochs : Epoch
            start and stop of epochs
        channel_indx : int/list
            channels by index location in the binary file

        Returns
        -------
        array
            concatenated frames
        """
        epochs_frames = (epochs.as_array() * self.sampling_rate).astype("int")
        frames = np.concatenate([np.arange(*e) for e in epochs_frames])
        return self._raw_traces[channel_indx, frames]

    def write_time_slice(self, write_filename, t_start, t_stop):

        duration = t_stop - t_start

        # read required chunk from the source file
        read_data = np.memmap(
            self.source_file,
            dtype=self.dtype,
            offset=2 * self.n_channels * self.sampling_rate * t_start,
            mode="r",
            shape=(self.n_channels * self.sampling_rate * duration),
        )

        # allocates space and writes data into the file
        write_data = np.memmap(
            write_filename, dtype=self.dtype, mode="w+", shape=(len(read_data))
        )
        write_data[: len(read_data)] = read_data
