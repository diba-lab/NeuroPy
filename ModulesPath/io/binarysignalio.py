import numpy as np
from ..core import Signal


class BinarysignalIO:
    def __init__(self, filename, dtype="int16", n_chans=2, sampling_rate=30000) -> None:
        pass

        self._raw_signal = np.memmap(filename, dtype=dtype)
        self.sampling_rate = sampling_rate
        self.n_chans = n_chans
        self.dtype = dtype
        self.source_file = filename

    def get_signal(self, chans=None, t_start=None, t_stop=None):

        if isinstance(chans, int):
            chans = [chans]

        if t_start is None:
            frame_start = int(t_start * self.sampling_rate)
        frame_stop = int(t_stop * self.sampling_rate)
        if chans is None:
            sig = self._raw_signal[frame_start:frame_stop, :]
        else:
            sig = self._raw_signal[frame_start:frame_stop, chans]

        return Signal(sig, self.sampling_rate, t_start, t_stop)

    def time_slice(self, chans, period=None):
        """Returns eeg signal for given channels. If multiple channels provided then it is list of lfps.

        Args:
            chans (list/array): channels required, index should in order of binary file
            timeRange (list, optional): In seconds and must have length 2.

        Returns:
            eeg: memmap, or list of memmaps
        """
        eegfile = self.recfiles.eegfile
        eegSrate = self.lfpSrate
        nChans = self.nChans

        if period is not None:
            assert len(period) == 2
            frameStart = int(period[0] * eegSrate)
            frameEnd = int(period[1] * eegSrate)
            eeg = np.memmap(
                eegfile,
                dtype="int16",
                mode="r",
                offset=2 * nChans * frameStart,
                shape=nChans * (frameEnd - frameStart),
            )
            eeg = np.memmap.reshape(eeg, (nChans, len(eeg) // nChans), order="F")

        else:
            eeg = np.memmap(eegfile, dtype="int16", mode="r")
            eeg = np.memmap.reshape(eeg, (nChans, len(eeg) // nChans), order="F")

        eeg_ = []
        if isinstance(chans, (list, np.ndarray)):
            for chan in chans:
                eeg_.append(eeg[chan, :])
        else:
            eeg_ = eeg[chans, :]

        return eeg_
