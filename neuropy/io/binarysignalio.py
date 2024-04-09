import numpy as np
from ..core import Signal
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

        return Signal(sig, self.sampling_rate, t_start, channel_id=channel_indx)

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


    # Persistance
    def __getstate__(self):
        """ Custom serialization/deserialization method to enable saving pipeline on macOS. Previously failed with the error:
            TypeError: cannot pickle 'mmap.mmap' object ' #11823
            Issue with macOS (Darwin) and multi-processing
            https://github.com/stamparm/maltrail/issues/11823
        """
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if '_raw_traces' in state:
	        del state['_raw_traces']
            
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # self.source_file = _raw_traces
        if (not self.source_file.exists()):
            original_source_file: Path = self.source_file.resolve()
            try:
                relative_source_file: Path = original_source_file.relative_to('/media/MAX/Data') # ValueError: '/nfs/turbo/umms-kdiba/Data/KDIBA/gor01/two/2006-6-07_16-40-19/2006-6-07_16-40-19.eeg' is not in the subpath of '/media/MAX/Data' OR one path is relative and the other is absolute.
                new_root: Path = Path('/media/halechr/MAX/Data').resolve()
                if new_root.exists():
                    proposed_new_source_path = new_root.joinpath(relative_source_file).resolve()
                    if proposed_new_source_path.exists():
                        print(f'updated source_file with {proposed_new_source_path}')
                        self.source_file = proposed_new_source_path

                # Re-load the raw traces from file on load from the saved properties. The same file must exist, meaning this solution isn't portable.
                self._raw_traces = (
                    np.memmap(self.source_file, dtype=self.dtype, mode="r").reshape(-1, self.n_channels).T
                ) # FileNotFoundError - FileNotFoundError: [Errno 2] No such file or directory: '/media/MAX/Data/KDIBA/gor01/one/2006-6-09_1-22-43/2006-6-09_1-22-43.eeg'
                
            except (ValueError, ) as e:
                print(f'failed to get the memory mapped file with err: {e}')
                # ## try to get the relative path:
                # data_split_parts = original_source_file.as_posix().split('Data/') # ['/nfs/turbo/umms-kdiba/', 'KDIBA/gor01/two/2006-6-07_16-40-19/2006-6-07_16-40-19.eeg']
                # original_data_root = Path(data_split_parts[0], 'Data').resolve() #
                # relative_source_file: Path = original_source_file.relative_to(original_data_root)
                # relative_source_file
                self._raw_traces = None
                pass 
                


        

