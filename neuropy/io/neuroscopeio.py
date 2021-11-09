import numpy as np
from pathlib import Path
import xml.etree.ElementTree as Etree
from .. import core


class NeuroscopeIO:
    def __init__(self, xml_filename) -> None:
        self.source_file = Path(xml_filename)
        self.eeg_filename = self.source_file.with_suffix(".eeg")
        self.dat_filename = self.source_file.with_suffix(".dat")
        self.skipped_channels = None
        self.channel_groups = None
        self.discarded_channels = None
        self._parse_xml_file()
        self._good_channels()

    def _parse_xml_file(self):

        tree = Etree.parse(self.source_file)
        myroot = tree.getroot()
        nbits = int(myroot.find("acquisitionSystem").find("nBits").text)

        dat_sampling_rate = n_channels = None
        for sf in myroot.findall("acquisitionSystem"):
            dat_sampling_rate = int(sf.find("samplingRate").text)
            n_channels = int(sf.find("nChannels").text)

        eeg_sampling_rate = None
        for val in myroot.findall("fieldPotentials"):
            eeg_sampling_rate = int(val.find("lfpSamplingRate").text)

        channel_groups, skipped_channels = [], []
        for x in myroot.findall("anatomicalDescription"):
            for y in x.findall("channelGroups"):
                for z in y.findall("group"):
                    chan_group = []
                    for chan in z.findall("channel"):
                        if int(chan.attrib["skip"]) == 1:
                            skipped_channels.append(int(chan.text))

                        chan_group.append(int(chan.text))
                    if chan_group:
                        channel_groups.append(np.array(chan_group))

        discarded_channels = np.setdiff1d(
            np.arange(n_channels), np.concatenate(channel_groups)
        )

        self.sig_dtype = nbits
        self.dat_sampling_rate = dat_sampling_rate
        self.eeg_sampling_rate = eeg_sampling_rate
        self.n_channels = n_channels
        self.channel_groups = np.array(channel_groups, dtype="object")
        self.discarded_channels = discarded_channels
        self.skipped_channels = np.array(skipped_channels)

    def _good_channels(self):
        good_chan = []
        for n in range(self.n_channels):
            if n not in self.discarded_channels and n not in self.skipped_channels:
                good_chan.append(n)

        self.good_channels = np.array(good_chan)

    def __str__(self) -> str:
        return (
            f"filename: {self.source_file} \n"
            f"# channels: {self.n_channels}\n"
            f"sampling rate: {self.dat_sampling_rate}\n"
            f"lfp Srate (downsampled): {self.eeg_sampling_rate}\n"
        )

    def set_datetime(self, datetime_epoch):
        """Often a resulting recording file is creating after concatenating different blocks.
        This method takes Epoch array containing datetime.
        """
        pass

    def write_neurons(self, neurons: core.Neurons):
        """To view spikes in neuroscope, spikes are exported to .clu.1 and .res.1 files in the basepath.
        You can order the spikes in a way to view sequential activity in neuroscope.

        Parameters
        ----------
        spks : list
            list of spike times.
        """

        spks = neurons.spiketrains
        srate = neurons.sampling_rate
        nclu = len(spks)
        spk_frame = np.concatenate([(cell * srate).astype(int) for cell in spks])
        clu_id = np.concatenate([[_] * len(spks[_]) for _ in range(nclu)])

        sort_ind = np.argsort(spk_frame)
        spk_frame = spk_frame[sort_ind]
        clu_id = clu_id[sort_ind]
        clu_id = np.append(nclu, clu_id)

        file_clu = self.source_file.with_suffix(".clu.1")
        file_res = self.source_file.with_suffix(".res.1")

        with file_clu.open("w") as f_clu, file_res.open("w") as f_res:
            for item in clu_id:
                f_clu.write(f"{item}\n")
            for frame in spk_frame:
                f_res.write(f"{frame}\n")

    def write_epochs(self, epochs: core.Epoch, ext=".epc"):
        with self.source_file.with_suffix(f".evt.{ext}").open("w") as a:
            for event in epochs.to_dataframe().itertuples():
                a.write(f"{event.start*1000} start\n{event.stop*1000} stop\n")

    def write_position(self, position: core.Position):
        # neuroscope only displays positive values so translating the coordinates
        x, y = position.x, position.y
        x = self.x + abs(min(self.x))
        y = self.y + abs(min(self.y))
        print(max(x))
        print(max(y))

        filename = self._obj.files.filePrefix.with_suffix(".pos")
        with filename.open("w") as f:
            for xpos, ypos in zip(x, y):
                f.write(f"{xpos} {ypos}\n")

    def to_dict(self):
        return {
            "source_file": self.source_file,
            "channel_groups": self.channel_groups,
            "skipped_channels": self.skipped_channels,
            "discarded_channels": self.discarded_channels,
            "n_channels": self.n_channels,
            "dat_sampling_rate": self.dat_sampling_rate,
            "eeg_sampling_rate": self.eeg_sampling_rate,
        }
