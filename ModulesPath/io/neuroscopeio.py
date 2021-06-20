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

    def __str__(self) -> str:
        return f"filename: {self.source_file} \n# channels: {self.n_channels}\nsampling rate: {self.dat_sampling_rate}\nlfp Srate (downsampled): {self.eeg_sampling_rate}"

    def write_neurons(self, neurons: core.Neurons):
        """To view spikes in neuroscope, spikes are exported to .clu.1 and .res.1 files in the basepath.
        You can order the spikes in a way to view sequential activity in neuroscope.

        Parameters
        ----------
        spks : list
            list of spike times.
        """

        spks = neurons.get_spiketrains()
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

    def write_epochs(self):
        pass

    def write_position(self):
        pass