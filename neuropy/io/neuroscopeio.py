import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as Etree
# from .. import core
from neuropy.core.neurons import Neurons
from neuropy.core.position import Position
from neuropy.core.epoch import Epoch


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

    def write_neurons(self, neurons: Neurons, suffix_num: int = 1):
        """To view spikes in neuroscope, spikes are exported to .clu.# and .res.# files in the basepath.
        You can order the spikes in a way to view sequential activity in neuroscope.

        Parameters
        ----------
        spks : list
            list of spike times.
        suffix_num: int
            number to tack onto end of clu and res files.
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

        file_clu = self.source_file.with_suffix(".clu." + str(suffix_num))
        file_res = self.source_file.with_suffix(".res." + str(suffix_num))

        with file_clu.open("w") as f_clu, file_res.open("w") as f_res:
            for item in clu_id:
                f_clu.write(f"{int(item)}\n")
            for frame in spk_frame:
                f_res.write(f"{frame}\n")

        return file_clu

    def write_epochs(self, epochs: Epoch, ext="epc"):
        with self.source_file.with_suffix(f".evt.{ext}").open("w") as a:
            for event in epochs.to_dataframe().itertuples():
                # First attempt to fix bug where Neuropy exported .evt files get broken after manual
                # adjustment in NeuroScope - does not seem to work
                event_start, event_stop = event.start * 1000, event.stop * 1000
                if np.mod(event_start, 1) == 0:
                    event_start += 0.2
                if np.mod(event_stop, 1) == 0:
                    event_stop += 0.2
                a.write(f"{event_start}\tstart\n{event_stop}\tstop\n")

    def write_position(self, position: Position):
        """Writes core.Position object to neuroscope compatible format

        Parameters
        ----------
        position : core.Position
        """
        # neuroscope only displays positive values so translating the coordinates
        x, y = position.x, position.y
        x = x + abs(min(x))
        y = y + abs(min(y))

        filename = self.source_file.with_suffix(".pos")
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

    def event_to_epochs(self, evt_file, label=""):
        """Read in an event file and convert to an epochs object"""
        with open(evt_file, "r") as f:
            Lines = f.readlines()

        # Neuropy output saves file without tab separators
        if Lines[0].find("\t") > -1:
            split_str = "\t"
        else:  # if you savne in Neuroscope the event file now has tab separators
            split_str = " "

        starts, stops = [], []
        for line in Lines:
            if line.find("start") > -1:
                starts.append(float(line.split(f"{split_str}start")[0]) / 1000)
            elif line.find("stop") > -1:
                stops.append(float(line.split(f"{split_str}stop")[0]) / 1000)

        return Epoch(
            pd.DataFrame({"start": starts, "stop": stops, "label": label})
        )
