import numpy as np
from pathlib import Path
import xml.etree.ElementTree as Etree


class NeuroscopeIO:
    def __init__(self, xml_filename) -> None:
        self.source_file = Path(xml_filename)
        self.eeg_filename = self.source_file.with_suffix(".eeg")
        self.dat_filename = self.source_file.with_suffix(".dat")
        self.skipped_channels = None
        self.channel_groups = None
        self.discarded_channels = None

    def _parse_xml_file(self):
        """Uses .xml file to parse anatomical groups

        Parameters
        ----------
        nShanks : int or list of int, optional
            number of shanks, if None then this equals to number of anatomical grps excluding channels mentioned in skulleeg, emg, motion
        skulleeg : list, optional
            any channels recorded from the skull, by default None
        emg : list, optional
            emg channels, by default None
        motion : list, optional
            channels recording accelerometer data or velocity, by default None
        """

        myroot = Etree.parse(self.recfiles.xmlfile).getroot()

        chan_session, channelgroups, badchans = [], [], []
        for x in myroot.findall("anatomicalDescription"):
            for y in x.findall("channelGroups"):
                for z in y.findall("group"):
                    chan_group = []
                    for chan in z.findall("channel"):
                        if int(chan.text) not in skulleeg + emg + motion:
                            chan_session.append(int(chan.text))
                            if int(chan.attrib["skip"]) == 1:
                                badchans.append(int(chan.text))

                            chan_group.append(int(chan.text))
                    if chan_group:
                        channelgroups.append(chan_group)

        sampfreq = nChans = None
        for sf in myroot.findall("acquisitionSystem"):
            sampfreq = int(sf.find("samplingRate").text)
            nChans = int(sf.find("nChannels").text)

        lfpSrate = None
        for val in myroot.findall("fieldPotentials"):
            lfpSrate = int(val.find("lfpSamplingRate").text)

        auxchans = np.setdiff1d(
            np.arange(nChans), np.array(chan_session + skulleeg + emg + motion)
        )
        if auxchans.size == 0:
            auxchans = None

        if nShanks is None:
            nShanks = len(channelgroups)

        nShanksProbe = [nShanks] if isinstance(nShanks, int) else nShanks
        nProbes = len(nShanksProbe)
        nShanks = np.sum(nShanksProbe)

        if motion is not None:
            pass

        basics = {
            "sampfreq": sampfreq,
            "channels": chan_session,
            "nChans": nChans,
            "channelgroups": channelgroups,
            "nShanks": nShanks,
            "nProbes": nProbes,
            "nShanksProbe": nShanksProbe,
            "subname": self.session.subname,
            "sessionName": self.session.sessionName,
            "lfpSrate": lfpSrate,
            "badchans": badchans,
            "auxchans": auxchans,
            "skulleeg": skulleeg,
            "emgChans": emg,
            "motionChans": motion,
        }

        np.save(self.files.basics, basics)
        print(f"_basics.npy created for {self.session.sessionName}")

        # laods attributes in runtime so doesn't lead reloading of entire class instance
