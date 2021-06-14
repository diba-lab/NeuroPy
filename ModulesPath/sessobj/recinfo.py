from dataclasses import dataclass
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy.ndimage import gaussian_filter
from .. import plotting


class Recinfo:
    """Parses .xml file to get a sense of recording parameters

    Attributes
    ----------
    basePath : str
        path to datafolder where .xml and .eeg files reside
    probemap : instance of Probemap class
        layout of recording channels
    sampfreq : int
        sampling rate of recording and is extracted from .xml file
    nChans : int,
        number of channels in the .dat/.eeg file
    channels : list
        list of recording channels in the .dat file from silicon probes and EXCLUDES any aux channels, skulleeg, emg, motion channels.
    channelgroups: list
        channels grouped in shanks.
    badchans : list
        list of bad channels
    skulleeg : list,
        channels in .dat/.eeg file from skull electrodes
    emgChans : list
        list of channels for emg
    nProbes : int,
        number of silicon probes used for this recording
    nShanksProbe : int or list of int,
        number of shanks in each probe. Example, [5,8], two probes with 5 and 8 shanks
    goodChans: list,
        list of channels excluding bad channels
    goodChangrp: list of lists,
        channels grouped in shanks and excludes bad channels. If all channels within a shank are bad, then it is represented as empty list within goodChangrp

    NOTE: len(channels) may not be equal to nChans.



    Methods
    ----------
    makerecinfo()
        creates a file containing basic infomation about the recording
    geteeg(chans, timeRange)
        returns lfp from .eeg file for given channels
    generate_xml(settingsPath)
        re-orders channels in the .xml file to reflect channel ordering in openephys settings.xml
    """

    def __init__(self, basePath):
        self.basePath = Path(basePath)

        filePrefix = None
        for file in os.listdir(basePath):
            if file.endswith(".xml"):
                xmlfile = self.basePath / file
                filePrefix = xmlfile.with_suffix("")
            elif file.endswith(".eeg"):
                eegfile = self.basePath / file
                filePrefix = eegfile.with_suffix("")
            elif file.endswith(".dat"):
                datfile = self.basePath / file
                filePrefix = datfile.with_suffix("")

        self.session = sessionname(filePrefix)
        self.files = files(filePrefix)
        self.recfiles = recfiles(filePrefix)

        self._intialize()
        self.animal = Animal(self)
        self.probemap = Probemap(self)

    def read_neuroscope(self, badchans=None):
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

        skulleeg = list(skulleeg)
        emg = list(emg)
        motion = list(motion)

        myroot = ET.parse(self.recfiles.xmlfile).getroot()

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
        self._intialize()

    def read_open_ephys(self):
        pass
