from dataclasses import dataclass
import os
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg


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
    channels: list
        list of recording channels in the .dat file
    badchans: list
        list of bad channels


    Methods
    ----------
    makerecinfo()
        creates a file containing basic infomation about the recording
    geteeg(chas, timeRange, period)
        returns lfp from .eeg file for given channels
    """

    def __init__(self, basePath):
        self.basePath = Path(basePath)

        filePrefix = None
        for file in os.listdir(basePath):
            if file.endswith(".xml"):
                xmlfile = self.basePath / file
                filePrefix = xmlfile.with_suffix("")

        self.session = sessionname(filePrefix)
        self.files = files(filePrefix)
        self.recfiles = recfiles(filePrefix)

        if self.files.basics.is_file():
            self._intialize()
        self.probemap = Probemap(self)

    def _intialize(self):

        myinfo = np.load(self.files.basics, allow_pickle=True).item()
        self.sampfreq = myinfo["sRate"]
        self.channels = myinfo["channels"]
        self.nChans = myinfo["nChans"]
        self.lfpSrate = myinfo["lfpSrate"]
        self.channelgroups = myinfo["channelgroups"]
        self.badchans = myinfo["badchans"]
        self.nShanks = myinfo["nShanks"]
        self.auxchans = myinfo["auxchans"]

        self.goodchans = np.setdiff1d(self.channels, self.badchans, assume_unique=True)
        self.goodchangrp = [
            list(np.setdiff1d(_, self.badchans, assume_unique=True).astype(int))
            for _ in self.channelgroups
        ]

    @property
    def metadata(self):
        metadatafile = Path(str(self.files.filePrefix) + "_metadata.csv")
        if metadatafile.is_file():
            metadata = pd.read_csv(metadatafile)

        else:
            val = input("Do you want to create metadata, Yes or No: ")
            if val in ["Y", "y", "yes", "Yes", "YES"]:

                def show_entry_fields():
                    print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))

                master = tk.Tk()
                tk.Label(master, text="First Name").grid(row=0)
                tk.Label(master, text="Last Name").grid(row=1)

                e1 = tk.Entry(master)
                e2 = tk.Entry(master)

                e1.grid(row=0, column=1)
                e2.grid(row=1, column=1)

                tk.Button(master, text="Quit", command=master.quit).grid(
                    row=3, column=0, sticky=tk.W, pady=4
                )
                tk.Button(master, text="Show", command=show_entry_fields).grid(
                    row=3, column=1, sticky=tk.W, pady=4
                )

                tk.mainloop()

        return metadata

    def __str__(self) -> str:
        return f"Name: {self.session.name} with {self.nChans} channels"

    # def __repr__(self) -> str:
    #     return "I am an animal"

    def makerecinfo(self):
        """Reads recording parameter from xml file"""

        myroot = ET.parse(self.recfiles.xmlfile).getroot()

        chan_session, channelgroups, badchans = [], [], []
        for x in myroot.findall("anatomicalDescription"):
            for y in x.findall("channelGroups"):
                for z in y.findall("group"):
                    chan_group = []
                    for chan in z.findall("channel"):
                        chan_session.append(int(chan.text))
                        if int(chan.attrib["skip"]) == 1:
                            badchans.append(int(chan.text))

                        chan_group.append(int(chan.text))
                    channelgroups.append(chan_group)

        sampfreq = nChans = None
        for sf in myroot.findall("acquisitionSystem"):
            sampfreq = int(sf.find("samplingRate").text)
            nChans = int(sf.find("nChannels").text)

        lfpSrate = None
        for val in myroot.findall("fieldPotentials"):
            lfpSrate = int(val.find("lfpSamplingRate").text)

        auxchans = np.setdiff1d(np.arange(nChans), np.concatenate(channelgroups))
        if auxchans.size == 0:
            auxchans = None

        basics = {
            "sRate": sampfreq,
            "channels": chan_session,
            "nChans": nChans,
            "channelgroups": channelgroups,
            "nShanks": len(channelgroups),
            "subname": self.session.subname,
            "sessionName": self.session.sessionName,
            "lfpSrate": lfpSrate,
            "badchans": badchans,
            "auxchans": auxchans,
        }

        np.save(self.files.basics, basics)
        print(f"_basics.npy created for {self.session.sessionName}")

        # laods attributes in runtime so doesn't lead reloading of entire class instance
        self._intialize()

    @property
    def getNframesDat(self):
        nChans = self.nChans
        datfile = self.recfiles.datfile
        data = np.memmap(datfile, dtype="int16", mode="r")

        return len(data) / nChans

    def geteeg(self, chans, timeRange=None, frames=None):
        """Returns eeg signal for given channels and timeperiod or selected frames

        Args:
            chans (list): list of channels required index should in order of binary file
            timeRange (list, optional): In seconds and must have length 2.
            frames (list, optional): Required frames from the eeg data.

        Returns:
            eeg: [array of channels x timepoints]
        """
        eegfile = self.recfiles.eegfile
        eegSrate = self.lfpSrate
        nChans = self.nChans

        eeg = np.memmap(eegfile, dtype="int16", mode="r")
        eeg = np.memmap.reshape(eeg, (nChans, len(eeg) // nChans), order="F")
        eeg = eeg[chans, :]

        if timeRange is not None:
            assert len(timeRange) == 2
            frameStart = int(timeRange[0] * eegSrate)
            frameEnd = int(timeRange[1] * eegSrate)
            eeg = eeg[..., frameStart:frameEnd]

        elif frames is not None:
            eeg = eeg[..., frames]

        return eeg

    def getPxx(self, chans, timeRange=None):
        eeg = self.geteeg(chans=chans, timeRange=timeRange)
        f, pxx = sg.welch(
            eeg, fs=self.lfpSrate, nperseg=2 * self.lfpSrate, noverlap=self.lfpSrate
        )
        return f, pxx


class files:
    def __init__(self, filePrefix):
        self.filePrefix = filePrefix
        self.probe = filePrefix.with_suffix(".probe.npy")
        self.basics = Path(str(filePrefix) + "_basics.npy")
        self.position = Path(str(filePrefix) + "_position.npy")
        self.epochs = Path(str(filePrefix) + "_epochs.npy")
        self.spindle_evt = Path(str(filePrefix) + "_spindles.npy")
        self.spindlelfp = Path(str(filePrefix) + "_BestSpindleChan.npy")
        self.hwsa_ripple = Path(str(filePrefix) + "_hswa_ripple.npy")
        self.slow_wave = Path(str(filePrefix) + "_hswa.npy")
        self.spectrogram = Path(str(filePrefix) + "_sxx.npy")


class recfiles:
    def __init__(self, f_prefix):

        self.xmlfile = f_prefix.with_suffix(".xml")
        self.eegfile = f_prefix.with_suffix(".eeg")
        self.datfile = f_prefix.with_suffix(".dat")


class sessionname:
    def __init__(self, f_prefix):
        basePath = str(f_prefix.parent)
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.name = basePath.split("/")[-2]
        self.day = basePath.split("/")[-1]
        # self.basePath = Path(basePath)
        self.subname = f_prefix.stem


class Probemap:
    def __init__(self, obj: Recinfo) -> None:

        self._obj = obj

        self.x, self.y = None, None
        if self._obj.files.probe.is_file():
            data = np.load(self._obj.files.probe, allow_pickle=True).item()
            self.x = data["x"]
            self.y = data["y"]
            self.coords = pd.DataFrame(
                {"chan": self._obj.channels, "x": self.x, "y": self.y}
            )

    def create(self, probetype="diagbio"):
        changroup = self._obj.channelgroups
        nShanks = self._obj.nShanks

        if len(changroup[0]) == 16:
            probetype = "diagbio"
        if len(changroup[0]) == 8:
            probetype = "buzsaki"

        changroup = changroup[:nShanks]
        xcoord, ycoord = [], []
        if probetype == "diagbio":

            for i in range(nShanks):
                xpos = [10 * (_ % 2) + i * 150 for _ in range(16)]
                ypos = [15 * 16 - _ * 15 for _ in range(16)]
                xcoord.extend(xpos)
                ycoord.extend(ypos)

        if probetype == "buzsaki":

            xp = [0, 37, 4, 33, 8, 29, 12, 20]
            yp = np.arange(160, 0, -20)
            for i in range(nShanks):
                xpos = [xp[_] + i * 200 for _ in range(8)]
                ypos = [yp[_] for _ in range(8)]
                xcoord.extend(xpos)
                ycoord.extend(ypos)

        if probetype == "linear":

            for i in range(nShanks):
                nchans = len(changroup[i])
                xpos = [10 * (_ % 2) + i * 150 for _ in range(nchans)]
                ypos = [15 * 128 - _ * 15 for _ in range(nchans)]
                xcoord.extend(xpos)
                ycoord.extend(ypos)

        coords = {"x": xcoord, "y": ycoord}
        np.save(self._obj.files.probe, coords)

    def get(self, chans):

        if isinstance(chans, int):
            chans = [chans]

        reqchans = self.coords[self.coords.chan.isin(chans)]

        return reqchans.x.values, reqchans.y.values

    def plot(self, chans=None, ax=None, colors=None):

        lfpchans = self._obj.channels

        chans2plot = chans
        chan_rank = np.where(np.isin(lfpchans, chans2plot))[0]
        xpos, ypos = self.x.copy(), self.y.copy()
        xpos = np.asarray(xpos)
        ypos = np.asarray(ypos)

        if ax is None:
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        ax.scatter(xpos, ypos, s=2, color="gray", zorder=1, linewidths=0.5)

        if len(self._obj.badchans) != 0:
            badchans = self._obj.badchans
            badchan_loc = np.where(np.isin(lfpchans, badchans))[0]
            ax.scatter(
                xpos[badchan_loc],
                ypos[badchan_loc],
                s=10,
                color="#FF5252",
                zorder=2,
                marker="x",
            )

        if colors is None:
            ax.scatter(xpos[chan_rank], ypos[chan_rank], c="#009688", s=20, zorder=2)
        else:
            ax.scatter(xpos[chan_rank], ypos[chan_rank], c=colors, s=40, zorder=2)

        # ax.legend(["channels", "bad", "chosen"])

        ax.axis("off")
