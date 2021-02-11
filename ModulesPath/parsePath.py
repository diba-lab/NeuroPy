from dataclasses import dataclass
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy import stats


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
        self.artifact = findartifact(self)

    def _intialize(self):

        self.sampfreq = None
        self.channels = None
        self.nChans = None
        self.lfpSrate = None
        self.channelgroups = None
        self.badchans = None
        self.nShanks = None
        self.auxchans = None
        self.skulleeg = None
        self.emgChans = None
        self.motionChans = None
        self.nShanksProbe = None
        self.nProbes = None

        if self.files.basics.is_file():
            myinfo = np.load(self.files.basics, allow_pickle=True).item()
            for attrib, val in myinfo.items():  # alternative list(epochs)
                setattr(self, attrib, val)  # .lower() will be removed

        if self.channels and self.badchans and self.channelgroups:
            self.goodchans = np.setdiff1d(
                self.channels, self.badchans, assume_unique=True
            )
            self.goodchangrp = [
                list(np.setdiff1d(_, self.badchans, assume_unique=True).astype(int))
                for _ in self.channelgroups
            ]

    def __str__(self) -> str:
        return f"Name: {self.session.name} \nChannels: {self.nChans}\nSampling Freq: {self.sampfreq}\nlfp Srate (downsampled): {self.lfpSrate}\n# bad channels: {len(self.badchans)}\nmotion channels: {self.motionChans}\nemg channels: {self.emgChans}\nskull eeg: {self.skulleeg}"

    def generate_xml(self, settingsPath):
        """Generates .xml for the data using openephys's settings.xml"""
        myroot = ET.parse(settingsPath).getroot()

        chanmap = []
        for elem in myroot[1][1][-1]:
            if "Mapping" in elem.attrib:
                chanmap.append(elem.attrib["Mapping"])

        neuroscope_xmltree = ET.parse(self.files.filePrefix.with_suffix(".xml"))
        neuroscope_xmlroot = neuroscope_xmltree.getroot()

        for i, chan in enumerate(neuroscope_xmlroot[2][0][0].iter("channel")):
            chan.text = str(int(chanmap[i]) - 1)

        neuroscope_xmltree.write(self.files.filePrefix.with_suffix(".xml"))

    def makerecinfo(self, nShanks=None, skulleeg=None, emg=None, motion=None):
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

        if skulleeg is None:
            skulleeg = []
        if emg is None:
            emg = []
        if motion is None:
            motion = []

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

    @property
    def getNframesDat(self):
        nChans = self.nChans
        datfile = self.recfiles.datfile
        data = np.memmap(datfile, dtype="int16", mode="r")

        return len(data) / nChans

    @property
    def getNframesEEG(self):
        nframes = len(self.geteeg(chans=0))

        return nframes

    def geteeg(self, chans, timeRange=None):
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

        if timeRange is not None:
            assert len(timeRange) == 2
            frameStart = int(timeRange[0] * eegSrate)
            frameEnd = int(timeRange[1] * eegSrate)
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

    def getPxx(self, chans, timeRange=None):
        eeg = self.geteeg(chans=chans, timeRange=timeRange)
        f, pxx = sg.welch(
            eeg, fs=self.lfpSrate, nperseg=2 * self.lfpSrate, noverlap=self.lfpSrate
        )
        return f, pxx

    def loadmetadata(self):
        metadatafile = Path(str(self.files.filePrefix) + "_metadata.csv")
        metadata = pd.read_csv(metadatafile)

        return metadata


class files:
    def __init__(self, filePrefix):
        self.filePrefix = filePrefix
        self.probe = filePrefix.with_suffix(".probe.npy")
        self.basics = Path(str(filePrefix) + "_basics.npy")
        self.position = Path(str(filePrefix) + "_position.npy")
        self.epochs = Path(str(filePrefix) + "_epochs.npy")
        self.spectrogram = Path(str(filePrefix) + "_sxx.npy")


class recfiles:
    def __init__(self, f_prefix):

        self.xmlfile = f_prefix.with_suffix(".xml")
        self.eegfile = f_prefix.with_suffix(".eeg")
        self.datfile = f_prefix.with_suffix(".dat")


class sessionname:
    def __init__(self, f_prefix):
        basePath = str(f_prefix.parent.as_posix())
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.name = basePath.split("/")[-2]
        self.day = basePath.split("/")[-1]
        # self.basePath = Path(basePath)
        self.subname = f_prefix.stem


class Animal:
    def __init__(self, obj: Recinfo) -> None:
        """
        Name:  name of the animal e.g. Roy, Ted, Kevin
        Alias:  code names as per lab standard e.g, RatN, Rat3254,
        Tag:  helps identifying category for analyses, e.g, sd, nsd, control, rollipramSD
        Sex: sex of animal
        Experimenter: Experimenter's name
        weight  : wieght of the animal during this recording
        Probe :  what probes were used
        BrainRegion :  which brain regions were implanted
        Day  : The number of experiments : Day1, Day2
        Date :  Date of experiment
        Experiment  : any name for experiment
        Track  : type of maze used, linear, l-shpaed
        files :
        StartTime : startime of the experiment in this format, 2019-10-11_03-58-54
        nFrames :
        deletedStart (minutes):  deleted from raw data to eliminate noise, required       for synching video tracking
        deletedEnd (minutes): deleted tiem from raw data
        Notes:  Any important notes

        """
        self._obj = obj

        metadatafile = Path(str(self._obj.files.filePrefix) + "_metadata.csv")

        if metadatafile.is_file():
            metadata = pd.read_csv(metadatafile)
            self.name = metadata.Name[0]
            self.alias = metadata.Alias[0]
            self.tag = metadata.Tag[0]
            self.sex = metadata.Sex[0]
            self.day = metadata.Day[0]
            self.date = metadata.Date[0]

        else:
            metadata = pd.DataFrame(
                columns=[
                    "Name",
                    "Alias",
                    "Tag",
                    "Sex",
                    "Experimenter",
                    "weight",
                    "Probe",
                    "BrainRegion",
                    "Day",
                    "Date",
                    "Experiment",
                    "Track",
                    "files",
                    "StartTime",
                    "nFrames",
                    "deletedStart (minutes)",
                    "deletedEnd (minutes)",
                    "Notes",
                ]
            )
            metadata.to_csv(metadatafile)
            print(
                f"Template metadata file {metadatafile.name} created. Please fill it accordingly."
            )


class Probemap:
    def __init__(self, obj: Recinfo) -> None:

        self._obj = obj

        self.x, self.y = None, None
        self._load()

    def _load(self):
        if self._obj.files.probe.is_file():
            data = np.load(self._obj.files.probe, allow_pickle=True).item()
            self.x = data["x"]
            self.y = data["y"]
            self.coords = pd.DataFrame(
                {"chan": self._obj.channels, "x": self.x, "y": self.y}
            )

    def create(self, xypitch=(15, 16), shankdist=150):
        """Probe layout, Assuming channels within each channelgroup/shank are oriented depthwise (dorsal --> ventral). Also supports mutiprobes. All distances are in um.

                      o
                    o
                      o
                    o
                      o
                    o   y
                      o
                    o
                     x

        Parameters
        ----------
        xypitch : tuple/list, optional
            x and y separation between electrodes
        shankdist : float,optional
            Distance between each shank.


        Examples
        ---------
        >>> Probemap.create(xypitch=[15,16]) # tetrode style
        >>> Probemap.create(xypitch=[[15,16],[20,14]]) # multiprobe
        >>> Probemap.create(xypich=[0,15]) # linear probe
        #TODO buzsaki style probe

        """
        nShanks = self._obj.nShanks
        changroup = self._obj.channelgroups[:nShanks]
        nProbes = self._obj.nProbes
        nShanksProbe = self._obj.nShanksProbe

        xcoord, ycoord = [], []
        probesid = np.concatenate([[_] * nShanksProbe[_] for _ in range(nProbes)])
        shankid = 0
        for probe in range(nProbes):

            if nProbes == 1:
                x, y = xypitch
            else:
                x, y = xypitch[probe]

            shanks_in_probe = [
                changroup[sh] for sh, _ in enumerate(probesid) if _ == probe
            ]
            for channels in shanks_in_probe:
                xpos = [x * (_ % 2) + shankid * shankdist for _ in range(len(channels))]
                ypos = [_ * y for _ in range(len(channels))]
                shankid += 1

                xcoord.extend(xpos)
                ycoord.extend(ypos[::-1])

        # if probetype == "buzsaki":

        #     xp = [0, 37, 4, 33, 8, 29, 12, 20]
        #     yp = np.arange(160, 0, -20)
        #     for i in range(nShanks):
        #         xpos = [xp[_] + i * 200 for _ in range(8)]
        #         ypos = [yp[_] for _ in range(8)]
        #         xcoord.extend(xpos)
        #         ycoord.extend(ypos)

        coords = {"x": xcoord, "y": ycoord}
        np.save(self._obj.files.probe, coords)
        print(".probe.npy file created")
        self._load()

    def get(self, chans):

        if isinstance(chans, int):
            chans = [chans]

        reqchans = self.coords[self.coords.chan.isin(chans)]

        return reqchans.x.values, reqchans.y.values

    def for_spyking_circus(self, rmv_badchans=True, shanksCombine=False):
        """Creates .prb file for spyking circus in the basepath folder

        Parameters
        ----------
        rmv_badchans : bool
            if True then removes badchannels from the .prb file, by default True
        shanksCombine : bool, optional
            if True then all shanks are combined in same channel group, by default False
        """
        nShanks = self._obj.nShanks
        nChans = self._obj.nChans
        channelgroups = self._obj.channelgroups[:nShanks]
        circus_prb = (self._obj.files.filePrefix).with_suffix(".prb")
        coords = self.coords.set_index("chan")

        if rmv_badchans:
            channelgroups = self._obj.goodchangrp[:nShanks]

        with circus_prb.open("w") as f:
            f.write(f"total_nb_channels = {nChans}\n")
            f.write(f"radius = 120\n")
            f.write("channel_groups = {\n")

            if shanksCombine:

                chan_list = np.concatenate(channelgroups[:nShanks])
                f.write(f"1: {{\n")
                f.write(f"'channels' : {[int(_) for _ in chan_list]},\n")
                f.write("'graph' : [],\n")
                f.write("'geometry' : {\n")

                for i, shank in enumerate(channelgroups):
                    if shank:
                        for chan in shank:
                            x, y = coords.loc[chan]
                            f.write(f"{chan}: [{x+i*300},{y+i*400}],\n")

                        f.write("\n")
                f.write("}\n")
                f.write("},\n")

                f.write("}\n")

            else:
                for i, shank in enumerate(channelgroups):
                    if shank:
                        f.write(f"{i+1}: {{\n")
                        f.write(f"'channels' : {[int(_) for _ in shank]},\n")
                        f.write("'graph' : [],\n")
                        f.write("'geometry' : {\n")

                        for chan in shank:
                            x, y = coords.loc[chan]
                            f.write(f"{chan}: [{x+i*300},{y+i*400}],\n")

                        f.write("}\n")
                        f.write("},\n\n")

                f.write("}\n")

        print(".prb file created for Spyking Circus")

    def plot(self, chans=None, ax=None, colors=None, annotate=False):

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
        if annotate:
            for coord in self.coords.itertuples():
                ax.annotate(coord.chan, (coord.x, coord.y), fontsize=8)

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


class findartifact:
    """Detects noisy periods using downsampled data

    Attributes
    ------------
    time: array,
        time periods which are noisy

    Methods
    ------------
    removefrom:
        removes noisy timestamps
    """

    def __init__(self, obj):

        if isinstance(obj, Recinfo):
            self._obj = obj
        else:
            self._obj = Recinfo(obj)

        self.time = None

        # ----- defining file names ---------
        filePrefix = self._obj.files.filePrefix

        @dataclass
        class files:
            dead: str = filePrefix.with_suffix(".dead")
            artifact: str = filePrefix.with_suffix(".artifact.npy")
            neuroscope: str = filePrefix.with_suffix(".evt.art")

        self.files = files()

        # ----- loading files --------
        if self.files.artifact.is_file():
            self._load()
        elif Path(self.files.dead).is_file():
            with self.files.dead.open("r") as f:
                noisy = []
                for line in f:
                    epc = line.split(" ")
                    epc = [float(_) for _ in epc]
                    noisy.append(epc)
                noisy = np.asarray(noisy) / 1000
                self.time = noisy  # in seconds

    def _load(self):
        data = np.load(self.files.artifact, allow_pickle=True).item()
        self.threshold = data["threshold"]
        self.chan = data["channel"]
        self.time = data["time"]

    def removefrom(self, lfp, timepoints):
        """Deletes detected artifacts from the 'lfp'

        Args:
            lfp ([array]): lfp signal
            timepoints ([array]): seconds, corresponding time stamps of the lfp

        Returns:
            [array]: artifact deleted lfp
        """
        # --- if a period is given, then convert it to timepoints------
        if len(timepoints) == 2:
            timepoints = np.linspace(timepoints[0], timepoints[1], len(lfp))

        if self.time is not None:
            dead_indx = np.concatenate(
                [
                    np.where((timepoints > start) & (timepoints < end))[0]
                    for (start, end) in self.time
                ]
            )
            lfp = np.delete(lfp, dead_indx, axis=-1)
        return lfp

    def getframes(self):
        eegSrate = self._obj.lfpSrate
        noisy_intervals = (self.time * eegSrate).astype(int) - 1  # zero indexing
        noisy_frames = np.concatenate(
            [np.arange(beg, end) for (beg, end) in noisy_intervals]
        )
        # correcting for any rounding error mostly an issue when artifacts are at end
        noisy_frames = noisy_frames[noisy_frames < self._obj.getNframesEEG]
        return noisy_frames

    def usingZscore(self, chans=None, thresh=5):
        """
        calculating periods to exclude for analysis using simple z-score measure
        """
        if chans is None:
            chans = np.random.choice(self._obj.goodchans, 4)

        eegSrate = self._obj.lfpSrate
        lfp = self._obj.geteeg(chans=chans)
        if isinstance(chans, list):
            lfp = np.asarray(lfp)
            lfp = np.median(lfp, axis=0)

        zsc = np.abs(stats.zscore(lfp))

        artifact_binary = np.where(zsc > thresh, 1, 0)
        artifact_binary = np.concatenate(([0], artifact_binary, [0]))
        artifact_diff = np.diff(artifact_binary)
        artifact_start = np.where(artifact_diff == 1)[0]
        artifact_end = np.where(artifact_diff == -1)[0]

        firstPass = np.vstack((artifact_start - 10, artifact_end + 2)).T

        minInterArtifactDist = 5 * eegSrate
        secondPass = []
        artifact = firstPass[0]
        for i in range(1, len(artifact_start)):
            if firstPass[i, 0] - artifact[1] < minInterArtifactDist:
                # Merging artifacts
                artifact = [artifact[0], firstPass[i, 1]]
            else:
                secondPass.append(artifact)
                artifact = firstPass[i]

        secondPass.append(artifact)

        artifact_s = np.asarray(secondPass) / eegSrate  # seconds

        data = {"channel": chans, "time": artifact_s, "threshold": thresh}
        np.save(self.files.artifact, data)

        self._load()
        return zsc

    def export2neuroscope(self):
        # --- converting to required time units for export ------
        artifact_ms = self.time * 1000  # ms

        # --- writing to file for neuroscope and spyking circus ----
        file_neuroscope = self.files.neuroscope
        with file_neuroscope.open("w") as file:
            for beg, stop in artifact_ms:
                file.write(f"{beg} start\n{stop} end\n")

    def export2circus(self):
        # --- converting to required time units for export ------
        artifact_ms = self.time * 1000  # ms

        # --- writing to file for neuroscope and spyking circus ----
        circus_file = self.files.dead
        with circus_file.open("w") as file:
            for beg, stop in artifact_ms:
                file.write(f"{beg} {stop}\n")

    def plot(self):

        chans = self.chan
        lfp = self._obj.geteeg(chans=chans)
        if not isinstance(chans, int):
            lfp = np.asarray(lfp)
            lfp = np.median(lfp, axis=0)

        zsc = np.abs(stats.zscore(lfp))
        artifact = self.time * self._obj.lfpSrate

        _, ax = plt.subplots(1, 1)
        ax.plot(zsc, "gray")
        ax.axhline(self.threshold, color="#37474F", ls="--")
        ax.plot(
            artifact[:, 0], self.threshold * np.ones(artifact.shape[0]), "r|", ms="10"
        )
        ax.plot(
            artifact[:, 1], self.threshold * np.ones(artifact.shape[0]), "k|", ms="10"
        )
        ax.set_xlabel("frames")
        ax.set_ylabel("Absolute zscore")

        ax.legend(["zsc-lfp", "threshold", "art-start", "art-end"])

    def createCleanDat(self):

        # for shankID in range(3, 9):
        #     print(shankID)

        #     DatFileOG = (
        #         folderPath
        #         + "Shank"
        #         + str(shankID)
        #         + "/RatJDay2_Shank"
        #         + str(shankID)
        #         + ".dat"
        #     )
        #     DestFolder = (
        #         folderPath
        #         + "Shank"
        #         + str(shankID)
        #         + "/RatJDay2_Shank"
        #         + str(shankID)
        #         + "_denoised.dat"
        #     )

        #     nChans = 8
        #     SampFreq = 30000

        #     b = []
        #     for i in range(len(Data_start)):

        #         start_time = Data_start[i]
        #         end_time = Data_end[i]

        #         duration = end_time - start_time  # in seconds
        #         b.append(
        #             np.memmap(
        #                 DatFileOG,
        #                 dtype="int16",
        #                 mode="r",
        #                 offset=2 * nChans * int(SampFreq * start_time),
        #                 shape=(nChans * int(SampFreq * duration)),
        #             )
        #         )

        #     c = np.memmap(
        #         DestFolder, dtype="int16", mode="w+", shape=sum([len(x) for x in b])
        #     )

        #     del c
        #     d = np.memmap(
        #         DestFolder, dtype="int16", mode="r+", shape=sum([len(x) for x in b])
        #     )

        #     sizeb = [0]
        #     sizeb.extend([len(x) for x in b])
        #     sizeb = np.cumsum(sizeb)

        #     for i in range(len(b)):

        #         d[sizeb[i] : sizeb[i + 1]] = b[i]
        #         # d[len(b[i]) : len(b1) + len(b2)] = b2
        #     del d
        #     del b
        pass
