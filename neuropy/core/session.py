from pathlib import Path
import os

# from neuropy.io.neuroscopeio import NeuroscopeIO
import neuropy.io.neuroscopeio as neuroscopeio
# from neuropy.io.binarysignalio import BinarysignalIO
import neuropy.io.binarysignalio as binarysignalio
import neuropy.core as core

class ProcessData:
    def __init__(self, basepath=os.getcwd()):
        basepath = Path(basepath)
        self.basepath = basepath
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found fewer/more than one .xml file"

        fp = xml_files[0].with_suffix("")
        self.filePrefix = fp     

        self.probegroup = core.ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))

        # self.recinfo = NeuroscopeIO(xml_files[0])
        self.recinfo = neuroscopeio.NeuroscopeIO(xml_files[0])    
        eegfiles = sorted(basepath.glob("*.eeg"))
        try:
            assert len(eegfiles) == 1, "Fewer/more than one .eeg file detected"
            self.eegfile = binarysignalio.BinarysignalIO(
                eegfiles[0],
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.eeg_sampling_rate,
            )
        except AssertionError:
            print("Fewer/more than one .eeg file detected, no eeg file loaded")
        try:
            self.datfile = binarysignalio.BinarysignalIO(
                eegfiles[0].with_suffix(".dat"),
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.dat_sampling_rate,
            )
        except (FileNotFoundError, IndexError):
            print("No dat file found, not loading")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})"
