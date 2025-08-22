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
        assert len(xml_files) == 1, f"Found fewer/more than one .xml file in {basepath.name}"

        fp = xml_files[0].with_suffix("")
        self.filePrefix = fp     

        self.probegroup = core.ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))

        # self.recinfo = NeuroscopeIO(xml_files[0])
        self.recinfo = neuroscopeio.NeuroscopeIO(xml_files[0])    
        # eegfiles = sorted(basepath.glob("*.eeg"))
        # # try:
        # #     assert len(eegfiles) == 1, f"Fewer/more than one .eeg file detected in {basepath.name}"
        # #     self.eegfile = binarysignalio.BinarysignalIO(
        # #         eegfiles[0],
        # #         n_channels=self.recinfo.n_channels,
        # #         sampling_rate=self.recinfo.eeg_sampling_rate,
        # #     )
        # # except AssertionError:
        # #     print(f"Fewer/more than one .eeg file detected in {basepath.name}, no eeg file loaded")
        # # datfiles = sorted(basepath.glob("*.dat"))
        # # try:
        # #     assert len(datfiles) == 1, f"Fewer/more than one .dat file detected in {basepath.name}"
        # #     self.datfile = binarysignalio.BinarysignalIO(
        # #         eegfiles[0].with_suffix(".dat"),
        # #         n_channels=self.recinfo.n_channels,
        # #         sampling_rate=self.recinfo.dat_sampling_rate,
        # #     )
        # # except (FileNotFoundError, IndexError):
        # #     print(f"No dat file found in {basepath.name}, not loading")

        fp = xml_files[0].with_suffix("")
        if self.recinfo.eeg_filename.is_file():
            self.eegfile = binarysignalio.BinarysignalIO(
                self.recinfo.eeg_filename,
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.eeg_sampling_rate,
            )

        if self.recinfo.dat_filename.is_file():
            self.datfile = binarysignalio.BinarysignalIO(
                self.recinfo.dat_filename,
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.dat_sampling_rate,
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})"
