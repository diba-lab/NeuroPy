from neuropy.core.datawriter import DataWriter
from pathlib import Path
import neuropy.io.openephysio as oeio
from neuropy.io.binarysignalio import BinarysignalIO
from neuropy.io.neuroscopeio import NeuroscopeIO


class ProcessData:
    def __init__(self, basepath):
        basepath = Path(basepath)
        self.basepath = basepath
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        fp = xml_files[0].with_suffix("")
        self.filePrefix = fp

        self.recinfo = NeuroscopeIO(xml_files[0])
        eegfiles = sorted(basepath.glob("*.eeg"))
        assert len(eegfiles) == 1, "Fewer/more than one .eeg file detected"
        self.eegfile = BinarysignalIO(
            eegfiles[0],
            n_channels=self.recinfo.n_channels,
            sampling_rate=self.recinfo.eeg_sampling_rate,
        )
        try:
            self.datfile = BinarysignalIO(
                eegfiles[0].with_suffix(".dat"),
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.dat_sampling_rate,
            )
        except FileNotFoundError:
            print("No dat file found, not loading")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})"


class OESync(DataWriter):
    """Class to synchronize different systems with ephys data collected with OpenEphys. Most useful when multiple dat
    files have been concatenated into one, but also useful for individual recordings."""

    def __init__(
        self,
        basepath,
        metadata=None,
    ) -> None:
        super().__init__(metadata=metadata)

        self.basepath = Path(basepath)
        self.sess = ProcessData(basepath)
        self.oe_sync_df = oeio.create_sync_df(self.basepath)
