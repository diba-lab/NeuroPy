import sys
from pathlib import Path
# print('sys.path: {}'.format(sys.path))
try:
    from neuropy import core
except ImportError:
    # sys.path.append(r'C:\Users\Pho\repos\NeuroPy')
    sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy')
    print('neuropy module not found, adding directory to sys.path. \nUpdated sys.path: {}'.format(sys.path))
    from neuropy import core

from neuropy.io import NeuroscopeIO, BinarysignalIO

class ProcessData:
    def __init__(self, basepath):
        basepath = Path(basepath)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        fp = xml_files[0].with_suffix("")
        self.filePrefix = fp

        self.recinfo = NeuroscopeIO(xml_files[0])
        self.eegfile = BinarysignalIO(
            self.recinfo.eeg_filename,
            n_channels=self.recinfo.n_channels,
            sampling_rate=self.recinfo.eeg_sampling_rate,
        )

        if self.recinfo.dat_filename.is_file():
            self.datfile = BinarysignalIO(
                self.recinfo.dat_filename,
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.dat_sampling_rate,
            )

        self.neurons = core.Neurons.from_file(fp.with_suffix(".neurons.npy"))
        self.probegroup = core.ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))
        self.position = core.Position.from_file(fp.with_suffix(".position.npy"))
        
        # self.paradigm = core.Epoch.from_file(fp.with_suffix(".paradigm.npy")) # "epoch" field of file
        self.epochs = core.Epoch.from_file(fp.with_suffix(".paradigm.npy")) # "epoch" field of file


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})"


# def ratN():
#     basepath='/data/Clustering/sessions/RatN_Day1_test_neuropy'
#     return ProcessData(basepath)

def processData(basedir='/Volumes/iNeo/Data/Bapun/Day5TwoNovel'):
    sess = ProcessData(basedir)
    return sess

