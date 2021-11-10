import sys
import numpy as np
from pathlib import Path
# print('sys.path: {}'.format(sys.path))
try:
    from neuropy import core
except ImportError:
    sys.path.append(r'C:\Users\Pho\repos\NeuroPy')
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy')
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

        # if self.recinfo.eeg_filename.is_file():
        try:
            self.eegfile = BinarysignalIO(
                self.recinfo.eeg_filename,
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.eeg_sampling_rate,
            )
        except ValueError:
            print('self.recinfo.eeg_filename exists ({}) but file cannot be loaded in the appropriate format. Skipping. \n'.format(self.recinfo.eeg_filename))
            self.eegfile = None
        # else:
        #     self.eegfile = None


        if self.recinfo.dat_filename.is_file():
            self.datfile = BinarysignalIO(
                self.recinfo.dat_filename,
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.dat_sampling_rate,
            )
        else:
            self.datfile = None

        self.neurons = core.Neurons.from_file(fp.with_suffix(".neurons.npy"))
        self.probegroup = core.ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))
        self.position = core.Position.from_file(fp.with_suffix(".position.npy"))
        
        # self.paradigm = core.Epoch.from_file(fp.with_suffix(".paradigm.npy")) # "epoch" field of file
        self.epochs = core.Epoch.from_file(fp.with_suffix(".paradigm.npy")) # "epoch" field of file


        # Load or compute linear positions if needed:
        if (not self.position.has_linear_pos):
            # compute linear positions:
            print('computing linear positions for all active epochs for session...')
            # end result will be self.computed_traces of the same length as self.traces in terms of frames, with all non-maze times holding NaN values
            self.position.computed_traces = np.full([1, self.position.traces.shape[1]], np.nan)
            acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = ProcessData.compute_linearized_position(self, 'maze1')
            acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = ProcessData.compute_linearized_position(self, 'maze2')
            self.position.computed_traces[0,  acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            self.position.computed_traces[0,  acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
            self.position.filename = self.filePrefix.with_suffix(".position.npy")
            print('Saving updated position results to {}...'.format(self.position.filename))
            self.position.save()
            print('done.\n')
        else:
            print('linearized position loaded from file.')

        # Extended properties:
        
        ## Ripples:
        active_file_suffix = '.ripple.npy'
        found_datafile = core.DataWriter.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            self.ripple = core.Epoch.from_dict(found_datafile)
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            self.ripple = ProcessData.compute_neurons_ripples(self)

        ## MUA:
        active_file_suffix = '.mua.npy'
        found_datafile = core.DataWriter.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            self.mua = core.Mua.from_dict(found_datafile)
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            self.mua = ProcessData.compute_neurons_mua(self)

        ## PBE Epochs:
        active_file_suffix = '.pbe.npy'
        found_datafile = core.DataWriter.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            self.pbe = core.Epoch.from_dict(found_datafile)
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            self.pbe = ProcessData.compute_pbe_epochs(self)



    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})"


    # @staticmethod
    # def load_or_compute(session, load_filepath, fallback_compute_function):
    #     found_datafile = DataWriter.from_file(load_filepath)
    #     if found_datafile is not None:
    #         return Epoch.from_dict(found_datafile)
    #     else:
    #         # Otherwise load failed, perform the fallback computation
    #         return None

    # def update_active_epochs_linearized_positions(self, activeEpochLabelNames):
    #     print('computing linear positions for all active epochs for session...\n')
    #      # compute linear positions:
    #     # end result will be self.computed_traces of the same length as self.traces in terms of frames, with all non-maze times holding NaN values
    #     self.computed_traces = np.full([1, traces.shape[1]], np.nan)

    #     acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = ProcessData.compute_linearized_position(self, 'maze1')
    #     acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = ProcessData.compute_linearized_position(self, 'maze2')

    #     self.computed_traces[0,  acitve_epoch_timeslice_indicies1] = linearized_positions_maze1
    #     self.computed_traces[0,  acitve_epoch_timeslice_indicies2] = linearized_positions_maze2
    #     self.positions.filename = session.filePrefix.with_suffix('.position.npy')
    #     print('Saving updated position results to {}...'.format(ripple_epochs.filename))
    #     ripple_epochs.save()
    #     print('done.\n')


    ## Linearize Position:
    @staticmethod
    def compute_linearized_position(session, epochLabelName='maze1'):
        # returns core.Position objects for active_epoch_pos and linear_pos
        from neuropy.utils import position_util
        active_epoch_times = session.epochs[epochLabelName] # array([11070, 13970], dtype=int64)
        acitve_epoch_timeslice_indicies = session.position.time_slice_indicies(active_epoch_times[0], active_epoch_times[1])
        active_epoch_pos = session.position.time_slice(active_epoch_times[0], active_epoch_times[1])
        linear_pos = position_util.linearize_position(active_epoch_pos)
        return acitve_epoch_timeslice_indicies, active_epoch_pos, linear_pos

    #  acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = compute_linearized_position(sess, 'maze1')
    #  acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = compute_linearized_position(sess, 'maze2')



    ## Ripple epochs
    #To detect ripples one also needs probegroup.
    @staticmethod
    def compute_neurons_ripples(session):
        print('computing ripple epochs for session...\n')
        from neuropy.analyses import oscillations
        signal = session.eegfile.get_signal()
        ripple_epochs = oscillations.detect_ripple_epochs(signal, session.probegroup)
        ripple_epochs.filename = session.filePrefix.with_suffix('.ripple.npy')
        print('Saving ripple epochs results to {}...'.format(ripple_epochs.filename))
        ripple_epochs.save()
        print('done.\n')
        return ripple_epochs

    # sess.ripple = compute_neurons_ripples(sess)

    ## BinnedSpiketrain and Mua objects using Neurons
    @staticmethod
    def compute_neurons_mua(session):
        print('computing neurons mua for session...\n')
        mua = session.neurons.get_mua()
        mua.filename = session.filePrefix.with_suffix(".mua.npy")
        print('Saving mua results to {}...'.format(mua.filename))
        mua.save()
        print('done.\n')
        return mua    
    # sess.mua = compute_neurons_mua(sess) # Set the .mua field on the session object once complete

    @staticmethod
    def compute_pbe_epochs(session):
        from neuropy.analyses import detect_pbe_epochs
        print('computing PBE epochs for session...\n')
        smth_mua = session.mua.get_smoothed(sigma=0.02) # Get the smoothed mua from the session's mua
        pbe = detect_pbe_epochs(smth_mua)
        pbe.filename = session.filePrefix.with_suffix('.pbe.npy')
        print('Saving pbe results to {}...'.format(pbe.filename))
        pbe.save()
        print('done.\n')
        return pbe

    # sess.pbe = compute_pbe_epochs(sess)

# def ratN():
#     basepath='/data/Clustering/sessions/RatN_Day1_test_neuropy'
#     return ProcessData(basepath)

def processData(basedir='/Volumes/iNeo/Data/Bapun/Day5TwoNovel'):
    sess = ProcessData(basedir)
    return sess



if __name__ == "__main__":
    # Now initiate the class
    # basedir = '/data/Working/Opto/Jackie671/Jackie_placestim_day2/Jackie_TRACK_2020-10-07_11-21-39'  # fill in here
    basedir = 'R:\data\Bapun\Day5TwoNovel'
    # basedir = '/Volumes/iNeo/Data/Bapun/Day5TwoNovel'
    sess = processData(basedir)
    print(sess.recinfo)
    sess.epochs.to_dataframe()
    sess.neurons.get_all_spikes()
    sess.position.sampling_rate # 60
    
    pass
