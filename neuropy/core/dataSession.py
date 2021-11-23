import sys
import numpy as np
from pathlib import Path

# Local imports:
## Core:
from .datawriter import DataWriter
from .neurons import Neurons, BinnedSpiketrain, Mua
from .probe import ProbeGroup
from .position import Position
from .epoch import Epoch
from .signal import Signal

from ..io import NeuroscopeIO, BinarysignalIO # from neuropy.io import NeuroscopeIO, BinarysignalIO


class DataSessionLoader:
    def __init__(self, load_function, load_arguments=dict()):        
        self.load_function = load_function
        self.load_arguments = load_arguments
        
    def load(self, updated_load_arguments=None):
        if updated_load_arguments is not None:
            self.load_arguments = updated_load_arguments
                 
        return self.load_function(self.load_arguments)
    
    
    @staticmethod
    def default_load_bapun_npy_session_folder(args_dict):
        basepath = args_dict['basepath']
        session = args_dict['session_obj']
        
        basepath = Path(basepath)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        fp = xml_files[0].with_suffix("")
        session.filePrefix = fp
        session.recinfo = NeuroscopeIO(xml_files[0])

        # if session.recinfo.eeg_filename.is_file():
        try:
            session.eegfile = BinarysignalIO(
                session.recinfo.eeg_filename,
                n_channels=session.recinfo.n_channels,
                sampling_rate=session.recinfo.eeg_sampling_rate,
            )
        except ValueError:
            print('session.recinfo.eeg_filename exists ({}) but file cannot be loaded in the appropriate format. Skipping. \n'.format(session.recinfo.eeg_filename))
            session.eegfile = None
        # else:
        #     session.eegfile = None


        if session.recinfo.dat_filename.is_file():
            session.datfile = BinarysignalIO(
                session.recinfo.dat_filename,
                n_channels=session.recinfo.n_channels,
                sampling_rate=session.recinfo.dat_sampling_rate,
            )
        else:
            session.datfile = None

        session.neurons = Neurons.from_file(fp.with_suffix(".neurons.npy"))
        session.probegroup = ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))
        session.position = Position.from_file(fp.with_suffix(".position.npy"))
        
        # session.paradigm = Epoch.from_file(fp.with_suffix(".paradigm.npy")) # "epoch" field of file
        session.paradigm = Epoch.from_file(fp.with_suffix(".paradigm.npy"))
        session.epochs = session.paradigm # "epoch" is an alias for "paradigm". 

        # Load or compute linear positions if needed:
        if (not session.position.has_linear_pos):
            # compute linear positions:
            print('computing linear positions for all active epochs for session...')
            # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
            session.position.computed_traces = np.full([1, session.position.traces.shape[1]], np.nan)
            acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, 'maze1')
            acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = DataSession.compute_linearized_position(session, 'maze2')
            session.position.computed_traces[0,  acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            session.position.computed_traces[0,  acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
            session.position.filename = session.filePrefix.with_suffix(".position.npy")
            print('Saving updated position results to {}...'.format(session.position.filename))
            session.position.save()
            print('done.\n')
        else:
            print('linearized position loaded from file.')

        # Extended properties:
        
        ## Ripples:
        active_file_suffix = '.ripple.npy'
        found_datafile = DataWriter.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.ripple = Epoch.from_dict(found_datafile)
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.ripple = DataSession.compute_neurons_ripples(session)

        ## MUA:
        active_file_suffix = '.mua.npy'
        found_datafile = DataWriter.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.mua = Mua.from_dict(found_datafile)
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.mua = DataSession.compute_neurons_mua(session)

        ## PBE Epochs:
        active_file_suffix = '.pbe.npy'
        found_datafile = DataWriter.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.pbe = Epoch.from_dict(found_datafile)
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.pbe = DataSession.compute_pbe_epochs(session)

        return session # returns the session when done

        


class DataSession:
    def __init__(self, filePrefix = None, recinfo = None,
                 eegfile = None, datfile = None,
                 neurons = None, probegroup = None, position = None, paradigm = None,
                 ripple = None, mua = None):
        self.filePrefix = filePrefix
        self.recinfo = recinfo
        
        self.eegfile = eegfile
        self.datfile = datfile
        
        self.neurons = neurons
        self.probegroup = probegroup
        self.position = position
        self.paradigm = paradigm
        self.ripple = ripple
        self.mua = mua
        
        # curr_args_dict = dict()
        # curr_args_dict['basepath'] = basepath
        # curr_args_dict['session'] = DataSession()
        # DataSessionLoader.default_load_bapun_npy_session_folder(curr_args_dict)
        
        
    # def __init__(self, basepath):
    #     basepath = Path(basepath)
    #     xml_files = sorted(basepath.glob("*.xml"))
    #     assert len(xml_files) == 1, "Found more than one .xml file"


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})"


    ## Linearize Position:
    @staticmethod
    def compute_linearized_position(session, epochLabelName='maze1'):
        # returns Position objects for active_epoch_pos and linear_pos
        from neuropy.utils import position_util
        active_epoch_times = session.epochs[epochLabelName] # array([11070, 13970], dtype=int64)
        acitve_epoch_timeslice_indicies = session.position.time_slice_indicies(active_epoch_times[0], active_epoch_times[1])
        active_epoch_pos = session.position.time_slice(active_epoch_times[0], active_epoch_times[1])
        linear_pos = position_util.linearize_position(active_epoch_pos)
        return acitve_epoch_timeslice_indicies, active_epoch_pos, linear_pos
    #  acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = compute_linearized_position(sess, 'maze1')
    #  acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = compute_linearized_position(sess, 'maze2')

    ## Ripple epochs
    #   To detect ripples one also needs probegroup.
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



# Helper function that processed the data in a given directory
def processDataSession(basedir='/Volumes/iNeo/Data/Bapun/Day5TwoNovel'):
    # sess = DataSession(basedir)
    curr_args_dict = dict()
    curr_args_dict['basepath'] = basedir
    curr_args_dict['session_obj'] = DataSession() # Create an empty session object
    sess = DataSessionLoader.default_load_bapun_npy_session_folder(curr_args_dict)
    return sess


## Main:
if __name__ == "__main__":
    # Now initiate the class
    # basedir = '/data/Working/Opto/Jackie671/Jackie_placestim_day2/Jackie_TRACK_2020-10-07_11-21-39'  # fill in here
    basedir = 'R:\data\Bapun\Day5TwoNovel'
    # basedir = '/Volumes/iNeo/Data/Bapun/Day5TwoNovel'
    sess = processDataSession(basedir)
    print(sess.recinfo)
    sess.epochs.to_dataframe()
    sess.neurons.get_all_spikes()
    sess.position.sampling_rate # 60
    
    pass
