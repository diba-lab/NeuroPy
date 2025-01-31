# Standard Library Imports
from copy import deepcopy

# Package Imports
import h5py
import numpy as np
import pandas as pd
import tables as tb

# Module-Specific Imports
from joblib import Parallel, delayed
from scipy import stats
from typing import Sequence, Union
from ..utils.mathutil import gaussian_kernel1D
from scipy.ndimage import gaussian_filter1d

# Local Imports
from .datawriter import DataWriter
from . import Epoch
from .. import core

from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol

from neuropy.utils.mixins.concatenatable import ConcatenationInitializable
from neuropy.utils.mixins.HDF5_representable import HDF_SerializationMixin
from neuropy.core.neuron_identities import NeuronIdentityTable, neuronTypesEnum, NeuronType




class Neurons(HDF_SerializationMixin, NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, TimeSlicableObjectProtocol, ConcatenationInitializable, DataWriter):
    """Class to hold a group of spiketrains and their labels, ids etc."""

    def __init__(
        self,
        spiketrains: np.ndarray,
        t_stop,
        t_start=0.0,
        sampling_rate=1,
        neuron_ids=None,
        neuron_type=None,
        waveforms=None,
        waveforms_amplitude=None,
        peak_channels=None,
        clu_q=None,
        shank_ids=None,
        extended_neuron_properties_df=None,
        metadata=None,
    ) -> None:
        """Initializes the Neurons instance
         Parameters
         ----------
         spiketrains : np.array/list of numpy arrays
             each array contains spiketimes in seconds, 5 arrays for 5 neurons
         t_stop : float
             time when the recording was stopped
         t_start : float, optional
             start time for the recording/spike trains, by default 0.0
         sampling_rate : int, optional
             at what sampling rate the spike times were recorded, by default 1
         neuron_ids : array, optional
             id for each spiketrain/neuron, by default None
         neuron_type : array of strings, optional
             what neuron type, by default None
         waveforms : (n_neurons x n_channels x n_timepoints), optional
             waveshape for each neuron, by default None
         waveforms_amplitude : list/array of arrays, optional
             the number of arrays should match spiketrains, each value gives scaling factor used for template waveform to extract that spike, by default None
         peak_channels : array, optional
             peak channel for waveform, by default None
         shank_ids : array of int, optional
             which shank of the probe each spiketrain was recorded from, by default None
         metadata : dict, optional
             any additional metadata, by default None
         """
        super().__init__(metadata=metadata)

        self.spiketrains = np.array(spiketrains, dtype="object")
        self._neuron_ids = None
        self._reverse_cellID_index_map = None
        self._extended_neuron_properties_df = extended_neuron_properties_df
        if neuron_ids is None:
            self._neuron_ids = np.arange(len(self.spiketrains))
        else:
            if neuron_ids is int:
                neuron_ids = [neuron_ids] # if it's a single element, wrap it in a list.
            self._neuron_ids = pd.Series([int(cell_id) for cell_id in neuron_ids], name="neuron_id")
            
        self._reverse_cellID_index_map = Neurons.__build_cellID_reverse_lookup_map(self._neuron_ids.copy())
        
        if waveforms is not None:
            assert (
                waveforms.shape[0] == self.n_neurons
            ), "Waveforms first dimension should match number of neurons"


        if waveforms_amplitude is not None:
            assert len(waveforms_amplitude) == len(
                self.spiketrains
            ), "length should match"
            self.waveforms_amplitude = waveforms_amplitude
        else:
            self.waveforms_amplitude = None

        self.waveforms = waveforms
        self.shank_ids = shank_ids
        self.neuron_type = neuron_type
        self.peak_channels = peak_channels
        self._sampling_rate = sampling_rate
        self.t_start = t_start
        self.t_stop = t_stop
        self.clu_q = clu_q

    @staticmethod
    def load(file):
        """Loads a previously saved Neurons class from a .npy file
        """
        neurons_dict = DataWriter.from_file(file)
        return Neurons.from_dict(neurons_dict)

    @property
    def neuron_ids(self):
        """The neuron_ids property."""
        return self._neuron_ids

    @neuron_ids.setter
    def neuron_ids(self, value):
        """ ensures the indicies are integers and builds the reverse index map upon setting this value """
        if value is not None:
            flat_cell_ids = np.array([int(cell_id) for cell_id in value]) # ensures integer indexes for IDs
            self._reverse_cellID_index_map = Neurons.__build_cellID_reverse_lookup_map(flat_cell_ids)
            self._neuron_ids = flat_cell_ids
        else:
            self._reverse_cellID_index_map = None
            self._neuron_ids = None

    @property
    def reverse_cellID_index_map(self):
        """The reverse_cellID_index_map property: Allows reverse indexing into the linear imported array using the original cell ID indicies."""
        return self._reverse_cellID_index_map
    
    @staticmethod
    def __build_cellID_reverse_lookup_map(cell_ids):
        # Allows reverse indexing into the linear imported array using the original cell ID indicies
        flat_cell_ids = np.array([int(cell_id) for cell_id in cell_ids]) # ensures integer indexes for IDs
        linear_flitered_ids = np.arange(len(flat_cell_ids))
        return dict(zip(flat_cell_ids, linear_flitered_ids))

    @property
    def neuron_type(self):
        """The neuron_type property."""
        return self._neuron_type

    @neuron_type.setter
    def neuron_type(self, value):
        if value is not None:
            if value is int:
                value = [value] # if it's a single element, wrap it in a list.
            if len(value) > 0:
                # check to see if the neuron_type is the correct class (should be NeuronType) by checking the first element
                if isinstance(value[0], NeuronType):
                    # neuron_type already the correct type (np.array of NeuronType)
                    pass
                elif isinstance(value[0], str):
                    # neuron_type is a raw string type, so it needs to be converted
                    print('converting neuron_type strings to core.neurons.NeuronType objects...')
                    neuron_type_str = value
                    value = NeuronType.from_any_string_series(neuron_type_str) ## Works
                    print('\t done.')
                else:
                    print('ERROR: neuron_type value was of unknown type!')
                    raise NotImplementedError
        self._neuron_type = value

    @property
    def aclu_to_neuron_type_map(self):
        """ builds a map from the neuron_id to the neuron_type """
        return dict(zip(self.neuron_ids, self.neuron_type))

    def __getitem__(self, i):    
        # print('Neuron.__getitem__(i: {}): \n\t n_neurons: {}'.format(i, self.n_neurons))
        # copy object
        spiketrains = self.spiketrains[i]
        if self.neuron_type is not None:
            neuron_type = self.neuron_type[i]
        else:
            neuron_type = self.neuron_type

        if self.waveforms is not None:
            waveforms = self.waveforms[i]
        else:
            waveforms = self.waveforms

        if self.waveforms_amplitude is not None:
            waveforms_amplitude = self.waveforms_amplitude[i]
        else:
            waveforms_amplitude = self.waveforms_amplitude

        if self.peak_channels is not None:
            peak_channels = self.peak_channels[i]
        else:
            peak_channels = self.peak_channels

        if self.shank_ids is not None:
            shank_ids = self.shank_ids[i]
        else:
            shank_ids = self.shank_ids

        return Neurons(
            spiketrains=spiketrains,
            t_start=self.t_start,
            t_stop=self.t_stop,
            sampling_rate=self.sampling_rate,
            neuron_ids=self.neuron_ids[i],
            neuron_type=neuron_type,
            waveforms=waveforms,
            waveforms_amplitude=waveforms_amplitude,
            peak_channels=peak_channels,
            shank_ids=shank_ids,
        )

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def n_neurons(self):
        return len(self.spiketrains)

    def __repr__(self) -> str:
        try:
            neuron_types = np.unique(self.neuron_type)
        except TypeError:
            neuron_types = "Error importing - check inputs"
        return f"{self.__class__.__name__}\n n_neurons: {self.n_neurons}\n n_total_spikes: {self.n_total_spikes}\n t_start: {self.t_start}\n t_stop: {self.t_stop}\n neuron_type: {neuron_types}"

    def time_slice(self, t_start=None, t_stop=None, zero_spike_times=False):
        """zero_spike_times = True will subtract t_start from all spike times"""
        t_start, t_stop = super()._time_slice_params(t_start, t_stop)
        neurons = deepcopy(self)
        if zero_spike_times:
            spiketrains = [t[(t >= t_start) & (t <= t_stop)] - t_start for t in neurons.spiketrains]
            t_stop = t_stop - t_start
            t_start = 0
        else:
            spiketrains = [t[(t >= t_start) & (t <= t_stop)] for t in neurons.spiketrains]

        return Neurons(
            spiketrains=spiketrains,
            t_stop=t_stop,
            t_start=t_start,
            sampling_rate=neurons.sampling_rate,
            neuron_ids=neurons.neuron_ids,
            neuron_type=neurons.neuron_type,
            waveforms=neurons.waveforms,
            waveforms_amplitude=neurons.waveforms_amplitude,
            peak_channels=neurons.peak_channels,
            clu_q=neurons.clu_q,
            shank_ids=neurons.shank_ids,
        )

    def neuron_slice(self, neuron_inds=None, neuron_ids=None):
        neurons = deepcopy(self)

        if neuron_inds is not None and neuron_ids is not None:
            raise ValueError("Specify either neuron indexes or neuron ids, but not both.")

        # Handle selection of neuron indices
        if neuron_inds is not None:
            if isinstance(neuron_inds, int):
                neuron_inds = [neuron_inds]

            # Get list of original neuron indices
            all_neurons = np.array(neurons.neuron_ids.index)
            # Find the positional indices of original indexes
            positions = np.where(np.isin(all_neurons, neuron_inds))[0]

        # Handle selection by neuron ids
        elif neuron_ids is not None:
            if isinstance(neuron_ids, int):
                neuron_ids = [neuron_ids]

            # Find positions corresponding to neuron ids
            positions = self.neuron_ids[self.neuron_ids.isin(neuron_ids)].index.to_list()
            if len(positions) == 0:
                raise ValueError(f"No neurons found for give ids: {neuron_ids}")

        else:
            raise ValueError("Must specify either neuron_inds or neuron_ids.")

        # Get spiketrains from original neuron index
        spiketrains = neurons.spiketrains[positions]

        # Get waveforms, peak channels, shank ids, from original neuron index
        neuron_type = (None if neurons.neuron_type is None else neurons.neuron_type[positions])
        waveforms = (None if neurons.waveforms is None else neurons.waveforms[positions])
        waveforms_amplitude = (None if neurons.waveforms_amplitude is None else neurons.waveforms_amplitude[positions])
        peak_channels = (None if neurons.peak_channels is None else neurons.peak_channels[positions])
        shank_ids = (None if neurons.shank_ids is None else neurons.shank_ids[positions])
        clu_q = (None if neurons.clu_q is None else neurons.clu_q[positions])

        # neuron_ids and neuron_type stay maintained because neuron_inds points to original indices.
        return Neurons(
            spiketrains=spiketrains,
            t_stop=neurons.t_stop,
            t_start=neurons.t_start,
            sampling_rate=neurons.sampling_rate,
            neuron_ids=neurons.neuron_ids.loc[positions],  # Keep correct neuron IDs
            neuron_type=neuron_type,  # Keep correct neuron types
            waveforms=waveforms,
            waveforms_amplitude=waveforms_amplitude,
            peak_channels=peak_channels,
            clu_q=clu_q,
            shank_ids=shank_ids,
        )

    def concatenate(self, neurons_to_add, index_to_add=0):
        """Add two neuron spike trains together. Adds 'index_to_add' to neuron_ids, shank_ids, and peak_channels
        to help differentiate different sessions (e.g. index_to_add=100 will make the cluster_ids from
        neurons_to_add be 101, 102, 103... """
        t_start = np.min((self.t_start, neurons_to_add.t_start))
        t_stop = np.max((self.t_stop, neurons_to_add.t_stop))

        # Check to make sure everything is compatible
        assert self.sampling_rate == neurons_to_add.sampling_rate
        feature_dict = {}
        for feature in ["spiketrains", "neuron_ids", "neuron_type", "waveforms",
                        "peak_channels", "shank_ids"]:
            print(f"{feature} with kind={getattr(self, feature).dtype.kind}")
            # try:
            if feature in ["spiketrains", "neuron_type", "waveforms"]:
                feature_dict[feature] = np.concatenate((getattr(self, feature),
                                                        getattr(neurons_to_add, feature)),
                                                        axis=0)
            else:  # only add to id related fields

                feature_dict[feature] = np.concatenate((getattr(self, feature),
                                                        getattr(neurons_to_add, feature) + index_to_add),
                                                       axis=0)

        return Neurons(spiketrains=feature_dict["spiketrains"],
                       t_start=t_start,
                       t_stop=t_stop,
                       sampling_rate=self.sampling_rate,
                       neuron_ids=feature_dict["neuron_ids"],
                       neuron_type=feature_dict["neuron_type"],
                       waveforms=feature_dict["waveforms"],
                       waveforms_amplitude=feature_dict["waveforms_amplitude"],
                       peak_channels=feature_dict["peak_channels"],
                       clu_q=feature_dict["clu_q"],
                       shank_ids=feature_dict["shank_ids"])

    def get_neuron_type(self, query_neuron_type):
        """ filters self by the specified query_neuron_type, only returning neurons that match. """
        if isinstance(query_neuron_type, NeuronType):
            query_neuron_type = query_neuron_type
        elif isinstance(query_neuron_type, str):
            query_neuron_type_str = query_neuron_type
            query_neuron_type = NeuronType.from_string(query_neuron_type_str) ## Works
        else:
            print('error!')
            return []
            
        indices = self.neuron_type == query_neuron_type
        return self[indices]

    def _check_integrity(self):
        assert isinstance(self.spiketrains, np.ndarray)

    def __len__(self):
        return self.n_neurons

    def add_metadata(self):
        pass

    def get_all_spikes(self):
        return np.concatenate(self.spiketrains)
    
    @property
    def n_total_spikes(self):
        return np.sum(self.n_spikes)
        
    @property
    def n_spikes(self):
        "number of spikes within each spiketrain"
        return np.asarray([len(_) for _ in self.spiketrains])

    @property
    def firing_rate(self):
        """Return average firing rate for each neuron over the entire duration of the recording """
        return self.n_spikes / (self.t_stop - self.t_start)

    def get_above_firing_rate(self, thresh: float):
        """Return neurons which have firing rate above thresh (on average, for the entire duration of the recording) """
        indices = self.firing_rate > thresh
        return self[indices]

    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids):
        """Returns neurons object with neuron_ids equal to ids"""
        indices = np.isin(self.neuron_ids, ids)
        return self[indices] # TODO 2023-01-01: should this use 'safe_pandas_get_group' to ensure that an empty dataframe is returned if no neurons are found?

    def get_isi(self, bin_size=0.001, n_bins=200):
        """Interspike interval

        Parameters
        ----------
        bin_size : float, optional
            [description], by default 0.001
        n_bins : int, optional
            [description], by default 200

        Returns
        -------
        [type]
            [description]
        """
        bins = np.arange(n_bins + 1) * bin_size
        return np.asarray(
            [np.histogram(np.diff(spktrn), bins=bins)[0] for spktrn in self.spiketrains]
        )

    def get_waveform_similarity(self):
        waveforms = np.reshape(self.waveforms, (self.n_neurons, -1)).astype(float)
        similarity = np.corrcoef(waveforms)
        np.fill_diagonal(similarity, 0)
        return similarity

    def get_binned_spiketrains(self, bin_size=0.25, ignore_epochs: Epoch = None):
        """Get binned spike counts

        Parameters
        ----------
        bin_size : float, optional
            bin size in seconds, by default 0.25

        Returns
        -------
        neuropy.core.BinnedSpiketrains

        """
        duration = self.t_stop - self.t_start
        n_bins = np.floor(duration / bin_size)
        # bins = np.arange(self.t_start, self.t_stop + bin_size, bin_size)
        bins = np.arange(n_bins + 1) * bin_size + self.t_start
        spike_counts = np.asarray(
            [np.histogram(_, bins=bins)[0] for _ in self.spiketrains]
        ).astype("float")
        if ignore_epochs is not None:
            ignore_bins = ignore_epochs.flatten()
            ignore_indices = np.digitize(bins[:-1], ignore_bins) % 2 == 1
            spike_counts[:, ignore_indices] = np.nan

        return BinnedSpiketrain(
            spike_counts,
            t_start=self.t_start,
            bin_size=bin_size,
            neuron_ids=self.neuron_ids,
            peak_channels=self.peak_channels,
            shank_ids=self.shank_ids,
        )

    def get_mua(self, bin_size=0.001):
        """Get mua between two time points

        Parameters
        ----------
        bin_size : float, optional
            [description], by default 0.001

        Returns
        -------
        MUA object
            [description]
        """

        spks = self.get_all_spikes()
        bins = np.arange(self.t_start, self.t_stop, bin_size)
        # spike_counts = np.histogram(all_spikes, bins=bins)[0] # super slow
        counts = stats.binned_statistic(spks, spks, bins=bins, statistic="count")[0]
        return Mua(counts.astype("int"), t_start=self.t_start, bin_size=bin_size)

    def get_psth(self, t: np.array, bin_size: float, n_bins: int, n_jobs=1):
        """Get peristimulus time histograms with respect to timepoints

        Parameters
        ----------
        t : np.array
            timepoints around which psths are computed, in seconds
        bin_size : float
            binsize in seconds
        n_bins : int
            number of bins before/after the timepoints, total number of bins= 2*n_bins
        n_jobs : int, optional
            number of cpus to speed up calculations, by default 1

        Returns
        -------
        psths: shape(n_neurons, 2*n_bins, len(t))
            number of spikes for each neuron around each timepoint
        """
        n_bins_around = 2 * n_bins
        n_t = len(t)
        bins = np.linspace(-n_bins, n_bins, n_bins_around + 1) * bin_size
        t_bins = np.tile(bins, len(t)) + np.repeat(t, n_bins_around + 1)

        def get_counts(spiketimes):
            indx_right = np.searchsorted(spiketimes, t_bins[:-1], side="right")
            indx_left = np.searchsorted(spiketimes, t_bins[1:], side="left")
            # count the number of spikes and skip time bins that represent bins between adjacent time points
            counts_in_bins = np.delete(
                indx_left - indx_right,
                np.arange(n_bins_around, indx_left.size, n_bins_around + 1),
            )
            return counts_in_bins.reshape(1, n_bins_around, n_t)

        psths = Parallel(n_jobs=n_jobs)(
            delayed(get_counts)(_) for _ in self.spiketrains
        )

        return np.vstack(psths)

    def add_jitter(self):
        pass

    def get_neurons_in_epochs(self, epochs: Epoch):
        """Remove spikes that lie outside of given epochs and return a new Neurons object with t_start and t_stop changed to start of first epoch and stop of last epoch.

        Parameters
        ----------
        epochs : Epoch
            epochs defining starts and stops
        """
        assert epochs.is_overlapping == False, "epochs should be non-overlapping"
        spktrns = self.spiketrains
        epochs_bins = epochs.flatten()

        new_spktrns = []
        for spktrn in spktrns:
            bin_loc = np.digitize(spktrn, epochs_bins)
            new_spktrns.append(spktrn[bin_loc % 2 == 1])

        new_spktrns = np.array(new_spktrns, dtype="object")

        return Neurons(
            spiketrains=new_spktrns,
            t_start=epochs.starts[0],
            t_stop=epochs.stops[-1],
            sampling_rate=self.sampling_rate,
            neuron_ids=self.neuron_ids,
            neuron_type=self.neuron_type,
            waveforms=self.waveforms,
            waveforms_amplitude=self.waveforms_amplitude,
            peak_channels=self.peak_channels,
            clu_q=self.clu_q,
            shank_ids=self.shank_ids,
        )

    def get_modulation_in_epochs(self, epochs: Epoch, n_bins):
        """Total number of across all epochs where each epoch is divided into equal number of bins

        Parameters
        ----------
        epochs : Epoch
            epochs for calculation
        n_bins : int
            number of bins to divide each epoch

        Returns
        -------
        2d array: n_neurons x n_bins
            total number of spikes within each bin across all epochs
        """
        assert epochs.is_overlapping == False, "epochs should be non-overlapping"
        assert isinstance(n_bins, int), "n_bins can only be integer"
        starts = epochs.starts.reshape(-1, 1)
        bin_size = (epochs.durations / n_bins).reshape(-1, 1)

        # create 2D-array (n_epochs x n_bins+1) with bin_size spacing along columns
        bins = np.arange(n_bins + 1) * bin_size

        epoch_bins = (starts + bins).flatten()

        # calculate spikes on flattened epochs and delete bins which represent spike counts between (not within) epochs and then sums across all epochs for each bin
        counts = [
            np.delete(
                np.histogram(_, epoch_bins)[0],
                np.arange(n_bins, epoch_bins.size, n_bins + 1)[:-1],
            )
            .reshape(-1, n_bins)
            .sum(axis=0)
            for _ in self.spiketrains
        ]

        return np.asarray(counts)

    def get_spikes_in_epochs(
            self, epochs: Epoch, bin_size=0.01, slideby=None, sigma=None
    ):
        """A list of 2D arrays containing spike counts

        Parameters
        ----------
        epochs : Epoch
            start and stop times of epochs
        bin_size : float, optional
            bin size to be used to within each epoch, by default 0.01
        slideby : [type], optional
            if spike counts should have sliding window, by default None
        sigma: float, optional
            standard deviation for gaussian kernel used for smoothing in seconds, by default None

        Returns
        -------
        spkcount, nbins
            list of arrays, number of bins within each epoch
        """
        spkcount = []
        nbins = np.zeros(epochs.n_epochs, dtype="int")

        # ----- little faster but requires epochs to be non-overlapping ------

        if (~epochs.is_overlapping) and (slideby is None):
            bins_epochs = []
            for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
                bins = np.arange(epoch.start, epoch.stop, bin_size)
                nbins[i] = len(bins) - 1
                bins_epochs.extend(bins)
            spkcount = np.asarray(
                [np.histogram(_, bins=bins_epochs)[0] for _ in self.spiketrains]
            )

            # deleting unwanted columns that represent time between events
            cumsum_nbins = np.cumsum(nbins)
            del_columns = cumsum_nbins[:-1] + np.arange(len(cumsum_nbins) - 1)
            spkcount = np.delete(spkcount, del_columns.astype(int), axis=1)
            spkcount = np.hsplit(spkcount, cumsum_nbins[:-1])

        else:
            if slideby is None:
                slideby = bin_size
            for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
                # first dividing in 1ms
                bins = np.arange(epoch.start, epoch.stop, 0.001)
                spkcount_ = np.asarray(
                    [np.histogram(_, bins=bins)[0] for _ in self.spiketrains]
                )

                # if signficant portion at end of epoch is not included then append zeros
                # if (frac := epoch.duration / bin_size % 1) > 0.7:
                #     extra_columns = int(100 * (1 - frac))
                #     spkcount_ = np.hstack(
                #         (spkcount_, np.zeros((neurons.n_neurons, extra_columns)))
                #     )

                slide_view = np.lib.stride_tricks.sliding_window_view(
                    spkcount_, int(bin_size * 1000), axis=1
                )[:, :: int(slideby * 1000), :].sum(axis=2)

                nbins[i] = slide_view.shape[1]
                spkcount.append(slide_view)

        if sigma is not None:
            kernel = gaussian_kernel1D(sigma=sigma, bin_size=bin_size)
            spkcount = [
                np.apply_along_axis(np.convolve, arr=_, v=kernel, mode="same", axis=1)
                for _ in spkcount
            ]

        return spkcount, nbins

    # DictionaryRepresentable Protocol:
    def to_dict(self, recurrsively=False):

        # self._check_integrity()

        return {
            "spiketrains": self.spiketrains,
            "t_stop": self.t_stop,
            "t_start": self.t_start,
            "sampling_rate": self.sampling_rate,
            "neuron_ids": self.neuron_ids,
            "neuron_type": self.neuron_type,
            "waveforms": self.waveforms,
            "waveforms_amplitude": self.waveforms_amplitude,
            "peak_channels": self.peak_channels,
            "clu_q": self.clu_q,
            "shank_ids": self.shank_ids,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d):
        return Neurons(
            d["spiketrains"],
            d["t_stop"],
            d["t_start"],
            d["sampling_rate"],
            d["neuron_ids"],
            d["neuron_type"],
            d["waveforms"],
            d["waveforms_amplitude"],
            d["peak_channels"],
            d["clu_q"],
            shank_ids=d["shank_ids"],
            metadata=d["metadata"],
        )

    def to_dataframe(self):
        df = self._spikes_df.copy()
        return df
        
    @classmethod
    def initialize_missing_spikes_df_columns(cls, spikes_df, debug_print=False):
        """ make sure the needed columns exist on spikes_df """
        if ('shank' not in spikes_df.columns):
            if debug_print:
                print('dataframe shank column does not exist. Initializing it to 1s')
            spikes_df['shank'] = 1
            
        if ('qclu' not in spikes_df.columns):
            if debug_print:
                print('dataframe qclu column does not exist. Initializing it to the same as aclu')
            spikes_df['qclu'] = 111 # spikes_df['aclu']
            
        if ('cluster' not in spikes_df.columns):
            if debug_print:
                print('dataframe cluster column does not exist. Initializing it to the same as aclu')
            spikes_df['cluster'] = 111 # spikes_df['aclu']

    @classmethod
    def from_dataframe(cls, spikes_df, dat_sampling_rate, time_variable_name='t_rel_seconds'):
        """ Builds a Neurons object from a spikes_df, such as the one belonging to its complementary FlattenedSpiketrains:
            Usage:
                neurons_obj = Neurons.from_dataframe(sess.flattened_spiketrains.spikes_df, sess.recinfo.dat_sampling_rate, time_variable_name='t_rel_seconds') 
        """
        ## Get unique cell ids to enable grouping flattened results by cell:
        unique_cell_ids = np.unique(spikes_df['aclu'])
        flat_cell_ids = [int(cell_id) for cell_id in unique_cell_ids]
        num_unique_cell_ids = len(flat_cell_ids)
        # print('flat_cell_ids: {}'.format(flat_cell_ids))
        # Group by the aclu (cluster indicator) column
        cell_grouped_spikes_df = spikes_df.groupby(['aclu'])
        spiketrains = list()
        shank_ids = np.zeros([num_unique_cell_ids, ]) # (108,) Array of float64
        cell_quality = np.zeros([num_unique_cell_ids, ]) # (108,) Array of float64
        neuron_type = list() # (108,) Array of float64
                
        for i in np.arange(num_unique_cell_ids):
            curr_cell_id = flat_cell_ids[i] # actual cell ID
            #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
            curr_cell_dataframe = cell_grouped_spikes_df.get_group(curr_cell_id)
            spiketrains.append(curr_cell_dataframe[time_variable_name].to_numpy())
            
            shank_ids[i] = curr_cell_dataframe['shank'].to_numpy()[0] # get the first shank identifier, which should be the same for all of this curr_cell_id
            cell_quality[i] = curr_cell_dataframe['qclu'].mean() # should be the same for all instances of curr_cell_id, but use mean just to make sure
            neuron_type.append(curr_cell_dataframe['neuron_type'].to_numpy()[0])

        spiketrains = np.array(spiketrains, dtype='object')
        t_stop = np.max(spikes_df[time_variable_name])
        flat_cell_ids = np.array(flat_cell_ids)
        neuron_type = np.array(neuron_type)
        out_neurons = Neurons(spiketrains, t_stop, t_start=0,
            sampling_rate=dat_sampling_rate,
            neuron_ids=flat_cell_ids,
            neuron_type=neuron_type,
            shank_ids=shank_ids
        )
        return out_neurons
                       
    # ConcatenationInitializable protocol:
    @classmethod
    def concat(cls, objList: Union[Sequence, np.array]):
        """ Concatenates the object list along the time axis """
        # objList = np.array(objList)
        t_start_times = np.array([obj.t_start for obj in objList])
        
        num_neurons_list = np.array([obj.n_neurons for obj in objList])
        #  test if all num_neurons are equal 
        assert np.array_equal(num_neurons_list, np.full_like(num_neurons_list, num_neurons_list[0])), " All objects must have the same number of neurons to be concatenated. The concatenation only occurs in respect to time."
        num_neurons = num_neurons_list[0]
        sort_idx = list(np.argsort(t_start_times))
        # print(sort_idx)
        # sort the objList by t_start
        # objList = objList[sort_idx]
        
        objList = [objList[i] for i in sort_idx]
        
        new_neuron_ids = objList[0].neuron_ids
        
        # Concatenate the elements:
        # spiketrains_list = np.concatenate([obj.spiketrains for obj in objList], axis=1)
        
        # spiketrains_list = np.hstack([obj.spiketrains for obj in objList])
        # spiketrains_list = list()
        
        # for neuron_idx in np.arange(num_neurons):
        #     curr_neuron_spiketrains_list = np.concatenate([obj.spiketrains[neuron_idx] for obj in objList], axis=0)
        #     spiketrains_list.append(curr_neuron_spiketrains_list)
            
        spiketrains_list = objList[0].spiketrains
        for neuron_idx in np.arange(num_neurons):
            for obj_idx in np.arange(1, len(objList)):
                # spiketrains_list[neuron_idx].append(objList[obj_idx].spiketrains[neuron_idx])
                spiketrains_list[neuron_idx] = np.append(spiketrains_list[neuron_idx], objList[obj_idx].spiketrains[neuron_idx]).astype(np.float64)
                
            # for obj_idx in np.arange(len(objList)):
            # # spiketrains_list[i] =  np.concatenate([obj.spiketrains for obj in objList], axis=0)
            # spiketrains_list.append(np.concatenate([obj.spiketrains[neuron_idx] for neuron_idx in np.arange(num_neurons)], axis=0))
        
        return Neurons(
            spiketrains=spiketrains_list,
            t_stop=objList[-1].t_stop,
            t_start=objList[0].t_start,
            sampling_rate=objList[0].sampling_rate,
            neuron_ids=new_neuron_ids,
            neuron_type=objList[0].neuron_type,
            waveforms=objList[0].waveforms,
            waveforms_amplitude=objList[0].waveforms_amplitude,
            peak_channels=objList[0].peak_channels,
            clu_q=objList[0].clu_q,
            shank_ids=objList[0].shank_ids,
            metadata=objList[0].metadata
        )

    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, session_uid:str="test_session_uid", **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
        """
        
        ## Serialize the dataframe:
        raise NotImplementedError # 2023-08-02 - This is complete except for the fact that for Diba sessions it doesn't have a spikes_df because it is computed from one unlike the other sessions where it is loaded from one.
    
        df_representation = self.to_dataframe()
        df_representation.spikes.to_hdf(file_path, key=f'{key}/spikes')
        

        unique_rows_df = df_representation.spikes.extract_unique_neuron_identities()
        # Extract the selected columns as NumPy arrays
        aclu_array = unique_rows_df['aclu'].values
        shank_array = unique_rows_df['shank'].values
        cluster_array = unique_rows_df['cluster'].values
        qclu_array = unique_rows_df['qclu'].values
        neuron_type_array = unique_rows_df['neuron_type'].values
        neuron_types_enum_array = np.array([neuronTypesEnum[a_type.hdfcodingClassName] for a_type in neuron_type_array]) # convert NeuronTypes to neuronTypesEnum



        # self.spiketrains = np.array(spiketrains, dtype="object")

        # self._extended_neuron_properties_df = extended_neuron_properties_df
        
        # h5f = tables.open_file('enum.h5', 'w')

        assert self.waveforms is None, f"waveforms are not HDF serialized and will be LOST"
        assert self._extended_neuron_properties_df is None, f"self._extended_neuron_properties_df are not yet serializable!"

        # Open the file with h5py to add attributes to the group. The pandas.HDFStore object doesn't provide a direct way to manipulate groups as objects, as it is primarily intended to work with datasets (i.e., pandas DataFrames)
        # with h5py.File(file_path, 'r+') as f:
        with tb.open_file(file_path, mode='r+') as f:
            
            # f.create_dataset(f'{key}/neuron_ids', data=self.neuron_ids)
            # f.create_dataset(f'{key}/shank_ids', data=self.shank_ids)
            if self.peak_channels is not None:
                f.create_dataset(f'{key}/peak_channels', data=self.peak_channels)

            ## Unfortunately, you cannot directly assign a dictionary to the attrs attribute of an h5py group or dataset. The attrs attribute is an instance of a special class that behaves like a dictionary in some ways but not in others. You must assign attributes individually
            group = f[key]

            table = f.create_table(group, 'table', NeuronIdentityTable, "Neuron identities")

            # Serialization
            row = table.row
            for i in np.arange(self.n_neurons):
                ## Build the row here from aclu_array, etc
                row['neuron_uid'] = f"{session_uid}-{aclu_array[i]}"
                row['session_uid'] = session_uid  # Provide an appropriate session identifier here
                row['neuron_id'] = aclu_array[i]
                row['neuron_type'] = neuron_types_enum_array[i]
                row['shank_index'] = shank_array[i]
                row['cluster_index'] = cluster_array[i] # self.peak_channels[i]
                row['qclu'] = qclu_array[i]  # Replace with appropriate value if available                
                row.append()
                
            table.flush()
            
            # Metadata:
            group.attrs['dat_sampling_rate'] = self.sampling_rate
            group.attrs['t_start'] = self.t_start
            group.attrs['t_start'] = self.t_start
            group.attrs['t_stop'] = self.t_stop
            group.attrs['n_neurons'] = self.n_neurons



    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> "Neurons":
        """ Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pfnd_obj = PfND.read_hdf(hdf5_output_path, key='test_pfnd')
            _reread_pfnd_obj


        # Neurons.__init__( self, spiketrains: np.ndarray, t_stop, t_start=0.0, sampling_rate=1, neuron_ids=None, neuron_type=None, waveforms=None, peak_channels=None, shank_ids=None, extended_neuron_properties_df=None, metadata=None, )
        """
        # Read DataFrames using pandas
        raise NotImplementedError
        position = Position.read_hdf(file_path, key=f'{key}/pos')
        try:
            epochs = Epoch.read_hdf(file_path, key=f'{key}/epochs')
        except KeyError as e:
            # epochs can be None, in which case the serialized object will not contain the f'{key}/epochs' key.  'No object named test_pfnd/epochs in the file'
            epochs = None
        except Exception as e:
            # epochs can be None, in which case the serialized object will not contain the f'{key}/epochs' key
            print(f'Unhandled exception {e}')
            raise e
        
        spikes_df = SpikesAccessor.read_hdf(file_path, key=f'{key}/spikes')

        # Open the file with h5py to read attributes
        with h5py.File(file_path, 'r') as f:
            group = f[key]
            position_srate = group.attrs['position_srate']
            ndim = group.attrs['ndim'] # Assuming you'll use it somewhere else if needed

            # Read the config attributes
            config_dict = {
                'speed_thresh': group.attrs['config/speed_thresh'],
                'grid_bin': tuple(group.attrs['config/grid_bin']),
                'grid_bin_bounds': tuple(group.attrs['config/grid_bin_bounds']),
                'smooth': tuple(group.attrs['config/smooth']),
                'frate_thresh': group.attrs['config/frate_thresh']
            }

        # Create a PlacefieldComputationParameters object from the config_dict
        config = PlacefieldComputationParameters(**config_dict)

        # Reconstruct the object using the from_config_values class method
        return cls(spikes_df=spikes_df, position=position, epochs=epochs, config=config, position_srate=position_srate)


class BinnedSpiketrain(NeuronUnitSlicableObjectProtocol, DataWriter):
    """Class to hold binned spiketrains"""

    def __init__(
        self,
        spike_counts: np.ndarray,
        bin_size: float,
        t_start=0.0,
        neuron_ids=None,
        peak_channels=None,
        shank_ids=None,
        metadata=None,
    ) -> None:
        super().__init__()
        self.spike_counts = spike_counts
        self.bin_size = bin_size
        self.t_start = t_start
        self.peak_channels = peak_channels
        self.shank_ids = shank_ids
        if neuron_ids is None:
            self.neuron_ids = np.arange(self.n_neurons)

        self.metadata = metadata

    @staticmethod
    def from_neurons(neurons: Neurons, t_start=None, t_stop=None, bin_size=0.25):
        pass

    @property
    def spike_counts(self):
        return self._spike_counts

    @spike_counts.setter
    def spike_counts(self, arr):
        self._spike_counts = arr

    @property
    def n_neurons(self):
        return self.spike_counts.shape[0]

    @property
    def n_bins(self):
        return self.spike_counts.shape[1]

    @property
    def duration(self):
        return self.n_bins * self.bin_size

    @property
    def t_stop(self):
        return self.t_start + self.duration

    def add_metadata(self):
        pass

    @property
    def time(self):
        return np.arange(self.n_bins) * self.bin_size + self.t_start

    def _get_nan_bins(self):
        return np.isnan(self.spike_counts).any(axis=0)

    def to_dict(self, recurrsively=False):
        return {
            "spike_counts": self.spike_counts,
            "t_start": self.t_start,
            "bin_size": self.bin_size,
            "neuron_ids": self.neuron_ids,
            "peak_channels": self.peak_channels,
            "shank_ids": self.shank_ids,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d):
        return BinnedSpiketrain(
            spike_counts=d["spike_counts"],
            bin_size=d["bin_size"],
            t_start=d["t_start"],
            neuron_ids=d["neuron_ids"],
            peak_channels=d["peak_channels"],
            shank_ids=d["shank_ids"],
            metadata=d["metadata"],
        )

    def get_pairwise_corr(self, pairs_bool=None, return_pair_id=False):
        """Pairwise correlation between pairs of binned of spiketrains

        Parameters
        ----------
        pairs_bool : 2D bool/logical array, optional
            Only these pairs are returned, by default None which means all pairs
        return_pair_id : bool, optional
            If true pair_ids are returned, by default False

        Returns
        -------
        corr
            1d vector of pairwise correlations
        """

        assert self.n_neurons > 1, "Should have more than 1 neuron"
        corr = np.corrcoef(self.spike_counts[:, ~self._get_nan_bins()])

        if pairs_bool is not None:
            assert (
                pairs_bool.shape[0] == pairs_bool.shape[1]
            ), "pairs_bool should be sqare shpae"
            assert (
                pairs_bool.shape[0] == self.n_neurons
            ), f"pairs_bool should be of {corr.shape} shape"
            pairs_bool = pairs_bool.astype("bool")
        else:
            pairs_bool = np.ones(corr.shape).astype("bool")

        pairs_bool = np.tril(pairs_bool, k=-1)

        return corr[pairs_bool]

    @property
    def firing_rate(self):
        return self.spike_counts / self.bin_size

    def __getitem__(self, i):
        # copy object
        spike_counts = self.spike_counts[i]
        if self.peak_channels is not None:
            peak_channels = self.peak_channels[i]
        else:
            peak_channels = self.peak_channels

        if self.shank_ids is not None:
            shank_ids = self.shank_ids[i]
        else:
            shank_ids = self.shank_ids

        return BinnedSpiketrain(
            spike_counts=spike_counts,
            bin_size=self.bin_size,
            t_start=self.t_start,
            neuron_ids=self.neuron_ids[i],
            peak_channels=peak_channels,
            shank_ids=shank_ids,
        )

    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids):
        """Returns BinnedSpiketrain object with neuron_ids equal to ids"""
        indices = np.isin(self.neuron_ids, ids)
        return self[indices]

    
class Mua(DataWriter):
    """Multi-Unit Activity"""
    def __init__(
        self,
        spike_counts: np.ndarray,
        bin_size: float,
        t_start: float = 0.0,
        metadata=None,
    ) -> None:
        super().__init__()
        self.spike_counts = spike_counts
        self.t_start = t_start
        self.bin_size = bin_size
        self.metadata = metadata

    @property
    def spike_counts(self):
        return self._spike_counts
    @spike_counts.setter
    def spike_counts(self, arr: np.ndarray):
        assert arr.ndim == 1, "only 1 dimensional arrays are allowed"
        self._spike_counts = arr

    @property
    def bin_size(self):
        return self._bin_size
    @bin_size.setter
    def bin_size(self, val):
        self._bin_size = val

    @property
    def n_bins(self):
        return len(self._spike_counts)

    @property
    def duration(self):
        return self.n_bins * self.bin_size

    @property
    def t_stop(self):
        return self.t_start + self.duration

    @property
    def time(self):
        return np.arange(self.n_bins) * self.bin_size + self.t_start

    @property
    def firing_rate(self):
        return self.spike_counts / self.bin_size

    def get_smoothed(self, sigma=0.02, **kwargs):
        """Smoothing of mua spike counts

        Parameters
        ----------
        sigma : float, optional
            gaussian kernel in seconds, by default 0.02 s (20 milliseconds)
        kwargs : float, optional
            keyword arguments for scipy.ndimage.gaussian_filter1d, by default 4.0

        Returns
        -------
        core.MUA object
            containing smoothed spike counts
        """

        dt = self.bin_size
        spike_counts = gaussian_filter1d(
            self.spike_counts, sigma=sigma / dt, output="float", **kwargs
        )
        return Mua(spike_counts, t_start=self.t_start, bin_size=self.bin_size)

    def time_slice(self, t_start, t_stop):
        indices = (self.time >= t_start) & (self.time <= t_stop)

        return Mua(
            spike_counts=self.spike_counts[indices],
            bin_size=self.bin_size,
            t_start=t_start,
        )

    def to_dataframe(self):
        return pd.DataFrame({"time": self.time, "spike_counts": self.spike_counts})


def pe_raster(
    neurons: Neurons,
    neuron_id: int,
    event_times: np.ndarray or list,
    buffer_sec=(5, 5),
):
    """Get peri-event raster of spike times"""
    spiketrain = neurons.spiketrains[neuron_id]
    rast = []
    for event_time in event_times:
        time_bool = (spiketrain > (event_time - buffer_sec[0])) & (
            spiketrain <= (event_time + buffer_sec[1])
        )
        rast.append(spiketrain[time_bool] - event_time)

    return Neurons(rast, t_stop=buffer_sec[1], t_start=-buffer_sec[0])


def binned_pe_raster(
    binned_spiketrain: (BinnedSpiketrain, Mua),
    event_times: np.ndarray or list,
    neuron_id: int = 0,
    buffer_sec=(5, 5),
):
    """Build a peri-event raster for a binned spiketrain or MUA. neuron_id only needed for
    binned_spiketrain class."""

    if isinstance(binned_spiketrain, BinnedSpiketrain):
        binned_fr = binned_spiketrain.firing_rate[neuron_id]
    elif isinstance(binned_spiketrain, Mua):
        binned_fr = binned_spiketrain.firing_rate

    firing_rate = []
    for event_time in event_times:
        time_bool = (binned_spiketrain.time > (event_time - buffer_sec[0])) & (
            binned_spiketrain.time
            <= (event_time + buffer_sec[1] + binned_spiketrain.bin_size * 0.5)
        )
        firing_rate.append(binned_fr[time_bool])

    fr_len = [len(f) for f in firing_rate]
    if np.max(fr_len) == np.min(fr_len):
        fr_array = np.array(firing_rate)
    elif (
        np.max(fr_len) - np.min(fr_len)
    ) == 1:  # append a 0 firing rate to last bin of any short
        for id in np.where(fr_len == np.min(fr_len))[0]:
            firing_rate[id] = np.append(firing_rate[id], np.nan)
        fr_array = np.array(firing_rate)
    else:
        fr_array = np.nan
        print(
            "Raster has uneven # of bins in one row, likely due to edge effects. Fix code or delete start/end event from input"
        )

    pe_times = np.arange(
        -buffer_sec[0],
        buffer_sec[1] + binned_spiketrain.bin_size * 0.5,
        binned_spiketrain.bin_size,
    )

    return fr_array, pe_times