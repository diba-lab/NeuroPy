from typing import Sequence, Union
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg

from neuropy.utils.mixins.print_helpers import SimplePrintable

from .datawriter import DataWriter
# from .flattened_spiketrains import FlattenedSpiketrains

from copy import deepcopy
from enum import Enum, unique, IntEnum
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol

from neuropy.utils.mixins.concatenatable import ConcatenationInitializable


@unique
class NeuronType(Enum):
    PYRAMIDAL = 0
    CONTAMINATED = 1
    INTERNEURONS = 2
    
    # [name for name, member in NeuronType.__members__.items() if member.name != name]    
    # longClassNames = ['pyramidal','contaminated','interneurons']
    # shortClassNames = ['pyr','cont','intr']
    # classCutoffValues = [0, 4, 7, 9]
    
    def describe(self):
        self.name, self.value
    
    
    # def __repr__(self) -> str:
    #     return super().__repr__()
    
    
    
    @property
    def shortClassName(self):
        return NeuronType.shortClassNames()[self.value]
        
    @property
    def longClassName(self):
        return NeuronType.longClassNames()[self.value]

    # def equals(self, string):
    #     # return self.name == string
    #     return ((self.shortClassName == string) or (self.longClassName == string))

    # Static properties
    @classmethod
    def longClassNames(cls):
        return np.array(['pyramidal','contaminated','interneurons'])
    
    @classmethod
    def shortClassNames(cls):
        return np.array(['pyr','cont','intr'])
    
    @classmethod
    def bapunNpyFileStyleShortClassNames(cls):
        return np.array(['pyr','mua','inter'])
    
    @classmethod
    def classCutoffValues(cls):
        return np.array([0, 4, 7, 9])
    
    @classmethod
    def from_short_string(cls, string_value):
        itemindex = np.where(cls.shortClassNames()==string_value)
        return NeuronType(itemindex[0])
    
    @classmethod
    def from_long_string(cls, string_value):
        itemindex = np.where(cls.longClassNames()==string_value)
        return NeuronType(itemindex[0])    
    
    @classmethod
    def from_string(cls, string_value):
        itemindex = np.where(cls.longClassNames()==string_value)
        if len(itemindex[0]) < 1:
            # if not found in longClassNames, try shortClassNames
            itemindex = np.where(cls.shortClassNames()==string_value)
            if len(itemindex[0]) < 1:
                # if not found in shortClassNames, try bapunNpyFileStyleShortClassNames
                itemindex = np.where(cls.bapunNpyFileStyleShortClassNames()==string_value)
        return NeuronType(itemindex[0])
        
    @classmethod
    def from_bapun_npy_style_string(cls, string_value):
        itemindex = np.where(cls.bapunNpyFileStyleShortClassNames()==string_value)
        return NeuronType(itemindex[0])
    
    
    @classmethod
    def from_qclu_series(cls, qclu_Series):
        # qclu_Series: a Pandas Series object, such as qclu_Series=spikes_df['qclu']
        # example: spikes_df['cell_type'] = pd.cut(x=spikes_df['qclu'], bins=classCutoffValues, labels=classNames)
        temp_neuronTypeStrings = pd.cut(x=qclu_Series, bins=cls.classCutoffValues(), labels=cls.shortClassNames())
        temp_neuronTypes = np.array([NeuronType.from_short_string(_) for _ in np.array(temp_neuronTypeStrings)])
        return temp_neuronTypes
        
    @classmethod
    def from_any_string_series(cls, neuron_types_strings):
        # neuron_types_strings: a np.ndarray containing any acceptable style strings, such as: ['mua', 'mua', 'inter', 'pyr', ...]
        return np.array([NeuronType.from_string(_) for _ in np.array(neuron_types_strings)])
    
    
    @classmethod
    def from_bapun_npy_style_series(cls, bapun_style_neuron_types):
        # bapun_style_neuron_types: a np.ndarray containing Bapun-style strings, such as: ['mua', 'mua', 'inter', 'pyr', ...]
        return np.array([NeuronType.from_bapun_npy_style_string(_) for _ in np.array(bapun_style_neuron_types)])
        

    
class Neurons(NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, TimeSlicableObjectProtocol, ConcatenationInitializable, DataWriter):
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
        peak_channels=None,
        shank_ids=None,
        metadata=None,
    ) -> None:
        super().__init__(metadata=metadata)

        self.spiketrains = np.array(spiketrains, dtype="object")
        self._neuron_ids = None
        self._reverse_cellID_index_map = None
        if neuron_ids is None:
            self._neuron_ids = np.arange(len(self.spiketrains))
        else:
            if neuron_ids is int:
                neuron_ids = [neuron_ids] # if it's a single element, wrap it in a list.
            self._neuron_ids = np.array([int(cell_id) for cell_id in neuron_ids]) # ensures integer indexes for IDs
            
        self._reverse_cellID_index_map = Neurons.__build_cellID_reverse_lookup_map(self._neuron_ids.copy())
        
        if waveforms is not None:
            assert (
                waveforms.shape[0] == self.n_neurons
            ), "Waveforms first dimension should match number of neurons"

        self.waveforms = waveforms
        self.shank_ids = shank_ids
        self.neuron_type = neuron_type
        self.peak_channels = peak_channels
        self._sampling_rate = sampling_rate
        self.t_start = t_start
        self.t_stop = t_stop

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
            peak_channels=peak_channels,
            shank_ids=shank_ids,
        )

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def n_neurons(self):
        return len(self.spiketrains)

    
    def time_slice(self, t_start=None, t_stop=None):
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        neurons = deepcopy(self)
        spiketrains = [
            spktrn[(spktrn > t_start) & (spktrn < t_stop)]
            for spktrn in neurons.spiketrains
        ]
        
        
        return Neurons(
            spiketrains=spiketrains,
            t_stop=t_stop,
            t_start=t_start,
            sampling_rate=neurons.sampling_rate,
            neuron_ids=neurons.neuron_ids,
            neuron_type=neurons.neuron_type,
            waveforms=neurons.waveforms,
            peak_channels=neurons.peak_channels,
            shank_ids=neurons.shank_ids,
        )

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
            
        # indices = self.neuron_type == neuron_type # old
        indices = self.neuron_type == query_neuron_type ## Works        
        return self[indices]

    def _check_integrity(self):
        assert isinstance(self.spiketrains, np.ndarray)
        # n_neurons = self.n_neurons
        # assert all(
        #     len(arr) == n_neurons
        #     for arr in [
        #         self.shankid,
        #         self.labels,
        #         self.ids,
        #         self.waveforms,
        #         self.instfiring,
        #     ]
        # )


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n n_neurons: {self.n_neurons}\n n_total_spikes: {self.n_total_spikes}\n t_start: {self.t_start}\n t_stop: {self.t_stop}"

    # def __str__(self) -> str:
    #     # num_original_total_spikes = np.sum(self.n_spikes)
    #     return f"# neurons = {self.n_neurons}"
    
    

    def __len__(self):
        return self.n_neurons


    def add_metadata(self):
        pass

    def get_all_spikes(self):
        return np.concatenate(self.spiketrains)

    # def get_flattened_spikes(self):
    #     # Gets the flattened spikes, sorted in ascending timestamp for all cells. Returns a FlattenedSpiketrains object
    #     flattened_spike_identities = np.concatenate([np.full((self.n_spikes[i],), self.neuron_ids[i]) for i in np.arange(self.n_neurons)]) # repeat the neuron_id for each spike that belongs to that neuron
    #     flattened_spike_times = np.concatenate(self.spiketrains)
    #     # Get the indicies required to sort the flattened_spike_times
    #     sorted_indicies = np.argsort(flattened_spike_times)
    #     return FlattenedSpiketrains(
    #         sorted_indicies,
    #         flattened_spike_identities[sorted_indicies],
    #         flattened_spike_times[sorted_indicies],
    #         t_start=self.t_start
    #     )
    
    @property
    def n_total_spikes(self):
        return np.sum(self.n_spikes)
        
    @property
    def n_spikes(self):
        "number of spikes within each spiketrain"
        return np.asarray([len(_) for _ in self.spiketrains])

    @property
    def firing_rate(self):
        return self.n_spikes / (self.t_stop - self.t_start)

    def get_above_firing_rate(self, thresh: float):
        """Return neurons which have firing rate above thresh"""
        indices = self.firing_rate > thresh
        return self[indices]

    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids):
        """Returns neurons object with neuron_ids equal to ids"""
        indices = np.isin(self.neuron_ids, ids)
        return self[indices]

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

    def get_binned_spiketrains(self, bin_size=0.25):

        """Get binned spike counts

        Parameters
        ----------
        bin_size : float, optional
            bin size in seconds, by default 0.25

        Returns
        -------
        neuropy.core.BinnedSpiketrains

        """

        bins = np.arange(self.t_start, self.t_stop + bin_size, bin_size)
        spike_counts = np.asarray([np.histogram(_, bins=bins)[0] for _ in self.spiketrains])

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

        all_spikes = self.get_all_spikes()
        bins = np.arange(self.t_start, self.t_stop, bin_size)
        spike_counts = np.histogram(all_spikes, bins=bins)[0]
        return Mua(spike_counts.astype("int"), t_start=self.t_start, bin_size=bin_size)

    def add_jitter(self):
        pass

    # def get_psth(self, t, bin_size, window=0.25):
    #     """Get peri-stimulus time histograms w.r.t time points in t"""

    #     time_diff = [np.histogram(spktrn - t) for spktrn in self.spiketrains]

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
            "peak_channels": self.peak_channels,
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
            d["peak_channels"],
            shank_ids=d["shank_ids"],
            metadata=d["metadata"],
        )


    def to_dataframe(self):
        df = self._spikes_df.copy()
        # df['t_start'] = self.t_start
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
            spikes_df['qclu'] = spikes_df['aclu']
            
        if ('cluster' not in spikes_df.columns):
            if debug_print:
                print('dataframe cluster column does not exist. Initializing it to the same as aclu')
            spikes_df['cluster'] = spikes_df['aclu']
    
    
            
        # return spikes_df

    @classmethod
    def from_dataframe(cls, spikes_df, dat_sampling_rate, time_variable_name='t_rel_seconds'):
        """ Builds a Neurons object from a spikes_df, such as the one belonging to its complementary FlattenedSpiketrains:
            Usage:
                neurons_obj = build_neurons_obj(sess.flattened_spiketrains.spikes_df, sess.recinfo.dat_sampling_rate, time_variable_name='t_rel_seconds') 
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
        cell_type = list() # (108,) Array of float64
                
        for i in np.arange(num_unique_cell_ids):
            curr_cell_id = flat_cell_ids[i] # actual cell ID
            #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
            curr_cell_dataframe = cell_grouped_spikes_df.get_group(curr_cell_id)
            spiketrains.append(curr_cell_dataframe[time_variable_name].to_numpy())
            
            shank_ids[i] = curr_cell_dataframe['shank'].to_numpy()[0] # get the first shank identifier, which should be the same for all of this curr_cell_id
            cell_quality[i] = curr_cell_dataframe['qclu'].mean() # should be the same for all instances of curr_cell_id, but use mean just to make sure
            cell_type.append(curr_cell_dataframe['cell_type'].to_numpy()[0])

        spiketrains = np.array(spiketrains, dtype='object')
        t_stop = np.max(spikes_df[time_variable_name])
        flat_cell_ids = np.array(flat_cell_ids)
        cell_type = np.array(cell_type)
        out_neurons = Neurons(spiketrains, t_stop, t_start=0,
            sampling_rate=dat_sampling_rate,
            neuron_ids=flat_cell_ids,
            neuron_type=cell_type,
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
            peak_channels=objList[0].peak_channels,
            shank_ids=objList[0].shank_ids,
            metadata=objList[0].metadata
        )



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
        corr = np.corrcoef(self.spike_counts)

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
    """ Mua stands for Multi-unit activity maybe? """
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

    def get_smoothed(self, sigma=0.02, truncate=4.0):
        t_gauss = np.arange(-truncate * sigma, truncate * sigma, self.bin_size)
        gaussian = np.exp(-(t_gauss ** 2) / (2 * sigma ** 2))
        gaussian /= np.sum(gaussian)

        spike_counts = sg.fftconvolve(self._spike_counts, gaussian, mode="same")
        # frate = gaussian_filter1d(self._frate, sigma=sigma, **kwargs)
        return Mua(spike_counts, t_start=self.t_start, bin_size=self.bin_size)

    # def _gaussian(self):
    #     # TODO fix gaussian smoothing binsize
    #     """Gaussian function for generating instantenous firing rate

    #     Returns:
    #         [array] -- [gaussian kernel centered at zero and spans from -1 to 1 seconds]
    #     """

    #     sigma = 0.020  # 20 ms
    #     binSize = 0.001  # 1 ms
    #     t_gauss = np.arange(-1, 1, binSize)
    #     gaussian = np.exp(-(t_gauss ** 2) / (2 * sigma ** 2))
    #     gaussian /= np.sum(gaussian)

    #     return gaussian

    def to_dict(self, recurrsively=False):
        return {
            "spike_counts": self._spike_counts,
            "t_start": self.t_start,
            "bin_size": self.bin_size,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d):
        return Mua(
            spike_counts=d["spike_counts"],
            t_start=d["t_start"],
            bin_size=d["bin_size"],
            metadata=d["metadata"],
        )

    def to_dataframe(self):
        return pd.DataFrame({"time": self.time, "spike_counts": self.spike_counts})

    