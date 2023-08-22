# neuron_identities
from collections import namedtuple
from enum import Enum
from typing import List
import numpy as np
import pandas as pd
import tables as tb
from tables import (
    Int8Col, Int16Col, Int32Col, Int64Col,
    UInt8Col, UInt16Col, UInt32Col, UInt64Col,
    Float32Col, Float64Col,
    TimeCol, ComplexCol, StringCol, BoolCol, EnumCol
)
import h5py
from neuropy.utils.mixins.print_helpers import SimplePrintable
from neuropy.utils.colors_util import get_neuron_colors
from matplotlib.colors import ListedColormap

NeuronExtendedIdentityTuple = namedtuple('NeuronExtendedIdentityTuple', 'shank cluster id')


"""
from neuropy.core.neuron_identities import NeuronExtendedIdentityTuple, neuronTypesEnum, NeuronIdentityTable
['shank', 'cluster', 'aclu', 'qclu', 'cell_type', 'fragile_linear_neuron_IDX']
"""

neuronTypesList: List[str] = ['pyr', 'bad', 'intr']
neuronTypesEnum = tb.Enum(neuronTypesList)

class NeuronIdentityTable(tb.IsDescription):
    """ represents a single neuron in the scope of multiple sessions for use in a PyTables table or HDF5 output file """
    global_uid = StringCol(68)  # TO REMOVE   # 68-character String, globally unique neuron identifier (across all sessions) composed of a session_uid and the neuron's (session-specific) aclu
    session_uid = StringCol(64)
    ## Session-Local Identifiers
    neuron_id = UInt16Col() # 65535 max neurons
    neuron_type = EnumCol(neuronTypesEnum, 'bad', base='uint8') # 
    shank_index  = UInt16Col() # specific to session
    cluster_index  = UInt16Col() # specific to session
    qclu = UInt8Col() # specific to session
    


@pd.api.extensions.register_dataframe_accessor("neuron_identity")
class NeuronIdentityDataframeAccessor:
    """ Describes a dataframe with at least a neuron_id (aclu) column. Provides functionality regarding building globally (across-sessions) unique neuron identifiers.
    
    #TODO 2023-08-22 15:34: - [ ] Finish implementation. Purpose is to easily add across-session-unique neuron identifiers to a result dataframe (as many result dataframes have an 'aclu' column).
        - [ ] find already implemented 'aclu' conversions, like for the JonathanFiringRateResult (I think)
        
    """
   
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('cell_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """       
        if "aclu" not in obj.columns or "cell_type" not in obj.columns:
            raise AttributeError("Must have unit id column 'aclu' and 'cell_type' column.")
        
    @property
    def neuron_ids(self):
        """ return the unique cell identifiers (given by the unique values of the 'aclu' column) for this DataFrame """
        unique_aclus = np.unique(self._obj['aclu'].values)
        return unique_aclus
    
    @property
    def neuron_probe_tuple_ids(self):
        """ returns a list of NeuronExtendedIdentityTuple tuples where the first element is the shank_id and the second is the cluster_id. Returned in the same order as self.neuron_ids """
        # groupby the multi-index [shank, cluster]:
        # shank_cluster_grouped_spikes_df = self._obj.groupby(['shank','cluster'])
        aclu_grouped_spikes_df = self._obj.groupby(['aclu'])
        shank_cluster_reference_df = aclu_grouped_spikes_df[['aclu','shank','cluster']].first() # returns a df indexed by 'aclu' with only the 'shank' and 'cluster' columns
        output_tuples_list = [NeuronExtendedIdentityTuple(an_id.shank, an_id.cluster, an_id.aclu) for an_id in shank_cluster_reference_df.itertuples()] # returns a list of tuples where the first element is the shank_id and the second is the cluster_id. Returned in the same order as self.neuron_ids
        return output_tuples_list
        
    @property
    def n_neurons(self):
        return len(self.neuron_ids)
    
    def extract_unique_neuron_identities(self):
        """ Tries to build information about the unique neuron identitiies from the (highly reundant) information in the spikes_df. """
        selected_columns = ['aclu', 'shank', 'cluster', 'qclu', 'cell_type']
        unique_rows_df = self._obj[selected_columns].drop_duplicates().reset_index(drop=True).sort_values(by='aclu') # Based on only these columns, remove all repeated rows. Since every spike from the same aclu must have the same values for all the rest of the values, there should only be one row for each aclu. 
        assert len(unique_rows_df) == self.n_neurons, f"if this were false that would suggest that there are multiple entries for aclus. n_neurons: {self.n_neurons}, {len(unique_rows_df) =}"
        return unique_rows_df

        # # Extract the selected columns as NumPy arrays
        # aclu_array = unique_rows_df['aclu'].values
        # shank_array = unique_rows_df['shank'].values
        # cluster_array = unique_rows_df['cluster'].values
        # qclu_array = unique_rows_df['qclu'].values
        # neuron_type_array = unique_rows_df['cell_type'].values
        # neuron_types_enum_array = np.array([neuronTypesEnum[a_type.hdfcodingClassName] for a_type in neuron_type_array]) # convert NeuronTypes to neuronTypesEnum
        


    # ==================================================================================================================== #
    # Additive Mutating Functions: Adds or Update Columns in the Dataframe                                                 #
    # ==================================================================================================================== #
    
    # def to_hdf(self, file_path, key: str, **kwargs):
    #     """ Saves the object to key in the hdf5 file specified by file_path 
    #     Usage:

    #     .spikes.to_hdf(
    #     """
    #     _spikes_df = deepcopy(self._obj)
    #     # Convert the 'cell_type' column of the dataframe to the categorical type if needed
    #     cat_type = NeuronType.get_pandas_categories_type()
    #     if _spikes_df["cell_type"].dtype != cat_type:
    #         # If this type check ever becomes a problem and we want a more liberal constraint, All instances of CategoricalDtype compare equal to the string 'category'.
    #         _spikes_df["cell_type"] = _spikes_df["cell_type"].apply(lambda x: x.hdfcodingClassName).astype(cat_type) # NeuronType can't seem to be cast directly to the new categorical type, it results in the column being filled with NaNs. Instead cast to string first.

    #     # Store DataFrame using pandas
    #     with pd.HDFStore(file_path) as store:
    #         _spikes_df.to_hdf(store, key=key, format=kwargs.pop('format', 'table'), data_columns=kwargs.pop('data_columns',True), **kwargs)

    #     # Open the file with h5py to add attributes
    #     with h5py.File(file_path, 'r+') as f:
    #         _ds = f[key]
    #         _ds.attrs['time_variable_name'] = self.time_variable_name
    #         _ds.attrs['n_neurons'] = self.n_neurons
    #         # You can add more attributes here as needed
    #         # _ds.attrs['neuron_ids'] = self.neuron_ids
    #         # _ds.attrs['neuron_probe_tuple_ids'] = self.neuron_probe_tuple_ids


       
    # @classmethod
    # def read_hdf(cls, file_path, key: str, **kwargs) -> pd.DataFrame:
    #     """  Reads the data from the key in the hdf5 file at file_path         
    #     # TODO 2023-07-30 13:05: - [ ] interestingly this leaves the dtype of this column as 'category' still, but _spikes_df["cell_type"].to_numpy() returns the correct array of objects... this is better than it started before saving, but not the same. 
    #         - UPDATE: I think adding `.astype(str)` to the end of the conversion resolves it and makes the type the same as it started. Still not sure if it would be better to leave it a categorical because I think it's more space efficient and better than it started anyway.
    #     """
    #     _spikes_df = pd.read_hdf(file_path, key=key, **kwargs)
    #     # Convert the 'cell_type' column back to its original type (e.g., a custom class NeuronType)
    #     # .astype(object)

    #     _spikes_df["cell_type"] = _spikes_df["cell_type"].apply(lambda x: NeuronType.from_hdf_coding_string(x)).astype(object) #.astype(str) # interestingly this leaves the dtype of this column as 'category' still, but _spikes_df["cell_type"].to_numpy() returns the correct array of objects... this is better than it started before saving, but not the same. 
        
    #     return _spikes_df








# NOTE: this is like visual identity also
class NeuronIdentity(SimplePrintable):
    """NeuronIdentity: A multi-facited identifier for a specific neuron/putative cell 
        Used to retain a the identity associated with a value or set of values even after filtering and such.

        cell_uid: (aclu) [2:65]
        shank_index: [1:12]
        cluster_index: [2:28]
        
        NOTE: also store 'color'
        ['shank','cluster']
    
    """
    @property
    def extended_identity_tuple(self):
        """The extended_identity_tuple property."""
        return NeuronExtendedIdentityTuple(self.shank_index, self.cluster_index, self.cell_uid) # returns self as a NeuronExtendedIdentityTuple 
    @extended_identity_tuple.setter
    def extended_identity_tuple(self, value):
        assert isinstance(value, NeuronExtendedIdentityTuple), "value should be a NeuronExtendedIdentityTuple"
        self.cell_uid = value.id
        self.shank_index = value.shank
        self.cluster_index = value.cluster
        
    @property
    def extended_id_string(self):
        """The extended_id_string property."""
        return self._extended_id_string
    
    
    
    def __init__(self, cell_uid, shank_index, cluster_index, color=None):
        self.cell_uid = cell_uid
        self.shank_index = shank_index
        self.cluster_index = cluster_index
        self.color = color

    @classmethod
    def init_from_NeuronExtendedIdentityTuple(cls, a_tuple: NeuronExtendedIdentityTuple, a_color=None):
        """Iniitalizes from a NeuronExtendedIdentityTuple and optionally a color
        Args:
            a_tuple (NeuronExtendedIdentityTuple): [description]
            a_color ([type], optional): [description]. Defaults to None.
        """
        return cls(a_tuple.id, a_tuple.shank, a_tuple.cluster, color=a_color)
        
        
        
class NeuronIdentityAccessingMixin:
    """ 
        Requires implementor overrides the neuron_ids property to provide an ordered list of unique cell identifiers (such as the 'aclu' values from a spikes_df)
        
        provides functions to map between unique cell_identifiers (cell_ids) and implementor specific indicies (neuron_IDXs)
    
        NOTE: 
            Based on how the neuron_IDXs are treated in self.get_neuron_id_and_idx(...), they are constrained to be:
                1. Monotonically increasing from 0 to (len(self.neuron_ids)-1)
                
                Note that if an implementor violates this definition, for example if they filtered out or excluded some of the neurons and so were left with fewer self.neuron_ids than were had when the self.neuron_IDXs were first built, then there can potentially be:
                    1. Indicies missing from the neuronIDXs (corresponding to the filtered out neuron_ids)
                    2. Too many indicies present in neuronIDXs (with the extras corresponding to the neuron_ids that were removed after the IDXs were built).
                    3. **IMPORTANT**: Values of neuronIDXs that are too large and would cause index out of bound errors when trying to get the corresponding to neuron_id value.
                    4. **CRITICAL**: VALUE SHIFTED reverse lookups! If any neuronIDX is removed with its corresponding neuron_id, it will cause all the neuron_IDXs after it to be 1 value too large and throw off reverse lookups. This is what's happening with the placefields/spikes getting shifted!
    
    
    CONCLUSIONS:
        Implementor must be sure to keep self.neuron_ids up-to-date with any other list of neuron_ids it might use (like the 'aclu' values from the spikes_df) AND be sure to not hold references to (or keep them up-to-date) the neuron_IDXs. Any time IDXs are used (such as those retrieved from the spikes_df's neuron_IDX column) they must be up-to-date to be referenced.
        
    """
    @property
    def neuron_ids(self):
        """ e.g. return np.array(active_epoch_placefields2D.cell_ids) """
        raise NotImplementedError
    
    def get_neuron_id_and_idx(self, neuron_i=None, neuron_id=None):
        """For a specified neuron_i (index) or neuron_id, returns the other quanity (or both)

        Args:
            neuron_i ([type], optional): [description]. Defaults to None.
            neuron_id ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        assert (neuron_i is not None) or (neuron_id is not None), "You must specify either neuron_i (index) or neuron_id, and the other will be returned"
        assert (neuron_i is None) or (neuron_id is None), "You cannot specify both neuron_i (index) and neuron_id, as it would be ambiguous which takes priority. Please remove one of the two arguments."
        if neuron_i is not None:
            neuron_i = int(neuron_i)
            neuron_id = self.neuron_ids[neuron_i]
        elif neuron_id is not None:
            neuron_id = int(neuron_id)
            neuron_i = np.where(self.neuron_ids == neuron_id)[0].item()
        # print(f'cell_i: {cell_i}, cell_id: {cell_id}')
        return neuron_i, neuron_id

    def find_cell_ids_from_neuron_IDXs(self, neuron_IDXs):
        """Finds the cell original IDs from the cell IDXs (not IDs)
        Args:
            neuron_IDXs ([type]): [description]
        """
        found_cell_ids = [self.get_neuron_id_and_idx(neuron_i=an_included_neuron_IDX)[1] for an_included_neuron_IDX in neuron_IDXs] # get the ids from the cell IDXs
        return found_cell_ids
    
    
    def find_neuron_IDXs_from_cell_ids(self, cell_ids):
        """Finds the cell IDXs (not IDs) from the cell original IDs (cell_ids)
        Args:
            cell_ids ([type]): [description]
        """
        found_cell_INDEXES = [self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in cell_ids] # get the indexes from the cellIDs
        return found_cell_INDEXES
    
    

# ==================================================================================================================== #
# Display and Render Helpers                                                                                           #
# ==================================================================================================================== #

class PlotStringBrevityModeEnum(Enum):
    """An enum of different modes that specify how verbose/brief the rendered strings should be on a given plot.
    More verbose means longer ouptuts with fewer abbreviations. For very brief modes, less important elements may be omitted entirely
    """
    VERBOSE = "VERBOSE"
    DEFAULT = "DEFAULT"
    CONCISE = "CONCISE"
    MINIMAL = "MINIMAL"
    NONE = "NONE"
    
    @property
    def extended_identity_labels(self):
        """The extended_identity_labels property."""
        if self == PlotStringBrevityModeEnum.VERBOSE:
            return {'cell_uid':'cell_uid', 'shank_index':'shank_index', 'cluster_index':'cluster_index'}
        elif self == PlotStringBrevityModeEnum.DEFAULT:
            return {'cell_uid':'id', 'shank_index':'shank', 'cluster_index':'cluster'}
        elif self == PlotStringBrevityModeEnum.CONCISE:
            return {'cell_uid':'', 'shank_index':'shk', 'cluster_index':'clu'}
        elif self == PlotStringBrevityModeEnum.MINIMAL:
            return {'cell_uid':'', 'shank_index':'s', 'cluster_index':'c'}
        elif self == PlotStringBrevityModeEnum.NONE:
            return {'cell_uid':'', 'shank_index':'', 'cluster_index':''}
        else:
            raise NameError

    def _basic_identity_formatting_string(self, neuron_extended_id):
        """Builds the string output for just the id (aclu) component of the neuron_extended_id """
        if self.name == PlotStringBrevityModeEnum.VERBOSE.name:
            return f'Cell cell_uid: {neuron_extended_id.id}'
        elif self.name == PlotStringBrevityModeEnum.DEFAULT.name:
            return f'Cell {neuron_extended_id.id}'
        elif self.name == PlotStringBrevityModeEnum.CONCISE.name:
            return f'Cell {neuron_extended_id.id}'
        elif self.name == PlotStringBrevityModeEnum.MINIMAL.name:
            return f'{neuron_extended_id.id}'
        elif self.name == PlotStringBrevityModeEnum.NONE.name:
            return f'{neuron_extended_id.id}'
        else:
            print(f'self: {self} with name {self.name} and value {self.value} is unknown type!')
            raise NameError

    def _extra_info_identity_formatting_string(self, neuron_extended_id):
        """Builds the string output for just the shank and cluster components of the neuron_extended_id."""
        if self.name == PlotStringBrevityModeEnum.VERBOSE.name:
            return f'(shank_index {neuron_extended_id.shank}, cluster_index {neuron_extended_id.cluster})'
        elif self.name == PlotStringBrevityModeEnum.DEFAULT.name:
            return f'(shank {neuron_extended_id.shank}, cluster {neuron_extended_id.cluster})'
        elif self.name == PlotStringBrevityModeEnum.CONCISE.name:
            return f'(shk {neuron_extended_id.shank}, clu {neuron_extended_id.cluster})'
        elif self.name == PlotStringBrevityModeEnum.MINIMAL.name:
            return f's{neuron_extended_id.shank}, c{neuron_extended_id.cluster}'
        elif self.name == PlotStringBrevityModeEnum.NONE.name:
            return f'{neuron_extended_id.shank},{neuron_extended_id.cluster}'
        else:
            print(f'self: {self} with name {self.name} and value {self.value} is unknown type!')
            raise NameError

    def extended_identity_formatting_string(self, neuron_extended_id):
        """The extended_identity_labels property."""
        if (self.name == PlotStringBrevityModeEnum.VERBOSE.name) or (self.name == PlotStringBrevityModeEnum.DEFAULT.name):
            return ' - '.join([self._basic_identity_formatting_string(neuron_extended_id), self._extra_info_identity_formatting_string(neuron_extended_id)])
        elif (self.name == PlotStringBrevityModeEnum.CONCISE.name) or (self.name == PlotStringBrevityModeEnum.MINIMAL.name):
            return '-'.join([self._basic_identity_formatting_string(neuron_extended_id), self._extra_info_identity_formatting_string(neuron_extended_id)])
        elif self.name == PlotStringBrevityModeEnum.NONE.name:
            # Show only the id label:
            return self._basic_identity_formatting_string(neuron_extended_id)
        else:
            print(f'self: {self} with name {self.name} and value {self.value} is unknown type!')
            raise NameError
        
    @property
    def should_show_firing_rate_label(self):
        """ Whether the firing rate in Hz should be showed on the plot """
        if self.name == PlotStringBrevityModeEnum.CONCISE.name:
            return True # was False
        elif self.name == PlotStringBrevityModeEnum.MINIMAL.name:
            return True # was False
        elif self.name == PlotStringBrevityModeEnum.NONE.name:
            return False # was False
        else:
            return True
        

## Plotting Colors:
def build_units_colormap(neuron_ids):
    """ 
    Usage:
        pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = build_units_colormap(good_placefield_neuronIDs)
    """
    pf_sort_ind = np.array([int(i) for i in np.arange(len(neuron_ids))]) # convert to integer scalar array
    pf_colors = get_neuron_colors(pf_sort_ind, cmap=None) # [4 x n_neurons]: colors are by ascending index ID
    pf_colormap = pf_colors.T # [n_neurons x 4] Make the colormap from the listed colors, used seemingly only by 'runAnalysis_PCAandICA(...)'
    pf_listed_colormap = ListedColormap(pf_colormap)
    return pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap


        
class NeuronIdentitiesDisplayerMixin:
    @property
    def neuron_ids(self):
        """ like self.neuron_ids """
        raise NotImplementedError
    @property
    def neuron_extended_ids(self):
        """ list of NeuronExtendedIdentityTuple named tuples) like tuple_neuron_ids """
        raise NotImplementedError

    @property
    def neuron_shank_ids(self):
        return [extended_id.shank for extended_id in self.neuron_extended_ids]
    
    @property
    def neuron_cluster_ids(self):
        return [extended_id.cluster for extended_id in self.neuron_extended_ids]

    def get_extended_neuron_id_string(self, neuron_i=None, neuron_id=None):
        assert (neuron_i is not None) or (neuron_id is not None), "You must specify either neuron_i (index) or neuron_id, and the other will be returned"
        assert (neuron_i is None) or (neuron_id is None), "You cannot specify both neuron_i (index) and neuron_id, as it would be ambiguous which takes priority. Please remove one of the two arguments."
        
        # TODO: make more general
        curr_cell_alt_id = self.neuron_extended_ids[neuron_i]
        curr_cell_shank = curr_cell_alt_id.shank
        curr_cell_cluster = curr_cell_alt_id.cluster
        return f'(shank {curr_cell_shank}, cluster {curr_cell_cluster})'
        
        # ax1.set_title(
        #     f"Cell {neuron_ids[cell]} - (shank {curr_cell_shank}, cluster {curr_cell_cluster}) \n{round(np.nanmax(pfmap),2)} Hz"
        # )
        
    def other_neuron_id_string(self, neuron_i):
        # TODO: implement as below
        raise NotImplementedError
        # # sorted_neuron_id_labels = [f'Cell[{a_neuron_id}]' for a_neuron_id in sorted_neuron_ids]
        # sorted_neuron_id_labels = [f'C[{sorted_neuron_ids[i]}]({sorted_alt_tuple_neuron_ids[i][0]}|{sorted_alt_tuple_neuron_ids[i][1]})' for i in np.arange(len(sorted_neuron_ids))]
