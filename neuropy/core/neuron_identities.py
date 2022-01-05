# neuron_identities
from collections import namedtuple
import numpy as np
from neuropy.utils.mixins.print_helpers import SimplePrintable
from neuropy.utils.colors_util import get_neuron_colors
from matplotlib.colors import ListedColormap

NeuronExtendedIdentityTuple = namedtuple('NeuronExtendedIdentity', 'shank cluster id')

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


class NeuronIdentity(SimplePrintable):
    """NeuronIdentity: A multi-facited identifier for a specific neuron/putative cell 
        Used to retain a the identity associated with a value or set of values even after filtering and such.

        cell_uid: (aclu) [2:65]
        shank_index: [1:12]
        cluster_index: [2:28]
        
        ['shank','cluster']
    
    """
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
