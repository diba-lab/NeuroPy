import numpy as np


class NeuronUnitSlicableObjectProtocol:
	""" Implementors are slicable the the number of units.
	
	from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
	
	
	"""
	
	def get_by_id(self, ids):
		"""Implementors return a copy of themselves with neuron_ids equal to ids"""
		# indices = np.isin(self.neuron_ids, ids)
		# return self[indices]
		raise NotImplementedError
