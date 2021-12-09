import numpy as np


class NeuronUnitSlicableObjectProtocol:
	def get_by_id(self, ids):
		"""Implementors return a copy of themselves with neuron_ids equal to ids"""
		indices = np.isin(self.neuron_ids, ids)
		# return self[indices]
		raise NotImplementedError
