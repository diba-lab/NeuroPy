class DictInitializable:
	""" Implementors can be initialized from a dict
	"""
	@staticmethod
	def from_dict(d: dict):
		raise NotImplementedError


class DictRepresentable(DictInitializable):
	def to_dict(self, recurrsively=False):
		raise NotImplementedError
