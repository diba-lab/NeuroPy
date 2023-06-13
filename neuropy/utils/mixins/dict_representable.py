from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage

class DictInitializable:
	""" Implementors can be initialized from a dict
	"""
	@staticmethod
	def from_dict(d: dict):
		raise NotImplementedError


class DictRepresentable(DictInitializable):
	def to_dict(self, recurrsively=False):
		raise NotImplementedError


class SubsettableDict(DictRepresentable):
	""" confomers can be subsettable
	requires `benedict` library: from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage

	"""
	def to_dict(self, subset_includelist=None, subset_excludelist=None):
		""" 
		Inputs:
			subset_includelist:<list?> a list of keys that specify the subset of the keys to be returned. If None, all are returned.
		"""
		if subset_excludelist is not None:
			# if we have a excludelist, assert that we don't have an includelist
			assert subset_includelist is None, f"subset_includelist MUST be None when a subset_excludelist is provided, but instead it's {subset_includelist}!"
			subset_includelist = self.keys(subset_excludelist=subset_excludelist) # get all but the excluded keys

		if subset_includelist is None:
			return benedict(self.__dict__)
		else:
			return benedict(self.__dict__).subset(subset_includelist)

	def keys(self, subset_includelist=None, subset_excludelist=None):
		if subset_includelist is None:
			return [a_key for a_key in benedict(self.__dict__).keys() if a_key not in (subset_excludelist or [])]
		else:
			assert subset_excludelist is None, f"subset_excludelist MUST be None when a subset_includelist is provided, but instead it's {subset_excludelist}!"
			return [a_key for a_key in benedict(self.__dict__).subset(subset_includelist).keys() if a_key not in (subset_excludelist or [])]

	def keypaths(self, subset_includelist=None, subset_excludelist=None): 
		if subset_includelist is None:
			return [a_key for a_key in benedict(self.__dict__).keys() if a_key not in (subset_excludelist or [])]
		else:
			assert subset_excludelist is None, f"subset_excludelist MUST be None when a subset_includelist is provided, but instead it's {subset_excludelist}!"
			return [a_key for a_key in benedict(self.__dict__).subset(subset_includelist).keys() if a_key not in (subset_excludelist or [])]
		