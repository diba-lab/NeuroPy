## Implementors can be faithfully (although maybe not completely) represented by a Pandas DataFrame. 
# dataframe_representable.py


class DataFrameInitializable:
	""" Implementors can be initialized from a Pandas DataFrame. 
	"""
	@classmethod
	def from_dataframe(cls, df):
		raise NotImplementedError

class DataFrameRepresentable(DataFrameInitializable):
	def to_dataframe(self):
		raise NotImplementedError
