from pathlib import Path

class FileInitializable:
	""" Implementors can be initialized from a file path (from which they are loaded)
	"""
	@classmethod
	def from_file(cls, f):
		assert isinstance(f, (str, Path))
		raise NotImplementedError


class FileRepresentable(FileInitializable):
	""" Implementors can be loaded or saved to a file
	"""
	@classmethod
	def to_file(cls, data: dict, f):
		raise NotImplementedError

 
	def save(self):
		raise NotImplementedError