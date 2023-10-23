
def flatten(A):
    """ safely flattens lists of lists without flattening top-level strings also. 
    https://stackoverflow.com/questions/17864466/flatten-a-list-of-strings-and-lists-of-strings-and-lists-in-python

	Usage:

		from neuropy.utils.indexing_helpers import flatten

    Example:
        list(flatten(['format_name', ('animal','exper_name', 'session_name')] ))
        >>> ['format_name', 'animal', 'exper_name', 'session_name']

    """
    rt = []
    for i in A:
        if isinstance(i, (list, tuple)): rt.extend(flatten(i))
        else: rt.append(i)
    return rt
