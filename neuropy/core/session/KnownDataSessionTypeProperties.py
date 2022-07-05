import dataclasses
from pathlib import Path
from typing import Callable, List
from neuropy.utils.mixins.print_helpers import SimplePrintable


class KnownDataSessionTypeProperties(SimplePrintable, object):
    """Docstring for KnownDataSessionTypeProperties."""
    load_function: Callable
    basedir: Path
    # Optional members
    post_load_functions: List[Callable]
    filter_functions: List[Callable]
    post_compute_functions: List[Callable]
    
    def __init__(self, load_function: Callable, basedir: Path, post_load_functions: List[Callable] = None, filter_functions: List[Callable] = None, post_compute_functions: List[Callable] = None) -> None:
        self.load_function = load_function
        self.basedir = basedir
        # Optional properties:
        if post_load_functions is not None:
            self.post_load_functions = post_load_functions
        else:
            self.post_load_functions = []

        if filter_functions is not None:
            self.filter_functions = filter_functions
        else:
            self.filter_functions = []

        if post_compute_functions is not None:
            self.post_compute_functions = post_compute_functions
        else:
            self.post_compute_functions = []


    # def update_basedir(self, updated_basedir):
    #     self.basedir = updated_basedir