import dataclasses
# from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List
from neuropy.utils.mixins.print_helpers import SimplePrintable

# @dataclass
class KnownDataSessionTypeProperties(SimplePrintable, object):
    """Docstring for KnownDataSessionTypeProperties."""
    load_function: Callable
    basedir: Path
    # Optional members
    post_load_functions: List[Callable] = dataclasses.field(default_factory=list)
    filter_functions: List[Callable] = dataclasses.field(default_factory=list)
    post_compute_functions: List[Callable] = dataclasses.field(default_factory=list)
    
    def __init__(self, load_function: Callable, basedir: Path, post_load_functions: List[Callable] = dataclasses.field(default_factory=list), filter_functions: List[Callable] = dataclasses.field(default_factory=list), post_compute_functions: List[Callable] = dataclasses.field(default_factory=list)) -> None:
        self.load_function = load_function
        self.basedir = basedir
        # Optional properties:
        self.post_load_functions = post_load_functions
        self.filter_functions = filter_functions
        self.post_compute_functions = post_compute_functions

    # def update_basedir(self, updated_basedir):
    #     self.basedir = updated_basedir