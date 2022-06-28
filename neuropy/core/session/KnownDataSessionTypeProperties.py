import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List


@dataclass
class KnownDataSessionTypeProperties(object):
    """Docstring for KnownDataSessionTypeProperties."""
    load_function: Callable
    basedir: Path
    # Optional members
    post_load_functions: List[Callable] = dataclasses.field(default_factory=list)
    filter_functions: List[Callable] = dataclasses.field(default_factory=list)
    post_compute_functions: List[Callable] = dataclasses.field(default_factory=list)
    # filter_function: Callable = None
