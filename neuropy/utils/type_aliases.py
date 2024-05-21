from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias  # "from typing_extensions" in Python 3.9 and earlier
from typing import NewType
from nptyping import NDArray

""" Usage:

from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types

"""

aclu_index: TypeAlias = int # an integer index that is an aclu
DecoderName = NewType('DecoderName', str)

