from __future__ import annotations # MUST add to start of file

# from https://www.stefaanlippens.net/circular-imports-type-hints-python.html to avoid circular import issues
# also you must add the following line to the beginning of this file:
#   from __future__ import annotations # otherwise have to do type like 'Ratemap'
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neuropy.analyses.placefields import PlacefieldComputationParameters
    
# from neuropy.analyses.placefields import PlacefieldComputationParameters # causes circular import errors unfortunately

def unwrap_placefield_computation_parameters(computation_config) -> PlacefieldComputationParameters:
    """ Extract the older PlacefieldComputationParameters from the newer computation_config (which is a dynamic_parameters object), which should have a field .pf_params
    If the computation_config is passed in the old-style (already a PlacefieldComputationParameters) it's returned unchanged.

    Usage:
        active_pf_computation_params = unwrap_placefield_computation_parameters(active_config.computation_config)
    """
    if hasattr(computation_config, 'pf_params'):
        return computation_config.pf_params
    else:
        return computation_config
        
    # if isinstance(computation_config, PlacefieldComputationParameters):
    #     # Older format:
    #     return computation_config
    # else:
    #     # Extract the older PlacefieldComputationParameters from pf_params:
    #     assert isinstance(computation_config.pf_params, PlacefieldComputationParameters), "computation_config.pf_params should exist and be of type PlacefieldComputationParameters!"
    #     return computation_config.pf_params
