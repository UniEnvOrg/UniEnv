from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal, Iterable
import numpy as np
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame

from unienv_interface.transformations.filter_dict import FilterDictTransformation
from .transformation import ContextObservationTransformWrapper

class FilterContextObsWrapper(
    ContextObservationTransformWrapper[
        ContextType, ObsType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    """
    This wrapper will filter the context and observation dictionaries to only include the specified keys.
    """

    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        keys_obs : Optional[Iterable[str]] = None,
        keys_context : Optional[Iterable[str]] = None,
    ):
        assert keys_obs is not None or keys_context is not None, "At least one of keys_obs and keys_context must be specified"
        if keys_obs is None:
            obs_transformation = None
        else:
            obs_transformation = FilterDictTransformation(
                env.observation_space,
                keys_obs
            )
        if keys_context is None:
            context_transformation = None
        else:
            context_transformation = FilterDictTransformation(
                env.context_space,
                keys_context
            )

        super().__init__(
            env,
            context_transformation,
            obs_transformation
        )
