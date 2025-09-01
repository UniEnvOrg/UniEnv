from typing import Sequence, List, Tuple, Union, Dict, Any, Optional, Generic, TypeVar, Iterable, Iterator, Callable
from types import EllipsisType
import os
import abc
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.space import Space, DictSpace, BoxSpace, BinarySpace, TupleSpace
import dataclasses
import copy
import functools

from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu

from ..base.common import BatchBase, IndexableType, BatchT
from .slicestack_batch import SliceStackedBatch

class FrameStackedBatch(SliceStackedBatch[
    BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    """
    A batch that stacks frames in a sliding window manner.
    This batch allows for prefetching and postfetching of frames, which can be useful for training models that require temporal context (e.g. Diffusion Policy, ACT, etc.)
    This is a read-only batch, since it is a view of the original batch. If you want to change the data, you should mutate the containing batch instead.
    """

    is_mutable = False

    def __init__(
        self,
        batch: BatchBase[
            BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
        ],
        prefetch_horizon : int = 0,
        postfetch_horizon : int = 0,
        get_valid_mask_function : Optional[Callable[["SliceStackedBatch", BArrayType, BatchT, Dict[str, Any]], BArrayType]] = None,
        fill_invalid_data : bool = True,
        stack_metadata : bool = False,
    ):
        assert prefetch_horizon >= 0, "Prefetch horizon must be a non-negative integer"
        assert postfetch_horizon >= 0, "Postfetch horizon must be a non-negative integer"
        assert prefetch_horizon > 0 or postfetch_horizon > 0, "At least one of prefetch_horizon and postfetch_horizon must be greater than 0"
        fixed_offset = batch.backend.arange(
            -prefetch_horizon, postfetch_horizon + 1, device=batch.device
        )
        super().__init__(
            batch,
            fixed_offset,
            get_valid_mask_function=get_valid_mask_function,
            fill_invalid_data=fill_invalid_data,
            stack_metadata=stack_metadata,
        )