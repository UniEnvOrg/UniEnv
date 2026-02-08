from unienv_interface.space.space_utils import batch_utils as sbu
from .transformation import DataTransformation, TargetDataT
from unienv_interface.space import Space
from typing import Union, Any, Optional, Dict
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.utils.symbol_util import get_full_class_name


class BatchifyTransformation(DataTransformation):
    has_inverse = True

    def __init__(
        self,
        axis: int = 0
    ) -> None:
        self.axis = axis

    def get_target_space_from_source(self, source_space):
        ret = sbu.batch_space(source_space, 1)
        if self.axis != 0:
            ret = sbu.swap_batch_dims(
                ret,
                0,
                self.axis
            )
        return ret

    def transform(self, source_space, data):
        return sbu.concatenate(
            source_space,
            [data],
            axis=self.axis
        )

    def direction_inverse(self, source_space=None):
        return UnBatchifyTransformation(axis=self.axis)

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": get_full_class_name(type(self)),
            "axis": self.axis,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        backend: Optional[ComputeBackend] = None,
        device: Optional[BDeviceType] = None,
    ) -> "BatchifyTransformation":
        return cls(
            axis=json_data.get("axis", 0),
        )


class UnBatchifyTransformation(DataTransformation):
    has_inverse = True

    def __init__(
        self,
        axis: int = 0
    ) -> None:
        self.axis = axis

    def get_target_space_from_source(self, source_space):
        if self.axis != 0:
            source_space = sbu.swap_batch_dims(
                source_space,
                0,
                self.axis
            )
        assert sbu.batch_size(source_space) == 1, "Cannot unbatch space with batch size > 1"
        return next(iter(sbu.unbatch_spaces(source_space)))

    def transform(self, source_space, data):
        if self.axis != 0:
            source_space = sbu.swap_batch_dims(
                source_space,
                0,
                self.axis
            )
            data = sbu.swap_batch_dims_in_data(
                source_space.backend,
                data,
                0,
                self.axis
            )
        return sbu.get_at(
            source_space,
            data,
            0
        )

    def direction_inverse(self, source_space=None):
        return BatchifyTransformation(axis=self.axis)

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": get_full_class_name(type(self)),
            "axis": self.axis,
        }

    @classmethod
    def deserialize_from(
        cls,
        json_data: Dict[str, Any],
        backend: Optional[ComputeBackend] = None,
        device: Optional[BDeviceType] = None,
    ) -> "UnBatchifyTransformation":
        return cls(
            axis=json_data.get("axis", 0),
        )