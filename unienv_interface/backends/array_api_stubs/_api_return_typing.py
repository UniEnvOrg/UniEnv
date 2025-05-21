from typing import Protocol, TypeVar, Optional, Any, Tuple, Union, Type, TypedDict, List
from dataclasses import dataclass
from abc import abstractmethod
from ._array_typing import Device, DType

__all__ = [
    "SupportsBufferProtocol",
    "DefaultDataTypes",
    "DataTypes",
    "Capabilities",
    "finfo_object",
    "iinfo_object",
    "NestedSequence",
    "Info"
]

SupportsBufferProtocol = TypeVar("SupportsBufferProtocol")

"""
https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/_types.py#L55
"""
DefaultDataTypes = TypedDict(
    "DefaultDataTypes",
    {
        "real floating": DType,
        "complex floating": DType,
        "integral": DType,
        "indexing": DType,
    },
)
DataTypes = TypedDict(
    "DataTypes",
    {
        "bool": DType,
        "float32": DType,
        "float64": DType,
        "complex64": DType,
        "complex128": DType,
        "int8": DType,
        "int16": DType,
        "int32": DType,
        "int64": DType,
        "uint8": DType,
        "uint16": DType,
        "uint32": DType,
        "uint64": DType,
    },
    total=False,
)
Capabilities = TypedDict(
    "Capabilities",
    {
        "boolean indexing": bool,
        "data-dependent shapes": bool,
        "max rank": Optional[int],
    },
)

@dataclass
class finfo_object:
    """Dataclass returned by `finfo`."""
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: DType

@dataclass
class iinfo_object:
    """Dataclass returned by `iinfo`."""
    bits: int
    max: int
    min: int
    dtype: DType

_T_co = TypeVar("_T_co", covariant=True)
class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> Union[_T_co, 'NestedSequence'[_T_co]]:
        ...

    def __len__(self, /) -> int:
        ...

class Info(Protocol):
    """Namespace returned by `__array_namespace_info__`."""

    def capabilities(self) -> Capabilities:
        ...

    def default_device(self) -> Device:
        ...

    def default_dtypes(self, *, device: Optional[Device]) -> DefaultDataTypes:
        ...

    def devices(self) -> List[Device]:
        ...

    def dtypes(
        self, *, device: Optional[Device], kind: Optional[Union[str, Tuple[str, ...]]]
    ) -> DataTypes:
        ...
