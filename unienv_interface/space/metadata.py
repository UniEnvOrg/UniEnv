"""Read-only, JSON-serializable metadata attached to spaces (see ``Space.metadata``)."""
from collections.abc import Mapping
from typing import Any, Dict, Iterator, Optional

import numpy as np

__all__ = [
    "SpaceMetadata",
    "EMPTY_METADATA",
]

def _freeze(value : Any, path : str) -> Any:
    """Validate ``value`` as JSON data and return an immutable copy of it.

    Mappings become ``SpaceMetadata``, lists / tuples become tuples and numpy scalars
    become the equivalent Python scalars.
    """
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, np.generic) and isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return SpaceMetadata._from_items(value, path)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{path}[{i}]") for i, item in enumerate(value))
    raise TypeError(
        f"Space metadata must be JSON-serializable (None, bool, int, float, str, lists and "
        f"str-keyed mappings of those), but {path} is a {type(value).__name__}: {value!r}"
    )

def _thaw(value : Any) -> Any:
    if isinstance(value, SpaceMetadata):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value

class SpaceMetadata(Mapping):
    """Immutable, JSON-serializable ``str``-keyed mapping describing a space.

    Created from any mapping of JSON data (validated and deep-copied); nested
    mappings are frozen into ``SpaceMetadata`` and lists into tuples. Compares
    equal to plain mappings with the same content (lists and tuples are
    interchangeable). ``to_dict()`` returns a plain, mutable JSON copy.
    """
    __slots__ = ("_data",)

    def __init__(self, data : Optional[Mapping[str, Any]] = None):
        if data is None:
            data = {}
        elif not isinstance(data, Mapping):
            raise TypeError(f"Space metadata must be a mapping, got {type(data).__name__}: {data!r}")
        object.__setattr__(self, "_data", self._freeze_items(data, "metadata"))

    @staticmethod
    def _freeze_items(data : Mapping[str, Any], path : str) -> Dict[str, Any]:
        frozen = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(f"Space metadata keys must be strings, but {path} has key {key!r} ({type(key).__name__})")
            frozen[str(key)] = _freeze(value, f"{path}[{key!r}]")
        return frozen

    @classmethod
    def _from_items(cls, data : Mapping[str, Any], path : str) -> "SpaceMetadata":
        self = object.__new__(cls)
        object.__setattr__(self, "_data", cls._freeze_items(data, path))
        return self

    def __setattr__(self, name, value):
        raise AttributeError("SpaceMetadata is read-only; use `Space.with_metadata` to get a space with other metadata")

    def __delattr__(self, name):
        raise AttributeError("SpaceMetadata is read-only")

    def __getitem__(self, key : str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key : object) -> bool:
        return key in self._data

    def get(self, key : str, default : Any = None) -> Any:
        return self._data.get(key, default)

    def __eq__(self, other : object) -> bool:
        if isinstance(other, SpaceMetadata):
            return self._data == other._data
        if isinstance(other, Mapping):
            try:
                return self._data == SpaceMetadata(other)._data
            except TypeError:
                return False
        return NotImplemented

    def __ne__(self, other : object) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._data.items())))

    def __repr__(self) -> str:
        return f"SpaceMetadata({self.to_dict()!r})"

    def __bool__(self) -> bool:
        return len(self._data) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain (mutable) ``dict`` / ``list`` copy, as written to JSON."""
        return {key: _thaw(value) for key, value in self._data.items()}

    # Immutable: copies can share the instance
    def __copy__(self) -> "SpaceMetadata":
        return self

    def __deepcopy__(self, memo) -> "SpaceMetadata":
        return self

    def __reduce__(self):
        return (SpaceMetadata, (self.to_dict(),))

EMPTY_METADATA = SpaceMetadata()

def coerce_metadata(metadata : Optional[Mapping[str, Any]]) -> SpaceMetadata:
    """``metadata`` as a ``SpaceMetadata`` (validated and copied unless it already is one)."""
    if metadata is None:
        return EMPTY_METADATA
    if type(metadata) is not SpaceMetadata:
        metadata = SpaceMetadata(metadata)
    return metadata if len(metadata) > 0 else EMPTY_METADATA
