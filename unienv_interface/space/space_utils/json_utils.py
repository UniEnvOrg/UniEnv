"""Spaces as JSON Schemas, and data of a space to and from plain JSON values.

For exposing environments and policies to tools that speak JSON, such as LLM tool-calling APIs and
MCP (``unienv_mcp``). The JSON form of a space's data:

* ``BoxSpace`` / ``DynamicBoxSpace``: numbers (``integer`` for integer dtypes), nested in lists for
  each axis. With ``named=True``, an axis described by ``metadata["names"]`` (one name per entry of
  the last axis, e.g. joint names) becomes an object keyed by those names, which language models
  read and write more reliably than positional lists.
* ``BinarySpace``: booleans, nested like boxes.
* ``TextSpace``: a string.
* ``DictSpace``: an object; ``TupleSpace``: a list; ``UnionSpace``: ``{"option": i, "value": v}``.
* ``BatchedSpace``: a list of its single space's values.

:func:`space_to_json_schema` describes that form (JSON Schema 2020-12) with bounds, lengths and a
description built from the space's metadata; :func:`to_json` and :func:`from_json` convert data.
:func:`from_json` checks structure and shapes, not bounds: use :func:`bound_violations` (or
``space.contains``) for those. ``GraphSpace`` has no JSON form.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..space import Space
from ..spaces import (BatchedSpace, BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, GraphSpace, TextSpace,
                      TupleSpace, UnionSpace)

__all__ = ["space_to_json_schema", "to_json", "from_json", "bound_violations", "describe_metadata"]

#: Metadata keys rendered into schema descriptions, in this order
_DESCRIBED_KEYS = ("label", "kind", "units", "frame", "rotation", "control", "mode", "normalized", "range", "layout",
                   "format", "source")


def describe_metadata(space: Space) -> Optional[str]:
    """A one-line description of a space from its metadata (label, kind, units, frame, ...)."""
    md = dict(getattr(space, "metadata", None) or {})
    parts = []
    for key in _DESCRIBED_KEYS:
        if key not in md or md[key] is None:
            continue
        value = md[key]
        if key == "label":
            parts.append(str(value))
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}: {', '.join('-' if v is None else str(v) for v in value)}")
        else:
            parts.append(f"{key}: {value}")
    return "; ".join(parts) or None


def _names(space: Space, named: bool) -> Optional[List[str]]:
    """The names of the last axis, if ``named`` and the metadata gives one name per entry."""
    if not named:
        return None
    names = (getattr(space, "metadata", None) or {}).get("names")
    shape = tuple(getattr(space, "shape", None) or ())
    if (isinstance(names, (list, tuple)) and shape and len(names) == shape[-1] and len(set(names)) == len(names)
            and all(isinstance(n, str) for n in names)):
        return [str(n) for n in names]
    return None


def _units(space: Space) -> Optional[List[Any]]:
    units = (getattr(space, "metadata", None) or {}).get("units")
    return list(units) if isinstance(units, (list, tuple)) else None


def _numpy(space: Space, array: Any) -> np.ndarray:
    return np.asarray(space.backend.to_numpy(array))


def _bounds(space: BoxSpace) -> Tuple[np.ndarray, np.ndarray]:
    shape = tuple(space.shape)
    return (np.broadcast_to(_numpy(space, space.low), shape), np.broadcast_to(_numpy(space, space.high), shape))


def _number(value: Any, integer: bool) -> Any:
    value = value.item() if isinstance(value, np.generic) else value
    if integer:
        return int(value)
    value = float(value)
    return value if math.isfinite(value) else None


def _leaf_schema(kind: str, low: Optional[float], high: Optional[float], integer: bool) -> Dict[str, Any]:
    if kind == "boolean":
        return {"type": "boolean"}
    schema: Dict[str, Any] = {"type": "integer" if integer else "number"}
    if low is not None and math.isfinite(low):
        schema["minimum"] = int(low) if integer else float(low)
    if high is not None and math.isfinite(high):
        schema["maximum"] = int(high) if integer else float(high)
    return schema


def _array_schema(shape: Tuple[int, ...], low: Optional[np.ndarray], high: Optional[np.ndarray], *, kind: str,
                  integer: bool, names: Optional[List[str]], units: Optional[List[Any]],
                  min_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
    """Nested arrays over ``shape`` (the last axis as an object when named) with per-entry bounds."""
    def leaf(index: Tuple[int, ...]) -> Dict[str, Any]:
        lo = None if low is None else float(low[index])
        hi = None if high is None else float(high[index])
        return _leaf_schema(kind, lo, hi, integer)

    def build(axis: int, prefix: Tuple[int, ...]) -> Dict[str, Any]:
        if axis == len(shape):
            return leaf(prefix)
        size = shape[axis]
        minimum = size if min_shape is None else min_shape[axis]
        last = axis == len(shape) - 1
        if last and names is not None:
            properties = {}
            for i, name in enumerate(names):
                item = leaf(prefix + (i,))
                if units is not None and i < len(units) and units[i] is not None:
                    item["description"] = f"units: {units[i]}"
                properties[name] = item
            return {"type": "object", "properties": properties, "required": list(names), "additionalProperties": False}
        items = [build(axis + 1, prefix + (i,)) for i in range(size)]
        schema: Dict[str, Any] = {"type": "array", "minItems": minimum, "maxItems": size}
        if size and all(item == items[0] for item in items):
            schema["items"] = items[0]
        elif size:
            schema["prefixItems"] = items
            schema["items"] = False
        return schema

    return build(0, ())


def space_to_json_schema(space: Space, *, named: bool = True) -> Dict[str, Any]:
    """The JSON Schema (2020-12) of the JSON form of ``space``'s data (see the module docs)."""
    if isinstance(space, DictSpace):
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {str(key): space_to_json_schema(sub, named=named) for key, sub in space.spaces.items()},
            "required": [str(key) for key in space.spaces],
            "additionalProperties": False,
        }
    elif isinstance(space, BoxSpace):
        low, high = _bounds(space)
        integer = space.backend.dtype_is_real_integer(space.dtype)
        schema = _array_schema(tuple(space.shape), low, high, kind="number", integer=integer,
                               names=_names(space, named), units=_units(space))
    elif isinstance(space, DynamicBoxSpace):
        integer = space.backend.dtype_is_real_integer(space.dtype)
        shape = tuple(space.shape_high)
        low = np.broadcast_to(_numpy(space, space.low), shape) if np.ndim(_numpy(space, space.low)) else \
            np.full(shape, float(_numpy(space, space.low)))
        high = np.broadcast_to(_numpy(space, space.high), shape) if np.ndim(_numpy(space, space.high)) else \
            np.full(shape, float(_numpy(space, space.high)))
        schema = _array_schema(shape, low, high, kind="number", integer=integer, names=None, units=None,
                               min_shape=tuple(space.shape_low))
    elif isinstance(space, BinarySpace):
        schema = _array_schema(tuple(space.shape), None, None, kind="boolean", integer=False,
                               names=_names(space, named), units=None)
    elif isinstance(space, TextSpace):
        schema = {"type": "string", "minLength": space.min_length, "maxLength": space.max_length}
        if space.charset is not None and len(space.charset) <= 64:
            schema["description"] = "characters: " + "".join(sorted(space.charset))
    elif isinstance(space, TupleSpace):
        items = [space_to_json_schema(sub, named=named) for sub in space.spaces]
        schema = {"type": "array", "prefixItems": items, "items": False, "minItems": len(items),
                  "maxItems": len(items)}
    elif isinstance(space, UnionSpace):
        schema = {"oneOf": [
            {"type": "object", "properties": {"option": {"const": i}, "value": space_to_json_schema(sub, named=named)},
             "required": ["option", "value"], "additionalProperties": False}
            for i, sub in enumerate(space.spaces)
        ]}
    elif isinstance(space, BatchedSpace):
        schema = space_to_json_schema(space.single_space, named=named)
        for size in reversed(tuple(space.batch_shape)):
            schema = {"type": "array", "items": schema, "minItems": size, "maxItems": size}
    elif isinstance(space, GraphSpace):
        raise TypeError("GraphSpace data has no JSON form")
    else:
        raise TypeError(f"No JSON form for {type(space).__name__}")
    description = describe_metadata(space)
    if description:
        schema["description"] = f"{description}; {schema['description']}" if "description" in schema else description
    return schema


# ---------------------------------------------------------------------------- data


def _nested(array: np.ndarray, names: Optional[List[str]], convert) -> Any:
    if array.ndim == 0:
        return convert(array[()])
    if array.ndim == 1 and names is not None:
        return {name: convert(value) for name, value in zip(names, array)}
    return [_nested(sub, names, convert) for sub in array]


def to_json(space: Space, data: Any, *, named: bool = True, digits: Optional[int] = None) -> Any:
    """``data`` of ``space`` as a JSON value; ``digits`` rounds floats (non-finite floats become ``None``)."""
    if isinstance(space, DictSpace):
        return {str(key): to_json(sub, data[key], named=named, digits=digits) for key, sub in space.spaces.items()}
    if isinstance(space, (BoxSpace, DynamicBoxSpace)):
        integer = space.backend.dtype_is_real_integer(space.dtype)
        array = _numpy(space, data)

        def convert(value):
            number = _number(value, integer)
            return round(number, digits) if digits is not None and isinstance(number, float) else number

        return _nested(array, _names(space, named) if isinstance(space, BoxSpace) else None, convert)
    if isinstance(space, BinarySpace):
        return _nested(_numpy(space, data).astype(bool), _names(space, named), lambda v: bool(v))
    if isinstance(space, TextSpace):
        return str(data)
    if isinstance(space, TupleSpace):
        return [to_json(sub, item, named=named, digits=digits) for sub, item in zip(space.spaces, data)]
    if isinstance(space, UnionSpace):
        index, value = data
        index = int(index)
        return {"option": index, "value": to_json(space.spaces[index], value, named=named, digits=digits)}
    if isinstance(space, BatchedSpace):
        return _nested_batched(space, data, len(tuple(space.batch_shape)), named, digits)
    raise TypeError(f"No JSON form for {type(space).__name__}")


def _nested_batched(space: BatchedSpace, data: Any, depth: int, named: bool, digits: Optional[int]) -> Any:
    if depth == 0:
        return to_json(space.single_space, data, named=named, digits=digits)
    return [_nested_batched(space, data[i], depth - 1, named, digits) for i in range(len(data))]


class _JSONError(ValueError):
    pass


def _at(path: str, message: str) -> _JSONError:
    return _JSONError(f"{path or '<value>'}: {message}")


def _unnest(value: Any, shape: Tuple[int, ...], names: Optional[List[str]], path: str,
            min_shape: Optional[Tuple[int, ...]] = None) -> Any:
    """Positional nested lists from ``value``, checking the shape (objects for the named last axis)."""
    if not shape:
        if isinstance(value, (list, dict)):
            raise _at(path, f"expected a single value, got {type(value).__name__}")
        return value
    if len(shape) == 1 and names is not None:
        if not isinstance(value, Mapping):
            raise _at(path, f"expected an object with keys {names}")
        missing = [name for name in names if name not in value]
        extra = [key for key in value if key not in names]
        if missing or extra:
            raise _at(path, "".join([f"missing {missing}" if missing else "",
                                     "; " if missing and extra else "", f"unexpected {extra}" if extra else ""]))
        return [value[name] for name in names]
    if not isinstance(value, (list, tuple)):
        raise _at(path, f"expected a list of {shape[0]}")
    low = shape[0] if min_shape is None else min_shape[0]
    if not low <= len(value) <= shape[0]:
        expected = shape[0] if low == shape[0] else f"{low} to {shape[0]}"
        raise _at(path, f"expected {expected} entries, got {len(value)}")
    inner_min = None if min_shape is None else min_shape[1:]
    return [_unnest(item, shape[1:], names, f"{path}[{i}]", inner_min) for i, item in enumerate(value)]


def from_json(space: Space, value: Any, *, named: bool = True, _path: str = "") -> Any:
    """Data of ``space`` (on its backend, device and dtype) from its JSON form.

    Raises ``ValueError`` naming the offending path when the structure or a shape does not match;
    bounds are not checked (see :func:`bound_violations`).
    """
    if isinstance(space, DictSpace):
        if not isinstance(value, Mapping):
            raise _at(_path, "expected an object")
        keys = [str(key) for key in space.spaces]
        missing = [key for key in keys if key not in value]
        extra = [key for key in value if key not in keys]
        if missing or extra:
            raise _at(_path, "".join([f"missing {missing}" if missing else "",
                                      "; " if missing and extra else "", f"unexpected {extra}" if extra else ""]))
        return {key: from_json(sub, value[str(key)], named=named, _path=f"{_path}/{key}".lstrip("/"))
                for key, sub in space.spaces.items()}
    if isinstance(space, (BoxSpace, BinarySpace)):
        nested = _unnest(value, tuple(space.shape), _names(space, named), _path)
        return _to_backend(space, nested, _path)
    if isinstance(space, DynamicBoxSpace):
        shape_high = tuple(space.shape_high)
        nested = _unnest(value, shape_high, None, _path, min_shape=tuple(space.shape_low))
        return _to_backend(space, nested, _path)
    if isinstance(space, TextSpace):
        if not isinstance(value, str):
            raise _at(_path, "expected a string")
        return value
    if isinstance(space, TupleSpace):
        if not isinstance(value, (list, tuple)) or len(value) != len(space.spaces):
            raise _at(_path, f"expected a list of {len(space.spaces)}")
        return tuple(from_json(sub, item, named=named, _path=f"{_path}[{i}]")
                     for i, (sub, item) in enumerate(zip(space.spaces, value)))
    if isinstance(space, UnionSpace):
        if not isinstance(value, Mapping) or set(value) != {"option", "value"}:
            raise _at(_path, 'expected {"option": index, "value": ...}')
        index = value["option"]
        if not isinstance(index, int) or not 0 <= index < len(space.spaces):
            raise _at(_path, f"option must be an integer in [0, {len(space.spaces)})")
        return index, from_json(space.spaces[index], value["value"], named=named, _path=f"{_path}/value")
    if isinstance(space, BatchedSpace):
        result = np.empty(tuple(space.batch_shape), dtype=object)
        _fill_batched(space, value, result, (), named, _path)
        return result
    raise TypeError(f"No JSON form for {type(space).__name__}")


def _fill_batched(space: BatchedSpace, value: Any, out: np.ndarray, index: Tuple[int, ...], named: bool,
                  path: str) -> None:
    if len(index) == out.ndim:
        out[index] = from_json(space.single_space, value, named=named, _path=path)
        return
    size = out.shape[len(index)]
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise _at(path, f"expected a list of {size}")
    for i, item in enumerate(value):
        _fill_batched(space, item, out, index + (i,), named, f"{path}[{i}]")


def _to_backend(space: Space, nested: Any, path: str) -> Any:
    backend = space.backend
    if isinstance(space, BinarySpace):
        if not all(isinstance(v, (bool, np.bool_)) for v in _flat(nested)):
            raise _at(path, "expected booleans")
        array = np.asarray(nested, dtype=bool)
    else:
        values = list(_flat(nested))
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            raise _at(path, "expected numbers")
        integer = backend.dtype_is_real_integer(space.dtype)
        if integer and not all(float(v).is_integer() for v in values):
            raise _at(path, "expected integers")
        array = np.asarray(nested, dtype=np.int64 if integer else np.float64)
        if isinstance(space, DynamicBoxSpace) and array.dtype == object:
            raise _at(path, "ragged lists are not supported")
    return backend.asarray(array, dtype=space.dtype, device=space.device)


def _flat(nested: Any):
    if isinstance(nested, list):
        for item in nested:
            yield from _flat(item)
    else:
        yield nested


def bound_violations(space: Space, data: Any, *, named: bool = True, limit: int = 10, _path: str = "") -> List[str]:
    """Messages for entries of ``data`` outside ``space``'s bounds (at most ``limit``), e.g.
    ``"arm/shoulder: 1.7 is above the maximum 1.0"``; empty when ``data`` is within them."""
    found: List[str] = []
    if isinstance(space, DictSpace):
        for key, sub in space.spaces.items():
            found += bound_violations(sub, data[key], named=named, limit=limit - len(found),
                                      _path=f"{_path}/{key}".lstrip("/"))
            if len(found) >= limit:
                break
        return found[:limit]
    if isinstance(space, TupleSpace):
        for i, (sub, item) in enumerate(zip(space.spaces, data)):
            found += bound_violations(sub, item, named=named, limit=limit - len(found), _path=f"{_path}[{i}]")
        return found[:limit]
    if isinstance(space, UnionSpace):
        index, value = data
        return bound_violations(space.spaces[int(index)], value, named=named, limit=limit, _path=f"{_path}/value")
    if isinstance(space, TextSpace):
        if not space.contains(data):
            found.append(f"{_path or '<value>'}: text of length {len(str(data))} outside "
                         f"[{space.min_length}, {space.max_length}] or with characters outside the charset")
        return found
    if not isinstance(space, BoxSpace):
        if not space.contains(data):
            found.append(f"{_path or '<value>'}: not within the space")
        return found
    array = _numpy(space, data)
    low, high = _bounds(space)
    names = _names(space, named)
    below, above = array < low, array > high
    for index in np.ndindex(*array.shape):
        if not (below[index] or above[index]):
            continue
        kind, bound = ("below the minimum", low) if below[index] else ("above the maximum", high)
        where = _path
        if index:
            if names is not None:
                where += "".join(f"[{i}]" for i in index[:-1]) + ("/" if where or len(index) > 1 else "") + names[index[-1]]
            else:
                where += "".join(f"[{i}]" for i in index)
        found.append(f"{where or '<value>'}: {array[index].item():.6g} is {kind} {bound[index].item():.6g}")
        if len(found) >= limit:
            return found
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        found.append(f"{_path or '<value>'}: contains NaN or infinite values")
    return found[:limit]
