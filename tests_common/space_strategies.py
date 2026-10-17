"""Hypothesis strategies for UniEnv spaces.

Strategies draw backend-independent *specs* (plain dicts) that :func:`build_space`
turns into a space on any backend/device. Data is never drawn by hypothesis:
tests draw an integer seed and let ``space.sample`` produce it.

Spec kinds: ``box``, ``binary``, ``dynbox``, ``text``, ``graph`` (leaves) and
``dict``, ``tuple``, ``union``, ``batched`` (containers).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np
from hypothesis import strategies as st

from unienv_interface.space import (
    BatchedSpace, BinarySpace, BoxSpace, DictSpace, DynamicBoxSpace, GraphSpace, Space, TextSpace,
    TupleSpace, UnionSpace,
)

LEAF_KINDS = ("box", "binary", "dynbox", "text", "graph")
CONTAINER_KINDS = ("dict", "tuple", "union", "batched")
ALL_KINDS = LEAF_KINDS + CONTAINER_KINDS
CHARSETS = ("ab", "xyz", "0123")

seeds = st.integers(min_value=0, max_value=2 ** 31 - 1)
_shapes = st.lists(st.integers(1, 3), min_size=0, max_size=2).map(tuple)
_dtypes = st.sampled_from(["float32", "int32"])


def backend_dtype(backend: Any, name: str) -> Any:
    """The backend's dtype object for a portable dtype name such as ``"float32"``."""
    return backend.__array_namespace_info__().dtypes()[name]


# ---------------------------------------------------------------------------
# Leaf specs
# ---------------------------------------------------------------------------

@st.composite
def _bounds(draw, shape, dtype, allow_inf=True, finite_low=False):
    """``(low, high)`` as python scalars or numpy arrays broadcastable to ``shape``."""
    elementwise = len(shape) > 0 and draw(st.booleans())
    bshape = shape if elementwise else ()
    size = int(np.prod(bshape)) if bshape else 1
    if dtype == "int32":
        low = np.array(draw(st.lists(st.integers(-20, 20), min_size=size, max_size=size)), dtype=np.int64)
        width = np.array(draw(st.lists(st.integers(0, 20), min_size=size, max_size=size)), dtype=np.int64)
        high = low + width
    else:
        # No subnormal bounds: JAX on CPU flushes subnormals to zero in arithmetic, so a box bounded
        # at e.g. 1.4e-45 could not contain its own rescaled or clipped data
        values = st.floats(-50, 50, allow_nan=False, allow_subnormal=False, width=32)
        low = np.array(draw(st.lists(values, min_size=size, max_size=size)), dtype=np.float64)
        # Widths are 0 or not tiny: rescaling a ~1e-38 wide box overflows float32 by design
        widths = st.one_of(st.just(0.0), st.floats(0.0009765625, 50, width=32))
        width = np.array(draw(st.lists(widths, min_size=size, max_size=size)), dtype=np.float64)
        high = low + width
        if allow_inf:
            if not finite_low:
                low = np.where(np.array(draw(st.lists(st.booleans(), min_size=size, max_size=size))), -np.inf, low)
            high = np.where(np.array(draw(st.lists(st.booleans(), min_size=size, max_size=size))), np.inf, high)
    low, high = low.reshape(bshape), high.reshape(bshape)
    if not elementwise:
        low, high = low.item(), high.item()
    return low, high


@st.composite
def box_specs(draw, shapes=_shapes, dtypes=_dtypes, allow_inf=True):
    shape = draw(shapes)
    dtype = draw(dtypes)
    low, high = draw(_bounds(shape, dtype, allow_inf=allow_inf))
    return {"kind": "box", "shape": shape, "dtype": dtype, "low": low, "high": high}


@st.composite
def binary_specs(draw, shapes=_shapes):
    return {"kind": "binary", "shape": draw(shapes)}


@st.composite
def dynbox_specs(draw, dtypes=_dtypes):
    ndim = draw(st.integers(1, 2))
    # Sizes start at 1: data with an empty axis is all padding, so it cannot be unpadded exactly
    shape_low = tuple(draw(st.lists(st.integers(1, 2), min_size=ndim, max_size=ndim)))
    extra = draw(st.lists(st.integers(0, 2), min_size=ndim, max_size=ndim).filter(any))
    shape_high = tuple(lo + e for lo, e in zip(shape_low, extra))
    dtype = draw(dtypes)
    low, high = draw(_bounds((), dtype, finite_low=True))
    # fill_value strictly below the bounds: data never equals it, so unpadding is exact
    fill_value = int(low) - 1 if dtype == "int32" else float(np.floor(low)) - 1.0
    return {"kind": "dynbox", "shape_low": shape_low, "shape_high": shape_high, "dtype": dtype,
            "low": low, "high": high, "fill_value": fill_value}


@st.composite
def text_specs(draw, allow_free_charset=True):
    max_length = draw(st.integers(0, 5))
    min_length = draw(st.integers(0, max_length))
    charsets = CHARSETS + ((None,) if allow_free_charset else ())
    return {"kind": "text", "min_length": min_length, "max_length": max_length,
            "charset": draw(st.sampled_from(charsets))}


@st.composite
def graph_specs(draw, batch_shapes=st.sampled_from([(), (2,)])):
    is_edge = draw(st.booleans())
    if is_edge:
        max_nodes = draw(st.one_of(st.none(), st.integers(2, 4)))
    else:
        max_nodes = draw(st.one_of(st.none(), st.integers(1, 4)))
    min_nodes = draw(st.integers(1, max_nodes if max_nodes is not None else 3))
    min_edges, max_edges = 1, None
    if is_edge:
        limit = max_nodes * (max_nodes - 1) if max_nodes is not None else 6
        max_edges = draw(st.one_of(st.none(), st.integers(1, min(limit, 6)))) if max_nodes is not None else draw(st.integers(1, 6))
        min_edges = draw(st.integers(1, max_edges if max_edges is not None else 2))
    return {
        "kind": "graph", "node_shape": draw(st.sampled_from([(), (2,)])),
        "edge_shape": draw(st.one_of(st.none(), st.sampled_from([(), (3,)]))) if is_edge else None,
        "is_edge": is_edge, "min_nodes": min_nodes, "max_nodes": max_nodes,
        "min_edges": min_edges, "max_edges": max_edges, "batch_shape": draw(batch_shapes),
    }


_LEAF_STRATEGIES: Dict[str, Callable[[], st.SearchStrategy]] = {
    "box": box_specs, "binary": binary_specs, "dynbox": dynbox_specs, "text": text_specs, "graph": graph_specs,
}
_keys = st.sampled_from(["a", "b", "c", "obs", "z"])


def space_specs(kinds: Iterable[str] = ALL_KINDS, max_leaves: int = 4,
                text_kwargs: Optional[dict] = None) -> st.SearchStrategy:
    """Recursive spec strategy restricted to ``kinds`` (leaf and container kinds)."""
    kinds = tuple(kinds)
    leaves = []
    for kind in LEAF_KINDS:
        if kind in kinds:
            leaves.append(text_specs(**(text_kwargs or {})) if kind == "text" else _LEAF_STRATEGIES[kind]())
    base = st.one_of(*leaves)

    def extend(children):
        options = []
        if "dict" in kinds:
            options.append(st.dictionaries(_keys, children, min_size=1, max_size=3).map(
                lambda d: {"kind": "dict", "spaces": d}))
        if "tuple" in kinds:
            options.append(st.lists(children, min_size=1, max_size=3).map(
                lambda s: {"kind": "tuple", "spaces": s}))
        if "union" in kinds:
            options.append(st.lists(children, min_size=1, max_size=3).map(
                lambda s: {"kind": "union", "spaces": s}))
        if "batched" in kinds:
            options.append(st.tuples(children, st.sampled_from([(2,), (1, 2)])).map(
                lambda t: {"kind": "batched", "single": t[0], "batch_shape": t[1]}))
        return st.one_of(*options) if options else children

    if not any(k in kinds for k in CONTAINER_KINDS):
        return base
    return st.recursive(base, extend, max_leaves=max_leaves)


# ---------------------------------------------------------------------------
# Spec utilities
# ---------------------------------------------------------------------------

def children(spec: dict) -> Sequence[dict]:
    if spec["kind"] == "dict":
        return list(spec["spaces"].values())
    if spec["kind"] in ("tuple", "union"):
        return list(spec["spaces"])
    if spec["kind"] == "batched":
        return [spec["single"]]
    return []


def kinds_in(spec: dict) -> set:
    found = {spec["kind"]}
    for child in children(spec):
        found |= kinds_in(child)
    return found


def all_specs(spec: dict) -> Iterable[dict]:
    yield spec
    for child in children(spec):
        yield from all_specs(child)


def build_space(spec: dict, backend: Any, device: Optional[Any] = None) -> Space:
    kind = spec["kind"]
    if kind == "box":
        dtype = backend_dtype(backend, spec["dtype"])
        low, high = spec["low"], spec["high"]
        if isinstance(low, np.ndarray):
            low = backend.from_numpy(low.astype(spec["dtype"]) if spec["dtype"] == "int32" else low.astype(np.float32), device=device)
        if isinstance(high, np.ndarray):
            high = backend.from_numpy(high.astype(spec["dtype"]) if spec["dtype"] == "int32" else high.astype(np.float32), device=device)
        return BoxSpace(backend, low, high, dtype, device=device, shape=spec["shape"])
    if kind == "binary":
        return BinarySpace(backend, spec["shape"], device=device)
    if kind == "dynbox":
        return DynamicBoxSpace(backend, spec["low"], spec["high"], spec["shape_low"], spec["shape_high"],
                               backend_dtype(backend, spec["dtype"]), device=device, fill_value=spec["fill_value"])
    if kind == "text":
        return TextSpace(backend, spec["max_length"], min_length=spec["min_length"], charset=spec["charset"], device=device)
    if kind == "graph":
        float32 = backend_dtype(backend, "float32")
        node_low, node_high = spec.get("node_bounds", (-1.0, 1.0))
        edge_low, edge_high = spec.get("edge_bounds", (0.0, 2.0))
        node = BoxSpace(backend, node_low, node_high, float32, shape=spec["node_shape"], device=device)
        edge = None if spec["edge_shape"] is None else BoxSpace(backend, edge_low, edge_high, float32, shape=spec["edge_shape"], device=device)
        return GraphSpace(backend, node, edge, is_edge=spec["is_edge"], min_nodes=spec["min_nodes"],
                          max_nodes=spec["max_nodes"], min_edges=spec["min_edges"], max_edges=spec["max_edges"],
                          batch_shape=spec["batch_shape"], device=device)
    if kind == "dict":
        return DictSpace(backend, {k: build_space(v, backend, device) for k, v in spec["spaces"].items()}, device=device)
    if kind == "tuple":
        return TupleSpace(backend, [build_space(v, backend, device) for v in spec["spaces"]], device=device)
    if kind == "union":
        return UnionSpace(backend, [build_space(v, backend, device) for v in spec["spaces"]], device=device)
    if kind == "batched":
        return BatchedSpace(build_space(spec["single"], backend, device), spec["batch_shape"])
    raise ValueError(f"Unknown spec kind {kind}")


@st.composite
def widened_specs(draw, spec: dict, extra_keys: bool = False):
    """A spec whose space is a (non-strict) superspace of ``spec``'s space.

    With ``extra_keys`` dicts may gain keys: still a superspace for
    ``is_subspaceeq``, but ``DictSpace.contains`` demands exact keys.
    """
    kind = spec["kind"]
    if kind == "box":
        grow = draw(st.integers(0, 3))
        low = spec["low"] - grow
        high = spec["high"] + draw(st.integers(0, 3))
        return {**spec, "low": low, "high": high}
    if kind == "dynbox":
        new_low = tuple(max(0, s - draw(st.integers(0, 1))) for s in spec["shape_low"])
        new_high = tuple(s + draw(st.integers(0, 1)) for s in spec["shape_high"])
        return {**spec, "shape_low": new_low, "shape_high": new_high, "low": spec["low"] - draw(st.integers(0, 2))}
    if kind == "text":
        if spec["charset"] is None:
            charset = None
        else:
            charset = draw(st.sampled_from([spec["charset"], spec["charset"] + "q", None]))
        return {**spec, "min_length": max(0, spec["min_length"] - draw(st.integers(0, 1))),
                "max_length": spec["max_length"] + draw(st.integers(0, 2)), "charset": charset}
    if kind == "graph":
        max_nodes = spec["max_nodes"]
        if max_nodes is not None:
            max_nodes = draw(st.sampled_from([max_nodes, max_nodes + 1, None]))
        max_edges = spec["max_edges"]
        if max_edges is not None:
            max_edges = draw(st.sampled_from([max_edges, None]))
        return {**spec, "min_nodes": 1, "max_nodes": max_nodes, "max_edges": max_edges,
                "min_edges": 1 if spec["is_edge"] else spec["min_edges"]}
    if kind == "dict":
        spaces = {k: draw(widened_specs(v, extra_keys)) for k, v in spec["spaces"].items()}
        if extra_keys and draw(st.booleans()):
            spaces["extra_key"] = {"kind": "binary", "shape": (1,)}
        return {"kind": "dict", "spaces": spaces}
    if kind in ("tuple", "union"):
        return {"kind": kind, "spaces": [draw(widened_specs(v, extra_keys)) for v in spec["spaces"]]}
    if kind == "batched":
        return {**spec, "single": draw(widened_specs(spec["single"], extra_keys))}
    return spec


# ---------------------------------------------------------------------------
# Narrowing (subspaces)
# ---------------------------------------------------------------------------

_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


@st.composite
def _narrowed_float(draw, low, high):
    """A float sub-interval ``[a, b]`` of ``[low, high]`` (either end may stay infinite)."""
    if np.isfinite(low):
        a = low + draw(st.floats(0, 1)) * ((high - low) if np.isfinite(high) else 10.0)
    elif draw(st.booleans()):
        a = -np.inf
    else:
        a = (high if np.isfinite(high) else 0.0) - draw(st.floats(0, 10))
    if np.isfinite(high):
        b = a + draw(st.floats(0, 1)) * (high - a) if np.isfinite(a) else high - draw(st.floats(0, 10))
    elif draw(st.booleans()):
        b = np.inf
    else:
        b = (a if np.isfinite(a) else 0.0) + draw(st.floats(0, 10))
    # Lands in [low, high] up to float64 rounding, which float32 storage then absorbs
    a, b = float(min(max(a, low), high)), float(min(max(b, a, low), high))
    # Flush float32-subnormal ends to zero (see the base bounds above); monotone, so a <= b holds
    # and both stay inside [low, high], whose finite ends are never subnormal
    return tuple(0.0 if 0.0 < abs(v) < _FLOAT32_TINY else v for v in (a, b))


@st.composite
def _narrowed_bounds(draw, low, high, dtype):
    """Bounds inside ``low``/``high`` (python scalars or arrays), keeping their form."""
    lows, highs = np.asarray(low, dtype=np.float64).reshape(-1), np.asarray(high, dtype=np.float64).reshape(-1)
    new_low, new_high = np.empty_like(lows), np.empty_like(highs)
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        if dtype == "int32":
            new_low[i] = draw(st.integers(int(lo), int(hi)))
            new_high[i] = draw(st.integers(int(new_low[i]), int(hi)))
        else:
            new_low[i], new_high[i] = draw(_narrowed_float(lo, hi))
    if isinstance(low, np.ndarray):
        return new_low.reshape(low.shape), new_high.reshape(low.shape)
    cast = int if dtype == "int32" else float
    return cast(new_low[0]), cast(new_high[0])


@st.composite
def narrowed_specs(draw, spec: dict):
    """A spec whose space is a (non-strict) subspace of ``spec``'s space.

    Shrinks value bounds, length / shape / count ranges and charsets, drops
    dict keys and recurses into containers.
    """
    kind = spec["kind"]
    if kind == "box":
        low, high = draw(_narrowed_bounds(spec["low"], spec["high"], spec["dtype"]))
        return {**spec, "low": low, "high": high}
    if kind == "dynbox":
        while True:
            shape_low = tuple(draw(st.integers(lo, hi)) for lo, hi in zip(spec["shape_low"], spec["shape_high"]))
            shape_high = tuple(draw(st.integers(lo, hi)) for lo, hi in zip(shape_low, spec["shape_high"]))
            if shape_low != shape_high:  # a dynamic box needs a dynamic axis
                break
            shape_low, shape_high = spec["shape_low"], spec["shape_high"]
            if draw(st.booleans()):
                break
        low, high = draw(_narrowed_bounds(spec["low"], spec["high"], spec["dtype"]))
        return {**spec, "shape_low": shape_low, "shape_high": shape_high, "low": low, "high": high}
    if kind == "text":
        min_length = draw(st.integers(spec["min_length"], spec["max_length"]))
        max_length = draw(st.integers(min_length, spec["max_length"]))
        if spec["charset"] is None:
            charset = draw(st.sampled_from((None,) + CHARSETS))
        else:
            chars = draw(st.lists(st.sampled_from(sorted(spec["charset"])), min_size=1, unique=True))
            charset = "".join(sorted(chars))
        return {**spec, "min_length": min_length, "max_length": max_length, "charset": charset}
    if kind == "graph":
        upper = spec["max_nodes"] if spec["max_nodes"] is not None else spec["min_nodes"] + 3
        min_nodes = draw(st.integers(spec["min_nodes"], upper))
        max_nodes = draw(st.integers(min_nodes, upper))
        if spec["max_nodes"] is None and draw(st.booleans()):
            max_nodes = None
        new = {**spec, "min_nodes": min_nodes, "max_nodes": max_nodes,
               "node_bounds": draw(_narrowed_float(*spec.get("node_bounds", (-1.0, 1.0))))}
        if spec["is_edge"]:
            if max_nodes is not None:
                limit = max_nodes * (max_nodes - 1)
                if limit < spec["min_edges"]:
                    # Too few nodes for the required edges: keep the source node limits
                    new["min_nodes"], new["max_nodes"] = spec["min_nodes"], spec["max_nodes"]
                    max_nodes = spec["max_nodes"]
                    limit = max_nodes * (max_nodes - 1) if max_nodes is not None else None
            else:
                limit = None
            cap = min(c for c in (limit, spec["max_edges"], spec["min_edges"] + 3) if c is not None)
            min_edges = draw(st.integers(spec["min_edges"], cap))
            edge_cap = min(c for c in (limit, spec["max_edges"], min_edges + 3) if c is not None)
            max_edges = draw(st.integers(min_edges, edge_cap))
            if spec["max_edges"] is None and draw(st.booleans()):
                max_edges = None
            new.update(min_edges=min_edges, max_edges=max_edges)
            if spec["edge_shape"] is not None:
                new["edge_bounds"] = draw(_narrowed_float(*spec.get("edge_bounds", (0.0, 2.0))))
        return new
    if kind == "dict":
        keys = draw(st.lists(st.sampled_from(sorted(spec["spaces"])), unique=True)) if spec["spaces"] else []
        return {"kind": "dict", "spaces": {k: draw(narrowed_specs(spec["spaces"][k])) for k in sorted(keys)}}
    if kind in ("tuple", "union"):
        return {"kind": kind, "spaces": [draw(narrowed_specs(v)) for v in spec["spaces"]]}
    if kind == "batched":
        return {**spec, "single": draw(narrowed_specs(spec["single"]))}
    return spec
