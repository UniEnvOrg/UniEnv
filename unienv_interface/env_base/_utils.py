"""Private helpers shared by the environment, wrapper and policy base classes."""
import inspect
from typing import Any, Optional

import numpy as np

from unienv_interface.backends import ComputeBackend
from unienv_interface.space import Space, BoxSpace, BinarySpace, DictSpace, TupleSpace
from unienv_interface.space.space_utils import batch_utils as sbu

__all__ = ["mask_to_numpy", "masked_space", "settable_here", "set_attr_on_first_layer"]


def mask_to_numpy(backend : Optional[ComputeBackend], mask : Any) -> np.ndarray:
    """A reset mask (any backend / device) as a 1-d numpy bool array."""
    if backend is not None and backend.is_backendarray(mask) and not isinstance(mask, np.ndarray):
        mask = backend.to_numpy(mask)
    elif hasattr(mask, "detach"):  # a torch tensor given to a non-torch env
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask, dtype=bool).reshape(-1)


def _masked_space_fast(space : Space, indices : np.ndarray) -> Optional[Space]:
    """``space`` restricted to the batch rows ``indices`` for Box / Binary leaves in Dict / Tuple
    spaces (``None`` if a leaf is of another type)."""
    if type(space) is DictSpace:
        subspaces = {}
        for key, subspace in space.spaces.items():
            masked = _masked_space_fast(subspace, indices)
            if masked is None:
                return None
            subspaces[key] = masked
        return DictSpace(space.backend, subspaces, device=space.device, metadata=space._metadata)
    if type(space) is TupleSpace:
        subspaces = []
        for subspace in space.spaces:
            masked = _masked_space_fast(subspace, indices)
            if masked is None:
                return None
            subspaces.append(masked)
        return TupleSpace(space.backend, subspaces, device=space.device, metadata=space._metadata)
    if type(space) is BinarySpace:
        if len(space.shape) == 0:
            return None
        return BinarySpace(
            space.backend, shape=(len(indices),) + tuple(space.shape[1:]),
            device=space.device, metadata=space._metadata, dtype=space.dtype,
        )
    if type(space) is BoxSpace:
        if len(space.shape) == 0:
            return None
        backend = space.backend

        def select(bound):
            # Bounds may be abbreviated: a leading axis of size 1 (or fewer dims) is shared by every row.
            if len(bound.shape) < len(space.shape) or bound.shape[0] == 1:
                return bound
            if len(indices) == 0:
                return bound[:1]  # keep a (broadcastable) row: empty bounds break e.g. ``repr``
            index = backend.asarray(indices.astype(np.int64), device=backend.device(bound))
            return backend.take(bound, index, axis=0)

        return BoxSpace._from_validated(
            backend=backend,
            low=select(space._low),
            high=select(space._high),
            dtype=space.dtype,
            shape=(len(indices),) + tuple(space.shape[1:]),
            device=space.device,
            metadata=space._metadata,
        )
    return None


def masked_space(space : Optional[Space], mask : Any) -> Optional[Space]:
    """The batched ``space`` restricted to the rows selected by the boolean reset ``mask``.

    A masked ``reset`` returns only the selected rows, so code that maps that data through a
    space-dependent function (e.g. a data transformation) must use this space instead of the
    full-batch one. Falls back to ``space`` itself if it cannot be restricted.
    """
    if space is None or mask is None:
        return space
    try:
        mask_np = mask_to_numpy(space.backend, mask)
    except Exception:
        return space
    if mask_np.all():
        return space
    indices = np.nonzero(mask_np)[0]
    masked = _masked_space_fast(space, indices)
    if masked is not None:
        return masked
    try:
        rows = list(sbu.unbatch_spaces(space))
        if len(indices) == 0:
            return sbu.batch_space(rows[0], 0)
        selected = [rows[i] for i in indices]
        if all(row == selected[0] for row in selected[1:]):
            return sbu.batch_space(selected[0], len(selected))
        return sbu.batch_differing_spaces(selected, device=space.device)
    except Exception:
        return space


def settable_here(obj : Any, name : str) -> bool:
    """Whether ``setattr(obj, name, ...)`` sets an existing attribute of ``obj`` itself
    (a read-only property, such as a wrapper forwarding ``batch_size``, does not count)."""
    if not hasattr(obj, name):
        return False
    static = inspect.getattr_static(type(obj), name, None)
    if isinstance(static, property):
        return static.fset is not None
    return True


def set_attr_on_first_layer(layer : Any, name : str, value : Any, owner_name : str) -> None:
    """Set ``name`` on the outermost layer (walking ``prev_wrapper_layer``) that defines it and
    can set it; layers that only expose it through a read-only property are skipped."""
    found = False
    start = layer
    while layer is not None:
        if settable_here(layer, name):
            setattr(layer, name, value)
            return
        found = found or hasattr(layer, name)
        inner = layer.prev_wrapper_layer
        if inner is None and layer is not start and layer.has_wrapper_attr(name):
            # Let the innermost layer apply its own lookup (e.g. FuncEnvBasedEnv searching its
            # functional stack, FuncPolicyBasedPolicy its functional policy)
            layer.set_wrapper_attr(name, value)
            return
        layer = inner
    if found:
        raise AttributeError(f"attribute {name!r} of wrapper {owner_name} is read-only on every layer")
    raise AttributeError(f"wrapper {owner_name} has no attribute {name!r}")
