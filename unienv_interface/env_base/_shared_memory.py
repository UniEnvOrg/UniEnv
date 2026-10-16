"""Shared-memory observation transfer for :class:`~unienv_interface.env_base.vec_env.AsyncVecEnv`.

The parent process allocates one buffer of shape ``(num_envs, *leaf_shape)`` per
leaf of the single-environment observation (and context) space. Worker ``i``
writes its observation into row ``i`` and only the small step results (reward,
termination flags, info) travel through the pipe.

Where the buffers live depends on the backend and device of the worker
environment (see :func:`select_mode`):

* ``HOST`` - one host shared-memory segment (:mod:`multiprocessing.shared_memory`)
  holding every leaf; the parent reads zero-copy numpy / torch views of it.
* ``HOST_STAGED`` - the same host segment, used for devices that cannot be shared
  between processes (JAX TPUs, CUDA without IPC support, other accelerators).
  Workers copy device -> host into the segment and the parent copies host ->
  device once per read.
* ``CUDA_IPC`` - one CUDA tensor per leaf allocated by the parent on the env's
  device and shared with the workers through ``torch.multiprocessing`` CUDA IPC.
  Workers copy device -> device and synchronise before acknowledging. JAX GPU
  arrays are moved in and out of these torch buffers with DLPack (zero-copy).

This module is private; the public switch is ``AsyncVecEnv(shared_memory=True)``.
"""

from __future__ import annotations

import ctypes
import math
import sys
import warnings
import weakref
from collections.abc import Mapping, Sequence
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unienv_interface.backends import ComputeBackend
from unienv_interface.space import BinarySpace, BoxSpace, DictSpace, Space, TupleSpace

__all__ = [
    "HOST",
    "HOST_STAGED",
    "CUDA_IPC",
    "space_leaves",
    "check_leaf_layouts_match",
    "select_mode",
    "ParentSharedBuffers",
    "WorkerSharedBuffers",
]

HOST = "host"
HOST_STAGED = "host_staged"
CUDA_IPC = "cuda_ipc"

# Every leaf block starts on a 64-byte boundary (cache line; also what XLA wants
# for zero-copy DLPack imports on the CPU).
_ALIGNMENT = 64

_warned_no_cuda_ipc = False

PathType = Tuple[Any, ...]


def _format_path(path : PathType) -> str:
    return "".join(f"[{key!r}]" for key in path) or "<root>"


# ---------------------------------------------------------------------------
# Space layout
# ---------------------------------------------------------------------------

def space_leaves(space : Space, what : str = "observation", path : PathType = ()) -> List[Tuple[PathType, Space]]:
    """Return ``[(path, leaf_space)]`` for the fixed-size leaves of ``space``.

    Only :class:`BoxSpace` and :class:`BinarySpace` leaves nested in
    :class:`DictSpace` / :class:`TupleSpace` are supported; anything else raises
    a ``ValueError``.
    """
    if type(space) in (BoxSpace, BinarySpace):
        return [(path, space)]
    if isinstance(space, DictSpace):
        leaves = []
        for key, sub_space in space.spaces.items():
            leaves.extend(space_leaves(sub_space, what, path + (key,)))
        return leaves
    if isinstance(space, TupleSpace):
        leaves = []
        for index, sub_space in enumerate(space.spaces):
            leaves.extend(space_leaves(sub_space, what, path + (index,)))
        return leaves
    raise ValueError(
        f"AsyncVecEnv(shared_memory=True) only supports fixed-size {what} spaces "
        f"(BoxSpace / BinarySpace leaves nested in DictSpace / TupleSpace), but the "
        f"{what} space has a {type(space).__name__} at {_format_path(path)}. "
        f"Use shared_memory=False for this environment."
    )


def _canonical_dtype(backend : ComputeBackend, dtype : Any) -> Any:
    """The dtype arrays created with ``dtype`` actually have (e.g. JAX without x64)."""
    return backend.zeros((0,), dtype=dtype).dtype


def _dtype_to_str(backend : ComputeBackend, dtype : Any) -> str:
    """Name of a backend dtype (also covers bfloat16, unlike the array-API dtype table)."""
    if backend.simplified_name == "pytorch":
        return str(dtype).removeprefix("torch.")
    return np.dtype(dtype).name


def _dtype_from_str(backend : ComputeBackend, name : str) -> Any:
    if backend.simplified_name == "pytorch":
        import torch
        return getattr(torch, name)
    if backend.simplified_name == "jax":
        import jax.numpy as jnp
        return jnp.dtype(name)
    return np.dtype(name)


def _leaf_layout(backend : ComputeBackend, space : Optional[Space], what : str) -> List[Tuple[PathType, Tuple[int, ...], str]]:
    if space is None:
        return []
    return [
        (path, tuple(leaf.shape), _dtype_to_str(backend, _canonical_dtype(backend, leaf.dtype)))
        for path, leaf in space_leaves(space, what)
    ]


def check_leaf_layouts_match(backend : ComputeBackend, spaces : Sequence[Optional[Space]], what : str) -> None:
    """Every worker must report the same leaf structure, shapes and dtypes."""
    first = _leaf_layout(backend, spaces[0], what)
    for index, space in enumerate(spaces[1:], start=1):
        layout = _leaf_layout(backend, space, what)
        if layout != first:
            raise ValueError(
                f"AsyncVecEnv(shared_memory=True) requires every worker to have the same "
                f"{what} leaves (paths, shapes and dtypes); worker 0 has {first} but worker "
                f"{index} has {layout}"
            )


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

def _cuda_mode(start_method : str) -> str:
    global _warned_no_cuda_ipc
    if start_method == "fork":
        raise ValueError(
            "AsyncVecEnv(shared_memory=True) shares CUDA observations through CUDA IPC, "
            "which needs the 'spawn' or 'forkserver' start method (CUDA cannot be used "
            "in forked children); pass ctx=multiprocessing.get_context('spawn') or "
            "leave ctx unset."
        )
    if not sys.platform.startswith("linux"):
        if not _warned_no_cuda_ipc:
            warnings.warn(
                "AsyncVecEnv(shared_memory=True): CUDA IPC is only supported on Linux; "
                "observations are staged through host shared memory instead (one "
                "device-to-host copy per worker and one host-to-device copy in the parent).",
                RuntimeWarning,
            )
            _warned_no_cuda_ipc = True
        return HOST_STAGED
    return CUDA_IPC


def select_mode(backend : ComputeBackend, device_description : Optional[Tuple[str, Any]], start_method : str) -> str:
    """Pick the buffer placement for a worker env on ``backend`` / ``device``.

    ``device_description`` is the worker-reported device (see
    ``vec_env._serialize_device``): ``("object", torch.device)`` or
    ``("str", "<platform>:<id>")`` for JAX.
    """
    name = backend.simplified_name
    if name == "numpy":
        return HOST
    if name == "pytorch":
        import torch
        device = torch.device("cpu") if device_description is None else torch.device(device_description[1])
        if device.type == "cpu":
            return HOST
        if device.type == "cuda":
            return _cuda_mode(start_method)
        return HOST_STAGED
    if name == "jax":
        if device_description is None:
            # Unknown placement (default device / sharding): go through the host.
            return HOST_STAGED
        platform = str(device_description[1]).split(":")[0]
        if platform == "cpu":
            return HOST
        if platform in ("gpu", "cuda"):
            try:
                import torch  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "AsyncVecEnv(shared_memory=True) with a JAX environment on a GPU needs "
                    "PyTorch, which carries the CUDA IPC buffers (JAX has no cross-process "
                    "buffer API). Install torch or use shared_memory=False."
                ) from e
            return _cuda_mode(start_method)
        # TPUs (and other accelerators) cannot be shared between processes.
        return HOST_STAGED
    return HOST_STAGED


# ---------------------------------------------------------------------------
# Helpers shared by both sides
# ---------------------------------------------------------------------------

def _segment_holder(segment : shared_memory.SharedMemory):
    """Buffer object for the views of ``segment``.

    numpy (and torch) views keep a reference to the object they were created
    from, but numpy does not keep a buffer *export* on it, so views created
    directly from ``segment.buf`` would not stop ``segment.close()`` from
    unmapping the memory under them. A ctypes array created with ``from_buffer``
    holds an export for as long as it lives; every view references it, so the
    mapping cannot be closed while any view exists (see :func:`_release_segment`).
    """
    return (ctypes.c_char * segment.size).from_buffer(segment.buf)


def _host_view(backend_name : str, buf : Any, offset : int, num_envs : int,
               shape : Tuple[int, ...], dtype : Any):
    """``(num_envs, *shape)`` view into a host segment holder (torch tensor for
    the pytorch backend, numpy array otherwise)."""
    full_shape = (num_envs, *shape)
    if backend_name == "pytorch":
        import torch
        count = math.prod(full_shape)
        if count == 0:
            return torch.empty(full_shape, dtype=dtype)
        return torch.frombuffer(buf, dtype=dtype, count=count, offset=offset).view(full_shape)
    return np.ndarray(full_shape, dtype=np.dtype(dtype), buffer=buf, offset=offset)


def _numpy_alias(tensor : Any) -> Optional[np.ndarray]:
    """numpy view sharing the memory of a CPU torch tensor (``None`` for dtypes
    numpy lacks, e.g. bfloat16).

    Host-side copies of torch tensors go through these views: ``torch`` copies
    of a few hundred KB are split over all intra-op threads, and N worker
    processes each spinning up an OpenMP pool oversubscribe the machine (a
    (3, 128, 128) uint8 image dropped from ~200 to ~10 steps/s with 8 workers).
    """
    try:
        return tensor.numpy()
    except (TypeError, RuntimeError):
        return None


def _torch_dtype_for(backend : ComputeBackend, dtype : Any):
    import torch
    if backend.simplified_name == "pytorch":
        return dtype
    np_dtype = np.dtype(dtype)
    if np_dtype.name == "bfloat16":
        return torch.bfloat16
    return torch.from_numpy(np.empty((0,), dtype=np_dtype)).dtype


def _itemsize(backend : ComputeBackend, dtype : Any) -> int:
    if backend.simplified_name == "pytorch":
        import torch
        return torch.empty((), dtype=dtype).element_size()
    return np.dtype(dtype).itemsize


# Segments whose mapping could neither be closed nor detached (see below); kept
# alive so that their views stay valid. Only used if CPython's internals change.
_UNDETACHABLE_SEGMENTS : List[shared_memory.SharedMemory] = []


def _release_segment(segment : shared_memory.SharedMemory, unlink : bool) -> None:
    """Close (and optionally unlink) a segment, tolerating live views.

    Views handed out with ``copy=False`` may outlive the vector env. Their mapping
    must stay valid until they are garbage collected: closing fails with a
    ``BufferError`` while views exist (they hold an export through
    :func:`_segment_holder`). In that case the mapping is detached from the
    ``SharedMemory`` object (whose ``__del__`` would retry and fail) and is
    unmapped by the views themselves once the last one is collected. The name is
    unlinked right away either way, so nothing is left in ``/dev/shm``.
    """
    if unlink:
        try:
            segment.unlink()
        except FileNotFoundError:
            pass
    try:
        segment.close()
    except BufferError:
        try:
            segment._buf = None
            segment._mmap = None
            segment.close()  # closes the file descriptor
        except Exception:
            _UNDETACHABLE_SEGMENTS.append(segment)


def _attach_segment(name : str) -> shared_memory.SharedMemory:
    try:
        # Python >= 3.13: the parent owns (and unlinks) the segment.
        return shared_memory.SharedMemory(name=name, create=False, track=False)
    except TypeError:
        # Python < 3.13 registers the segment with the resource tracker, which the
        # workers share with the parent, so the parent's unlink unregisters it.
        return shared_memory.SharedMemory(name=name, create=False)


def _get_leaf(data : Any, path : PathType, what : str) -> Any:
    value = data
    for depth, key in enumerate(path):
        if isinstance(key, int):
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or key >= len(value):
                raise ValueError(
                    f"AsyncVecEnv(shared_memory=True): expected a sequence with index {key} at "
                    f"{what}{_format_path(path[:depth])}, got {type(value).__name__}"
                )
        elif not isinstance(value, Mapping) or key not in value:
            raise ValueError(
                f"AsyncVecEnv(shared_memory=True): expected a mapping with key {key!r} at "
                f"{what}{_format_path(path[:depth])}, got {type(value).__name__}"
            )
        value = value[key]
    return value


def _build_tree(space : Space, leaves : "iter") -> Any:
    if isinstance(space, DictSpace):
        return {key: _build_tree(sub_space, leaves) for key, sub_space in space.spaces.items()}
    if isinstance(space, TupleSpace):
        return tuple(_build_tree(sub_space, leaves) for sub_space in space.spaces)
    return next(leaves)


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------

class ParentSharedBuffers:
    """Buffers owned by the parent process (see module docstring).

    ``groups`` maps a group name (``"obs"``, ``"context"``) to the
    single-environment space of that group.
    """

    def __init__(
        self,
        backend : ComputeBackend,
        device : Any,
        num_envs : int,
        groups : Dict[str, Space],
        mode : str,
        copy : bool,
    ):
        self.backend = backend
        self.backend_name = backend.simplified_name
        self.device = device
        self.num_envs = num_envs
        self.mode = mode
        self.copy = copy
        self.spaces = dict(groups)
        self.segment : Optional[shared_memory.SharedMemory] = None
        self._finalizer = None
        self._ipc_aliases : Dict[str, List[Any]] = {}
        self._numpy_buffers : Dict[str, List[Optional[np.ndarray]]] = {}
        self.torch_device = None

        # (path, shape, serialized dtype, offset) per leaf and group
        self.layout : Dict[str, List[Tuple[PathType, Tuple[int, ...], str, int]]] = {}
        dtypes : Dict[str, List[Any]] = {}
        offset = 0
        for group, space in self.spaces.items():
            self.layout[group] = []
            dtypes[group] = []
            for path, leaf in space_leaves(space, "context" if group == "context" else "observation"):
                dtype = _canonical_dtype(backend, leaf.dtype)
                shape = tuple(leaf.shape)
                self.layout[group].append((path, shape, _dtype_to_str(backend, dtype), offset))
                dtypes[group].append(dtype)
                nbytes = num_envs * math.prod(shape) * _itemsize(backend, dtype)
                offset += -(-nbytes // _ALIGNMENT) * _ALIGNMENT

        self.buffers : Dict[str, List[Any]] = {}
        if mode == CUDA_IPC:
            import torch
            if self.backend_name == "pytorch":
                self.torch_device = torch.device(device)
            else:
                if device is None or getattr(device, "platform", None) not in ("gpu", "cuda"):
                    raise RuntimeError(
                        "AsyncVecEnv(shared_memory=True): the workers' JAX env runs on a GPU, "
                        "but the parent process cannot see that GPU through JAX (is a "
                        "CUDA-enabled jaxlib installed in the parent environment?)."
                    )
                ordinal = getattr(device, "local_hardware_id", None)
                self.torch_device = torch.device("cuda", device.id if ordinal is None else ordinal)
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "AsyncVecEnv(shared_memory=True): the workers' env runs on a CUDA device, "
                    "but torch cannot use CUDA in the parent process."
                )
            for group, leaves in self.layout.items():
                self.buffers[group] = [
                    torch.zeros((num_envs, *shape), dtype=_torch_dtype_for(backend, dtype), device=self.torch_device)
                    for (_, shape, _, _), dtype in zip(leaves, dtypes[group])
                ]
            # The workers receive DLPack aliases of the buffers (same memory, separate
            # storage objects): torch tracks CUDA IPC lifetimes per storage and warns
            # at exit about every *shared* storage that is still alive. Only the
            # aliases are shared, and they are dropped in ``close()`` once the
            # workers have exited, so buffers handed out with ``copy=False`` may
            # outlive the env without that warning.
            self._ipc_aliases = {
                group: [torch.from_dlpack(buffer) for buffer in buffers]
                for group, buffers in self.buffers.items()
            }
        else:
            self.segment = shared_memory.SharedMemory(create=True, size=max(offset, 1))
            self._finalizer = weakref.finalize(self, _release_segment, self.segment, True)
            holder = _segment_holder(self.segment)
            for group, leaves in self.layout.items():
                self.buffers[group] = [
                    _host_view(self.backend_name, holder, leaf_offset, num_envs, shape, dtype)
                    for (_, shape, _, leaf_offset), dtype in zip(leaves, dtypes[group])
                ]
                if self.backend_name == "pytorch":
                    self._numpy_buffers[group] = [_numpy_alias(buffer) for buffer in self.buffers[group]]

    @property
    def segment_name(self) -> Optional[str]:
        return None if self.segment is None else self.segment.name

    def worker_spec(self) -> Dict[str, Any]:
        """Picklable description sent to every worker (CUDA buffers are shared via IPC)."""
        return {
            "mode": self.mode,
            "num_envs": self.num_envs,
            "segment": self.segment_name,
            "layout": self.layout,
            "cuda_buffers": self._ipc_aliases if self.mode == CUDA_IPC else None,
        }

    def before_dispatch(self) -> None:
        """Make sure no pending parent-side work still reads the buffers before
        the workers overwrite them (CUDA reads are asynchronous)."""
        if self.mode == CUDA_IPC:
            import torch
            torch.cuda.synchronize(self.torch_device)

    def read(self, group : str, indices : Optional[Sequence[int]] = None) -> Any:
        """Return the batched data of ``group`` (only rows ``indices`` if given)."""
        numpy_buffers = self._numpy_buffers.get(group) or [None] * len(self.buffers[group])
        leaves = iter([
            self._read_leaf(buffer, numpy_buffer, indices)
            for buffer, numpy_buffer in zip(self.buffers[group], numpy_buffers)
        ])
        return _build_tree(self.spaces[group], leaves)

    def _read_leaf(self, buffer : Any, numpy_buffer : Optional[np.ndarray], indices : Optional[Sequence[int]]) -> Any:
        mode, name = self.mode, self.backend_name
        if name == "pytorch":
            import torch
            if numpy_buffer is not None and mode == HOST_STAGED and indices is None:
                # a single host -> device transfer straight from the shared buffer
                return torch.from_numpy(numpy_buffer).to(self.device, copy=True)
            if numpy_buffer is not None and (mode == HOST_STAGED or indices is not None or self.copy):
                # host copy through numpy (see ``_numpy_alias``)
                host = numpy_buffer.copy() if indices is None else numpy_buffer[np.asarray(indices, dtype=np.intp)]
                data = torch.from_numpy(host)
                return data.to(self.device) if mode == HOST_STAGED else data
            if indices is not None:
                data = buffer.index_select(0, torch.as_tensor(indices, dtype=torch.long, device=buffer.device))
                return data.to(self.device) if mode == HOST_STAGED else data
            if mode == HOST_STAGED:
                # one host -> device transfer (``copy=True`` so that it never aliases
                # the host buffer, even if the device turns out to be the CPU)
                return buffer.to(self.device, copy=True)
            return buffer.clone() if self.copy else buffer
        if name == "jax":
            import jax
            import jax.numpy as jnp
            if mode == CUDA_IPC:
                import torch
                if indices is None:
                    data = buffer.clone() if self.copy else buffer
                else:
                    data = buffer.index_select(0, torch.as_tensor(indices, dtype=torch.long, device=buffer.device))
                if data is not buffer:
                    torch.cuda.current_stream(self.torch_device).synchronize()
                return jax.dlpack.from_dlpack(data)
            host = buffer if indices is None else buffer[np.asarray(indices, dtype=np.intp)]
            if mode == HOST and indices is None and not self.copy:
                try:
                    return jax.dlpack.from_dlpack(host)
                except Exception:
                    pass
            # ``jnp.array(copy=True)`` never aliases the host memory (``device_put``
            # may on the CPU); block so the source can be overwritten afterwards.
            return jnp.array(host, copy=True, device=self.device).block_until_ready()
        # numpy
        if indices is None:
            return buffer.copy() if self.copy else buffer
        return buffer[np.asarray(indices, dtype=np.intp)]

    def close(self) -> None:
        """Release the buffers. Call only once the workers have exited (CUDA IPC
        buffers must outlive every consumer)."""
        self.buffers = {}
        self._numpy_buffers = {}
        self._ipc_aliases = {}
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self.segment = None


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

class WorkerSharedBuffers:
    """Worker ``index``'s rows of the parent's buffers."""

    def __init__(self, spec : Dict[str, Any], index : int, backend : ComputeBackend):
        self.backend = backend
        self.backend_name = backend.simplified_name
        self.mode = spec["mode"]
        self.index = index
        self.segment : Optional[shared_memory.SharedMemory] = None
        self.cuda_device = None
        self.rows : Dict[str, List[Any]] = {}
        self._numpy_rows : Dict[str, List[Optional[np.ndarray]]] = {}
        self.leaves : Dict[str, List[Tuple[PathType, Tuple[int, ...], Any]]] = {}
        for group, layout in spec["layout"].items():
            self.leaves[group] = [
                (path, shape, _dtype_from_str(backend, dtype))
                for path, shape, dtype, _ in layout
            ]
        if self.mode == CUDA_IPC:
            for group, buffers in spec["cuda_buffers"].items():
                self.rows[group] = [buffer[index, ...] for buffer in buffers]
                if len(buffers) > 0:
                    self.cuda_device = buffers[0].device
        else:
            self.segment = _attach_segment(spec["segment"])
            holder = _segment_holder(self.segment)
            view_backend = "pytorch" if self.backend_name == "pytorch" else "numpy"
            for group, layout in spec["layout"].items():
                self.rows[group] = [
                    _host_view(view_backend, holder, offset, spec["num_envs"], shape, dtype)[index, ...]
                    for (_, shape, _, offset), (_, _, dtype) in zip(layout, self.leaves[group])
                ]
                if view_backend == "pytorch":
                    self._numpy_rows[group] = [_numpy_alias(row) for row in self.rows[group]]

    def has_group(self, group : str) -> bool:
        return group in self.rows

    def write(self, group : str, data : Any) -> None:
        """Validate ``data`` against the declared space and write it into this worker's row."""
        what = "context" if group == "context" else "observation"
        values = []
        for path, shape, dtype in self.leaves[group]:
            value = _get_leaf(data, path, what)
            value_shape = getattr(value, "shape", None)
            value_dtype = getattr(value, "dtype", None)
            if value_shape is None or value_dtype is None:
                raise TypeError(
                    f"AsyncVecEnv(shared_memory=True): {what}{_format_path(path)} must be an array "
                    f"of shape {shape} and dtype {dtype}, got {type(value).__name__}"
                )
            if tuple(value_shape) != shape or value_dtype != dtype:
                raise ValueError(
                    f"AsyncVecEnv(shared_memory=True): {what}{_format_path(path)} does not match "
                    f"the declared space: expected shape {shape} and dtype {dtype}, got shape "
                    f"{tuple(value_shape)} and dtype {value_dtype} (shared buffers are never cast "
                    f"or reshaped silently)"
                )
            values.append(value)
        if self.backend_name == "jax" and self.mode == CUDA_IPC:
            import jax
            jax.block_until_ready(values)
        numpy_rows = self._numpy_rows.get(group) or [None] * len(values)
        for row, numpy_row, value in zip(self.rows[group], numpy_rows, values):
            self._write_leaf(row, numpy_row, value)

    def _write_leaf(self, row : Any, numpy_row : Optional[np.ndarray], value : Any) -> None:
        if self.backend_name == "pytorch":
            if numpy_row is not None and value.device.type == "cpu":
                # host copy through numpy (see ``_numpy_alias``)
                numpy_value = _numpy_alias(value.detach())
                if numpy_value is not None:
                    np.copyto(numpy_row, numpy_value)
                    return
            row.copy_(value)
        elif self.mode == CUDA_IPC:  # jax on a GPU -> torch buffer, zero-copy import
            import torch
            row.copy_(torch.from_dlpack(value))
        else:  # numpy / jax (device -> host for staged devices)
            row[...] = np.asarray(value)

    def finish(self) -> None:
        """Wait until every write has landed before the parent is told so."""
        if self.mode == CUDA_IPC and self.cuda_device is not None:
            import torch
            torch.cuda.synchronize(self.cuda_device)

    def close(self) -> None:
        self.rows = {}
        self._numpy_rows = {}
        if self.segment is not None:
            _release_segment(self.segment, unlink=False)
            self.segment = None
