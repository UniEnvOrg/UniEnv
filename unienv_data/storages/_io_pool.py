"""A process-wide, bounded thread pool for the per-file IO of file-backed storages."""

from typing import Any, Callable, Iterable, List, Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor

import os
import threading

T = TypeVar("T")

#: Worker threads of the shared pool. Image / video decoding (PIL, zlib, PyAV) and file IO release the GIL,
#: so per-file work scales with cores up to here.
IO_POOL_WORKERS : int = min(8, os.cpu_count() or 1)
#: Batched reads / writes of fewer items than this run in the calling thread.
PARALLEL_IO_MIN_ITEMS : int = 4

_THREAD_NAME_PREFIX = "unienv-storage-io"

_pool : Optional[ThreadPoolExecutor] = None
_pool_pid : Optional[int] = None
_pool_lock = threading.Lock()


def _forget_pool_after_fork() -> None:
    # Worker threads do not survive fork(), and the lock may have been held by a thread that no longer exists
    global _pool, _pool_pid, _pool_lock
    _pool, _pool_pid, _pool_lock = None, None, threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_pool_after_fork)


def shared_io_pool() -> ThreadPoolExecutor:
    """The shared pool of this process, created on first use (and again in a forked child)."""
    global _pool, _pool_pid
    pid = os.getpid()
    if _pool is None or _pool_pid != pid:
        with _pool_lock:
            if _pool is None or _pool_pid != pid:
                _pool = ThreadPoolExecutor(max_workers=IO_POOL_WORKERS, thread_name_prefix=_THREAD_NAME_PREFIX)
                _pool_pid = pid
    return _pool


def in_io_pool_thread() -> bool:
    return threading.current_thread().name.startswith(_THREAD_NAME_PREFIX)


def map_io(fn : Callable[[Any], T], items : Iterable[Any], min_parallel : int = PARALLEL_IO_MIN_ITEMS) -> List[T]:
    """
    `list(map(fn, items))`, run on the shared pool when there are at least `min_parallel` items.
    Calls made from a pool worker run serially: waiting on the bounded pool from inside it could deadlock.
    """
    items = list(items)
    if len(items) < max(2, min_parallel) or IO_POOL_WORKERS <= 1 or in_io_pool_thread():
        return [fn(item) for item in items]
    return list(shared_io_pool().map(fn, items))
