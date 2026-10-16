from typing import Any, Tuple, Union, Optional, List, Dict, Type, TypeVar, Generic, Callable, Sequence
from unienv_data.base import BatchBase, BatchT, SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType, BatchSampler
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.utils.seed_util import next_seed_rng
import multiprocessing as mp
from multiprocessing.context import BaseContext as MPContext
from multiprocessing.synchronize import Event as MPEvent
import copy
import queue
import sys
import time

def _limit_worker_intraop_threads() -> None:
    """
    Run each worker's tensor ops single-threaded, as torch's ``DataLoader`` workers do.

    A ``fork``-ed child inherits the parent's OpenMP thread-pool state but not its threads,
    so once the parent has run a parallel torch op, the worker's first parallel op (e.g.
    gathering a large image batch) waits forever for threads that do not exist. The workers
    provide the parallelism instead. torch is only touched if it is already imported.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)

def worker_loop(
    fetch_fn : Callable[[BatchSampler[
        SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ], Any], Any],
    workid_queue : mp.Queue,
    result_queue : mp.Queue,
    done_event : MPEvent,
    doze_time : float = 0.005,
    init_fn : Optional[Callable[[int], None]] = None,
    seed : Optional[int] = None,
):
    _limit_worker_intraop_threads()
    # Every worker starts from a copy of the parent's sampler (and RNG state); reseed it so
    # that workers do not all draw the same batches.
    if init_fn is not None and seed is not None:
        init_fn(seed)

    try:
        while True:
            if done_event.is_set():
                break
            try:
                work_info = workid_queue.get(timeout=doze_time)
            except queue.Empty:
                continue
            except ValueError:
                break
            
            result = fetch_fn(work_info)
            result_queue.put(result)
            del result
    except KeyboardInterrupt:
        pass
    finally:
        # Results nobody will fetch any more must not keep this process alive on exit.
        result_queue.cancel_join_thread()

class _SamplerTask:
    """
    Picklable worker task calling ``fn()`` (or ``fn(work_info)``) on a sampler.

    Local closures cannot be sent to ``spawn``/``forkserver`` workers, so the task is a
    module-level object. When pickled, the sampler is shallow-copied without its RNG
    state (``torch.Generator`` state cannot travel through multiprocessing pickling);
    :meth:`reseed` re-creates those RNGs in the worker.
    """
    _RNG_ATTRS = ("rng", "data_rng")

    def __init__(
        self,
        sampler : BatchSampler,
        fn : Callable[..., Any],
        takes_work_info : bool,
    ):
        self.sampler = sampler
        if getattr(fn, "__self__", None) is sampler:
            self.fn_name, self.fn = fn.__name__, None
        else:
            self.fn_name, self.fn = None, fn
        self.takes_work_info = takes_work_info
        self._stripped_rngs : Tuple[str, ...] = ()

    def __call__(self, work_info : Any) -> Any:
        fn = getattr(self.sampler, self.fn_name) if self.fn_name is not None else self.fn
        return fn(work_info) if self.takes_work_info else fn()

    def reseed(self, seed : int) -> None:
        sampler = self.sampler
        if not self._stripped_rngs:
            sampler.manual_seed(seed)
            return
        for name in self._stripped_rngs:
            device = sampler.device if name == "rng" else sampler.data.device
            setattr(sampler, name, sampler.backend.random.random_number_generator(seed, device=device))

    def __getstate__(self):
        state = self.__dict__.copy()
        sampler_copy = copy.copy(self.sampler)
        instance_attrs = vars(sampler_copy)
        stripped = tuple(
            name for name in self._RNG_ATTRS if instance_attrs.get(name, None) is not None
        )
        for name in stripped:
            instance_attrs[name] = None
        state["sampler"] = sampler_copy
        state["_stripped_rngs"] = stripped
        return state

def _draw_worker_seeds(
    sampler : BatchSampler,
    n_workers : int,
) -> Optional[List[int]]:
    """Draw one seed per worker from the sampler's RNG (advancing it), or None if it has no RNG."""
    use_data_rng = sampler.data_rng is not None
    rng = sampler.data_rng if use_data_rng else sampler.rng
    if rng is None:
        return None
    seeds = []
    for _ in range(n_workers):
        rng, seed = next_seed_rng(rng, sampler.backend)
        seeds.append(seed % (2 ** 31))
    if use_data_rng:
        sampler.data_rng = rng
    else:
        sampler.rng = rng
    return seeds

TaskInfoT = TypeVar('TaskInfoT')
ResultT = TypeVar('ResultT')
class MultiProcessingSampleManager(Generic[TaskInfoT, ResultT]):
    def __init__(
        self,
        target_fn : Callable[[TaskInfoT], ResultT],
        n_workers : int = 4,
        ctx : Optional[MPContext] = None,
        done_event = None,
        doze_time : float = 0.001,
        daemon : Optional[bool] = None,
        worker_init_fn : Optional[Callable[[int], None]] = None,
        worker_seeds : Optional[Sequence[int]] = None,
    ):
        # Set first so that close() (also called from __del__) works on a half-built manager.
        self.closed = True
        self.workers : List[mp.Process] = []
        assert n_workers > 0
        assert worker_seeds is None or len(worker_seeds) == n_workers
        if ctx is None:
            ctx = mp.get_context()

        self.sampler_result_queue : mp.Queue = ctx.Queue()
        self.sampler_work_queue : mp.Queue = ctx.Queue()
        self.done_event = done_event or ctx.Event()
        self.target_fn = target_fn
        self.doze_time = doze_time
        self.remaining_work = 0

        for i in range(n_workers):
            worker = ctx.Process(
                target=worker_loop,
                args=(
                    self.target_fn,
                    self.sampler_work_queue, 
                    self.sampler_result_queue, 
                    self.done_event,
                    self.doze_time,
                    worker_init_fn,
                    None if worker_seeds is None else worker_seeds[i],
                ),
                daemon=daemon,
            )
            self.workers.append(worker)
        
        self.closed = False
        try:
            for worker in self.workers:
                worker.start()
        except BaseException:
            self.close()
            raise

    @property
    def n_workers(self):
        return len(self.workers)

    def close(self):
        if getattr(self, "closed", True):
            return
        
        self.done_event.set()
        self.sampler_result_queue.cancel_join_thread()
        self.sampler_result_queue.close()
        started = [worker for worker in self.workers if worker.pid is not None]
        for worker in started:
            worker.join(timeout=2)
        for worker in started:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1)
        self.closed = True
    
    def fetch(self, block : bool = True, timeout : Optional[float] = None) -> Optional[ResultT]:
        assert not self.closed, "Cannot fetch from a closed MultiProcessingSampleManager."
        assert self.remaining_work >= 1, "No work to fetch."
        try:
            ret = self.sampler_result_queue.get(block=block, timeout=timeout)
            self.remaining_work -= 1
        except queue.Empty:
            return None
        return ret

    def add_work(self, work_info : TaskInfoT):
        if self.closed:
            raise RuntimeError(f"Cannot add work to a closed {__class__.__name__}.")
        
        self.sampler_work_queue.put(work_info)
        self.remaining_work += 1
    
    def __del__(self):
        self.close()

def close_fn_for_wrapped_fn(wrapped_fn):
    def fn():
        if hasattr(wrapped_fn, "sample_manager") and wrapped_fn.sample_manager is not None:
            wrapped_fn.sample_manager.close()
            wrapped_fn.sample_manager = None
    return fn

def wrap_sample_function(
    sampler : BatchSampler[
        SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ],
    n_workers : int,
    n_buffers : int,
    fn : Callable[[], Any],
    ctx : Optional[mp.context.BaseContext] = None,
    doze_time : float = 0.001,
    daemon : Optional[bool] = None,
    done_event = None,
):
    fn_fetch = _SamplerTask(sampler, fn, takes_work_info=False)

    def target_fn():
        if hasattr(target_fn, "sample_manager") and target_fn.sample_manager is not None:
            sample_manager : MultiProcessingSampleManager = target_fn.sample_manager
        else:
            sample_manager = MultiProcessingSampleManager(
                fn_fetch, 
                n_workers=n_workers, 
                ctx=ctx,
                done_event=done_event,
                doze_time=doze_time,
                daemon=daemon,
                worker_init_fn=fn_fetch.reseed,
                worker_seeds=_draw_worker_seeds(sampler, n_workers),
            )
            for _ in range(n_buffers):
                sample_manager.add_work(1)
            target_fn.sample_manager = sample_manager
        
        sample_manager.add_work(1)
        return sample_manager.fetch()

    target_fn.close = close_fn_for_wrapped_fn(target_fn)
    return target_fn

def wrap_epoch_iter_function(
    sampler : BatchSampler[
        SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ],
    n_workers : int,
    n_buffers : int,
    getter_fn : Callable[[Any], Any],
    ctx : Optional[mp.context.BaseContext] = None,
    doze_time : float = 0.005,
    daemon : Optional[bool] = None,
    done_event = None,
):
    fn_fetch = _SamplerTask(sampler, getter_fn, takes_work_info=True)

    def target_fn():
        sample_manager = MultiProcessingSampleManager(
            fn_fetch, 
            n_workers=n_workers, 
            ctx=ctx,
            done_event=done_event,
            doze_time=doze_time,
            daemon=daemon,
        )
        target_fn.sample_manager = sample_manager

        try:
            if sampler.data_rng is not None:
                sampler.data_rng, idx = sampler.backend.random.random_permutation(len(sampler.data), rng=sampler.data_rng, device=sampler.data.device)
            else:
                sampler.rng, idx = sampler.backend.random.random_permutation(len(sampler.data), rng=sampler.rng, device=sampler.data.device)
            n_batches = len(sampler.data) // sampler.batch_size
            num_left = len(sampler.data) % sampler.batch_size
            
            total_batches = n_batches + (1 if num_left > 0 else 0)

            added_batch_i = 0

            def try_add_batch():
                nonlocal added_batch_i
                if added_batch_i < total_batches:
                    if added_batch_i < n_batches:
                        sample_manager.add_work(idx[added_batch_i*sampler.batch_size:(added_batch_i+1)*sampler.batch_size])
                    else:
                        sample_manager.add_work(idx[-num_left:])
                    added_batch_i += 1

            for _ in range(n_buffers):
                try_add_batch()

            for i in range(n_batches):
                try_add_batch()
                yield sample_manager.fetch()

            if num_left > 0:
                try_add_batch()
                yield sample_manager.fetch()
        finally:
            # Also runs when the consumer stops iterating early (generator close / GC).
            sample_manager.close()
            if getattr(target_fn, "sample_manager", None) is sample_manager:
                target_fn.sample_manager = None
    
    target_fn.close = close_fn_for_wrapped_fn(target_fn)

    return target_fn

class MultiprocessingSampler(
    BatchSampler[
        SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        sampler : BatchSampler[
            SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
            BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
        ],
        n_workers : int = 4,
        n_buffers : int = 8,
        ctx : Optional[mp.context.BaseContext] = None,
        doze_time : float = 0.005,
        daemon : Optional[bool] = None,
    ):
        #TODO: Implement metadata space for this sampler
        super().__init__(
            sampler.single_space,
            sampler.single_metadata_space,
            sampler.batch_size
        )
        assert n_workers > 0
        assert n_buffers > 0
        
        self.mp_ctx = ctx
        self.sampler = sampler
        self.closed = False
        self.n_workers = n_workers
        self.n_buffers = n_buffers

        # Iterate through the wrapper sampler's methods and wrap them with the multiprocessing methods
        self.functions_to_close = []
        for name in dir(sampler):
            attr = getattr(sampler, name)
            if callable(attr) and name.startswith("sample"):
                wrapped_fn = wrap_sample_function(
                    sampler, self.n_workers, self.n_buffers, attr,
                    ctx=self.mp_ctx, doze_time=doze_time, daemon=daemon,
                )
                setattr(self, name, wrapped_fn)
                self.functions_to_close.append(wrapped_fn)
            elif callable(attr) and name.startswith("get_at"):
                new_name = "epoch_iter" + name[len("get_at"):]
                wrapped_fn = wrap_epoch_iter_function(
                    sampler, self.n_workers, self.n_buffers, attr,
                    ctx=self.mp_ctx, doze_time=doze_time, daemon=daemon,
                )
                setattr(self, new_name, wrapped_fn)
                self.functions_to_close.append(wrapped_fn)
            elif callable(attr) and name.startswith("get_flattened_at"):
                new_name = "epoch_flat_iter" + name[len("get_flattened_at"):]
                wrapped_fn = wrap_epoch_iter_function(
                    sampler, self.n_workers, self.n_buffers, attr,
                    ctx=self.mp_ctx, doze_time=doze_time, daemon=daemon,
                )
                setattr(self, new_name, wrapped_fn)
                self.functions_to_close.append(wrapped_fn)
        assert hasattr(self.sample, "close")

    def manual_seed(self, seed):
        self.sampler.manual_seed(seed)

    @property
    def sampled_space(self):
        return self.sampler.sampled_space
    
    @property
    def sampled_metadata_space(self):
        return self.sampler.sampled_metadata_space

    @property
    def backend(self):
        return self.sampler.backend
    
    @property
    def device(self):
        return self.sampler.device
    
    @property
    def data(self):
        return self.sampler.data

    @property
    def rng(self):
        return self.sampler.rng
    
    @rng.setter
    def rng(self, rng):
        self.sampler.rng = rng

    @property
    def data_rng(self):
        return self.sampler.data_rng
    
    @data_rng.setter
    def data_rng(self, data_rng):
        self.sampler.data_rng = data_rng

    def get_at(self, idx):
        return self.sampler.get_at(idx)
    
    def get_at_with_metadata(self, idx):
        return self.sampler.get_at_with_metadata(idx)
    
    def get_flattened_at(self, idx):
        return self.sampler.get_flattened_at(idx)
    
    def get_flattened_at_with_metadata(self, idx):
        return self.sampler.get_flattened_at_with_metadata(idx)

    def close(self):
        if getattr(self, "closed", True):
            return
        
        for fn in self.functions_to_close:
            fn.close()
        self.closed = True
    