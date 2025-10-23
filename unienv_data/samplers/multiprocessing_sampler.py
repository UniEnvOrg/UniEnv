from typing import Any, Tuple, Union, Optional, List, Dict, Type, TypeVar, Generic, Callable
from unienv_data.base import BatchBase, BatchT, SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType, BatchSampler
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
from unienv_interface.utils.seed_util import next_seed_rng
import multiprocessing as mp
from multiprocessing.context import BaseContext as MPContext
from multiprocessing.synchronize import Event as MPEvent
import queue
import time

# try:
#     import torch
# except ImportError:
#     torch = None

def worker_loop(
    fetch_fn : Callable[[BatchSampler[
        SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ], Any], Any],
    workid_queue : mp.Queue,
    result_queue : mp.Queue,
    done_event : MPEvent,
    doze_time : float = 0.005,
):
    # if seed is not None:
    #     sampler.rng = sampler.backend.random.random_number_generator(seed, device=sampler.device)
    #     sampler.data_rng = sampler.backend.random.random_number_generator(seed, device=sampler.data.device)

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
        # if seed is not None:
        #     del sampler.rng
        #     del sampler.data_rng
        #     sampler.rng = None
        #     sampler.data_rng = None
        pass

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
    ):
        assert n_workers > 0
        if ctx is None:
            ctx = mp.get_context()

        self.sampler_result_queue : mp.Queue = ctx.Queue()
        self.sampler_work_queue : mp.Queue = ctx.Queue()
        self.done_event = done_event or ctx.Event()
        self.target_fn = target_fn
        self.doze_time = doze_time
        self.remaining_work = 0

        # ===== Cache RNGs =====
        # tmp_rng = sampler.rng
        # tmp_data_rng = sampler.data_rng
        # cache_rng = torch is not None and (isinstance(tmp_rng, torch.Generator) or isinstance(tmp_data_rng, torch.Generator))
        # if cache_rng:
        #     # For some reason we cannot pickle pytorch's Generator object
        #     sampler.rng = None
        #     sampler.data_rng = None
        # ===== End Cache RNGs =====

        self.workers : List[mp.Process] = []
        for i in range(n_workers):
            # if cache_rng and (tmp_rng is not None or tmp_data_rng is not None):
            #     if tmp_rng is not None:
            #         tmp_rng, seed = next_seed_rng(tmp_rng, sampler.backend)
            #     else:
            #         tmp_data_rng, seed = next_seed_rng(tmp_data_rng, sampler.backend)
            # else:
            #     seed = None

            worker = ctx.Process(
                target=worker_loop,
                args=(
                    self.target_fn,
                    self.sampler_work_queue, 
                    self.sampler_result_queue, 
                    self.done_event,
                    self.doze_time,
                ),
                daemon=daemon,
            )
            self.workers.append(worker)
        
        for worker in self.workers:
            worker.start()
        
        # if cache_rng:
        #     sampler.rng = tmp_rng
        #     sampler.data_rng = tmp_data_rng
        
        self.closed = False

    @property
    def n_workers(self):
        return len(self.workers)

    def close(self):
        if self.closed:
            return
        
        self.done_event.set()
        self.sampler_result_queue.cancel_join_thread()
        self.sampler_result_queue.close()
        for worker in self.workers:
            worker.join(timeout=2)
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
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
    def fn_fetch(_: Any):
        return fn()

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
    def fn_fetch(index: Any):
        ret = getter_fn(index)
        return ret

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

        if sampler.data_rng is not None:
            sampler.data_rng, idx = sampler.backend.random.random_permutation(len(sampler.data), rng=sampler.data_rng, device=sampler.data.device)
        else:
            sampler.rng, idx = sampler.backend.random.random_permutation(len(sampler.data), rng=sampler.rng, device=sampler.data.device)
        n_batches = len(sampler.data) // sampler.batch_size
        num_left = len(sampler.data) % sampler.batch_size
        
        total_batches = n_batches + (1 if num_left > 0 else 0)

        added_batch_i = 0
        fetch_batch_i = 0

        def try_add_batch():
            nonlocal added_batch_i
            nonlocal fetch_batch_i
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
        
        sample_manager.close()
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
        if self.closed:
            return
        
        for fn in self.functions_to_close:
            fn.close()
        self.closed = True
    