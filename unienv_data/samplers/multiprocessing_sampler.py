from typing import Any, Tuple, Union, Optional, List, Dict, Type, TypeVar, Generic
from unienv_data.base import BatchBase, BatchT, SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType, BatchSampler
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, flatten_utils as sfu, batch_utils as sbu
from unienv_interface.utils.seed_util import next_seed_rng
import multiprocessing as mp
import queue
import time

try:
    import torch
except ImportError:
    torch = None

def worker_loop(
    sampler : BatchSampler[
        SamplerBatchT, SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType,
        BatchT, BArrayType, BDeviceType, BDtypeType, BRNGType
    ],
    seed : Optional[int],
    workid_queue : mp.Queue,
    sampler_result_queue : mp.Queue,
    done_event,
):
    if seed is not None:
        sampler.rng = sampler.backend.random_number_generator(seed, device=sampler.device)
        sampler.data_rng = sampler.backend.random_number_generator(seed, device=sampler.data.device)

    try:
        while True:
            try:
                work_info = workid_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except ValueError:
                break

            if done_event.is_set():
                break

            sample = sampler.sample()
            sampler_result_queue.put(sample)
            del sample
    except KeyboardInterrupt:
        pass
    finally:
        if seed is not None:
            del sampler.rng
            del sampler.data_rng
            sampler.rng = None
            sampler.data_rng = None
        
        del sampler
        sampler_result_queue.cancel_join_thread()
        sampler_result_queue.close()

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
    ):
        assert n_workers > 0
        assert n_buffers > 0
        if ctx is None:
            ctx = mp

        self.sampler = sampler
        self.sampler_result_queue = ctx.Queue()
        self.sampler_work_queue = ctx.Queue()
        self.done_event = ctx.Event()

        # ===== Cache RNGs =====
        tmp_rng = sampler.rng
        tmp_data_rng = sampler.data_rng
        cache_rng = torch is not None and (isinstance(tmp_rng, torch.Generator) or isinstance(tmp_data_rng, torch.Generator))
        if cache_rng:
            # For some reason we cannot pickle pytorch's Generator object
            sampler.rng = None
            sampler.data_rng = None
        # ===== End Cache RNGs =====

        self.workers = []
        for i in range(n_workers):
            if cache_rng and (tmp_rng is not None or tmp_data_rng is not None):
                if tmp_rng is not None:
                    tmp_rng, seed = next_seed_rng(tmp_rng, sampler.backend)
                else:
                    tmp_data_rng, seed = next_seed_rng(tmp_data_rng, sampler.backend)
            else:
                seed = None
            
            worker = ctx.Process(
                target=worker_loop,
                args=(sampler, seed, self.sampler_work_queue, self.sampler_result_queue, self.done_event)
            )
            self.workers.append(worker)
        
        self.n_buffers = n_buffers
        for i in range(n_buffers):
            self.sampler_work_queue.put(1)
        
        for worker in self.workers:
            worker.start()
        
        if cache_rng:
            sampler.rng = tmp_rng
            sampler.data_rng = tmp_data_rng
        
        self.closed = False

    @property
    def batch_size(self):
        return self.sampler.batch_size
    
    @property
    def sampled_space(self):
        return self.sampler.sampled_space
    
    @property
    def sampled_space_flat(self):
        return self.sampler.sampled_space_flat

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

    @property
    def n_workers(self):
        return len(self.workers)

    def close(self):
        if self.closed:
            return
        
        self.sampler_result_queue.cancel_join_thread()
        self.sampler_result_queue.close()
        self.done_event.set()
        for _ in range(self.n_buffers):
            self.sampler_work_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=2)
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
        self.closed = True
    
    def sample(self):
        samp = self.sampler_result_queue.get()
        self.sampler_work_queue.put(1)
        return samp