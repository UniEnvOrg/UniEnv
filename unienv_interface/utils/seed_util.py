import numpy as np
from typing import Optional, Type, Tuple
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

def next_seed(np_rng : np.random.Generator) -> int:
    return np_rng.integers(0, 2**32)

def next_seed_rng(rng : BRNGType, backend : ComputeBackend) -> Tuple[
    BRNGType,
    int
]:
    iinfo = backend.iinfo(backend.default_integer_dtype)
    rng, sample = backend.random.random_discrete_uniform(
        (1,),
        rng=rng,
        from_num=iinfo.min,
        to_num=iinfo.max,
        from_num=0,
        to_num=iinfo.max,
        dtype=backend.default_integer_dtype,
        device=None if not hasattr(rng, 'device') else rng.device
    )
    return rng, int(sample[0])