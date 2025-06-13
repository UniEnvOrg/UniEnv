import numpy as np
from typing import Optional, Type, Tuple
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

def next_seed(np_rng : np.random.Generator) -> int:
    return np_rng.integers(0, 2**32)

def next_seed_rng(rng : BRNGType, backend : ComputeBackend) -> Tuple[
    BRNGType,
    int
]:
    iinfo = backend.iinfo(backend.default_integer_dtype)
    rng, sample = backend.random.random_discrete_uniform(
        (1,),
        iinfo.min,
        iinfo.max,
        rng=rng,
        dtype=backend.default_integer_dtype,
        device=None if not hasattr(rng, 'device') else rng.device
    )
    return rng, int(sample[0])