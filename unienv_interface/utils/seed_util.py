import numpy as np
from typing import Optional, Type, Tuple
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

def next_seed(np_rng : np.random.Generator) -> int:
    return np_rng.integers(0, 2**32)

def next_seed_rng(rng : BRNGType, backend : Type[ComputeBackend]) -> Tuple[
    BRNGType,
    int
]:
    rng, sample = backend.random_discrete_uniform(
        rng,
        shape=(1,),
        from_num=0,
        to_num=2**32,
    )
    return rng, int(sample[0])