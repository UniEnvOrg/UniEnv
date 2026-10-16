import numpy as np
from typing import Optional, Type, Tuple
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

# Seeds handed out by ``next_seed_rng`` are drawn from [0, 2**31 - 1) so that they
# are accepted by every backend's RNG constructor (numpy rejects negative seeds,
# JAX without x64 only handles 32-bit integers).
_MAX_PORTABLE_SEED = 2**31 - 1

def next_seed(np_rng : np.random.Generator) -> int:
    return np_rng.integers(0, 2**32)

def next_seed_rng(rng : BRNGType, backend : ComputeBackend) -> Tuple[
    BRNGType,
    int
]:
    """Draw a non-negative integer seed from ``rng``, returning the advanced rng and the seed."""
    iinfo = backend.iinfo(backend.default_integer_dtype)
    rng, sample = backend.random.random_discrete_uniform(
        (1,),
        0,
        min(int(iinfo.max), _MAX_PORTABLE_SEED),
        rng=rng,
        dtype=backend.default_integer_dtype,
        device=None if not hasattr(rng, 'device') else rng.device
    )
    return rng, int(sample[0])
