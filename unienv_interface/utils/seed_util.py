import numpy as np

def next_seed(np_rng : np.random.Generator) -> int:
    return np_rng.integers(0, 2**32)