"""Unit tests for ``DynamicBoxSpace``."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import DynamicBoxSpace
from tests_common.backends import cpu_backend_params


def _rng(seed=0, backend=B, device=None):
    return backend.random.random_number_generator(seed, device=device)


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_full_range_integer_samples_are_not_constant(backend, device):
    dtype = backend.default_integer_dtype
    # iinfo of the materialised dtype (jax without x64 stores ``int`` as int32)
    info = backend.iinfo(backend.zeros((1,), dtype=dtype).dtype)
    space = DynamicBoxSpace(backend, int(info.min), int(info.max), (1,), (8,), dtype, device=device)
    rng = _rng(0, backend, device)
    values = []
    for _ in range(5):
        rng, x = space.sample(rng)
        assert space.contains(x)
        values.extend(np.asarray(backend.to_numpy(x)).tolist())
    assert len(set(values)) > 1


def test_small_integer_range_hits_both_ends():
    space = DynamicBoxSpace(B, np.array([-2]), np.array([1]), (4,), (6,), np.int64)
    rng = _rng()
    values = set()
    for _ in range(20):
        rng, x = space.sample(rng)
        assert space.contains(x)
        values.update(x.tolist())
    assert values == {-2, -1, 0, 1}
