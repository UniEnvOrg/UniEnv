"""Unit tests for ``UnionSpace``."""

import numpy as np
import pytest

from unienv_interface.space import BoxSpace, TextSpace, UnionSpace
from tests_common.backends import backend_params


def _union(backend, device=None):
    return UnionSpace(backend, [
        BoxSpace(backend, 0.0, 1.0, backend.default_floating_dtype, shape=(2,), device=device),
        TextSpace(backend, 3, charset="ab", device=device),
    ], device=device)


@pytest.mark.parametrize("backend,device", backend_params())
def test_sample_on_every_backend(backend, device):
    space = _union(backend, device)
    rng = backend.random.random_number_generator(0, device=device)
    seen = set()
    for _ in range(12):
        rng, x = space.sample(rng)
        assert space.contains(x)
        seen.add(x[0])
    assert seen == {0, 1}


def test_contains_rejects_malformed():
    from unienv_interface.backends.numpy import NumpyComputeBackend as B
    space = _union(B)
    assert not space.contains([0, np.zeros(2)])
    assert not space.contains((2, "a"))
    assert not space.contains((1, np.zeros(2)))
    assert not space.contains((0, np.zeros(2), 1))


def test_contains_accepts_any_integer_index():
    # Regression: only ``int`` / ``np.int64`` indices were accepted, e.g. ``np.int32`` was rejected
    from unienv_interface.backends.numpy import NumpyComputeBackend as NB
    space = _union(NB)
    assert space.contains((np.int32(1), "ab"))
    assert space.contains((np.uint8(0), np.zeros(2, dtype=NB.default_floating_dtype)))
    assert not space.contains((np.int32(2), "ab"))
    assert not space.contains((1.0, "ab"))
