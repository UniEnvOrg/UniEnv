"""Unit tests for ``BinarySpace``."""

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BinarySpace
from tests_common.backends import backend_params


@pytest.mark.parametrize("backend,device", backend_params())
@pytest.mark.parametrize("shape", [(), (3,), (2, 2)])
def test_sample_is_a_boolean_array(backend, device, shape):
    space = BinarySpace(backend, shape, device=device)
    rng = backend.random.random_number_generator(0, device=device)
    for _ in range(3):
        rng, x = space.sample(rng)
        assert backend.is_backendarray(x) and tuple(x.shape) == shape
        assert space.contains(x)


def test_contains_and_equality():
    space = BinarySpace(B, (2,))
    assert space.contains(np.array([True, False]))
    assert not space.contains(np.array([1, 0]))
    assert not space.contains(np.array([True]))
    assert not space.contains([True, False])
    assert space == BinarySpace(B, (2,)) and space != BinarySpace(B, (3,))
    assert space.is_bounded() and space.create_empty().shape == (2,)
    assert "BinarySpace" in repr(space)
    with pytest.raises(AssertionError):
        BinarySpace(B, (2,), dtype=np.float32)
