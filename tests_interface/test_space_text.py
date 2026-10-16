"""Unit tests for ``TextSpace``."""

import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import TextSpace
from tests_common.backends import backend_params


@pytest.mark.parametrize("backend,device", backend_params())
def test_sample_on_every_backend_and_device(backend, device):
    space = TextSpace(backend, 6, min_length=2, charset="xyz", device=device)
    rng = backend.random.random_number_generator(0, device=device)
    for _ in range(5):
        rng, x = space.sample(rng)
        assert isinstance(x, str) and 2 <= len(x) <= 6 and set(x) <= set("xyz")
        assert space.contains(x)


def test_contains_and_bounds():
    space = TextSpace(B, 3, min_length=1, charset="ab")
    assert space.contains("ab") and not space.contains("") and not space.contains("abab")
    assert not space.contains("ac") and not space.contains(3)
    assert space.is_bounded("both")
    free = TextSpace(B, 3)
    assert free.contains("Z9") and free.is_bounded("below") and not free.is_bounded("above")
    assert space.character_index("b") == 1 and space.character_index("q") is None
    assert free.character_index("a") is None
    # Long strings may exceed the charset size.
    _, x = TextSpace(B, 10, min_length=10, charset="a").sample(B.random.random_number_generator(0))
    assert x == "a" * 10
