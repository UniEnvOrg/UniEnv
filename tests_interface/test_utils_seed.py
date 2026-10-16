"""Tests for ``unienv_interface.utils.seed_util``."""

import numpy as np
import pytest

from unienv_interface.utils import seed_util
from tests_common.backends import backend_params, backends_and_devices


def test_next_seed_is_in_uint32_range():
    rng = np.random.default_rng(0)
    seeds = [seed_util.next_seed(rng) for _ in range(100)]
    assert all(0 <= s < 2**32 for s in seeds)
    assert len(set(seeds)) > 90


@pytest.mark.parametrize("backend,device", backend_params())
def test_next_seed_rng_yields_seeds_usable_by_every_backend(backend, device):
    rng = backend.random.random_number_generator(1, device=device)
    seeds = []
    for _ in range(50):
        rng, seed = seed_util.next_seed_rng(rng, backend)
        assert isinstance(seed, int)
        assert seed >= 0
        seeds.append(seed)
        for target_backend, target_device in backends_and_devices():
            target_backend.random.random_number_generator(seed, device=target_device)
    assert len(set(seeds)) > 40


@pytest.mark.parametrize("backend,device", backend_params())
def test_next_seed_rng_is_deterministic(backend, device):
    a = backend.random.random_number_generator(7, device=device)
    b = backend.random.random_number_generator(7, device=device)
    _, seed_a = seed_util.next_seed_rng(a, backend)
    _, seed_b = seed_util.next_seed_rng(b, backend)
    assert seed_a == seed_b
