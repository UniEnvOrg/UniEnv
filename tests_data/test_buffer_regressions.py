"""Regressions of ReplayBufferCollectionWrapper (buffer swaps, reset boundaries, devices)."""
import os

import numpy as np
import pytest

from unienv_data.replay_buffer import ReplayBuffer, ReplayBufferCollectionWrapper
from unienv_data.storages.hdf5 import HDF5Storage
from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace

from tests_common.envs import StubEnv, make_spaces

ACTION, OBS, CONTEXT = make_spaces(None)
STEP_SPACE = DictSpace(B, {"obs": OBS, "next_obs": OBS, "action": ACTION,
                           "reward": BoxSpace(B, -np.inf, np.inf, np.float32, shape=())})


def _rb(tmp_path, name, **kwargs):
    return ReplayBuffer.create(HDF5Storage, STEP_SPACE, cache_path=str(tmp_path / name), **kwargs)


def _dumped(path):
    return os.path.exists(os.path.join(path, "metadata.json"))


def _run_unbatched_episode(wrapper):
    done = False
    while not done:
        _, _, terminated, truncated, _ = wrapper.step(np.ones(2, np.float32))
        done = terminated or truncated


# ---------------------------------------------------------------------------
# set_replay_buffers(dump_old=True)
# ---------------------------------------------------------------------------

def test_dump_old_dumps_shared_batched_buffers(tmp_path):
    # Regression: the dump only happened when an old buffer had an open segment, which
    # buffers recorded in shared B>1 mode (extend) never have: nothing was dumped.
    dump = str(tmp_path / "dump")
    rb = _rb(tmp_path, "old")
    wrapper = ReplayBufferCollectionWrapper(StubEnv(batch_size=2), rb, dump_path=dump)
    wrapper.reset(seed=0)
    for _ in range(3):
        wrapper.step(np.zeros((2, 2), np.float32))
    wrapper.set_replay_buffers(_rb(tmp_path, "new"), dump_old=True)
    assert _dumped(dump)
    assert len(ReplayBuffer.load_from(dump, backend=B)) == 6


def test_dump_old_dumps_after_a_closed_episode(tmp_path):
    dump = str(tmp_path / "dump")
    rb = _rb(tmp_path, "old")
    wrapper = ReplayBufferCollectionWrapper(StubEnv(terminate_above=0.5), rb, dump_path=dump)
    wrapper.reset(seed=0)
    _run_unbatched_episode(wrapper)
    assert not rb.storage.has_open_segment
    wrapper.set_replay_buffers(_rb(tmp_path, "new"), dump_old=True)
    assert _dumped(dump)
    assert len(ReplayBuffer.load_from(dump, backend=B)) == len(rb)


def test_dump_old_still_finalizes_open_segments_first(tmp_path):
    dump = str(tmp_path / "dump")
    rb = _rb(tmp_path, "old", maintain_segment_metadata=True)
    wrapper = ReplayBufferCollectionWrapper(StubEnv(terminate_above=100), rb, dump_path=dump)
    wrapper.reset(seed=0)
    wrapper.step(np.ones(2, np.float32))
    wrapper.set_replay_buffers(_rb(tmp_path, "new", maintain_segment_metadata=True), dump_old=True)
    assert not rb.storage.has_open_segment and rb.get_segments() == [(0, 1)]
    assert ReplayBuffer.load_from(dump, backend=B).get_segments() == [(0, 1)]


# ---------------------------------------------------------------------------
# set_replay_buffers(replay_mode="auto")
# ---------------------------------------------------------------------------

def test_auto_mode_on_swap_is_resolved_for_the_new_buffers(tmp_path):
    # Regression: "auto" was resolved from the old buffers, so swapping between one shared
    # buffer and one buffer per env slot raised a layout error.
    wrapper = ReplayBufferCollectionWrapper(StubEnv(batch_size=2), _rb(tmp_path, "shared"))
    assert wrapper.replay_mode == "shared"
    wrapper.reset(seed=0)
    wrapper.step(np.zeros((2, 2), np.float32))
    per_env = [_rb(tmp_path, f"env{i}", maintain_segment_metadata=True) for i in range(2)]
    wrapper.set_replay_buffers(per_env, replay_mode="auto")
    assert wrapper.replay_mode == "per_env"
    wrapper.step(np.zeros((2, 2), np.float32))
    assert [len(rb) for rb in per_env] == [1, 1]
    shared = _rb(tmp_path, "shared2")
    wrapper.set_replay_buffers(shared, replay_mode="auto", finalize_old_segments=True)
    assert wrapper.replay_mode == "shared"
    wrapper.step(np.zeros((2, 2), np.float32))
    assert len(shared) == 2


# ---------------------------------------------------------------------------
# Reset boundaries
# ---------------------------------------------------------------------------

def test_masked_reset_of_no_slot_keeps_the_shared_episode_open(tmp_path):
    # Regression: with batch_size=1, reset(mask=[False]) (e.g. reset(mask=done) every
    # step) closed the episode although the env slot was not reset.
    env = StubEnv(batch_size=1, terminate_above=100)
    rb = _rb(tmp_path, "rb", maintain_segment_metadata=True)
    wrapper = ReplayBufferCollectionWrapper(env, rb, dump_on_reset_boundary=True,
                                            dump_path=str(tmp_path / "dump"))
    wrapper.reset(seed=0)
    for _ in range(4):
        _, _, terminated, truncated, _ = wrapper.step(np.zeros((1, 2), np.float32))
        wrapper.reset(mask=terminated | truncated)
    assert rb.storage.has_open_segment and rb.get_segments() == []
    assert not _dumped(str(tmp_path / "dump"))
    wrapper.reset(mask=np.array([True]), seed=1)  # resetting the slot closes it
    assert rb.get_segments() == [(0, 4)]
    assert _dumped(str(tmp_path / "dump"))


def test_per_env_reset_boundary_dumps_after_a_natural_episode_end(tmp_path):
    # Regression: per_env mode only dumped on a reset that cut an episode short; after an
    # episode ended with terminated / truncated, the reset dumped nothing (shared mode did).
    dump = str(tmp_path / "dump")
    rbs = [_rb(tmp_path, f"env{i}", maintain_segment_metadata=True) for i in range(2)]
    wrapper = ReplayBufferCollectionWrapper(StubEnv(batch_size=2, terminate_above=0.5), rbs,
                                            replay_mode="per_env", dump_on_reset_boundary=True,
                                            dump_path=dump)
    wrapper.reset(seed=0)
    done = np.zeros(2, bool)
    while not done.any():
        _, _, terminated, truncated, _ = wrapper.step(np.ones((2, 2), np.float32))
        done = terminated | truncated
    assert not _dumped(dump)
    wrapper.reset(mask=done, seed=1)
    assert _dumped(dump)


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------

def _cuda_env_and_space():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    from unienv_interface.env_base.env import Env
    device = torch.device("cuda:0")

    class CudaEnv(Env):
        backend = TB
        render_mode = None
        render_fps = None
        metadata = {}

        def __init__(self, n=2):
            self.device = device
            self.batch_size = n
            self.action_space = BoxSpace(TB, -1, 1, torch.float32, shape=(n, 2), device=device)
            self.observation_space = BoxSpace(TB, -1e6, 1e6, torch.float32, shape=(n, 2), device=device)
            self.context_space = None
            self.x = torch.zeros(n, 2, device=device)

        def reset(self, *, mask=None, seed=None, **kwargs):
            self.x = torch.zeros_like(self.x)
            return None, self.x.clone(), {}

        def step(self, action):
            self.x = self.x + action
            flags = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
            return self.x.clone(), self.x.sum(-1), flags, flags.clone(), {}

        def render(self):
            return None

        def close(self):
            pass

    space = DictSpace(TB, {
        "obs": BoxSpace(TB, -1e6, 1e6, torch.float32, shape=(2,), device=device),
        "reward": BoxSpace(TB, -1e6, 1e6, torch.float32, shape=(), device=device),
        "env_index": BoxSpace(TB, 0, 100, torch.int64, shape=(), device=device),
    }, device=device)
    return torch, CudaEnv, space


@pytest.mark.parametrize("mode", ["shared", "per_env"])
@pytest.mark.parametrize("validate", [False, True], ids=["plain", "validated"])
def test_coerced_fields_follow_the_buffer_device(mode, validate):
    # Regression: env_index (and other coerced scalars) were created on the CPU, so
    # recording from a CUDA env into a CUDA buffer failed with a device mismatch.
    torch, CudaEnv, space = _cuda_env_and_space()
    from unienv_data.storages.flattened import FlattenedStorage
    from unienv_data.storages.pytorch import PytorchTensorStorage

    def make():
        return ReplayBuffer.create(FlattenedStorage, space, inner_storage_cls=PytorchTensorStorage,
                                   capacity=10, is_memmap=False,
                                   maintain_segment_metadata=mode == "per_env")

    rbs = [make(), make()] if mode == "per_env" else make()
    wrapper = ReplayBufferCollectionWrapper(CudaEnv(), rbs, replay_mode=mode, validate_transition=validate)
    wrapper.reset()
    wrapper.step(torch.ones(2, 2, device="cuda:0"))
    wrapper.step(torch.ones(2, 2, device="cuda:0"))
    if mode == "shared":
        assert rbs.get_at(Ellipsis)["env_index"].tolist() == [0, 1, 0, 1]
    else:
        assert [rb.get_at(Ellipsis)["env_index"].tolist() for rb in rbs] == [[0, 0], [1, 1]]
