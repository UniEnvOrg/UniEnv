"""Tests for ``SyncVecEnv`` / ``AsyncVecEnv`` and ``data_stack``."""

import functools
import multiprocessing as mp
import os

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.vec_env import AsyncVecEnv, SyncVecEnv, data_stack
from tests_common.envs import StubEnv
from tests_common.vecenv_envs import ArrayEnv
from tests_common.trees import assert_tree_equal


# ---------------------------------------------------------------------------
# data_stack
# ---------------------------------------------------------------------------

def test_data_stack_scalars_and_arrays():
    out = data_stack([1, 2, 3], B)
    assert isinstance(out, np.ndarray) and out.tolist() == [1, 2, 3]
    out = data_stack([True, False], B)
    assert out.dtype == B.default_boolean_dtype
    out = data_stack([0.5, 1.5], B)
    assert out.dtype == B.default_floating_dtype
    out = data_stack([np.float32(0.5), np.float32(1.5)], B)
    np.testing.assert_allclose(out, [0.5, 1.5])
    out = data_stack([np.zeros(2), np.ones(2)], B)
    assert out.shape == (2, 2)
    assert data_stack([], B) == []


def test_data_stack_list_of_infos_becomes_dict_of_stacked_values():
    infos = [
        {"step": 1, "success": True, "pos": np.zeros(2), "nested": {"x": 0.5}},
        {"step": 2, "success": False, "pos": np.ones(2), "nested": {"x": 1.5}},
    ]
    out = data_stack(infos, B)
    assert isinstance(out, dict)
    np.testing.assert_array_equal(out["step"], [1, 2])
    np.testing.assert_array_equal(out["success"], [True, False])
    assert out["pos"].shape == (2, 2)
    np.testing.assert_allclose(out["nested"]["x"], [0.5, 1.5])


def test_data_stack_strings_do_not_recurse():
    out = data_stack([{"name": "a", "step": 0}, {"name": "bb", "step": 1}], B)
    assert out["name"] == ["a", "bb"]
    assert data_stack(["x", "y"], B) == ["x", "y"]
    assert data_stack([None, None], B) == [None, None]


def test_data_stack_mismatched_info_keys_are_kept_per_env():
    infos = [{"a": 1}, {"b": 2}]
    out = data_stack(infos, B)
    assert out == [{"a": 1}, {"b": 2}]


# ---------------------------------------------------------------------------
# SyncVecEnv
# ---------------------------------------------------------------------------

@pytest.fixture
def sync_env():
    env = SyncVecEnv([StubEnv for _ in range(3)])
    yield env
    env.close()


def test_sync_spaces_are_batched(sync_env):
    assert sync_env.batch_size == 3
    assert sync_env.action_space.shape == (3, 2)
    assert sync_env.observation_space["position"].shape == (3, 2)
    assert sync_env.context_space.shape == (3, 2)
    assert sync_env.render_fps == StubEnv.render_fps
    assert sync_env.render_mode == "rgb_array"
    assert sync_env.metadata == StubEnv.metadata
    assert sync_env.backend is B and sync_env.device is None


def test_sync_reset_forwards_per_env_seeds(sync_env):
    context, obs, info = sync_env.reset(seed=10)
    for i, env in enumerate(sync_env.envs):
        assert env.reset_calls[-1]["seed"] == 10 + i
        np.testing.assert_array_equal(obs["position"][i], StubEnv().reset(seed=10 + i)[1]["position"])
    np.testing.assert_array_equal(context[:, 0], [10, 11, 12])
    np.testing.assert_array_equal(info["step"], [0, 0, 0])
    # same seed -> same result; unseeded reset does not reuse the seed
    _, obs2, _ = sync_env.reset(seed=10)
    assert_tree_equal(obs2, obs)
    sync_env.reset()
    assert all(env.reset_calls[-1]["seed"] is None for env in sync_env.envs)


def test_sync_step_and_masked_reset(sync_env):
    sync_env.reset(seed=0)
    action = np.full((3, 2), 0.5, np.float32)
    obs, reward, terminated, truncated, info = sync_env.step(action)
    assert sync_env.observation_space.contains(obs)
    assert reward.shape == terminated.shape == truncated.shape == (3,)
    assert terminated.dtype == bool and truncated.dtype == bool
    np.testing.assert_array_equal(info["step"], [1, 1, 1])
    for i, env in enumerate(sync_env.envs):
        np.testing.assert_allclose(env.last_action, action[i])

    mask = np.array([False, True, False])
    context, masked_obs, info = sync_env.reset(mask=mask, seed=7)
    assert masked_obs["position"].shape == (1, 2)
    np.testing.assert_array_equal(context, [[8.0, 8.0]])  # global index 1 -> seed 7 + 1
    assert [len(env.reset_calls) for env in sync_env.envs] == [1, 2, 1]
    np.testing.assert_array_equal(info["step"], [0])
    merged = sync_env.update_observation_post_reset(obs, masked_obs, mask)
    assert sync_env.observation_space.contains(merged)
    np.testing.assert_array_equal(merged["position"][0], obs["position"][0])
    np.testing.assert_array_equal(merged["position"][1], masked_obs["position"][0])


def test_sync_render_and_close():
    envs = [StubEnv() for _ in range(2)]
    vec = SyncVecEnv([lambda e=e: e for e in envs])
    vec.reset(seed=0)
    frames = vec.render()
    assert len(frames) == 2 and frames[0].shape == (4, 6, 3)
    with vec:
        pass
    assert [e.closed for e in envs] == [1, 1]


def test_sync_rejects_batched_or_single_env():
    with pytest.raises(AssertionError):
        SyncVecEnv([StubEnv])
    with pytest.raises(AssertionError):
        SyncVecEnv([lambda: StubEnv(2), lambda: StubEnv(2)])


def test_sync_batches_differing_sub_env_spaces():
    vec = SyncVecEnv([lambda: StubEnv(action_high=1.0), lambda: StubEnv(action_high=2.0)])
    assert vec.action_space.shape == (2, 2)
    np.testing.assert_allclose(vec.action_space.high, [[1.0, 1.0], [2.0, 2.0]])


# ---------------------------------------------------------------------------
# AsyncVecEnv
# ---------------------------------------------------------------------------

class _FailingStepEnv(StubEnv):
    def step(self, action):
        raise ValueError("boom in worker")


class _StringInfoEnv(StubEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["status"] = "ok"
        return obs, reward, terminated, truncated, info


@pytest.fixture(params=["forkserver", "spawn"])
def mp_context(request):
    if request.param not in mp.get_all_start_methods():
        pytest.skip(f"{request.param} not available")
    return mp.get_context(request.param)


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_async_matches_sync(mp_context):
    sync = SyncVecEnv([_StringInfoEnv for _ in range(3)])
    async_env = AsyncVecEnv([_StringInfoEnv for _ in range(3)], ctx=mp_context)
    try:
        assert async_env.batch_size == 3
        assert async_env.action_space == sync.action_space
        assert async_env.observation_space == sync.observation_space
        assert async_env.context_space == sync.context_space
        assert_tree_equal(async_env.reset(seed=3), sync.reset(seed=3))
        action = np.full((3, 2), 0.25, np.float32)
        for _ in range(2):
            assert_tree_equal(async_env.step(action), sync.step(action))
        mask = np.array([True, False, True])
        assert_tree_equal(async_env.reset(mask=mask, seed=9), sync.reset(mask=mask, seed=9))
        assert_tree_equal(async_env.step(action), sync.step(action))
        frames = async_env.render()
        assert_tree_equal(frames, sync.render())
    finally:
        async_env.close()
        sync.close()
    assert async_env.processes == []


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_async_surfaces_worker_errors():
    async_env = AsyncVecEnv([_FailingStepEnv for _ in range(2)], ctx=mp.get_context("spawn"))
    try:
        async_env.reset(seed=0)
        with pytest.raises(RuntimeError, match="worker") as excinfo:
            async_env.step(np.zeros((2, 2), np.float32))
        assert "boom in worker" in repr(excinfo.value.__cause__) + str(excinfo.value)
    finally:
        async_env.close()


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_async_reads_spaces_from_workers_without_building_an_env_in_the_parent():
    factory = functools.partial(ArrayEnv, image_shape=(3, 4, 5), forbidden_pid=os.getpid())
    reference = ArrayEnv(image_shape=(3, 4, 5))
    async_env = AsyncVecEnv([factory for _ in range(2)], ctx=mp.get_context("spawn"))
    try:
        assert async_env.backend is reference.backend and async_env.device is None
        assert async_env.metadata == reference.metadata
        assert async_env.render_mode is None and async_env.render_fps is None
        assert async_env.observation_space == SyncVecEnv([lambda: ArrayEnv(image_shape=(3, 4, 5))] * 2).observation_space
        assert async_env.observation_space["image"].shape == (2, 3, 4, 5)
        assert async_env.context_space.shape == (2, 2)
        context, obs, _ = async_env.reset(seed=1)
        np.testing.assert_array_equal(context, [[1, 1], [2, 2]])
    finally:
        async_env.close()


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_async_batches_differing_sub_env_spaces_like_sync():
    fns = [functools.partial(StubEnv, action_high=1.0), functools.partial(StubEnv, action_high=2.0)]
    async_env = AsyncVecEnv(fns, ctx=mp.get_context("spawn"))
    try:
        assert async_env.action_space == SyncVecEnv(fns).action_space
    finally:
        async_env.close()


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_async_torch_spaces_keep_backend_and_device():
    torch = pytest.importorskip("torch")
    async_env = AsyncVecEnv([functools.partial(ArrayEnv, "pytorch")] * 2, ctx=mp.get_context("spawn"))
    try:
        reference = ArrayEnv("pytorch")
        assert async_env.backend is reference.backend
        assert async_env.device == torch.device("cpu")
        assert async_env.observation_space == SyncVecEnv([lambda: ArrayEnv("pytorch")] * 2).observation_space
        _, obs, _ = async_env.reset(seed=0)
        assert isinstance(obs["vec"], torch.Tensor) and obs["vec"].shape == (2, 4)
    finally:
        async_env.close()


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_async_construction_errors_are_raised_by_the_constructor():
    with pytest.raises(RuntimeError, match="worker") as excinfo:
        AsyncVecEnv([functools.partial(ArrayEnv, "no-such-backend")] * 2, ctx=mp.get_context("spawn"))
    assert "no-such-backend" in repr(excinfo.value.__cause__) + str(excinfo.value)


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_async_rejects_workers_with_different_backends():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="same backend"):
        AsyncVecEnv([ArrayEnv, functools.partial(ArrayEnv, "pytorch")], ctx=mp.get_context("spawn"))


# ---------------------------------------------------------------------------
# Worker loop, exercised in-process (no subprocess) through a thread
# ---------------------------------------------------------------------------

def _run_worker_in_thread(env_fn):
    import queue
    import threading

    from unienv_interface.env_base.vec_env import _async_worker_fn

    parent, child = mp.Pipe()
    dummy_parent, _ = mp.Pipe()
    errors = queue.Queue()
    thread = threading.Thread(target=_async_worker_fn, args=(0, env_fn, child, dummy_parent, errors), daemon=True)
    thread.start()
    return parent, errors, thread


def test_worker_loop_serves_commands_and_closes_env():
    env = StubEnv()
    parent, errors, thread = _run_worker_in_thread(lambda: env)
    parent.send(("reset", (), {"seed": 2}))
    (context, obs, info), ok = parent.recv()
    assert ok and info == {"step": 0}
    np.testing.assert_array_equal(context, [2.0, 2.0])
    parent.send(("step", (np.full(2, 0.5, np.float32),), {}))
    (obs, reward, terminated, truncated, info), ok = parent.recv()
    assert ok and info == {"step": 1}
    parent.send(("render", (), {}))
    frame, ok = parent.recv()
    assert ok and frame.shape == (4, 6, 3)
    parent.send(("close", (), {}))
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert env.closed == 1 and errors.empty()


def test_worker_loop_reports_errors():
    parent, errors, thread = _run_worker_in_thread(_FailingStepEnv)
    parent.send(("step", (np.zeros(2, np.float32),), {}))
    (name, message, tb), ok = parent.recv()
    assert not ok and name == "ValueError" and message == "boom in worker"
    assert "Traceback" in tb
    index, exc = errors.get(timeout=5)
    assert index == 0 and isinstance(exc, ValueError)
    thread.join(timeout=10)


def test_worker_loop_reports_construction_errors():
    def broken():
        raise RuntimeError("cannot build env")

    parent, errors, thread = _run_worker_in_thread(broken)
    (name, message, _), ok = parent.recv()
    assert not ok and name == "RuntimeError" and message == "cannot build env"
    thread.join(timeout=10)
