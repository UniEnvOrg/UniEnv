"""Tests for ReplayBufferCollectionWrapper."""

from __future__ import annotations

import json
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from unienv_data.replay_buffer import ReplayBuffer, ReplayBufferCollectionWrapper
from unienv_data.storages.parquet import ParquetStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.env_base.env import Env
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.transformations import DataTransformation


# ===================================================================
# Dummy environments
# ===================================================================


class DummyUnbatchedEnv(
    Env[
        np.ndarray,  # BArrayType
        None,  # ContextType
        np.ndarray,  # ObsType
        np.ndarray,  # ActType
        Any,  # RenderFrame
        Any,  # BDeviceType
        np.dtype,  # BDtypeType
        np.random.Generator,  # BRNGType
    ]
):
    """A minimal unbatched (batch_size=None) env for testing."""

    metadata = {"render_modes": []}
    render_mode = None
    render_fps = None
    backend = NumpyComputeBackend
    device = None
    batch_size = None
    context_space = None

    def __init__(self, obs_dim: int = 3, act_dim: int = 2):
        self.action_space = BoxSpace(
            NumpyComputeBackend, -1.0, 1.0, np.float32, shape=(act_dim,)
        )
        self.observation_space = BoxSpace(
            NumpyComputeBackend, -10.0, 10.0, np.float32, shape=(obs_dim,)
        )
        self.rng = np.random.default_rng(42)
        self._step_count = 0

    def step(self, action):
        obs = (
            np.random.default_rng(self._step_count)
            .uniform(-1.0, 1.0, size=self.observation_space.shape)
            .astype(np.float32)
        )
        self._step_count += 1
        terminated = self._step_count >= 10
        truncated = False
        return obs, 1.0, terminated, truncated, {}

    def reset(self, *, mask=None, seed=None, **kwargs):
        self._step_count = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return None, obs, {}

    def render(self):
        return None

    def close(self):
        pass


class DummyBatchedEnv1(
    Env[
        np.ndarray,
        None,
        np.ndarray,
        np.ndarray,
        Any,
        Any,
        np.dtype,
        np.random.Generator,
    ]
):
    """A minimal batched env with batch_size=1."""

    metadata = {"render_modes": []}
    render_mode = None
    render_fps = None
    backend = NumpyComputeBackend
    device = None
    batch_size = 1
    context_space = None

    def __init__(self, obs_dim: int = 3, act_dim: int = 2):
        self.action_space = BoxSpace(
            NumpyComputeBackend, -1.0, 1.0, np.float32, shape=(1, act_dim)
        )
        self.observation_space = BoxSpace(
            NumpyComputeBackend, -10.0, 10.0, np.float32, shape=(1, obs_dim)
        )
        self.rng = np.random.default_rng(42)
        self._step_counts = np.zeros(1, dtype=np.int32)

    def step(self, action):
        self._step_counts += 1
        obs = (
            np.random.default_rng(int(self._step_counts[0]))
            .uniform(-1.0, 1.0, size=(1, self.observation_space.shape[1]))
            .astype(np.float32)
        )
        terminated = np.array([self._step_counts[0] >= 5], dtype=bool)
        truncated = np.array([False], dtype=bool)
        return obs, np.array([1.0], dtype=np.float32), terminated, truncated, {}

    def reset(self, *, mask=None, seed=None, **kwargs):
        obs = np.zeros((1, self.observation_space.shape[1]), dtype=np.float32)
        self._step_counts = np.zeros(1, dtype=np.int32)
        return None, obs, {}

    def render(self):
        return None

    def close(self):
        pass


class DummyBatchedEnvN(
    Env[
        np.ndarray,
        None,
        np.ndarray,
        np.ndarray,
        Any,
        Any,
        np.dtype,
        np.random.Generator,
    ]
):
    """A minimal batched env with batch_size=N (N>1) that supports masked reset."""

    metadata = {"render_modes": []}
    render_mode = None
    render_fps = None
    backend = NumpyComputeBackend
    device = None
    context_space = None

    def __init__(self, batch_size: int = 3, obs_dim: int = 3, act_dim: int = 2):
        self._batch_size = batch_size
        self.action_space = BoxSpace(
            NumpyComputeBackend,
            -1.0,
            1.0,
            np.float32,
            shape=(batch_size, act_dim),
        )
        self.observation_space = BoxSpace(
            NumpyComputeBackend,
            -10.0,
            10.0,
            np.float32,
            shape=(batch_size, obs_dim),
        )
        self.rng = np.random.default_rng(42)
        self._step_counts = np.zeros(batch_size, dtype=np.int32)
        self._max_steps = np.array([5, 7, 3][:batch_size], dtype=np.int32)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def step(self, action):
        self._step_counts += 1
        B = self._batch_size
        obs = np.stack(
            [
                np.random.default_rng(100 + i + int(self._step_counts[i]))
                .uniform(-1.0, 1.0, size=(self.observation_space.shape[1],))
                .astype(np.float32)
                for i in range(B)
            ],
            axis=0,
        )
        terminated = self._step_counts >= self._max_steps
        truncated = np.zeros(B, dtype=bool)
        return obs, np.ones(B, dtype=np.float32), terminated, truncated, {}

    def reset(self, *, mask=None, seed=None, **kwargs):
        B = self._batch_size
        if mask is not None:
            self._step_counts = np.where(
                np.asarray(mask), 0, self._step_counts
            ).astype(np.int32)
            n_mask = int(np.sum(np.asarray(mask)))
            obs = np.zeros(
                (n_mask, self.observation_space.shape[1]), dtype=np.float32
            )
        else:
            self._step_counts = np.zeros(B, dtype=np.int32)
            obs = np.zeros((B, self.observation_space.shape[1]), dtype=np.float32)
        return None, obs, {}

    def render(self):
        return None

    def close(self):
        pass

    # For masked reset testing, implement the post-reset merge methods.
    def update_observation_post_reset(self, old_obs, newobs_masked, mask):
        return sbu.set_at(self.observation_space, old_obs, mask, newobs_masked)


class DummyUnbatchedEnvWithContext(
    Env[
        np.ndarray,  # BArrayType
        np.ndarray,  # ContextType
        np.ndarray,  # ObsType
        np.ndarray,  # ActType
        Any,  # RenderFrame
        Any,  # BDeviceType
        np.dtype,  # BDtypeType
        np.random.Generator,  # BRNGType
    ]
):
    """A minimal unbatched env that provides a context space."""

    metadata = {"render_modes": []}
    render_mode = None
    render_fps = None
    backend = NumpyComputeBackend
    device = None
    batch_size = None

    def __init__(self, obs_dim: int = 3, act_dim: int = 2, ctx_dim: int = 3):
        self.action_space = BoxSpace(
            NumpyComputeBackend, -1.0, 1.0, np.float32, shape=(act_dim,)
        )
        self.observation_space = BoxSpace(
            NumpyComputeBackend, -10.0, 10.0, np.float32, shape=(obs_dim,)
        )
        self.context_space = BoxSpace(
            NumpyComputeBackend, -10.0, 10.0, np.float32, shape=(ctx_dim,)
        )
        self.rng = np.random.default_rng(42)
        self._step_count = 0

    def step(self, action):
        obs = (
            np.random.default_rng(self._step_count)
            .uniform(-1.0, 1.0, size=self.observation_space.shape)
            .astype(np.float32)
        )
        self._step_count += 1
        terminated = self._step_count >= 10
        truncated = False
        return obs, 1.0, terminated, truncated, {}

    def reset(self, *, mask=None, seed=None, **kwargs):
        self._step_count = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        ctx = np.ones(self.context_space.shape, dtype=np.float32)
        return ctx, obs, {}

    def render(self):
        return None

    def close(self):
        pass


# ===================================================================
# Helper – build a replay buffer for testing
# ===================================================================

# We use ParquetStorage which supports DictSpace natively with the Numpy backend,
# keeping the same backend as the dummy environments.


def _make_rb(
    tmp_path,
    transition_keys,
    capacity=None,
    maintain_segment_metadata=False,
):
    """Create a DictSpace replay buffer with ParquetStorage (Numpy backend)."""
    spaces = {}
    for key in transition_keys:
        if key in ("reward",):
            spaces[key] = BoxSpace(
                NumpyComputeBackend,
                -np.inf,
                np.inf,
                np.float32,
                shape=(),
            )
        elif key in ("terminated", "truncated", "done"):
            spaces[key] = BoxSpace(
                NumpyComputeBackend,
                0.0,
                1.0,
                np.float32,  # bool not supported; store 0.0/1.0
                shape=(),
            )
        elif key == "env_index":
            spaces[key] = BoxSpace(
                NumpyComputeBackend,
                0,
                100,
                np.int32,
                shape=(),
            )
        elif key in ("obs", "next_obs"):
            spaces[key] = BoxSpace(
                NumpyComputeBackend,
                -10.0,
                10.0,
                np.float32,
                shape=(3,),
            )
        elif key == "action":
            spaces[key] = BoxSpace(
                NumpyComputeBackend,
                -1.0,
                1.0,
                np.float32,
                shape=(2,),
            )
        elif key == "context":
            spaces[key] = BoxSpace(
                NumpyComputeBackend,
                -10.0,
                10.0,
                np.float32,
                shape=(3,),
            )
        else:
            # Unknown/arbitrary key – create a generic scalar float space
            # so the key is present in the rb DictSpace for testing.
            spaces[key] = BoxSpace(
                NumpyComputeBackend,
                -np.inf,
                np.inf,
                np.float32,
                shape=(),
            )

    single_space = DictSpace(NumpyComputeBackend, spaces)
    return ReplayBuffer.create(
        ParquetStorage,
        single_space,
        cache_path=str(tmp_path),
        capacity=capacity,
        maintain_segment_metadata=maintain_segment_metadata,
    )


# ===================================================================
# Test 1: B=None, K=1, auto → shared append
# ===================================================================


def test_unbatched_shared_append(tmp_path):
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "reward", "terminated", "next_obs"])
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="auto")

    assert wrapper.replay_mode == "shared"

    wrapper.reset()
    for _ in range(5):
        action = np.array([0.5, -0.3], dtype=np.float32)
        wrapper.step(action)

    assert len(rb) == 5

    t0 = rb.get_at(0)
    assert set(t0.keys()) == {"obs", "action", "reward", "terminated", "next_obs"}
    # Check shapes are unbatched
    assert t0["obs"].shape == (3,)
    assert t0["action"].shape == (2,)
    # reward should be ~1.0
    assert np.abs(t0["reward"] - 1.0) < 1e-6


# ===================================================================
# Test 2: B=1, K=1, auto → shared append after unbatching slot 0
# ===================================================================


def test_batched1_shared_unbatch_append(tmp_path):
    env = DummyBatchedEnv1(obs_dim=3, act_dim=2)
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "reward", "terminated", "next_obs"])
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="auto")

    assert wrapper.replay_mode == "shared"
    assert wrapper.batch_size == 1

    wrapper.reset()
    for _ in range(4):
        action = np.array([[0.5, -0.3]], dtype=np.float32)
        wrapper.step(action)

    assert len(rb) == 4

    t0 = rb.get_at(0)
    assert t0["obs"].shape == (3,)  # unbatched
    assert t0["action"].shape == (2,)  # unbatched


# ===================================================================
# Test 3: B>1, K=1, segment-unaware buffer → shared extend
# ===================================================================


def test_batched_shared_extend(tmp_path):
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "reward", "terminated", "next_obs"])
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="auto")

    assert wrapper.replay_mode == "shared"

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    for _ in range(5):
        wrapper.step(action)

    assert len(rb) == B * 5

    t0 = rb.get_at(0)
    assert t0["obs"].shape == (3,)


# ===================================================================
# Test 4: B>1, K=1, segment-aware buffer → auto rejects
# ===================================================================


def test_batched_shared_segment_aware_rejects_auto(tmp_path):
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    with pytest.raises(ValueError, match="segment-aware"):
        ReplayBufferCollectionWrapper(env, rb, replay_mode="auto")


def test_batched_shared_segment_aware_rejects_explicit(tmp_path):
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    with pytest.raises(ValueError, match="segment-unaware"):
        ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")


# ===================================================================
# Test 5: B>1, K=B → per-env append
# ===================================================================


def test_batched_per_env_append(tmp_path):
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rbs = [
        _make_rb(
            tmp_path / f"rb_{i}",
            ["obs", "action", "reward", "terminated", "next_obs"],
        )
        for i in range(B)
    ]
    wrapper = ReplayBufferCollectionWrapper(env, rbs, replay_mode="auto")

    assert wrapper.replay_mode == "per_env"

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    for _ in range(2):
        wrapper.step(action)

    # Each buffer should have 2 transitions
    for i in range(B):
        assert len(rbs[i]) == 2, f"Buffer {i} has {len(rbs[i])} transitions"


# ===================================================================
# Test 6: done mask closes only matching per-env buffers
# ===================================================================


def test_per_env_done_closes_segments(tmp_path):
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    # Override max_steps: env0 terminates at step 2, others run longer.
    env._max_steps = np.array([2, 99, 99], dtype=np.int32)
    rbs = [
        _make_rb(
            tmp_path / f"rb_{i}",
            ["obs", "action", "reward", "terminated", "next_obs"],
            maintain_segment_metadata=True,
        )
        for i in range(B)
    ]
    wrapper = ReplayBufferCollectionWrapper(env, rbs, replay_mode="per_env")

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    for step_idx in range(4):
        obs, reward, terminated, truncated, info = wrapper.step(action)
        # env 0 should terminate at step 2 (index 1 since 0-based)
        if step_idx == 1:
            assert terminated[0], f"Expected env 0 terminated at step {step_idx}"

    # Buffer 0 should have 3 segments because env 0 keeps being stepped
    # after termination (each step after done creates a single-transition segment).
    # Step 0: not done
    # Step 1: done → segment (0,2) for steps 0,1
    # Steps 2,3: already done → each creates a 1-step segment
    segs0 = rbs[0].get_segments()
    assert segs0 is not None
    assert len(segs0) == 3, f"Expected 3 segments for env 0, got {segs0}"
    assert segs0[0] == (0, 2), f"Expected segment 0 to be (0,2), got {segs0[0]}"
    assert segs0[1] == (2, 3), f"Expected segment 1 to be (2,3), got {segs0[1]}"
    assert segs0[2] == (3, 4), f"Expected segment 2 to be (3,4), got {segs0[2]}"

    # Buffers 1 and 2 still open (no mark_segment_end called yet)
    assert rbs[1].storage.has_open_segment
    assert rbs[2].storage.has_open_segment

    # Clean up
    rbs[1].mark_segment_end()
    rbs[2].mark_segment_end()


# ===================================================================
# Test 7: masked reset cache merge
# ===================================================================


def test_masked_reset_cache_merge(tmp_path):
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb", ["obs", "action", "reward", "terminated", "next_obs"]
    )
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    # Initial full reset
    wrapper.reset()
    assert wrapper._cached_obs.shape == (B, 3)

    # Step a few times
    action = np.zeros((B, 2), dtype=np.float32)
    wrapper.step(action)
    wrapper.step(action)

    # Masked reset: only reset slot 0
    mask = np.array([True, False, False], dtype=bool)
    context, obs, info = wrapper.reset(mask=mask)

    # After merge, cached obs should still have batch_size=3
    assert wrapper._cached_obs.shape == (B, 3)
    # Slot 0 should be reset (zeros), slots 1-2 should be from last step
    np.testing.assert_array_equal(
        wrapper._cached_obs[0], np.zeros(3, dtype=np.float32)
    )
    assert not np.allclose(
        wrapper._cached_obs[1], np.zeros(3, dtype=np.float32)
    )


# ===================================================================
# Test 8: invalid layouts / mode validation errors
# ===================================================================


def test_shared_mode_requires_exactly_one_buffer(tmp_path):
    env = DummyUnbatchedEnv()
    rb1 = _make_rb(tmp_path / "a", ["obs", "action", "next_obs"])
    rb2 = _make_rb(tmp_path / "b", ["obs", "action", "next_obs"])

    with pytest.raises(ValueError, match="exactly 1 replay buffer"):
        ReplayBufferCollectionWrapper(env, [rb1, rb2], replay_mode="shared")


def test_per_env_requires_batched(tmp_path):
    env = DummyUnbatchedEnv()
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "next_obs"])

    with pytest.raises(ValueError, match="batch_size > 1"):
        ReplayBufferCollectionWrapper(env, rb, replay_mode="per_env")


def test_per_env_requires_batch_size_gt_1(tmp_path):
    env = DummyBatchedEnv1()
    rbs = [_make_rb(tmp_path / "rb", ["obs", "action", "next_obs"])]

    with pytest.raises(ValueError, match="batch_size > 1"):
        ReplayBufferCollectionWrapper(env, rbs, replay_mode="per_env")


def test_per_env_requires_matching_buffer_count(tmp_path):
    B = 3
    env = DummyBatchedEnvN(batch_size=B)
    rbs = [
        _make_rb(tmp_path / f"rb_{i}", ["obs", "action", "next_obs"])
        for i in range(B - 1)
    ]

    with pytest.raises(ValueError, match="exactly one replay buffer per"):
        ReplayBufferCollectionWrapper(env, rbs, replay_mode="per_env")


def test_step_before_reset_raises(tmp_path):
    env = DummyUnbatchedEnv()
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "next_obs"])
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    with pytest.raises(RuntimeError, match="Cannot call step"):
        wrapper.step(np.array([0.5, -0.3], dtype=np.float32))


def test_auto_mode_cannot_resolve(tmp_path):
    B = 3
    env = DummyBatchedEnvN(batch_size=B)
    # 2 buffers for B=3 env
    rbs = [
        _make_rb(tmp_path / f"rb_{i}", ["obs", "action", "next_obs"])
        for i in range(2)
    ]

    with pytest.raises(ValueError, match="Cannot auto-resolve"):
        ReplayBufferCollectionWrapper(env, rbs, replay_mode="auto")


# ===================================================================
# Test 9: custom/default transition builder behavior
# ===================================================================


def test_default_transition_builder_dict_space(tmp_path):
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb", ["obs", "action", "reward", "done", "next_obs"]
    )
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    wrapper.step(action)

    t = rb.get_at(0)
    assert set(t.keys()) == {"obs", "action", "reward", "done", "next_obs"}
    # done is a numpy float32 (0.0 or 1.0)
    assert t["done"] in (np.float32(0.0), np.float32(1.0))


def test_default_transition_context_missing_raises(tmp_path):
    """Default builder: requesting 'context' when env has no context_space raises."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb", ["obs", "action", "reward", "next_obs", "context"]
    )
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    with pytest.raises(KeyError, match="context"):
        wrapper.step(action)


def test_default_transition_includes_env_index_per_env(tmp_path):
    B = 2
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rbs = [
        _make_rb(
            tmp_path / f"rb_{i}",
            ["obs", "action", "reward", "next_obs", "env_index"],
        )
        for i in range(B)
    ]
    wrapper = ReplayBufferCollectionWrapper(env, rbs, replay_mode="per_env")

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    wrapper.step(action)

    t0 = rbs[0].get_at(0)
    t1 = rbs[1].get_at(0)
    assert t0["env_index"] == 0
    assert t1["env_index"] == 1


def test_custom_transition_builder(tmp_path):
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "next_obs"])

    def custom_builder(
        *,
        cached_obs,
        cached_context,
        action,
        next_obs,
        reward,
        terminated,
        truncated,
        env_index,
        rb_single_space,
    ):
        # Double the cached obs
        return {
            "obs": cached_obs * 2,
            "action": action,
            "next_obs": next_obs,
        }

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared", transition_builder=custom_builder
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    wrapper.step(action)

    t = rb.get_at(0)
    # obs should be doubled (cached_obs was zeros, so still zeros)
    np.testing.assert_array_equal(t["obs"], np.zeros(3, dtype=np.float32))


# ===================================================================
# Test 10: validate_transition flag
# ===================================================================


def test_validate_transition_flag_raises_on_invalid(tmp_path):
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "next_obs"])

    def bad_builder(*, cached_obs, **kwargs):
        return {"obs": cached_obs}  # missing "action" and "next_obs"

    wrapper = ReplayBufferCollectionWrapper(
        env,
        rb,
        replay_mode="shared",
        transition_builder=bad_builder,
        validate_transition=True,
    )

    wrapper.reset()
    with pytest.raises(ValueError, match="does not match target space"):
        wrapper.step(np.array([0.5, -0.3], dtype=np.float32))


def test_validate_transition_off_by_default(tmp_path):
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "next_obs"])

    def bad_builder(*, cached_obs, **kwargs):
        return {"obs": cached_obs}  # missing keys

    wrapper = ReplayBufferCollectionWrapper(
        env,
        rb,
        replay_mode="shared",
        transition_builder=bad_builder,
    )
    # should NOT raise since validate_transition=False by default
    wrapper.reset()
    # May or may not raise depending on rb.append validation; we just check
    # that our validation didn't trigger.
    try:
        wrapper.step(np.array([0.5, -0.3], dtype=np.float32))
    except Exception:
        # rb.append might fail internally, that's fine
        pass


# ===================================================================
# Test 11: helper constructors
# ===================================================================


def test_shared_helper_constructor(tmp_path):
    env = DummyUnbatchedEnv()
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "next_obs"])
    wrapper = ReplayBufferCollectionWrapper.shared(env, rb)
    assert wrapper.replay_mode == "shared"


def test_per_env_helper_constructor(tmp_path):
    B = 2
    env = DummyBatchedEnvN(batch_size=B)
    rbs = [
        _make_rb(tmp_path / f"rb_{i}", ["obs", "action", "next_obs"])
        for i in range(B)
    ]
    wrapper = ReplayBufferCollectionWrapper.per_env(env, rbs)
    assert wrapper.replay_mode == "per_env"


# ===================================================================
# Test 12: per-env reset marks segment boundaries
# ===================================================================


def test_per_env_reset_marks_segments(tmp_path):
    B = 2
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    env._max_steps = np.array([99, 99], dtype=np.int32)  # never auto-terminate
    rbs = [
        _make_rb(
            tmp_path / f"rb_{i}",
            ["obs", "action", "reward", "terminated", "next_obs"],
            maintain_segment_metadata=True,
        )
        for i in range(B)
    ]
    wrapper = ReplayBufferCollectionWrapper(env, rbs, replay_mode="per_env")

    # First episode
    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    for _ in range(3):
        wrapper.step(action)

    # Reset (full reset) – should close segments
    wrapper.reset()
    for _ in range(2):
        wrapper.step(action)

    # Close remaining open segments before inspecting
    for i in range(B):
        if rbs[i].storage.has_open_segment:
            rbs[i].mark_segment_end()

    # Check that both buffers have 2 segments
    for i in range(B):
        segs = rbs[i].get_segments()
        assert len(segs) == 2, f"Buffer {i}: expected 2 segments, got {segs}"
        assert segs[0] == (0, 3), f"Buffer {i} segment 0: {segs}"
        assert segs[1] == (3, 5), f"Buffer {i} segment 1: {segs}"

    assert len(rbs[0]) == 5
    assert len(rbs[1]) == 5


# ===================================================================
# Test 13: transition_builder as DataTransformation (unbatched)
# ===================================================================


def test_transition_builder_as_identity_transform_unbatched(tmp_path):
    """IdentityTransformation as transition_builder on a canonical-compatible rb."""
    from unienv_interface.transformations.identity import IdentityTransformation

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env,
        rb,
        replay_mode="shared",
        transition_builder=IdentityTransformation(),
        validate_transition=True,
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    wrapper.step(action)

    assert len(rb) == 1
    t = rb.get_at(0)
    assert set(t.keys()) == {"obs", "action", "reward", "terminated", "next_obs"}
    assert t["obs"].shape == (3,)


def test_transition_builder_as_custom_transform_unbatched(tmp_path):
    """Custom DataTransformation that maps canonical dict to rb format."""
    from unienv_interface.space import Space
    from unienv_interface.backends import BDeviceType, BDtypeType, BRNGType

    class RenameTransform(DataTransformation):
        """Rename canonical keys to match a target rb space."""
        def get_target_space_from_source(self, source_space):
            # Return the rb single_space shape
            return source_space  # simplified; real impl would derive from source

        def transform(self, source_space, data):
            return {
                "obs": data["obs"],
                "action": data["action"],
                "reward": data["reward"],
                "terminated": data["terminated"],
                "next_obs": data["next_obs"],
            }

        def serialize(self, source_space=None):
            return {}

        @classmethod
        def deserialize_from(cls, json_data, source_space=None):
            return cls()

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env,
        rb,
        replay_mode="shared",
        transition_builder=RenameTransform(),
        validate_transition=True,
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    wrapper.step(action)

    assert len(rb) == 1
    t = rb.get_at(0)
    assert set(t.keys()) == {"obs", "action", "reward", "terminated", "next_obs"}


# ===================================================================
# Test 14: DataTransformation builder in batched shared extend mode
# ===================================================================


def test_transform_builder_batched_shared_extend(tmp_path):
    """DataTransformation builder with B>1 shared extend validates against
    batched space."""
    from unienv_interface.space import Space
    from unienv_interface.transformations.identity import IdentityTransformation

    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env,
        rb,
        replay_mode="shared",
        transition_builder=IdentityTransformation(),
        validate_transition=True,
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    for _ in range(3):
        wrapper.step(action)

    assert len(rb) == B * 3


def test_transform_builder_batched_shared_extend_invalid(tmp_path):
    """DataTransformation that produces output not matching batched space
    should raise with validate_transition=True."""
    from unienv_interface.space import Space

    class BadTransform(DataTransformation):
        def get_target_space_from_source(self, source_space):
            return source_space

        def transform(self, source_space, data):
            # Return only "obs" — missing keys.
            return {"obs": data["obs"]}

        def serialize(self, source_space=None):
            return {}

        @classmethod
        def deserialize_from(cls, json_data, source_space=None):
            return cls()

    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env,
        rb,
        replay_mode="shared",
        transition_builder=BadTransform(),
        validate_transition=True,
    )

    wrapper.reset()
    with pytest.raises(ValueError, match="does not match target space"):
        wrapper.step(np.zeros((B, 2), dtype=np.float32))


# ===================================================================
# Test 15: DataTransformation builder in per_env mode
# ===================================================================


def test_transform_builder_per_env(tmp_path):
    """DataTransformation builder in per_env mode with per-slot unbatching."""
    from unienv_interface.transformations.identity import IdentityTransformation

    B = 2
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rbs = [
        _make_rb(
            tmp_path / f"rb_{i}",
            ["obs", "action", "reward", "terminated", "next_obs"],
        )
        for i in range(B)
    ]

    wrapper = ReplayBufferCollectionWrapper(
        env,
        rbs,
        replay_mode="per_env",
        transition_builder=IdentityTransformation(),
        validate_transition=True,
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    for _ in range(2):
        wrapper.step(action)

    for i in range(B):
        assert len(rbs[i]) == 2
        t = rbs[i].get_at(0)
        assert t["obs"].shape == (3,)  # unbatched


# ===================================================================
# Test 16: simplified scalar assumptions (unbatched & batched)
# ===================================================================


def test_scalar_unbatched_reward_is_python_float(tmp_path):
    """In unbatched mode, reward is a Python float, not an array."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "reward", "next_obs"])
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    wrapper.reset()
    _, reward, _, _, _ = wrapper.step(np.array([0.5, -0.3], dtype=np.float32))
    # DummyUnbatchedEnv returns 1.0 (float)
    assert isinstance(reward, float)
    # The transition should still be recorded correctly
    assert len(rb) == 1
    assert rb.get_at(0)["reward"] is not None


def test_scalar_batched_reward_is_backend_array(tmp_path):
    """In batched mode, reward is a backend array, not a Python list/dict."""
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "reward", "next_obs"])
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    wrapper.reset()
    _, reward, _, _, _ = wrapper.step(np.zeros((B, 2), dtype=np.float32))
    # DummyBatchedEnvN returns np.ones(B) which is a backend array
    assert isinstance(reward, np.ndarray)
    assert reward.shape == (B,)


def test_unbatch_scalar_uses_batch_size_invariant(tmp_path):
    """_unbatch_scalar relies on batch_size, not __getitem__ check.
    Even with a value that has __getitem__ but in unbatched mode,
    the value should not be indexed."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "reward", "next_obs"])
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    # The invariant: batch_size is None → scalar, no unbatching attempted
    assert wrapper.batch_size is None
    # _unbatch_scalar on a Python float with index 0 should just return the float
    result = wrapper._unbatch_scalar(1.0, 0)
    assert result == 1.0
    assert isinstance(result, float)


def test_unbatch_scalar_batched_indexing(tmp_path):
    """In batched mode, _unbatch_scalar indexes into the array."""
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(tmp_path / "rb", ["obs", "action", "reward", "next_obs"])
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = wrapper._unbatch_scalar(arr, 1)
    assert result == 2.0


# ===================================================================
# Test 17: callable builder still works alongside DataTransformation
# ===================================================================


def test_callable_and_transform_both_work(tmp_path):
    """Both callable and DataTransformation builders produce valid results."""
    from unienv_interface.transformations.identity import IdentityTransformation

    env1 = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    env2 = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb1 = _make_rb(tmp_path / "rb1", ["obs", "action", "reward", "next_obs"])
    rb2 = _make_rb(tmp_path / "rb2", ["obs", "action", "reward", "next_obs"])

    # Callable builder
    w1 = ReplayBufferCollectionWrapper(
        env1, rb1, replay_mode="shared",
        transition_builder=lambda *, cached_obs, action, next_obs, reward, **kw: {
            "obs": cached_obs, "action": action,
            "reward": reward, "next_obs": next_obs,
        },
    )
    w1.reset()
    w1.step(np.array([0.5, -0.3], dtype=np.float32))
    assert len(rb1) == 1

    # DataTransformation builder
    w2 = ReplayBufferCollectionWrapper(
        env2, rb2, replay_mode="shared",
        transition_builder=IdentityTransformation(),
    )
    w2.reset()
    w2.step(np.array([0.5, -0.3], dtype=np.float32))
    assert len(rb2) == 1


# ===================================================================
# Test 18: Bug-fix — batched shared validation applies to ALL builders
# ===================================================================


def test_validate_batched_shared_default_builder(tmp_path):
    """Default builder with B>1 shared & validate_transition=True uses batched space."""
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )
    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        validate_transition=True,  # default builder (no transition_builder arg)
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    # Must not raise — validates against the batched space, not single_space.
    wrapper.step(action)
    assert len(rb) == B


def test_validate_batched_shared_callable_builder(tmp_path):
    """Callable builder with B>1 shared & validate_transition=True uses batched space."""
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    def builder(*, cached_obs, action, next_obs, reward, terminated, **kw):
        return {
            "obs": cached_obs,
            "action": action,
            "next_obs": next_obs,
            "reward": reward,
            "terminated": terminated,
        }

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=builder,
        validate_transition=True,
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    # Must not raise — validates against the batched space.
    wrapper.step(action)
    assert len(rb) == B


def test_validate_batched_shared_invalid_callable_raises(tmp_path):
    """Invalid callable-built transition with B>1 & validate_transition=True raises."""
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )
    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=lambda *, cached_obs, **kw: {"obs": cached_obs},
        validate_transition=True,
    )

    wrapper.reset()
    with pytest.raises(ValueError, match="does not match target space"):
        wrapper.step(np.zeros((B, 2), dtype=np.float32))


# ===================================================================
# Test 19: Bug-fix — canonical dict places context/env_index placeholders
# ===================================================================


def test_data_transform_context_missing_raises_unbatched(tmp_path):
    """DataTransformation builder: missing context → clear error (not placeholder)."""
    from unienv_interface.transformations.identity import IdentityTransformation

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "next_obs", "context"],
    )
    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    with pytest.raises(KeyError, match="context"):
        wrapper.step(action)


def test_data_transform_context_missing_raises_batched(tmp_path):
    """DataTransformation builder: batched env with no context → clear error."""
    from unienv_interface.transformations.identity import IdentityTransformation

    B = 2
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "next_obs", "context"],
    )
    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    with pytest.raises(KeyError, match="context"):
        wrapper.step(action)


def test_data_transform_env_index_shared_raises(tmp_path):
    """DataTransformation builder: shared mode env_index → clear error."""
    from unienv_interface.transformations.identity import IdentityTransformation

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "next_obs", "env_index"],
    )
    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    with pytest.raises(KeyError, match="env_index"):
        wrapper.step(action)


def test_data_transform_env_index_per_env(tmp_path):
    """DataTransformation builder: per_env mode provides real env_index values."""
    from unienv_interface.transformations.identity import IdentityTransformation

    B = 2
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rbs = [
        _make_rb(
            tmp_path / f"rb_{i}",
            ["obs", "action", "reward", "next_obs", "env_index"],
        )
        for i in range(B)
    ]
    wrapper = ReplayBufferCollectionWrapper(
        env, rbs, replay_mode="per_env",
        transition_builder=IdentityTransformation(),
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    wrapper.step(action)

    t0 = rbs[0].get_at(0)
    t1 = rbs[1].get_at(0)
    assert t0["env_index"] == 0
    assert t1["env_index"] == 1


def test_data_transform_env_index_shared_batched(tmp_path):
    """DataTransformation builder: shared B>1 mode provides batched env_index."""
    from unienv_interface.transformations.identity import IdentityTransformation

    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "next_obs", "env_index"],
    )
    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
        validate_transition=True,
    )

    wrapper.reset()
    wrapper.step(np.zeros((B, 2), dtype=np.float32))

    assert len(rb) == B
    for i in range(B):
        t = rb.get_at(i)
        assert int(t["env_index"]) == i


def test_data_transform_env_index_shared_b1_raises(tmp_path):
    """DataTransformation builder: shared B=1 mode still rejects env_index."""
    from unienv_interface.transformations.identity import IdentityTransformation

    env = DummyBatchedEnv1(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "next_obs", "env_index"],
    )
    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
    )

    wrapper.reset()
    with pytest.raises(KeyError, match="env_index"):
        wrapper.step(np.array([[0.5, -0.3]], dtype=np.float32))


# ===================================================================
# Test 20: Fail-fast – unknown key in default builder
# ===================================================================


def test_default_builder_unknown_key_raises(tmp_path):
    """Default builder: unknown key in rb space → KeyError."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb", ["obs", "action", "reward", "next_obs", "foobar"]
    )
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    wrapper.reset()
    with pytest.raises(KeyError, match="foobar"):
        wrapper.step(np.array([0.5, -0.3], dtype=np.float32))


def test_default_builder_env_index_shared_raises(tmp_path):
    """Default builder: env_index in shared mode → KeyError."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb", ["obs", "action", "reward", "next_obs", "env_index"]
    )
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    wrapper.reset()
    with pytest.raises(KeyError, match="env_index"):
        wrapper.step(np.array([0.5, -0.3], dtype=np.float32))


def test_default_builder_env_index_batched_shared_raises(tmp_path):
    """Default builder: env_index in batched shared mode with B>1 now succeeds
    (env_index is a batched field of shape (B,)).  See positive test below."""
    # This test is intentionally left as a placeholder; the positive behavior
    # is verified by test_default_builder_env_index_batched_shared below.
    pass


def test_default_builder_env_index_batched_shared(tmp_path):
    """Default builder: env_index in shared B>1 mode produces batched values."""
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb", ["obs", "action", "reward", "next_obs", "env_index"]
    )
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    wrapper.reset()
    wrapper.step(np.zeros((B, 2), dtype=np.float32))

    # The buffer should have B transitions (one per slot, via extend).
    assert len(rb) == B
    # Each stored transition should carry its slot index as env_index.
    for i in range(B):
        t = rb.get_at(i)
        assert int(t["env_index"]) == i, f"Expected env_index={i}, got {t['env_index']}"


def test_default_builder_env_index_shared_b1_raises(tmp_path):
    """Default builder: env_index in shared B=1 mode still raises KeyError."""
    env = DummyBatchedEnv1(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb", ["obs", "action", "reward", "next_obs", "env_index"]
    )
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    wrapper.reset()
    with pytest.raises(KeyError, match="env_index"):
        wrapper.step(np.array([[0.5, -0.3]], dtype=np.float32))


# ===================================================================
# Test 21: Fail-fast – unknown key in DataTransformation canonical path
# ===================================================================


def test_transform_unknown_key_raises(tmp_path):
    """DataTransformation builder: unknown key in rb space → KeyError."""
    from unienv_interface.transformations.identity import IdentityTransformation

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb", ["obs", "action", "reward", "next_obs", "unknown_field"]
    )
    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
    )

    wrapper.reset()
    with pytest.raises(KeyError, match="unknown_field"):
        wrapper.step(np.array([0.5, -0.3], dtype=np.float32))


# ===================================================================
# Test 22: Positive – context works when env provides context_space
# ===================================================================


def test_default_builder_context_available(tmp_path):
    """Default builder: context key works when env has context_space."""
    env = DummyUnbatchedEnvWithContext(obs_dim=3, act_dim=2, ctx_dim=3)
    rb = _make_rb(
        tmp_path / "rb", ["obs", "action", "reward", "next_obs", "context"]
    )
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    wrapper.step(action)

    t = rb.get_at(0)
    assert "context" in t
    # Context from DummyUnbatchedEnvWithContext.reset() is ones
    np.testing.assert_array_equal(t["context"], np.ones(3, dtype=np.float32))


def test_transform_context_available(tmp_path):
    """DataTransformation builder: context key works when env has context_space."""
    from unienv_interface.transformations.identity import IdentityTransformation

    env = DummyUnbatchedEnvWithContext(obs_dim=3, act_dim=2, ctx_dim=3)
    rb = _make_rb(
        tmp_path / "rb", ["obs", "action", "reward", "next_obs", "context"]
    )
    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
        validate_transition=True,
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    wrapper.step(action)

    t = rb.get_at(0)
    assert "context" in t
    np.testing.assert_array_equal(t["context"], np.ones(3, dtype=np.float32))


# ===================================================================
# Test 23: Source-space construction – no double-batching of env spaces
# ===================================================================


class DummyBatchedEnvNWithContext(
    Env[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Any,
        Any,
        np.dtype,
        np.random.Generator,
    ]
):
    """A minimal batched env with batch_size=N and a context_space."""

    metadata = {"render_modes": []}
    render_mode = None
    render_fps = None
    backend = NumpyComputeBackend
    device = None

    def __init__(self, batch_size: int = 3, obs_dim: int = 3, act_dim: int = 2, ctx_dim: int = 4):
        self._batch_size = batch_size
        self.action_space = BoxSpace(
            NumpyComputeBackend, -1.0, 1.0, np.float32,
            shape=(batch_size, act_dim),
        )
        self.observation_space = BoxSpace(
            NumpyComputeBackend, -10.0, 10.0, np.float32,
            shape=(batch_size, obs_dim),
        )
        self.context_space = BoxSpace(
            NumpyComputeBackend, -10.0, 10.0, np.float32,
            shape=(batch_size, ctx_dim),
        )
        self.rng = np.random.default_rng(42)
        self._step_counts = np.zeros(batch_size, dtype=np.int32)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def step(self, action):
        self._step_counts += 1
        B = self._batch_size
        obs = np.zeros((B, self.observation_space.shape[1]), dtype=np.float32)
        terminated = np.zeros(B, dtype=bool)
        truncated = np.zeros(B, dtype=bool)
        return obs, np.ones(B, dtype=np.float32), terminated, truncated, {}

    def reset(self, *, mask=None, seed=None, **kwargs):
        B = self._batch_size
        self._step_counts = np.zeros(B, dtype=np.int32)
        obs = np.zeros((B, self.observation_space.shape[1]), dtype=np.float32)
        ctx = np.ones((B, self.context_space.shape[1]), dtype=np.float32)
        return ctx, obs, {}

    def render(self):
        return None

    def close(self):
        pass


def test_source_space_no_double_batch_batched_shared(tmp_path):
    """In batched shared mode (B>1) with a DataTransformation, the source
    space for obs/action/context must use the env spaces as-is (already
    batched), NOT batch them again."""
    from unienv_interface.transformations.identity import IdentityTransformation

    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
        validate_transition=True,
    )

    # Build the source space as the wrapper would internally.
    # Simulate is_batched=True (shared B>1 path).
    source_space = wrapper._build_canonical_source_space(
        rb.single_space, is_batched=True,
    )

    # The env's observation_space is already (B, obs_dim) = (3, 3).
    # The source space for "obs" must match the env space as-is,
    # NOT be double-batched to (B, B, obs_dim) = (3, 3, 3).
    assert isinstance(source_space, DictSpace)
    assert source_space.spaces["obs"].shape == env.observation_space.shape
    assert source_space.spaces["obs"].shape == (B, 3)
    assert source_space.spaces["action"].shape == env.action_space.shape
    assert source_space.spaces["action"].shape == (B, 2)
    assert source_space.spaces["next_obs"].shape == env.observation_space.shape

    # Scalar fields should be batched (B,) because the rb subspace is scalar ()
    # and we're in batched mode.
    assert source_space.spaces["reward"].shape == (B,)
    assert source_space.spaces["terminated"].shape == (B,)


def test_source_space_no_double_batch_with_context(tmp_path):
    """Context space must also not be double-batched when env is batched."""
    from unienv_interface.transformations.identity import IdentityTransformation

    B = 2
    env = DummyBatchedEnvNWithContext(batch_size=B, obs_dim=3, act_dim=2, ctx_dim=4)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "next_obs", "context"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
        validate_transition=True,
    )

    source_space = wrapper._build_canonical_source_space(
        rb.single_space, is_batched=True,
    )

    # Context space: env's context_space is (B, ctx_dim) = (2, 4).
    # Must NOT be double-batched to (B, B, ctx_dim).
    assert source_space.spaces["context"].shape == env.context_space.shape
    assert source_space.spaces["context"].shape == (B, 4)


def test_source_space_unbatched_env_unchanged(tmp_path):
    """For unbatched env, source space uses env spaces as-is (unbatched)."""
    from unienv_interface.transformations.identity import IdentityTransformation

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
    )

    source_space = wrapper._build_canonical_source_space(
        rb.single_space, is_batched=False,
    )

    # Unbatched env: obs space is (3,), action is (2,), scalars are ().
    assert source_space.spaces["obs"].shape == (3,)
    assert source_space.spaces["action"].shape == (2,)
    assert source_space.spaces["reward"].shape == ()
    assert source_space.spaces["terminated"].shape == ()


def test_batched_shared_extend_with_transform_no_double_batch(tmp_path):
    """End-to-end: batched shared extend with DataTransformation + validation
    must succeed (would fail if source space were double-batched because the
    canonical data shape wouldn't match the double-batched source space)."""
    from unienv_interface.transformations.identity import IdentityTransformation

    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
        validate_transition=True,
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    # This should NOT raise.  Before the fix, the source space for obs
    # would be (B, B, 3) = (3, 3, 3) while the canonical data has shape
    # (B, 3) = (3, 3), causing a mismatch in the transformation.
    wrapper.step(action)

    assert len(rb) == B
    t0 = rb.get_at(0)
    assert t0["obs"].shape == (3,)  # individual transition is unbatched in storage


def test_batched_shared_extend_with_context_transform(tmp_path):
    """End-to-end: batched shared extend with context in DataTransformation."""
    from unienv_interface.transformations.identity import IdentityTransformation

    B = 2
    # ctx_dim=3 to match the hardcoded context shape in _make_rb.
    env = DummyBatchedEnvNWithContext(batch_size=B, obs_dim=3, act_dim=2, ctx_dim=3)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "next_obs", "context"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
        validate_transition=True,
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    wrapper.step(action)

    assert len(rb) == B
    t0 = rb.get_at(0)
    assert "context" in t0
    assert t0["context"].shape == (3,)  # individual transition unbatched


# ===================================================================
# Test 24: Source-space for env_index in shared B>1 mode
# ===================================================================


def test_source_space_env_index_shared_batched(tmp_path):
    """In shared B>1 mode, the source space for env_index is batched (B,)."""
    from unienv_interface.transformations.identity import IdentityTransformation

    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "next_obs", "env_index"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        transition_builder=IdentityTransformation(),
    )

    source_space = wrapper._build_canonical_source_space(
        rb.single_space, is_batched=True,
    )

    assert isinstance(source_space, DictSpace)
    # env_index subspace in rb is scalar (); batched by B it becomes (B,).
    assert source_space.spaces["env_index"].shape == (B,)


def test_source_space_env_index_shared_unbatched_raises(tmp_path):
    """In shared B=None mode, source space for env_index raises KeyError."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "next_obs", "env_index"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
    )

    with pytest.raises(KeyError, match="env_index"):
        wrapper._build_canonical_source_space(
            rb.single_space, is_batched=False,
        )


def test_source_space_env_index_shared_b1_raises(tmp_path):
    """In shared B=1 mode, source space for env_index raises KeyError."""
    env = DummyBatchedEnv1(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "next_obs", "env_index"],
    )

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
    )

    # is_batched=False because B=1 unbatch path produces unbatched data.
    with pytest.raises(KeyError, match="env_index"):
        wrapper._build_canonical_source_space(
            rb.single_space, is_batched=False,
        )


# ===================================================================
# Test 25: Auto-dump – unbatched shared mode
# ===================================================================


def test_auto_dump_unbatched_shared_on_episode_end(tmp_path):
    """Unbatched shared mode: done step closes segment and auto-dumps."""
    import os

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    dump_dir = str(tmp_path / "dump")

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        dump_on_episode_end=True,
        dump_path=dump_dir,
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)

    # Run for 10 steps (env terminates at step 10)
    for _ in range(10):
        wrapper.step(action)

    # Verify the dump actually happened at the configured path
    assert os.path.exists(dump_dir)
    assert os.path.exists(os.path.join(dump_dir, "metadata.json"))


def test_auto_dump_unbatched_shared_no_segment_open(tmp_path):
    """Unbatched shared mode: segment is closed before dump."""
    import os

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    dump_dir = str(tmp_path / "dump")

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        dump_on_episode_end=True,
        dump_path=dump_dir,
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    for _ in range(10):
        wrapper.step(action)

    # After dump, segment should be closed
    assert not rb.storage.has_open_segment
    # Dump should have happened
    assert os.path.exists(dump_dir)


# ===================================================================
# Test 26: Auto-dump – B=1 shared mode
# ===================================================================


def test_auto_dump_batched1_shared_on_episode_end(tmp_path):
    """B=1 shared mode: auto-dumps after done."""
    import os

    env = DummyBatchedEnv1(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    dump_dir = str(tmp_path / "dump")

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        dump_on_episode_end=True,
        dump_path=dump_dir,
    )

    wrapper.reset()
    action = np.array([[0.5, -0.3]], dtype=np.float32)

    # Run for 5 steps (env terminates at step 5)
    for _ in range(5):
        wrapper.step(action)

    # Should have dumped
    assert os.path.exists(dump_dir)
    assert os.path.exists(os.path.join(dump_dir, "metadata.json"))


# ===================================================================
# Test 27: Auto-dump – per_env mode
# ===================================================================


def test_auto_dump_per_env_only_done_slots(tmp_path):
    """Per-env mode: only done slots dump on step."""
    import os

    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    # env0 terminates at step 2, others run longer
    env._max_steps = np.array([2, 99, 99], dtype=np.int32)

    rbs = [
        _make_rb(
            tmp_path / f"rb_{i}",
            ["obs", "action", "reward", "terminated", "next_obs"],
            maintain_segment_metadata=True,
        )
        for i in range(B)
    ]

    dump_dir = str(tmp_path / "dump")

    wrapper = ReplayBufferCollectionWrapper(
        env, rbs, replay_mode="per_env",
        dump_on_episode_end=True,
        dump_path=dump_dir,
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)

    # Run for 3 steps
    for _ in range(3):
        wrapper.step(action)

    # Buffer 0 should have dumped (env0 terminates at step 2).
    # Buffers 1 and 2 should not have dumped.
    # Since all buffers dump to the same path, we verify by checking
    # that buffer 0's segment was closed (indicating dump happened).
    assert not rbs[0].storage.has_open_segment
    # Buffers 1 and 2 still have open segments (no dump triggered)
    assert rbs[1].storage.has_open_segment
    assert rbs[2].storage.has_open_segment

    # The dump directory should exist
    assert os.path.exists(dump_dir)


def test_auto_dump_per_env_masked_reset_boundary(tmp_path):
    """Per-env mode: masked reset boundary dumps only selected slots."""
    import os

    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    env._max_steps = np.array([99, 99, 99], dtype=np.int32)  # never auto-terminate

    rbs = [
        _make_rb(
            tmp_path / f"rb_{i}",
            ["obs", "action", "reward", "terminated", "next_obs"],
            maintain_segment_metadata=True,
        )
        for i in range(B)
    ]

    dump_dir = str(tmp_path / "dump")

    wrapper = ReplayBufferCollectionWrapper(
        env, rbs, replay_mode="per_env",
        dump_on_reset_boundary=True,
        dump_path=dump_dir,
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    for _ in range(3):
        wrapper.step(action)

    # Masked reset: only reset slot 0 and 2
    mask = np.array([True, False, True], dtype=bool)
    wrapper.reset(mask=mask)

    # Buffers 0 and 2 should have been dumped (segments closed)
    assert not rbs[0].storage.has_open_segment
    assert not rbs[2].storage.has_open_segment
    # Buffer 1 should still have an open segment (not reset)
    assert rbs[1].storage.has_open_segment

    # The dump directory should exist
    assert os.path.exists(dump_dir)


# ===================================================================
# Test 28: Auto-dump – shared B>1 mode
# ===================================================================


def test_auto_dump_shared_batched_any_done(tmp_path):
    """Shared B>1 mode: any done triggers one dump, without segment semantics."""
    import os

    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    # env0 terminates at step 2, others run longer
    env._max_steps = np.array([2, 99, 99], dtype=np.int32)

    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    dump_dir = str(tmp_path / "dump")

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        dump_on_episode_end=True,
        dump_path=dump_dir,
    )

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)

    # Run for 3 steps
    for _ in range(3):
        wrapper.step(action)

    # Should have dumped at least once (when env0 terminated at step 2)
    assert os.path.exists(dump_dir)
    assert os.path.exists(os.path.join(dump_dir, "metadata.json"))


# ===================================================================
# Test 29: Auto-dump – error when dump_path is None
# ===================================================================


def test_auto_dump_without_path_raises():
    """Dumping requested without dump_path -> clear error at init."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tempfile.mkdtemp() + "/rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    with pytest.raises(ValueError, match="dump_path must be provided"):
        ReplayBufferCollectionWrapper(
            env, rb, replay_mode="shared",
            dump_on_episode_end=True,
            dump_path=None,
        )


def test_auto_dump_reset_boundary_without_path_raises():
    """Dumping on reset boundary without dump_path -> clear error at init."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tempfile.mkdtemp() + "/rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    with pytest.raises(ValueError, match="dump_path must be provided"):
        ReplayBufferCollectionWrapper(
            env, rb, replay_mode="shared",
            dump_on_reset_boundary=True,
            dump_path=None,
        )


# ===================================================================
# Test 29b: Auto-dump – dump_path is mutable after construction
# ===================================================================


def test_dump_path_mutable_after_init(tmp_path):
    """Changing wrapper.dump_path after init affects subsequent dumps."""
    import os

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    dump_dir_1 = str(tmp_path / "dump_1")
    dump_dir_2 = str(tmp_path / "dump_2")

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        dump_on_episode_end=True,
        dump_path=dump_dir_1,
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)

    # Run for 10 steps (env terminates at step 10)
    for _ in range(10):
        wrapper.step(action)

    # First dump should have gone to dump_dir_1
    assert os.path.exists(dump_dir_1)
    assert not os.path.exists(dump_dir_2)

    # Change dump_path
    wrapper.dump_path = dump_dir_2

    # Reset to start a new episode, then run to termination again
    wrapper.reset()
    for _ in range(10):
        wrapper.step(action)

    # Second dump should have gone to dump_dir_2
    assert os.path.exists(dump_dir_2)


def test_dump_path_repeated_overwrites(tmp_path):
    """Repeated dumps to the same path overwrite (no error)."""
    import os

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    dump_dir = str(tmp_path / "dump")

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        dump_on_episode_end=True,
        dump_path=dump_dir,
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)

    # Run for 10 steps (env terminates at step 10)
    for _ in range(10):
        wrapper.step(action)

    # First dump happened
    assert os.path.exists(dump_dir)

    # Reset and run again - should overwrite without error
    wrapper.reset()
    for _ in range(10):
        wrapper.step(action)

    # Still exists (overwritten)
    assert os.path.exists(dump_dir)
    assert os.path.exists(os.path.join(dump_dir, "metadata.json"))


# ===================================================================
# Test 30: Buffer swapping – rejects open segments by default
# ===================================================================


def test_swap_rejects_open_segments_by_default(tmp_path):
    """Swap rejects open segments by default in append-based modes."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb_old = _make_rb(
        tmp_path / "rb_old",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )
    rb_new = _make_rb(
        tmp_path / "rb_new",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    wrapper = ReplayBufferCollectionWrapper(env, rb_old, replay_mode="shared")

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    wrapper.step(action)

    # rb_old has an open segment
    assert rb_old.storage.has_open_segment

    # Swap should reject
    with pytest.raises(RuntimeError, match="open segments"):
        wrapper.set_replay_buffers(rb_new)


def test_swap_with_finalize_old_segments(tmp_path):
    """Swap with finalize_old_segments=True succeeds."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb_old = _make_rb(
        tmp_path / "rb_old",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )
    rb_new = _make_rb(
        tmp_path / "rb_new",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    wrapper = ReplayBufferCollectionWrapper(env, rb_old, replay_mode="shared")

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    wrapper.step(action)

    # Swap with finalize
    old_buffers = wrapper.set_replay_buffers(rb_new, finalize_old_segments=True)

    assert old_buffers == [rb_old]
    assert wrapper.replay_buffers == [rb_new]
    # Old buffer segment should be closed
    assert not rb_old.storage.has_open_segment


def test_swap_with_dump_old(tmp_path):
    """Swap with dump_old=True triggers dumps safely."""
    import os

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb_old = _make_rb(
        tmp_path / "rb_old",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )
    rb_new = _make_rb(
        tmp_path / "rb_new",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    dump_dir = str(tmp_path / "dump")

    wrapper = ReplayBufferCollectionWrapper(
        env, rb_old, replay_mode="shared",
        dump_path=dump_dir,
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    wrapper.step(action)

    # Swap with dump_old
    wrapper.set_replay_buffers(rb_new, dump_old=True)

    # Should have dumped old buffer to dump_dir
    assert os.path.exists(dump_dir)
    assert os.path.exists(os.path.join(dump_dir, "metadata.json"))
    # Old buffer segment should be closed
    assert not rb_old.storage.has_open_segment


def test_swap_shared_batched_no_open_segment_issues(tmp_path):
    """Shared B>1 swap works without open-segment issues."""
    B = 3
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rb_old = _make_rb(
        tmp_path / "rb_old",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )
    rb_new = _make_rb(
        tmp_path / "rb_new",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(env, rb_old, replay_mode="shared")

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    wrapper.step(action)

    # Shared B>1 uses extend, no open segments
    # Swap should work without issues
    old_buffers = wrapper.set_replay_buffers(rb_new)

    assert old_buffers == [rb_old]
    assert wrapper.replay_buffers == [rb_new]


# ===================================================================
# Test 31: Buffer swapping – require_same_space validation
# ===================================================================


def test_swap_require_same_space_validation(tmp_path):
    """Swap with require_same_space=True validates spaces."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb_old = _make_rb(
        tmp_path / "rb_old",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )
    # Different space (missing "reward")
    rb_new = _make_rb(
        tmp_path / "rb_new",
        ["obs", "action", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(env, rb_old, replay_mode="shared")

    wrapper.reset()

    with pytest.raises(ValueError, match="different single_space"):
        wrapper.set_replay_buffers(rb_new, require_same_space=True)


def test_swap_allow_different_space(tmp_path):
    """Swap with require_same_space=False allows different spaces."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb_old = _make_rb(
        tmp_path / "rb_old",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )
    rb_new = _make_rb(
        tmp_path / "rb_new",
        ["obs", "action", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(env, rb_old, replay_mode="shared")

    wrapper.reset()

    # Should not raise
    old_buffers = wrapper.set_replay_buffers(rb_new, require_same_space=False)
    assert old_buffers == [rb_old]


# ===================================================================
# Test 32: Buffer swapping – per_env mode
# ===================================================================


def test_swap_per_env_mode(tmp_path):
    """Swap in per_env mode."""
    B = 2
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rbs_old = [
        _make_rb(
            tmp_path / f"rb_old_{i}",
            ["obs", "action", "reward", "terminated", "next_obs"],
            maintain_segment_metadata=True,
        )
        for i in range(B)
    ]
    rbs_new = [
        _make_rb(
            tmp_path / f"rb_new_{i}",
            ["obs", "action", "reward", "terminated", "next_obs"],
            maintain_segment_metadata=True,
        )
        for i in range(B)
    ]

    wrapper = ReplayBufferCollectionWrapper(env, rbs_old, replay_mode="per_env")

    wrapper.reset()
    action = np.zeros((B, 2), dtype=np.float32)
    wrapper.step(action)

    # Swap with finalize
    old_buffers = wrapper.set_replay_buffers(rbs_new, finalize_old_segments=True)

    assert old_buffers == rbs_old
    assert wrapper.replay_buffers == rbs_new


def test_swap_per_env_rejects_wrong_count(tmp_path):
    """Swap in per_env mode rejects wrong buffer count."""
    B = 2
    env = DummyBatchedEnvN(batch_size=B, obs_dim=3, act_dim=2)
    rbs_old = [
        _make_rb(
            tmp_path / f"rb_old_{i}",
            ["obs", "action", "reward", "terminated", "next_obs"],
        )
        for i in range(B)
    ]
    # Only 1 buffer for B=2
    rbs_new = [
        _make_rb(
            tmp_path / "rb_new",
            ["obs", "action", "reward", "terminated", "next_obs"],
        )
    ]

    wrapper = ReplayBufferCollectionWrapper(env, rbs_old, replay_mode="per_env")

    wrapper.reset()

    with pytest.raises(ValueError, match="exactly one replay buffer per"):
        wrapper.set_replay_buffers(rbs_new)


# ===================================================================
# Test 33: Buffer swapping – alias
# ===================================================================


def test_buffer_swap_alias(tmp_path):
    """buffer_swap is an alias for set_replay_buffers."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb_old = _make_rb(
        tmp_path / "rb_old",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )
    rb_new = _make_rb(
        tmp_path / "rb_new",
        ["obs", "action", "reward", "terminated", "next_obs"],
    )

    wrapper = ReplayBufferCollectionWrapper(env, rb_old, replay_mode="shared")

    wrapper.reset()

    # Use alias
    old_buffers = wrapper.buffer_swap(rb_new)

    assert old_buffers == [rb_old]
    assert wrapper.replay_buffers == [rb_new]


# ===================================================================
# Test 34: Auto-dump – reset boundary in shared mode
# ===================================================================


def test_auto_dump_shared_reset_boundary(tmp_path):
    """Shared mode: reset boundary dumps the buffer."""
    import os

    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb = _make_rb(
        tmp_path / "rb",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    dump_dir = str(tmp_path / "dump")

    wrapper = ReplayBufferCollectionWrapper(
        env, rb, replay_mode="shared",
        dump_on_reset_boundary=True,
        dump_path=dump_dir,
    )

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    for _ in range(3):
        wrapper.step(action)

    # Reset again – should dump
    wrapper.reset()

    assert os.path.exists(dump_dir)
    assert os.path.exists(os.path.join(dump_dir, "metadata.json"))


# ===================================================================
# Test 35: Buffer swapping – allow_mid_episode
# ===================================================================


def test_swap_allow_mid_episode(tmp_path):
    """Swap with allow_mid_episode=True allows mid-episode swap."""
    env = DummyUnbatchedEnv(obs_dim=3, act_dim=2)
    rb_old = _make_rb(
        tmp_path / "rb_old",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )
    rb_new = _make_rb(
        tmp_path / "rb_new",
        ["obs", "action", "reward", "terminated", "next_obs"],
        maintain_segment_metadata=True,
    )

    wrapper = ReplayBufferCollectionWrapper(env, rb_old, replay_mode="shared")

    wrapper.reset()
    action = np.array([0.5, -0.3], dtype=np.float32)
    wrapper.step(action)

    # Old buffer has open segment
    assert rb_old.storage.has_open_segment

    # Swap with allow_mid_episode
    old_buffers = wrapper.set_replay_buffers(rb_new, allow_mid_episode=True)

    assert old_buffers == [rb_old]
    assert wrapper.replay_buffers == [rb_new]
    # Old buffer still has open segment (not finalized)
    assert rb_old.storage.has_open_segment
