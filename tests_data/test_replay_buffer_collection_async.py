"""Async (reset_async / step_async) bridge of ReplayBufferCollectionWrapper."""

import numpy as np
import pytest

from unienv_data.replay_buffer import ReplayBuffer, ReplayBufferCollectionWrapper
from unienv_data.storages.parquet import ParquetStorage
from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace

from tests_common.envs import StubEnv


class AsyncStubEnv(StubEnv):
    """StubEnv with a deferred-result async API."""

    def reset_async(self, *args, mask=None, seed=None, **kwargs):
        self._pending_reset = self.reset(*args, mask=mask, seed=seed, **kwargs)

    def reset_wait(self):
        result, self._pending_reset = self._pending_reset, None
        return result

    def step_async(self, action):
        self._pending_step = self.step(action)

    def step_wait(self):
        result, self._pending_step = self._pending_step, None
        return result


def _rb(tmp_path, name, maintain_segment_metadata=False):
    space = DictSpace(B, {
        "obs": DictSpace(B, {
            "position": BoxSpace(B, -np.inf, np.inf, np.float32, shape=(2,)),
            "velocity": BoxSpace(B, -np.inf, np.inf, np.float32, shape=(2,)),
        }),
        "action": BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,)),
        "reward": BoxSpace(B, -np.inf, np.inf, np.float32, shape=()),
    })
    return ReplayBuffer.create(_FlatParquet, space, cache_path=str(tmp_path / name), capacity=None,
                               maintain_segment_metadata=maintain_segment_metadata)


class _FlatParquet:
    """Parquet only supports flat dicts; flatten 'obs' through a DictStorage of Parquet children."""
    single_file_ext = None

    @staticmethod
    def create(space, *args, **kwargs):
        from unienv_data.storages.dict_storage import DictStorage
        return DictStorage.create(space, {"obs/position": ParquetStorage, "obs/velocity": ParquetStorage,
                                          "*": ParquetStorage}, *args, **kwargs)


def test_unbatched_async_steps_are_recorded(tmp_path):
    env = AsyncStubEnv()
    rb = _rb(tmp_path, "rb")
    wrapper = ReplayBufferCollectionWrapper(env, rb, replay_mode="shared")
    wrapper.reset_async(seed=1)
    wrapper.reset_wait()
    for k in range(3):
        action = np.full(2, 0.1 * (k + 1), np.float32)
        wrapper.step_async(action)
        obs, reward, terminated, truncated, info = wrapper.step_wait()
    assert len(rb) == 3
    np.testing.assert_allclose(rb.get_at(-1)["action"], np.full(2, 0.3, np.float32))
    np.testing.assert_allclose(rb.get_at(-1)["obs"]["position"], obs["position"] - np.full(2, 0.3, np.float32))


def test_batched_async_masked_reset_marks_per_env_segments(tmp_path):
    env = AsyncStubEnv(batch_size=2)
    rbs = [_rb(tmp_path, f"rb{i}", maintain_segment_metadata=True) for i in range(2)]
    wrapper = ReplayBufferCollectionWrapper(env, rbs, replay_mode="per_env")
    wrapper.reset_async(seed=0)
    wrapper.reset_wait()
    for _ in range(2):
        wrapper.step_async(np.full((2, 2), 0.5, np.float32))
        wrapper.step_wait()
    wrapper.reset_async(mask=np.array([True, False]), seed=3)
    context, obs, info = wrapper.reset_wait()
    assert obs["position"].shape == (2, 2)  # compact reset merged back into the full batch
    wrapper.step_async(np.full((2, 2), 0.5, np.float32))
    wrapper.step_wait()
    assert len(rbs[0]) == 3 and len(rbs[1]) == 3
    assert rbs[0].get_segments() == [(0, 2)]
    for rb in rbs:
        rb.mark_segment_end()
    assert rbs[0].get_segments() == [(0, 2), (2, 3)]
    assert rbs[1].get_segments() == [(0, 3)]


def test_async_methods_require_support_from_the_env(tmp_path):
    env = StubEnv()
    wrapper = ReplayBufferCollectionWrapper(env, _rb(tmp_path, "rb"), replay_mode="shared")
    for call in (lambda: wrapper.reset_async(), wrapper.reset_wait,
                 lambda: wrapper.step_async(np.zeros(2, np.float32)), wrapper.step_wait):
        with pytest.raises(AttributeError):
            call()


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_masked_reset_leaves_previously_returned_arrays_untouched(tmp_path, use_async):
    # The merge of a masked reset must not write the reset rows into the observation / context the
    # caller got from the last step / reset (e.g. the terminal observation of the envs being reset)
    import copy
    env = AsyncStubEnv(batch_size=2)
    rbs = [_rb(tmp_path, f"rb{i}") for i in range(2)]
    wrapper = ReplayBufferCollectionWrapper(env, rbs, replay_mode="per_env")

    def reset(**kwargs):
        if use_async:
            wrapper.reset_async(**kwargs)
            return wrapper.reset_wait()
        return wrapper.reset(**kwargs)

    first_context, _, _ = reset(seed=0)
    first_context_before = copy.deepcopy(first_context)
    for _ in range(2):
        last_obs, *_ = wrapper.step(np.full((2, 2), 0.5, np.float32))
    last_obs_before = copy.deepcopy(last_obs)

    context, obs, info = reset(mask=np.array([True, False]), seed=3)

    for key in last_obs:
        np.testing.assert_array_equal(last_obs[key], last_obs_before[key])  # caller's arrays unchanged
        np.testing.assert_array_equal(obs[key][1], last_obs_before[key][1])  # env 1 keeps its observation
        assert not np.array_equal(obs[key][0], last_obs_before[key][0])  # env 0 was reset
    np.testing.assert_array_equal(first_context, first_context_before)
    np.testing.assert_array_equal(context[1], first_context_before[1])
