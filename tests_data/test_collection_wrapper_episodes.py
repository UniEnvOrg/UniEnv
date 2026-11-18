"""ReplayBufferCollectionWrapper writing episode records into buffers with an ``episode_space``."""
import numpy as np
import pytest

from unienv_data.replay_buffer import EpisodeSummary, ReplayBuffer, ReplayBufferCollectionWrapper
from unienv_data.storages.hdf5 import HDF5Storage
from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace

from tests_common.envs import StubEnv, make_spaces

ACTION, OBS, CONTEXT = make_spaces(None)
STEP_SPACE = DictSpace(B, {"obs": OBS, "action": ACTION, "reward": BoxSpace(B, -np.inf, np.inf, np.float32, shape=())})
EPISODE_SPACE = DictSpace(B, {
    "context": CONTEXT,
    "return": BoxSpace(B, -np.inf, np.inf, np.float64, shape=()),
    "length": BoxSpace(B, 0, np.inf, np.int64, shape=()),
    "terminated": BoxSpace(B, 0, 1, np.int64, shape=()),
    "truncated": BoxSpace(B, 0, 1, np.int64, shape=()),
})


class SuccessEnv(StubEnv):
    """StubEnv whose info reports ``success`` (= terminated)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info = dict(info, success=np.array(terminated) if self.batch_size else bool(terminated))
        return obs, reward, terminated, truncated, info


def _rb(tmp_path, name, episode_space=EPISODE_SPACE):
    return ReplayBuffer.create(
        HDF5Storage, STEP_SPACE, cache_path=str(tmp_path / name),
        episode_space=episode_space, episode_storage_cls=HDF5Storage,
    )


def _records(rb):
    return [(start, end, rb.get_episode_data(episode_id)) for start, end, episode_id in rb.get_episode_segments()]


def test_unbatched_episodes_get_context_return_length_and_flags(tmp_path):
    env = StubEnv()
    rb = _rb(tmp_path, "rb")
    wrapper = ReplayBufferCollectionWrapper(env, rb)
    rewards = []
    for seed in (2, 5):
        wrapper.reset(seed=seed)
        episode_rewards, done = [], False
        while not done:
            _, reward, terminated, truncated, _ = wrapper.step(np.ones(2, np.float32))
            episode_rewards.append(reward)
            done = terminated or truncated
        rewards.append(episode_rewards)

    records = _records(rb)
    assert [end - start for start, end, _ in records] == [len(r) for r in rewards]
    for (start, end, record), seed, episode_rewards in zip(records, (2, 5), rewards):
        np.testing.assert_array_equal(record["context"], np.full(2, seed, np.float32))
        assert int(record["length"]) == len(episode_rewards)
        assert float(record["return"]) == pytest.approx(sum(episode_rewards))
        assert (int(record["terminated"]), int(record["truncated"])) == (1, 0)


def test_a_reset_mid_episode_closes_it_as_truncated(tmp_path):
    env = StubEnv()
    rb = _rb(tmp_path, "rb")
    wrapper = ReplayBufferCollectionWrapper(env, rb)
    wrapper.reset(seed=1)
    for _ in range(2):
        wrapper.step(np.zeros(2, np.float32))
    wrapper.reset(seed=3)  # cuts the first episode
    wrapper.step(np.zeros(2, np.float32))
    wrapper.reset(seed=4)
    (s0, e0, first), (s1, e1, second) = _records(rb)
    assert (s0, e0, s1, e1) == (0, 2, 2, 3)
    assert (int(first["terminated"]), int(first["truncated"]), int(first["length"])) == (0, 1, 2)
    np.testing.assert_array_equal(first["context"], np.full(2, 1, np.float32))
    np.testing.assert_array_equal(second["context"], np.full(2, 3, np.float32))


def test_context_space_as_episode_space_stores_each_context_once(tmp_path):
    env = StubEnv()
    rb = _rb(tmp_path, "rb", episode_space=CONTEXT)
    wrapper = ReplayBufferCollectionWrapper(env, rb)
    for seed in (7, 8):
        wrapper.reset(seed=seed)
        for _ in range(3):
            wrapper.step(np.zeros(2, np.float32))
    wrapper.reset(seed=0)
    contexts = [record for _, _, record in _records(rb)]
    np.testing.assert_array_equal(np.stack(contexts), [[7, 7], [8, 8]])


def test_per_env_records_per_slot_with_masked_resets(tmp_path):
    env = SuccessEnv(batch_size=2)
    space = DictSpace(B, {
        "context": CONTEXT,
        "length": BoxSpace(B, 0, np.inf, np.int64, shape=()),
        "success": BoxSpace(B, 0, 1, np.int64, shape=()),
        "env_index": BoxSpace(B, 0, np.inf, np.int64, shape=()),
        "truncated": BoxSpace(B, 0, 1, np.int64, shape=()),
    })
    buffers = [_rb(tmp_path, f"rb{i}", space) for i in range(2)]
    wrapper = ReplayBufferCollectionWrapper(env, buffers, replay_mode="per_env")
    wrapper.reset(seed=1)
    for _ in range(2):
        wrapper.step(np.zeros((2, 2), np.float32))
    wrapper.reset(mask=np.array([False, True]), seed=9)  # cuts slot 1's episode only
    done = np.zeros(2, bool)
    while not done[0]:
        _, _, terminated, truncated, _ = wrapper.step(np.ones((2, 2), np.float32))
        done = terminated | truncated

    (start, end, slot0), = _records(buffers[0])
    assert (int(slot0["success"]), int(slot0["truncated"]), int(slot0["env_index"])) == (1, 0, 0)
    assert int(slot0["length"]) == end - start
    np.testing.assert_array_equal(slot0["context"], [1, 1])
    slot1_records = _records(buffers[1])
    _, _, cut = slot1_records[0]
    assert (int(cut["success"]), int(cut["truncated"]), int(cut["length"]), int(cut["env_index"])) == (0, 1, 2, 1)
    np.testing.assert_array_equal(cut["context"], [1, 1])
    if len(slot1_records) > 1:  # slot 1 also finished its second episode
        np.testing.assert_array_equal(slot1_records[1][2]["context"], [9, 9])
    assert buffers[1].get_episode_ids(len(buffers[1]) - 1) in (-1, slot1_records[-1][2] is not None and buffers[1].get_episode_ids(len(buffers[1]) - 1))


def test_custom_episode_builder_gets_the_summary(tmp_path):
    env = SuccessEnv()
    space = DictSpace(B, {"score": BoxSpace(B, -np.inf, np.inf, np.float64, shape=())})
    rb = _rb(tmp_path, "rb", space)
    seen = []

    def builder(summary: EpisodeSummary):
        seen.append(summary)
        return {"score": summary.total_reward * 10 + summary.length}

    wrapper = ReplayBufferCollectionWrapper(env, rb, episode_builder=builder)
    wrapper.reset(seed=2)
    done = False
    while not done:
        _, _, terminated, truncated, _ = wrapper.step(np.ones(2, np.float32))
        done = terminated or truncated
    summary, = seen
    assert summary.terminated and not summary.truncated and summary.info["success"] is True
    assert summary.env_index is None and summary.length == len(rb)
    np.testing.assert_array_equal(summary.context, [2, 2])
    assert float(rb.get_episode_data(0)["score"]) == pytest.approx(summary.total_reward * 10 + summary.length)


def test_default_record_errors_name_what_is_missing(tmp_path):
    env = StubEnv()
    rb = _rb(tmp_path, "rb", DictSpace(B, {"success": BoxSpace(B, 0, 1, np.int64, shape=())}))
    wrapper = ReplayBufferCollectionWrapper(env, rb)
    wrapper.reset(seed=0)
    wrapper.step(np.zeros(2, np.float32))
    with pytest.raises(KeyError, match="info has no 'success'"):
        wrapper.reset(seed=1)

    rb2 = _rb(tmp_path, "rb2", DictSpace(B, {"difficulty": BoxSpace(B, 0, 1, np.int64, shape=())}))
    wrapper2 = ReplayBufferCollectionWrapper(env, rb2)
    wrapper2.reset(seed=0)
    wrapper2.step(np.zeros(2, np.float32))
    with pytest.raises(KeyError, match="pass an episode_builder"):
        wrapper2.reset(seed=1)


def test_validate_transition_checks_episode_records(tmp_path):
    env = StubEnv()
    space = DictSpace(B, {"length": BoxSpace(B, 0, 3, np.int64, shape=())})
    rb = _rb(tmp_path, "rb", space)
    wrapper = ReplayBufferCollectionWrapper(env, rb, validate_transition=True)
    wrapper.reset(seed=0)
    for _ in range(5):
        wrapper.step(np.zeros(2, np.float32))
    with pytest.raises(ValueError, match="does not match target space"):
        wrapper.reset(seed=1)  # length 5 is outside [0, 3]


def test_swapping_buffers_finalizes_with_a_truncated_record(tmp_path):
    env = StubEnv()
    old, new = _rb(tmp_path, "old"), _rb(tmp_path, "new")
    wrapper = ReplayBufferCollectionWrapper(env, old)
    wrapper.reset(seed=4)
    for _ in range(3):
        wrapper.step(np.zeros(2, np.float32))
    wrapper.set_replay_buffers(new, finalize_old_segments=True)
    (_, _, record), = _records(old)
    assert (int(record["length"]), int(record["truncated"])) == (3, 1)
    wrapper.step(np.zeros(2, np.float32))
    wrapper.reset(seed=5)
    (_, _, continued), = _records(new)
    assert int(continued["length"]) == 1  # the new buffer's part of the episode
    np.testing.assert_array_equal(continued["context"], [4, 4])


def test_resets_close_segments_of_buffers_without_records(tmp_path):
    env = StubEnv()
    rb = ReplayBuffer.create(HDF5Storage, STEP_SPACE, cache_path=str(tmp_path / "rb"), maintain_segment_metadata=True)
    wrapper = ReplayBufferCollectionWrapper(env, rb)
    wrapper.reset(seed=0)
    for _ in range(2):
        wrapper.step(np.zeros(2, np.float32))
    wrapper.reset(seed=1)
    wrapper.step(np.zeros(2, np.float32))
    wrapper.reset(seed=2)
    assert rb.get_segments() == [(0, 2), (2, 3)]


START_SPACE = DictSpace(B, {"context": CONTEXT, "env_index": BoxSpace(B, 0, np.inf, np.int64, shape=())})


def test_auto_writes_context_records_at_the_first_step(tmp_path):
    env = StubEnv()
    rb = _rb(tmp_path, "rb", episode_space=CONTEXT)
    wrapper = ReplayBufferCollectionWrapper(env, rb)
    wrapper.reset(seed=7)
    wrapper.step(np.zeros(2, np.float32))
    assert int(rb.get_episode_ids(0)) == 0  # readable while the episode is open
    np.testing.assert_array_equal(rb.get_episode_data(0), [7, 7])
    assert wrapper._episode_accumulators[0].context is None  # the wrapper keeps no copy
    wrapper.step(np.zeros(2, np.float32))
    wrapper.reset(seed=8)  # closes the segment without a second record
    wrapper.step(np.zeros(2, np.float32))
    assert rb.get_episode_segments()[0] == (0, 2, 0)
    np.testing.assert_array_equal(rb.get_episode_data(1), [8, 8])


def test_per_env_start_records_with_env_index(tmp_path):
    env = StubEnv(batch_size=2)
    buffers = [_rb(tmp_path, f"rb{i}", START_SPACE) for i in range(2)]
    wrapper = ReplayBufferCollectionWrapper(env, buffers, replay_mode="per_env")
    wrapper.reset(seed=3)
    wrapper.step(np.zeros((2, 2), np.float32))
    for i, rb in enumerate(buffers):
        record = rb.get_episode_data(int(rb.get_episode_ids(0)))
        assert int(record["env_index"]) == i
        np.testing.assert_array_equal(record["context"], [3, 3])
    wrapper.reset(mask=np.array([True, False]), seed=4)
    wrapper.step(np.zeros((2, 2), np.float32))
    assert buffers[0].get_episode_segments()[0] == (0, 1, 0)
    np.testing.assert_array_equal(buffers[0].get_episode_data(1)["context"], [4, 4])
    assert buffers[1].get_episode_ids(np.arange(2)).tolist() == [0, 0]  # still the first episode


def test_auto_waits_for_the_end_when_the_record_needs_it(tmp_path):
    rb = _rb(tmp_path, "rb")  # EPISODE_SPACE has return / length / flags
    wrapper = ReplayBufferCollectionWrapper(StubEnv(), rb)
    wrapper.reset(seed=1)
    wrapper.step(np.zeros(2, np.float32))
    assert int(rb.get_episode_ids(0)) == -1
    builder_rb = _rb(tmp_path, "rb2", episode_space=CONTEXT)
    wrapper = ReplayBufferCollectionWrapper(StubEnv(), builder_rb, episode_builder=lambda summary: summary.context)
    wrapper.reset(seed=1)
    wrapper.step(np.zeros(2, np.float32))
    assert int(builder_rb.get_episode_ids(0)) == -1


def test_start_mode_builder_and_errors(tmp_path):
    space = DictSpace(B, {"tag": BoxSpace(B, -np.inf, np.inf, np.float32, shape=(2,))})
    rb = _rb(tmp_path, "rb", space)
    seen = []

    def builder(summary):
        seen.append(summary)
        return {"tag": summary.context * 2}

    wrapper = ReplayBufferCollectionWrapper(StubEnv(), rb, episode_builder=builder, episode_data_at="start")
    wrapper.reset(seed=5)
    for _ in range(3):
        wrapper.step(np.zeros(2, np.float32))
    wrapper.reset(seed=6)
    summary, = seen
    assert summary.at_start and summary.length == 0
    np.testing.assert_array_equal(rb.get_episode_data(0)["tag"], [10, 10])
    assert rb.get_episode_segments() == [(0, 3, 0)]

    with pytest.raises(ValueError, match="cannot provide"):
        ReplayBufferCollectionWrapper(StubEnv(), _rb(tmp_path, "rb2"), episode_data_at="start")
    with pytest.raises(ValueError, match="episode_data_at must be"):
        ReplayBufferCollectionWrapper(StubEnv(), _rb(tmp_path, "rb3"), episode_data_at="middle")


def test_end_mode_keeps_context_records_until_the_segment_closes(tmp_path):
    rb = _rb(tmp_path, "rb", episode_space=CONTEXT)
    wrapper = ReplayBufferCollectionWrapper(StubEnv(), rb, episode_data_at="end")
    wrapper.reset(seed=2)
    wrapper.step(np.zeros(2, np.float32))
    assert int(rb.get_episode_ids(0)) == -1
    wrapper.reset(seed=3)
    np.testing.assert_array_equal(rb.get_episode_data(0), [2, 2])


def test_swap_in_start_mode_finalizes_without_a_second_record(tmp_path):
    old, new = _rb(tmp_path, "old", CONTEXT), _rb(tmp_path, "new", CONTEXT)
    wrapper = ReplayBufferCollectionWrapper(StubEnv(), old)
    wrapper.reset(seed=4)
    for _ in range(2):
        wrapper.step(np.zeros(2, np.float32))
    wrapper.set_replay_buffers(new, finalize_old_segments=True)
    assert old.get_episode_segments() == [(0, 2, 0)]
    wrapper.step(np.zeros(2, np.float32))
    np.testing.assert_array_equal(new.get_episode_data(0), [4, 4])  # the new buffer's part gets its own record
