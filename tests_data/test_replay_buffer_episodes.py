"""Per-episode records of a ReplayBuffer (``episode_space``): writing, eviction, persistence and the view."""
import random
from collections import deque

import numpy as np
import pytest
import torch

from unienv_data.replay_buffer import ReplayBuffer, ReplayBufferEpisodeView
from unienv_data.samplers import StepSampler
from unienv_data.storages.hdf5 import HDF5Storage
from unienv_data.storages.pytorch import PytorchTensorStorage
from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
from unienv_interface.space import BoxSpace, DictSpace, TextSpace

STEP = BoxSpace(NB, -1e6, 1e6, np.float64, shape=(2,))
EPISODE = DictSpace(NB, {
    "task": TextSpace(NB, max_length=64),
    "success": BoxSpace(NB, 0, 1, np.int64, shape=()),
    "goal": BoxSpace(NB, -10.0, 10.0, np.float32, shape=(3,)),
})


def _steps(n, start=0.0):
    return np.stack([np.arange(n) + start, np.zeros(n)], -1).astype(np.float64)


def _episode(i):
    return {"task": f"task {i}", "success": i % 2, "goal": np.full(3, i, np.float32)}


def _buffer(tmp_path, capacity=None, **kwargs):
    return ReplayBuffer.create(
        HDF5Storage, STEP, cache_path=str(tmp_path / "rb"), capacity=capacity,
        episode_space=EPISODE, episode_storage_cls=HDF5Storage, **kwargs,
    )


# --------------------------------------------------------------------------- creation / arguments

def test_episode_space_needs_a_storage_and_storage_needs_a_space(tmp_path):
    with pytest.raises(ValueError, match="episode_storage_cls"):
        ReplayBuffer.create(HDF5Storage, STEP, cache_path=str(tmp_path / "a"), episode_space=EPISODE)
    with pytest.raises(ValueError, match="need an episode_space"):
        ReplayBuffer.create(HDF5Storage, STEP, cache_path=str(tmp_path / "b"), episode_storage_cls=HDF5Storage)


def test_every_segment_requires_episode_data(tmp_path):
    rb = _buffer(tmp_path)
    assert rb.maintain_segment_metadata and rb.episode_space is EPISODE or rb.episode_space == EPISODE
    with pytest.raises(ValueError, match="every segment needs its episode_data"):
        rb.extend(_steps(3))
    assert len(rb) == 0  # nothing written
    rb.append(_steps(1)[0])
    with pytest.raises(ValueError, match="every segment needs its episode_data"):
        rb.mark_segment_end()
    rb.mark_segment_end(episode_data=_episode(0))  # the segment stayed open and can still be closed
    with pytest.raises(ValueError, match="no open segment"):
        rb.mark_segment_end(episode_data=_episode(1))
    rb.mark_segment_end()  # nothing open: a no-op without data
    writer = rb.open_segment_writer()
    writer.append(_steps(1)[0])
    with pytest.raises(ValueError, match="every segment needs its episode_data"):
        rb.commit_segment(writer)
    rb.commit_segment(writer, episode_data=_episode(1))
    assert rb.get_episode_segments() == [(0, 1, 0), (1, 2, 1)]


def test_buffers_without_episode_space_reject_episode_data(tmp_path):
    rb = ReplayBuffer.create(HDF5Storage, STEP, cache_path=str(tmp_path / "rb"), maintain_segment_metadata=True)
    assert rb.episode_space is None
    with pytest.raises(ValueError, match="no episode_space"):
        rb.extend(_steps(2), episode_data=_episode(0))
    rb.append(_steps(1)[0])
    with pytest.raises(ValueError, match="no episode_space"):
        rb.mark_segment_end(episode_data=_episode(0))
    with pytest.raises(RuntimeError, match="no episode_space"):
        rb.get_episode_ids(0)


# --------------------------------------------------------------------------- reading

def test_records_by_step_and_by_id(tmp_path):
    rb = _buffer(tmp_path)
    rb.extend(_steps(3), episode_data=_episode(0))
    for i in range(2):
        rb.append(_steps(1, 10 + i)[0])
    assert rb.get_episode_ids(4) == -1  # still open
    rb.mark_segment_end(episode_data=_episode(1))
    rb.extend(_steps(4), episode_data=_episode(2))

    assert rb.get_episode_segments() == [(0, 3, 0), (3, 5, 1), (5, 9, 2)]
    np.testing.assert_array_equal(rb.get_episode_ids(np.arange(9)), [0, 0, 0, 1, 1, 2, 2, 2, 2])
    assert rb.get_episode_ids(-1) == 2
    single = rb.get_episode_data(1)
    assert single["task"] == "task 1" and int(single["success"]) == 1
    batch = rb.get_episode_data(rb.get_episode_ids(np.array([8, 0, 4])))
    assert list(batch["task"]) == ["task 2", "task 0", "task 1"]
    np.testing.assert_array_equal(batch["goal"][:, 0], [2, 0, 1])
    with pytest.raises(IndexError, match="no such episode"):
        rb.get_episode_data(3)


def test_unbounded_buffer_grows_its_episode_ids(tmp_path):
    rb = _buffer(tmp_path)
    for i in range(30):
        rb.extend(_steps(3), episode_data=_episode(i))
    assert len(rb) == 90
    np.testing.assert_array_equal(rb.get_episode_ids(np.arange(0, 90, 3)), np.arange(30))


def test_column_views_read_episodes_of_their_parent(tmp_path):
    space = DictSpace(NB, {"a": BoxSpace(NB, -1e6, 1e6, np.float64, shape=()), "b": BoxSpace(NB, -1e6, 1e6, np.float64, shape=())})
    rb = ReplayBuffer.create(HDF5Storage, space, cache_path=str(tmp_path / "rb"), episode_space=EPISODE, episode_storage_cls=HDF5Storage)
    rb.extend({"a": np.arange(2.0), "b": np.arange(2.0)}, episode_data=_episode(0))
    rb.extend({"a": np.arange(3.0), "b": np.arange(3.0)}, episode_data=_episode(1))
    column = rb.get_column(["a"])
    np.testing.assert_array_equal(column.get_episode_ids(np.arange(5)), [0, 0, 1, 1, 1])
    assert column.episode_space == EPISODE and column.get_episode_data(1)["task"] == "task 1"


# --------------------------------------------------------------------------- eviction

def test_fixed_capacity_keeps_records_while_any_step_remains(tmp_path):
    rb = _buffer(tmp_path, capacity=5)
    rb.extend(_steps(3), episode_data=_episode(0))  # steps 0..2
    rb.extend(_steps(2), episode_data=_episode(1))  # full
    rb.extend(_steps(2), episode_data=_episode(2))  # overwrites 2 steps of episode 0
    assert rb.get_episode_segments() == [(0, 1, 0), (1, 3, 1), (3, 5, 2)]
    assert rb.get_episode_data(0)["task"] == "task 0"  # one of its steps is still there
    rb.extend(_steps(1), episode_data=_episode(3))  # the last step of episode 0 goes
    assert [segment[2] for segment in rb.get_episode_segments()] == [1, 2, 3]
    with pytest.raises(IndexError, match="all their steps were overwritten"):
        rb.get_episode_data(0)  # its record is still stored, but the episode is gone
    rb.extend(_steps(7), episode_data=_episode(4))  # larger than the buffer: one episode is left
    assert rb.get_episode_segments() == [(0, 5, 4)]
    with pytest.raises(IndexError, match="all their steps were overwritten"):
        rb.get_episode_data(3)


def test_a_too_small_episode_capacity_evicts_live_records(tmp_path):
    rb = _buffer(tmp_path, capacity=10, episode_capacity=2)
    for i in range(3):
        rb.extend(_steps(2), episode_data=_episode(i))
    assert rb.get_episode_ids(0) == 0
    with pytest.raises(IndexError, match="episode_capacity is too small"):
        rb.get_episode_data(rb.get_episode_ids(0))
    assert rb.get_episode_data(2)["task"] == "task 2"


def test_clear_keeps_ids_increasing(tmp_path):
    rb = _buffer(tmp_path, capacity=8)
    rb.extend(_steps(2), episode_data=_episode(0))
    rb.clear()
    rb.extend(_steps(2), episode_data=_episode(1))
    assert rb.get_episode_segments() == [(0, 2, 1)]
    assert rb.get_episode_data(1)["task"] == "task 1"
    with pytest.raises(IndexError):
        rb.get_episode_data(0)


@pytest.mark.parametrize("capacity", [1, 2, 3, 5, 8, 13])
def test_randomized_episode_ids_match_a_reference(capacity):
    """Random appends (records at the start or the end) / segment ends / extends / writer commits / clears
    against a per-step reference."""
    step = BoxSpace(TB, -1e9, 1e9, torch.float64, shape=(1,))
    record = BoxSpace(TB, -1e9, 1e9, torch.int64, shape=())
    rng = random.Random(capacity)
    for trial in range(15):
        rb = ReplayBuffer.create(
            PytorchTensorStorage, step, capacity=capacity, is_memmap=False,
            episode_space=record, episode_storage_cls=PytorchTensorStorage,
            episode_storage_kwargs={"is_memmap": False},
        )
        state = {"reference": deque(maxlen=capacity), "next_id": 0, "open_id": None}  # reference: id per step, -1 open

        def close_open():
            if state["open_id"] is not None:  # the record came with the first append
                rb.mark_segment_end()
                state["open_id"] = None
            elif rb._active_segment_length > 0:
                rb.mark_segment_end(episode_data=torch.tensor(state["next_id"]))
                state["reference"] = deque((state["next_id"] if e == -1 else e for e in state["reference"]), maxlen=capacity)
                state["next_id"] += 1
            else:
                rb.mark_segment_end()

        for _ in range(rng.randint(5, 120)):
            op = rng.random()
            reference = state["reference"]
            if op < 0.45:
                opening = rb._active_segment_length == 0 and state["open_id"] is None
                if opening and rng.random() < 0.5:
                    rb.append(torch.zeros(1, dtype=torch.float64), episode_data=torch.tensor(state["next_id"]))
                    state["open_id"] = state["next_id"]
                    state["next_id"] += 1
                else:
                    rb.append(torch.zeros(1, dtype=torch.float64))
                reference.append(-1 if state["open_id"] is None else state["open_id"])
            elif op < 0.65:
                close_open()
            elif op < 0.85:
                close_open()
                size = rng.randint(1, capacity + 3)
                rb.extend(torch.zeros(size, 1, dtype=torch.float64), episode_data=torch.tensor(state["next_id"]))
                state["reference"].extend([state["next_id"]] * size)
                state["next_id"] += 1
            elif op < 0.97:
                close_open()
                size = rng.randint(1, capacity)
                writer = rb.open_segment_writer()
                for _ in range(size):
                    writer.append(torch.zeros(1, dtype=torch.float64))
                rb.commit_segment(writer, episode_data=torch.tensor(state["next_id"]))
                state["reference"].extend([state["next_id"]] * size)
                state["next_id"] += 1
            else:
                close_open()
                rb.clear()
                state["reference"].clear()
            reference = state["reference"]
            assert len(rb) == len(reference)
            if len(rb):
                ids = rb.get_episode_ids(torch.arange(len(rb)))
                np.testing.assert_array_equal(ids, list(reference))
                known = ids[ids >= 0]
                if known.size:
                    np.testing.assert_array_equal(rb.get_episode_data(known).numpy(), known)
                    if known.min() > 0:  # older ids have no step left
                        with pytest.raises(IndexError):
                            rb.get_episode_data(int(known.min()) - 1)
            for start, end, episode_id in rb.get_episode_segments():
                assert set(np.asarray(list(reference)[start:end]).tolist()) == {episode_id}


def test_record_with_the_first_append(tmp_path):
    rb = _buffer(tmp_path)
    rb.append(_steps(1)[0], episode_data=_episode(0))
    rb.append(_steps(1, 1)[0])
    assert rb.get_episode_ids(np.arange(2)).tolist() == [0, 0]  # known while the segment is open
    assert rb.get_episode_data(0)["task"] == "task 0"
    data, metadata = rb.with_episode_data().get_at_with_metadata(np.arange(2))
    assert list(metadata["episode"]["task"]) == ["task 0", "task 0"]
    with pytest.raises(ValueError, match="already open"):
        rb.append(_steps(1)[0], episode_data=_episode(1))
    with pytest.raises(ValueError, match="takes none"):
        rb.mark_segment_end(episode_data=_episode(1))
    rb.mark_segment_end()
    rb.append(_steps(1)[0])  # this one gets its record at the end
    with pytest.raises(ValueError, match="already open"):
        rb.append(_steps(1)[0], episode_data=_episode(1))
    rb.mark_segment_end(episode_data=_episode(1))
    assert rb.get_episode_segments() == [(0, 2, 0), (2, 3, 1)]
    rb.dumps(str(tmp_path / "rb"))
    loaded = ReplayBuffer.load_from(str(tmp_path / "rb"), backend=NB)
    assert loaded.get_episode_segments() == [(0, 2, 0), (2, 3, 1)]


# --------------------------------------------------------------------------- persistence

def test_save_and_load_keep_episodes_and_continue_ids(tmp_path):
    rb = _buffer(tmp_path, capacity=6)
    for i in range(4):
        rb.extend(_steps(2), episode_data=_episode(i))  # episode 0 and half of 1 overwritten
    rb.dumps(str(tmp_path / "rb"))
    segments = rb.get_episode_segments()

    loaded = ReplayBuffer.load_from(str(tmp_path / "rb"), backend=NB)
    assert loaded.episode_space == EPISODE and loaded.get_episode_segments() == segments
    np.testing.assert_array_equal(loaded.get_episode_ids(np.arange(6)), rb.get_episode_ids(np.arange(6)))
    assert loaded.get_episode_data(3)["task"] == "task 3"

    writable = ReplayBuffer.load_from(str(tmp_path / "rb"), backend=NB, read_only=False)
    writable.extend(_steps(2), episode_data=_episode(4))
    assert writable.get_episode_segments()[-1] == (4, 6, 4)
    assert writable.get_episode_data(4)["task"] == "task 4"


def test_buffers_saved_without_episodes_still_load(tmp_path):
    rb = ReplayBuffer.create(HDF5Storage, STEP, cache_path=str(tmp_path / "rb"), maintain_segment_metadata=True)
    rb.extend(_steps(3))
    rb.dumps(str(tmp_path / "rb"))
    loaded = ReplayBuffer.load_from(str(tmp_path / "rb"), backend=NB)
    assert loaded.episode_space is None and loaded.get_segments() == [(0, 3)]


# --------------------------------------------------------------------------- the view

def test_view_joins_steps_with_their_episodes(tmp_path):
    rb = _buffer(tmp_path)
    rb.extend(_steps(3), episode_data=_episode(0))
    rb.extend(_steps(2, 100), episode_data=_episode(1))
    view = rb.with_episode_data()
    assert isinstance(view, ReplayBufferEpisodeView) and len(view) == 5
    data, metadata = view.get_at_with_metadata(np.array([4, 0, 3]))
    np.testing.assert_array_equal(data[:, 0], [101, 0, 100])
    np.testing.assert_array_equal(metadata["episode_id"], [1, 0, 1])
    assert list(metadata["episode"]["task"]) == ["task 1", "task 0", "task 1"]
    assert view.single_metadata_space["episode"] == EPISODE
    single, single_md = view.get_at_with_metadata(1)
    assert single_md["episode"]["task"] == "task 0" and int(single_md["episode_id"]) == 0

    sampler = StepSampler(view, 4, seed=0)
    batch, sampled = sampler.sample_with_metadata()
    for i in range(4):
        assert sampled["episode"]["task"][i] == f"task {int(sampled['episode_id'][i])}"
    assert rb.get_at_with_metadata(np.arange(2))[1] is None  # the buffer itself is unchanged

    rb.append(_steps(1)[0])
    with pytest.raises(ValueError, match="still open"):
        view.get_at_with_metadata(np.array([5]))


def test_view_converts_records_to_the_buffer_backend(tmp_path):
    step = BoxSpace(TB, -1e6, 1e6, torch.float32, shape=(2,))
    record = DictSpace(NB, {"goal": BoxSpace(NB, -10.0, 10.0, np.float32, shape=(3,)), "task": TextSpace(NB, max_length=16)})
    rb = ReplayBuffer.create(
        PytorchTensorStorage, step, capacity=8, is_memmap=False,
        episode_space=record, episode_storage_cls=HDF5Storage, episode_storage_kwargs={},
        cache_path=str(tmp_path / "rb"),
    )
    rb.extend(torch.zeros(3, 2), episode_data={"goal": np.full(3, 7, np.float32), "task": "push"})
    data, metadata = rb.with_episode_data().get_at_with_metadata(torch.arange(3))
    assert isinstance(metadata["episode"]["goal"], torch.Tensor) and metadata["episode"]["goal"].shape == (3, 3)
    assert isinstance(metadata["episode_id"], torch.Tensor)
    assert list(metadata["episode"]["task"]) == ["push"] * 3


def test_view_requires_episode_space(tmp_path):
    rb = ReplayBuffer.create(HDF5Storage, STEP, cache_path=str(tmp_path / "rb"))
    with pytest.raises(RuntimeError, match="no episode_space"):
        rb.with_episode_data()
