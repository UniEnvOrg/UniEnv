"""
ParquetStorage growth: only complete pieces are written while the storage grows (the partial last piece is
written by dumps() / close()), the on-disk row count always describes a loadable prefix, and reads assemble
the same results as before for every index type.
"""

import json
import os

import numpy as np
import pytest

from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.parquet import ParquetStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BinarySpace, BoxSpace, DictSpace, TextSpace

B = NumpyComputeBackend
PIECE = 4


def _space() -> DictSpace:
    return DictSpace(B, {
        "obs": BoxSpace(B, -1000.0, 1000.0, np.float32, shape=(2,)),
        "step": BoxSpace(B, 0, 10_000, np.int64, shape=()),
        "done": BinarySpace(B, shape=(), dtype=np.bool_),
        "note": TextSpace(B, max_length=16),
    })


def _rows(start: int, n: int) -> dict:
    steps = np.arange(start, start + n)
    return {
        "obs": np.stack([steps, -steps], axis=1).astype(np.float32),
        "step": steps.astype(np.int64),
        "done": steps % 3 == 0,
        "note": np.array([f"n{i}" for i in steps], dtype=object),
    }


def _take(batch, index):
    return {key: value[index] for key, value in batch.items()}


def _assert_batch_equal(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        if expected[key].dtype == object:
            assert list(np.atleast_1d(actual[key])) == list(np.atleast_1d(expected[key])), key
        else:
            assert np.asarray(actual[key]).dtype == expected[key].dtype, key
            np.testing.assert_array_equal(actual[key], expected[key])


def _disk_rows(path: str) -> int:
    with open(os.path.join(path, "_metadata.json")) as f:
        return json.load(f)["num_rows"]


def _load_snapshot(path: str):
    reader = ParquetStorage.load_from(path, _space(), read_only=True)
    try:
        return len(reader), (reader.get(slice(None)) if len(reader) > 0 else None)
    finally:
        reader.close()


def test_growth_writes_complete_pieces_and_disk_is_always_a_loadable_prefix(tmp_path):
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_space(), capacity=None, cache_path=path, piece_size=PIECE)
    data = _rows(0, 23)
    for i in range(23):
        storage.extend_length(1)
        # Only complete pieces are on disk while growing, as of the last length change: rows set before it
        # are on disk, the new row (still zero there) is written with its piece by a later length change
        assert _disk_rows(path) == ((i + 1) // PIECE) * PIECE
        assert sorted(f for f in os.listdir(path) if f.endswith(".parquet")) == \
            [f"part-{p:06d}.parquet" for p in range((i + 1) // PIECE)]
        n, snapshot = _load_snapshot(path)
        assert n == _disk_rows(path)
        if n:
            _assert_batch_equal(_take(snapshot, slice(0, min(n, i))), _take(data, slice(0, min(n, i))))
        storage.set(i, _take(data, i))
    _assert_batch_equal(storage.get(slice(None)), data)

    storage.dumps(path)  # writes the partial piece
    assert _disk_rows(path) == 23
    n, snapshot = _load_snapshot(path)
    _assert_batch_equal(snapshot, data)

    # Growing on after dumps() keeps the longer prefix it made durable ...
    storage.extend_length(2)
    assert _disk_rows(path) == 24  # piece 5 completed and written; piece 6 (one row) only in memory
    storage.set(slice(23, 25), _rows(23, 2))
    storage.extend_length(1)
    assert _disk_rows(path) == 24
    storage.shrink_length(3)
    assert _disk_rows(path) == 23  # shrank below the completed piece: back to the prefix dumps() wrote
    storage.extend_length(6)
    storage.set(slice(23, 29), _rows(23, 6))
    assert _disk_rows(path) == 28
    # ... and close() writes everything
    storage.close()
    n, snapshot = _load_snapshot(path)
    assert n == 29
    _assert_batch_equal(snapshot, _rows(0, 29))


def test_shrink_keeps_disk_loadable(tmp_path):
    path = str(tmp_path / "pq")
    storage = ParquetStorage.create(_space(), capacity=None, cache_path=path, piece_size=PIECE)
    storage.extend_length(14)
    storage.set(slice(None), _rows(0, 14))
    storage.dumps(path)
    assert _disk_rows(path) == 14
    storage.shrink_length(5)  # 9 rows: piece 3 is removed, piece 2 keeps one row
    assert _disk_rows(path) == 9
    n, snapshot = _load_snapshot(path)
    _assert_batch_equal(snapshot, _rows(0, 9))
    storage.shrink_length(2)
    assert _disk_rows(path) == 7
    storage.extend_length(3)
    storage.set(slice(7, 10), _rows(100, 3))
    assert _disk_rows(path) == 8  # pieces 0 and 1 complete; the 2 rows of piece 2 wait for close()
    storage.close()
    n, snapshot = _load_snapshot(path)
    expected = _rows(0, 7)
    expected = {key: np.concatenate([expected[key], _rows(100, 3)[key]]) for key in expected}
    _assert_batch_equal(snapshot, expected)


def test_unbounded_replay_buffer_round_trip(tmp_path):
    rb = ReplayBuffer.create(ParquetStorage, _space(), cache_path=str(tmp_path / "rb"), capacity=None, piece_size=PIECE)
    data = _rows(0, 11)
    for i in range(11):
        rb.append(_take(data, i))
        if i % 4 == 3:
            rb.mark_segment_end()
    rb.mark_segment_end()
    rb.dumps(rb.cache_path)
    rb.storage.close()
    loaded = ReplayBuffer.load_from(str(tmp_path / "rb"), backend=B)
    _assert_batch_equal(loaded.get_at(slice(None)), data)
    loaded.storage.close()


@pytest.mark.parametrize("read_only", [False, True])
def test_random_reads_match_numpy(tmp_path, read_only):
    path = str(tmp_path / "pq")
    n = 37
    data = _rows(0, n)
    writer = ParquetStorage.create(_space(), capacity=n, cache_path=path, piece_size=PIECE)
    writer.set(slice(None), data)
    if read_only:
        writer.close()
        storage = ParquetStorage.load_from(path, _space(), capacity=n, read_only=True)
    else:
        storage = writer
    rng = np.random.default_rng(int(read_only))
    indices = [
        5, -1, np.int64(9), slice(None), slice(3, 30, 4), slice(10, 11), Ellipsis,
        np.array([3]), np.array([36, 0, 0, 17, -2]), rng.random(n) < 0.4,
        np.arange(8, 12), rng.integers(-n, n, 50), [1, 2, 30],
    ]
    for index in indices:
        if index is Ellipsis:
            expected = data
        elif isinstance(index, (int, np.integer)):
            expected = {key: value[int(index)] for key, value in data.items()}
        else:
            expected = _take(data, index if not isinstance(index, list) else np.array(index))
        result = storage.get(index)
        if isinstance(index, (int, np.integer)):
            assert result["note"] == expected["note"]
            for key in ("obs", "step", "done"):
                np.testing.assert_array_equal(result[key], expected[key])
                assert np.asarray(result[key]).shape == np.asarray(expected[key]).shape
        else:
            _assert_batch_equal(result, expected)
    empty = storage.get(np.zeros(0, np.int64))
    assert empty["obs"].shape == (0, 2)
    storage.close()
