"""
HDF5Storage dataset layouts and its sparse-read / run-write fast paths: randomized equivalence with a numpy
model for every layout (incl. files in the layout older versions wrote), and on-disk compatibility.
"""

import h5py
import numpy as np
import pytest

from unienv_data.replay_buffer import ReplayBuffer
from unienv_data.storages.hdf5 import HDF5Storage, read_sorted_unique_rows
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BinarySpace, BoxSpace, DictSpace, TextSpace

from tests_common.storage_hdf5_legacy import legacy_layout_is_loadable, write_legacy_layout

B = NumpyComputeBackend
CAPACITY = 40


def _space() -> DictSpace:
    return DictSpace(B, {
        "image": BoxSpace(B, 0, 255, np.uint8, shape=(6, 5, 3)),
        "obs": DictSpace(B, {
            "pos": BoxSpace(B, -100.0, 100.0, np.float32, shape=(3,)),
            "flag": BinarySpace(B, shape=(), dtype=np.bool_),
        }),
        "reward": BoxSpace(B, -100.0, 100.0, np.float64, shape=()),
    })


def _batch(rng, n: int) -> dict:
    return {
        "image": rng.integers(0, 256, (n, 6, 5, 3), dtype=np.uint8),
        "obs": {
            "pos": rng.normal(size=(n, 3)).astype(np.float32),
            "flag": rng.random(n) < 0.5,
        },
        "reward": rng.normal(size=n),
    }


def _take(tree, index):
    if isinstance(tree, dict):
        return {key: _take(value, index) for key, value in tree.items()}
    return tree[index]


def _put(tree, index, value):
    if isinstance(tree, dict):
        for key in tree:
            _put(tree[key], index, value[key])
        return
    tree[index] = value


def _assert_tree_equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
        return
    actual = np.asarray(actual)
    assert actual.shape == np.shape(expected)
    assert actual.dtype == np.asarray(expected).dtype
    np.testing.assert_array_equal(actual, expected)


def _random_read_index(rng, n):
    kind = rng.integers(0, 9)
    if kind == 0:
        return int(rng.integers(-n, n))
    if kind == 1:
        start, stop = sorted(rng.integers(0, n + 1, 2).tolist())
        return slice(start, stop)
    if kind == 2:
        return slice(int(rng.integers(0, n)), None, int(rng.integers(2, 5)))
    if kind == 3:
        return rng.random(n) < 0.3  # boolean mask
    if kind == 4:  # dense run
        start = int(rng.integers(0, n - 8))
        return np.arange(start, start + 8)
    if kind == 5:  # sparse, unsorted, with duplicates and negative entries
        index = rng.integers(-n, n, int(rng.integers(2, 20)))
        return index
    if kind == 6:  # a few runs
        starts = rng.choice(n - 3, 3, replace=False)
        return np.concatenate([np.arange(s, s + 3) for s in starts])
    if kind == 7:
        return Ellipsis
    return np.array([int(rng.integers(0, n))])


def _random_write(rng, n):
    """(index, numpy-model index, batch size or None for a single sample)."""
    kind = rng.integers(0, 6)
    if kind == 0:
        i = int(rng.integers(-n, n))
        return i, i, None
    if kind == 1:
        start = int(rng.integers(0, n - 1))
        stop = int(rng.integers(start + 1, n + 1))
        return slice(start, stop), slice(start, stop), stop - start
    if kind == 2:  # unique, unsorted, possibly negative
        index = rng.choice(np.arange(-n, 0), int(rng.integers(2, 12)), replace=False)
        return index, index, index.shape[0]
    if kind == 3:  # runs
        start = int(rng.integers(0, n - 10))
        index = np.concatenate([np.arange(start, start + 4), np.arange(start + 6, start + 10)])
        return index, index, index.shape[0]
    if kind == 4:  # wrap-around style
        index = (np.arange(7) + n - 3) % n
        return index, index, 7
    mask = rng.random(n) < 0.25
    return mask, mask, int(mask.sum())


LAYOUTS = {
    "contiguous": dict(),
    "gzip": dict(compression="gzip"),
    "chunks-true-lzf": dict(chunks=True, compression="lzf"),
    "chunks-per-step": dict(chunks={"image": (1,), "*": (8,)}),
    "legacy-auto-chunks": dict(legacy=True),
    "legacy-gzip": dict(legacy=True, compression="gzip"),
}


def _create(tmp_path, layout, capacity=CAPACITY, reduce_io=True):
    kwargs = dict(LAYOUTS[layout])
    path = str(tmp_path / f"{layout}.hdf5")
    if kwargs.pop("legacy", False):
        with h5py.File(path, "w") as root:
            write_legacy_layout(root, _space(), capacity, **kwargs)
        return HDF5Storage.load_from(path, _space(), capacity=capacity, read_only=False, reduce_io=reduce_io)
    return HDF5Storage.create(_space(), capacity=capacity, cache_path=path, reduce_io=reduce_io, **kwargs)


@pytest.mark.parametrize("reduce_io", [True, False])
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_random_reads_and_writes_match_numpy_model(tmp_path, layout, reduce_io):
    rng = np.random.default_rng(abs(hash((layout, reduce_io))) % 2**32)
    storage = _create(tmp_path, layout, reduce_io=reduce_io)
    model = _batch(rng, CAPACITY)
    storage.set(slice(None), model)
    for step in range(150):
        if rng.random() < 0.4:
            index, model_index, size = _random_write(rng, CAPACITY)
            value = _batch(rng, 1 if size is None else size)
            if size is None:
                value = _take(value, 0)
            elif rng.random() < 0.3:
                # Writes with other (castable) dtypes go through the same paths
                value["reward"] = value["reward"].astype(np.float32)
                value["image"] = value["image"].astype(np.int64)
            storage.set(index, value)
            if isinstance(model_index, np.ndarray) and model_index.dtype != np.bool_:
                model_index = np.where(model_index < 0, model_index + CAPACITY, model_index)
            _put(model, model_index, value)
        index = _random_read_index(rng, CAPACITY)
        _assert_tree_equal(storage.get(index), _take(model, index))
    storage.close()
    reopened = HDF5Storage.load_from(str(tmp_path / f"{layout}.hdf5"), _space(), capacity=CAPACITY)
    _assert_tree_equal(reopened.get(slice(None)), model)
    reopened.close()


def test_append_writes_single_rows(tmp_path):
    rng = np.random.default_rng(0)
    storage = _create(tmp_path, "contiguous", capacity=5)
    batch = _batch(rng, 7)
    storage.mark_segment_start(3)
    for i in range(7):
        storage.append(_take(batch, i))
    storage.mark_segment_end()
    expected_rows = [2, 3, 4, 5, 6]  # rows 3, 4, 0, 1, 2, 3, 4 got samples 0..6
    _assert_tree_equal(storage.get(slice(None)), _take(batch, np.array(expected_rows)))
    storage.close()


def _leaves(root):
    out = {}
    root.visititems(lambda name, item: out.__setitem__(name, item) if isinstance(item, h5py.Dataset) else None)
    return out


def test_default_layouts(tmp_path):
    fixed = HDF5Storage.create(_space(), capacity=CAPACITY, cache_path=str(tmp_path / "fixed.hdf5"))
    for name, dataset in _leaves(fixed.root).items():
        assert dataset.chunks is None, name  # contiguous
        assert dataset.maxshape == dataset.shape and dataset.maxshape[0] == CAPACITY
    fixed.close()

    images = DictSpace(B, {"image": BoxSpace(B, 0, 255, np.uint8, shape=(128, 128, 3)),
                           "pos": BoxSpace(B, -1.0, 1.0, np.float32, shape=(7,))})
    for kwargs in (dict(compression="gzip"), dict(chunks=True), dict(chunks=None, compression="lzf")):
        storage = HDF5Storage.create(images, capacity=1000, cache_path=str(tmp_path / "c.hdf5"), **kwargs)
        datasets = _leaves(storage.root)
        # whole rows per chunk, about 64 KiB: one 49 KiB image, or 1000 (the capacity) 28-byte rows
        assert datasets["image"].chunks == (1, 128, 128, 3)
        assert datasets["pos"].chunks == (1000, 7)
        assert datasets["image"].maxshape[0] == 1000
        storage.close()

    explicit = HDF5Storage.create(images, capacity=1000, cache_path=str(tmp_path / "e.hdf5"),
                                  chunks={"image": (4, 64, 64, 3), "*": (1,)}, compression="gzip")
    datasets = _leaves(explicit.root)
    assert datasets["image"].chunks == (4, 64, 64, 3)
    assert datasets["pos"].chunks == (1, 7)
    explicit.close()

    unbounded = HDF5Storage.create(images, capacity=None, cache_path=str(tmp_path / "u.hdf5"))
    datasets = _leaves(unbounded.root)
    assert datasets["image"].chunks == (1, 128, 128, 3) and datasets["image"].maxshape[0] is None
    assert datasets["pos"].chunks == (65536 // 28, 7)
    unbounded.extend_length(3)
    assert len(unbounded) == 3
    unbounded.close()


@pytest.mark.parametrize("layout", ["contiguous", "gzip", "chunks-per-step"])
def test_new_files_pass_previous_load_checks(tmp_path, layout):
    """Files written now keep `maxshape[0] == capacity`, so earlier versions of HDF5Storage still load them."""
    storage = _create(tmp_path, layout)
    storage.set(slice(None), _batch(np.random.default_rng(0), CAPACITY))
    storage.close()
    with h5py.File(str(tmp_path / f"{layout}.hdf5"), "r") as root:
        legacy_layout_is_loadable(root, _space(), CAPACITY)
        assert HDF5Storage.build_space_from_hdf5_file(root)[:2] == (CAPACITY, CAPACITY)


def test_legacy_replay_buffer_directory_loads(tmp_path):
    """A replay buffer directory whose storage file has the previous chunked layout loads and extends."""
    rng = np.random.default_rng(3)
    rb = ReplayBuffer.create(HDF5Storage, _space(), cache_path=str(tmp_path / "rb"), capacity=CAPACITY)
    rb.extend(_batch(rng, 10))
    rb.dumps(rb.cache_path)
    storage_path = rb.storage.cache_filename
    rb.storage.close()
    # Rewrite the storage file in the previous layout, with the same contents
    data = HDF5Storage.load_from(storage_path, _space(), capacity=CAPACITY).get(slice(None))
    with h5py.File(storage_path, "w") as root:
        write_legacy_layout(root, _space(), CAPACITY)
    legacy = HDF5Storage.load_from(storage_path, _space(), capacity=CAPACITY, read_only=False)
    legacy.set(slice(None), data)
    legacy.close()

    loaded = ReplayBuffer.load_from(str(tmp_path / "rb"), backend=B, read_only=False)
    assert all(d.chunks is not None for d in _leaves(loaded.storage.root).values())
    more = _batch(rng, 35)
    loaded.extend(more)
    first = _take(data, slice(0, 10))
    full = {
        "image": np.concatenate([first["image"], more["image"]])[-CAPACITY:],
        "obs": {key: np.concatenate([first["obs"][key], more["obs"][key]])[-CAPACITY:] for key in ("pos", "flag")},
        "reward": np.concatenate([first["reward"], more["reward"]])[-CAPACITY:],
    }
    _assert_tree_equal(loaded.get_at(slice(None)), full)
    loaded.storage.close()


def test_text_fields_with_fast_paths(tmp_path):
    space = DictSpace(B, {"text": TextSpace(B, max_length=16), "x": BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,))})
    storage = HDF5Storage.create(space, capacity=6, cache_path=str(tmp_path / "t.hdf5"))
    storage.set(slice(None), {"text": np.array([f"s{i}" for i in range(6)], dtype=object),
                              "x": np.arange(12, dtype=np.float32).reshape(6, 2)})
    storage.set(np.array([4, 1]), {"text": np.array(["four", "one"], dtype=object),
                                   "x": np.full((2, 2), -1, np.float32)})
    storage.set(2, {"text": "two", "x": np.zeros(2, np.float32)})
    result = storage.get(np.array([4, 4, 1, 2, 0]))
    assert list(result["text"]) == ["four", "four", "one", "two", "s0"]
    np.testing.assert_array_equal(result["x"][:3], -np.ones((3, 2), np.float32))
    assert storage.get(2)["text"] == "two"
    storage.close()


@pytest.mark.parametrize("n_rows, row_shape", [(5000, ()), (300, (32, 32, 3))])
def test_read_sorted_unique_rows_matches_fancy_indexing(tmp_path, n_rows, row_shape):
    rng = np.random.default_rng(1)
    with h5py.File(str(tmp_path / "r.hdf5"), "w") as root:
        data = rng.integers(0, 255, (n_rows, *row_shape)).astype(np.uint8)
        dataset = root.create_dataset("d", data=data)
        for size in (1, 2, 5, 50, 200):
            index = np.unique(rng.integers(0, n_rows, size))
            np.testing.assert_array_equal(read_sorted_unique_rows(dataset, index), data[index])
        np.testing.assert_array_equal(read_sorted_unique_rows(dataset, np.arange(10, 30)), data[10:30])
        assert read_sorted_unique_rows(dataset, np.zeros(0, np.int64)).shape == (0, *row_shape)
