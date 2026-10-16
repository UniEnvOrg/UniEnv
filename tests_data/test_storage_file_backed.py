"""File-per-sample (NPZ/Image) and file-per-episode (Video) storage internals."""

import json
import os

import numpy as np
import pytest

from unienv_data.storages.image_storage import ImageStorage
from unienv_data.storages.npz_storage import NPZStorage
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace

B = NumpyComputeBackend

av = pytest.importorskip("av")
from unienv_data.storages.video_storage import VideoStorage  # noqa: E402

from tests_common.storage_video import LOSSLESS_RGB_KWARGS, clean_video_codec_cache  # noqa: E402,F401

pytestmark = pytest.mark.usefixtures("clean_video_codec_cache")


def _space() -> BoxSpace:
    return BoxSpace(B, 0, 255, np.uint8, shape=(4, 4, 3))


def _frames(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, 4, 4, 3), dtype=np.uint8)


FILE_BACKED = {
    "npz": (NPZStorage, {}),
    "image-png": (ImageStorage, {"format": "PNG"}),
    "video": (VideoStorage, LOSSLESS_RGB_KWARGS),
}


@pytest.mark.parametrize("name", list(FILE_BACKED))
@pytest.mark.parametrize("capacity", [None, 4], ids=["unbounded", "fixed"])
@pytest.mark.parametrize("index", [slice(0, 0), slice(3, 1), np.array([], dtype=np.int64)],
                         ids=["empty-slice", "reversed-slice", "empty-array"])
def test_empty_selection_returns_empty_batch(tmp_path, name, capacity, index):
    cls, kwargs = FILE_BACKED[name]
    storage = cls.create(_space(), capacity=capacity, cache_path=str(tmp_path / name), **kwargs)
    for _ in range(2):  # both before and after data exists
        result = storage.get(index)
        assert result.shape == (0, 4, 4, 3)
        assert result.dtype == np.uint8
        storage.set(index, np.zeros((0, 4, 4, 3), np.uint8))
        if capacity is None:
            storage.extend_length(4)
        storage.set(slice(0, 4), _frames(4))
    storage.close()


@pytest.mark.parametrize("name", list(FILE_BACKED))
@pytest.mark.parametrize("target_exists", [False, True], ids=["missing-target", "existing-target"])
def test_dumps_elsewhere_fails_with_clear_assertion(tmp_path, name, target_exists):
    cls, kwargs = FILE_BACKED[name]
    storage = cls.create(_space(), capacity=2, cache_path=str(tmp_path / name), **kwargs)
    target = tmp_path / "elsewhere"
    if target_exists:
        target.mkdir()
    with pytest.raises(AssertionError, match="does not match cache filename"):
        storage.dumps(str(target))
    storage.dumps(str(tmp_path / name))  # dumping in place works
    storage.close()


# ---------------------------------------------------------------------------
# Episode storage file bookkeeping (exercised through VideoStorage)
# ---------------------------------------------------------------------------

def _covered(ranges, capacity):
    covered = []
    for start, end in ranges:
        if start <= end:
            covered.extend(range(start, end + 1))
        else:
            covered.extend(list(range(start, capacity)) + list(range(0, end + 1)))
    return covered


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("capacity", [None, 7], ids=["unbounded", "fixed"])
def test_episode_overwrites_keep_files_disjoint(tmp_path, seed, capacity):
    n = 7
    rng = np.random.default_rng(seed)
    storage = VideoStorage.create(_space(), capacity=capacity, cache_path=str(tmp_path / "v"), **LOSSLESS_RGB_KWARGS)
    if capacity is None:
        storage.extend_length(n)
    reference = _frames(n, seed=100 + seed)
    storage.set(slice(None), reference)
    for step in range(12):
        length = int(rng.integers(1, n + 1))
        start = int(rng.integers(0, n))
        if capacity is None:
            length = min(length, n - start)
        rows = (start + np.arange(length)) % n
        values = _frames(length, seed=1000 * seed + step)
        how = step % 3
        if how == 0:
            storage.set(rows, values)
        elif how == 1 and (capacity is not None or start + length <= n):
            if rows[-1] >= rows[0]:
                storage.set(slice(int(rows[0]), int(rows[-1]) + 1), values)
            else:
                storage.set(rows, values)
        else:
            storage.mark_segment_start(start)
            for frame in values:
                storage.append(frame)
            storage.mark_segment_end()
        reference[rows] = values
        covered = _covered(storage.get_segments(), n)
        assert sorted(covered) == list(range(n)), storage.get_segments()
        np.testing.assert_array_equal(storage.get(slice(None)), reference)
        np.testing.assert_array_equal(storage.get(np.array([n - 1, 0, 3])), reference[[n - 1, 0, 3]])
    storage.dumps(storage.cache_filename)
    storage.close()
    loaded = VideoStorage.load_from(str(tmp_path / "v"), _space(), capacity=capacity,
                                    hardware_acceleration=None, decode_backend="pyav", codec="ffv1")
    np.testing.assert_array_equal(loaded.get(slice(None)), reference)


def _video_with_two_files(tmp_path):
    storage = VideoStorage.create(_space(), capacity=None, cache_path=str(tmp_path / "v"), **LOSSLESS_RGB_KWARGS)
    storage.extend_length(4)
    frames = _frames(4)
    storage.set(slice(0, 2), frames[:2])
    storage.set(slice(2, 4), frames[2:])
    storage.dumps(storage.cache_filename)
    storage.close()
    return str(tmp_path / "v"), frames


def _load_video(path):
    return VideoStorage.load_from(path, _space(), hardware_acceleration=None, decode_backend="pyav", codec="ffv1")


def test_episode_index_metadata_is_used_on_load(tmp_path):
    path, frames = _video_with_two_files(tmp_path)
    with open(os.path.join(path, "episode_index.json")) as f:
        assert json.load(f) == {"file_ranges": [[0, 1], [2, 3]]}
    loaded = _load_video(path)
    assert loaded.get_segments() == [(0, 1), (2, 3)]
    np.testing.assert_array_equal(loaded.get(slice(None)), frames)


@pytest.mark.parametrize("corruption", ["missing-file", "bad-json", "no-index"])
def test_episode_index_falls_back_to_scanning_files(tmp_path, corruption, caplog):
    path, frames = _video_with_two_files(tmp_path)
    index_path = os.path.join(path, "episode_index.json")
    if corruption == "missing-file":
        with open(index_path, "w") as f:
            json.dump({"file_ranges": [[0, 1], [2, 3], [4, 9]]}, f)
    elif corruption == "bad-json":
        with open(index_path, "w") as f:
            f.write("{not json")
    else:
        os.remove(index_path)
    # unrelated files in the directory are ignored by the scan
    open(os.path.join(path, "notes_about.txt"), "w").close()
    open(os.path.join(path, "a_b.mkv"), "w").close()
    loaded = _load_video(path)
    assert loaded.get_segments() == [(0, 1), (2, 3)]
    np.testing.assert_array_equal(loaded.get(slice(None)), frames)


def test_episode_lookup_errors_for_missing_data(tmp_path):
    storage = VideoStorage.create(_space(), capacity=6, cache_path=str(tmp_path / "v"), **LOSSLESS_RGB_KWARGS)
    with pytest.raises(AssertionError, match="No episode files"):
        storage.get(0)
    storage.set(slice(2, 4), _frames(2))
    with pytest.raises(AssertionError, match="not found"):
        storage.get(5)
    with pytest.raises(AssertionError, match="not found"):
        storage.get(0)
    with pytest.raises(AssertionError):
        storage.get(np.array([2, 5]))
    with pytest.raises(AssertionError):
        storage.get(np.array([0, 2]))


def test_episode_set_rejects_non_contiguous_writes(tmp_path):
    storage = VideoStorage.create(_space(), capacity=6, cache_path=str(tmp_path / "v"), **LOSSLESS_RGB_KWARGS)
    with pytest.raises(AssertionError):
        storage.set(np.array([1, 3]), _frames(2))
    with pytest.raises(AssertionError):
        storage.set(np.array([0, 2, 4]), _frames(3))


def test_list_storage_shrink_removes_files_and_clear_resets(tmp_path):
    storage = NPZStorage.create(_space(), capacity=None, cache_path=str(tmp_path / "npz"), compressed=False)
    storage.extend_length(4)
    storage.set(slice(None), _frames(4))
    assert sorted(os.listdir(storage.cache_filename)) == ["0.npz", "1.npz", "2.npz", "3.npz"]
    storage.shrink_length(3)
    assert os.listdir(storage.cache_filename) == ["0.npz"]
    storage.clear()
    assert len(storage) == 0 and os.listdir(storage.cache_filename) == []
    with pytest.raises(AssertionError):
        storage.get(0)
    fixed = NPZStorage.create(_space(), capacity=2, cache_path=str(tmp_path / "fixed"))
    np.testing.assert_array_equal(fixed.get(1), np.zeros((4, 4, 3), np.uint8))  # unwritten slots read as zeros
    with pytest.raises(AssertionError):
        fixed.extend_length(1)
    with pytest.raises(AssertionError):
        fixed.shrink_length(1)


def test_image_storage_jpeg_is_lossy_but_close(tmp_path):
    smooth = np.broadcast_to(np.linspace(0, 255, 4, dtype=np.uint8)[None, :, None], (4, 4, 3)).copy()
    storage = ImageStorage.create(_space(), capacity=1, cache_path=str(tmp_path / "jpg"), format="JPEG", quality=95)
    storage.set(0, smooth)
    np.testing.assert_allclose(storage.get(0).astype(int), smooth.astype(int), atol=12)
    assert os.listdir(storage.cache_filename) == ["0.jpeg"]
    with pytest.raises(ValueError):
        ImageStorage.create(_space(), capacity=1, cache_path=None)


def test_image_storage_rejects_non_rgb_shapes(tmp_path):
    gray_space = BoxSpace(B, 0, 255, np.uint8, shape=(4, 4))
    storage = ImageStorage.create(gray_space, capacity=1, cache_path=str(tmp_path / "gray"), format="PNG")
    with pytest.raises(AssertionError, match="RGB"):
        storage.set(0, np.zeros((4, 4), np.uint8))
