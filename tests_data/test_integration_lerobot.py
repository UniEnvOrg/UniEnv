"""LeRobotAsUniEnvDataset on small synthetic LeRobot v2.1 / v3.0 datasets (incl. video)."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as NB
from unienv_data.integrations.lerobot import LeRobotAsUniEnvDataset

from tests_common.trees import to_numpy

FPS = 10
H = W = 16
EPISODE_LENGTHS = (3, 2, 4)
TOTAL = sum(EPISODE_LENGTHS)
STARTS = np.concatenate([[0], np.cumsum(EPISODE_LENGTHS)[:-1]]).astype(int)
IMAGE_KEY = "observation.image"


def pixel_value(global_index):
    """Frame ``k`` of the dataset is a solid image of this gray value."""
    return 20 + 25 * int(global_index)


def _write_video(path: Path, global_indices):
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        for k in global_indices:
            frame = av.VideoFrame.from_ndarray(np.full((H, W, 3), pixel_value(k), np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _frame_table(global_indices, with_index_column=True):
    g = np.asarray(global_indices)
    columns = {
        "observation.state": pa.array([[float(k)] for k in g], type=pa.list_(pa.float32(), 1)),
        "action": pa.array([[float(k), -float(k)] for k in g], type=pa.list_(pa.float32(), 2)),
        "next.done": pa.array([bool(k + 1 in STARTS[1:] or k == TOTAL - 1) for k in g]),
        "episode_index": pa.array(np.searchsorted(STARTS, g, side="right") - 1, type=pa.int64()),
    }
    if with_index_column:
        columns["index"] = pa.array(g, type=pa.int64())
    return pa.table(columns)


def _features(with_video, with_index_column=True):
    features = {
        "observation.state": {"dtype": "float32", "shape": [1]},
        "action": {"dtype": "float32", "shape": [2]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
    }
    if with_index_column:
        features["index"] = {"dtype": "int64", "shape": [1]}
    if with_video:
        features[IMAGE_KEY] = {"dtype": "video", "shape": [H, W, 3],
                               "video_info": {"video.fps": FPS, "video.codec": "mpeg4"}}
    return features


def _info(version, with_video, **extra):
    return {
        "codebase_version": version, "robot_type": "testbot", "fps": FPS,
        "total_episodes": len(EPISODE_LENGTHS), "total_frames": TOTAL, "total_tasks": 1,
        "chunks_size": 1000, "features": _features(with_video, extra.pop("with_index_column", True)), **extra,
    }


def write_v3(root: Path, data_files, video_files=None, with_index_column=True, nested_episodes=False) -> Path:
    """Write a v3.0 dataset.

    ``data_files`` / ``video_files`` list, per file, the episodes packed into it (in order).
    LeRobot itself writes episode metadata to ``meta/episodes/chunk-XXX/file-XXX.parquet``
    (``nested_episodes=True``); the parser currently only reads ``meta/episodes/*.parquet``.
    """
    ds = root / "lerobot_v3"
    episodes_dir = ds / "meta" / "episodes" / ("chunk-000" if nested_episodes else "")
    episodes_dir.mkdir(parents=True)
    info = _info("v3.0", video_files is not None, with_index_column=with_index_column,
                 data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                 video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    (ds / "meta" / "info.json").write_text(json.dumps(info))

    episodes = {name: [] for name in ("episode_index", "length", "task_index", "data/chunk_index",
                                      "data/file_index", "dataset_from_index", "dataset_to_index")}
    video_cols = {f"videos/{IMAGE_KEY}/{c}": [None] * len(EPISODE_LENGTHS)
                  for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")}
    rows = {}
    for file_index, eps in enumerate(data_files):
        frames = np.concatenate([np.arange(STARTS[e], STARTS[e] + EPISODE_LENGTHS[e]) for e in eps])
        path = ds / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_frame_table(frames, with_index_column), path)
        for e in eps:
            rows[e] = file_index
    for e, n in enumerate(EPISODE_LENGTHS):
        episodes["episode_index"].append(e)
        episodes["length"].append(n)
        episodes["task_index"].append(0)
        episodes["data/chunk_index"].append(0)
        episodes["data/file_index"].append(rows[e])
        episodes["dataset_from_index"].append(int(STARTS[e]))
        episodes["dataset_to_index"].append(int(STARTS[e] + n))
    if video_files is not None:
        for file_index, eps in enumerate(video_files):
            frames = np.concatenate([np.arange(STARTS[e], STARTS[e] + EPISODE_LENGTHS[e]) for e in eps])
            _write_video(ds / "videos" / IMAGE_KEY / "chunk-000" / f"file-{file_index:03d}.mp4", frames)
            offset = 0
            for e in eps:
                video_cols[f"videos/{IMAGE_KEY}/chunk_index"][e] = 0
                video_cols[f"videos/{IMAGE_KEY}/file_index"][e] = file_index
                video_cols[f"videos/{IMAGE_KEY}/from_timestamp"][e] = offset / FPS
                video_cols[f"videos/{IMAGE_KEY}/to_timestamp"][e] = (offset + EPISODE_LENGTHS[e]) / FPS
                offset += EPISODE_LENGTHS[e]
        episodes.update(video_cols)
    pq.write_table(pa.table(episodes), episodes_dir / "file-000.parquet")
    return ds


def write_v2(root: Path, with_video=True) -> Path:
    ds = root / "lerobot_v2"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(json.dumps(_info("v2.1", with_video)))
    with open(ds / "meta" / "episodes.jsonl", "w") as f:
        for e, n in enumerate(EPISODE_LENGTHS):
            f.write(json.dumps({"episode_index": e, "length": n, "tasks": ["stack"]}) + "\n")
    with open(ds / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "stack"}) + "\n")
    for e, n in enumerate(EPISODE_LENGTHS):
        frames = np.arange(STARTS[e], STARTS[e] + n)
        path = ds / "data" / "chunk-000" / f"episode_{e:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_frame_table(frames), path)
        if with_video:
            _write_video(ds / "videos" / "chunk-000" / IMAGE_KEY / f"episode_{e:06d}.mp4", frames)
    return ds


def _state_ids(batch):
    return to_numpy(batch["observation.state"]).reshape(-1).astype(int).tolist()


def _assert_frames(images, global_indices, atol=6):
    images = to_numpy(images)
    assert images.shape == (len(global_indices), H, W, 3)
    means = images.reshape(len(global_indices), -1).mean(axis=1)
    np.testing.assert_allclose(means, [pixel_value(k) for k in global_indices], atol=atol)


PICKS = np.array([0, 2, 3, 4, 5, 8, 6])


def test_v3_multiple_data_files_use_file_local_rows(tmp_path):
    # Regression: dataset_from_index is a *global* frame index; rows in later data
    # files start at 0, so episodes packed into file-001 read out-of-range rows.
    ds = LeRobotAsUniEnvDataset.from_local(str(write_v3(tmp_path, data_files=[[0, 1], [2]])), backend=NB)
    assert len(ds) == TOTAL and ds.num_episodes == 3
    assert _state_ids(ds.get_at(slice(None))) == list(range(TOTAL))
    assert _state_ids(ds.get_episode(2)) == [5, 6, 7, 8]
    assert _state_ids(ds.get_at(PICKS)) == PICKS.tolist()


def test_v3_nested_episode_metadata_layout(tmp_path):
    path = write_v3(tmp_path, data_files=[[0, 1, 2]], nested_episodes=True)
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB)
    assert len(ds) == TOTAL


def test_v3_single_file_without_index_column_keeps_offset(tmp_path):
    ds = LeRobotAsUniEnvDataset.from_local(
        str(write_v3(tmp_path, data_files=[[0, 1, 2]], with_index_column=False)), backend=NB)
    assert _state_ids(ds.get_at(...)) == list(range(TOTAL))


@pytest.mark.slow
def test_v3_video_uses_video_file_and_timestamps(tmp_path):
    # Regression: v3 video frames were looked up in the *data* file index with the
    # data row offset; videos are packed independently (videos/<key>/file_index,
    # from_timestamp), so episodes in other video files decoded the wrong frames.
    path = write_v3(tmp_path, data_files=[[0, 1, 2]], video_files=[[0, 1], [2]])
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB, video_backend="pyav")
    batch = ds.get_at(PICKS)
    assert _state_ids(batch) == PICKS.tolist()
    _assert_frames(batch[IMAGE_KEY], PICKS)
    _assert_frames(ds.get_episode(2)[IMAGE_KEY], [5, 6, 7, 8])
    single = ds.get_at(4)
    _assert_frames(single[IMAGE_KEY][None], [4])


@pytest.mark.slow
def test_v2_video_and_tabular_features(tmp_path):
    ds = LeRobotAsUniEnvDataset.from_local(str(write_v2(tmp_path)), backend=NB, video_backend="pyav")
    assert ds.fps == FPS and ds.schema.codebase_version.value == "v2.1"
    assert [m.length for m in ds.episode_metadata] == list(EPISODE_LENGTHS)
    assert ds.single_space[IMAGE_KEY].shape == (H, W, 3)
    batch = ds.get_at(PICKS)
    assert _state_ids(batch) == PICKS.tolist()
    np.testing.assert_array_equal(to_numpy(batch["action"]), np.stack([PICKS, -PICKS], -1).astype(np.float32))
    assert to_numpy(batch["next.done"]).reshape(-1).tolist() == [False, True, False, True, False, True, False]
    _assert_frames(batch[IMAGE_KEY], PICKS)
    mask = np.zeros(TOTAL, dtype=bool)
    mask[[1, 7]] = True
    assert _state_ids(ds.get_at(mask)) == [1, 7]
    assert _state_ids(ds.get_at(np.array([-1, -TOTAL]))) == [TOTAL - 1, 0]


def test_feature_filters_and_column(tmp_path):
    path = write_v2(tmp_path, with_video=False)
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB, include_features=["action", "observation.state"])
    assert set(ds.single_space.keys()) == {"action", "observation.state"}
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB, exclude_features=["action"])
    assert "action" not in ds.single_space.keys()
    col = ds.get_column(["observation", "state"])
    np.testing.assert_array_equal(to_numpy(col.get_at(np.array([3, 4]))), np.array([[3.0], [4.0]], np.float32))
    assert col.single_space == ds.single_space["observation.state"]


def test_parquet_cache_is_bounded(tmp_path):
    ds = LeRobotAsUniEnvDataset.from_local(str(write_v2(tmp_path, with_video=False)), backend=NB)
    ds._parquet_cache_max = 2
    for e in range(3):
        ds.get_episode(e)
    assert list(ds._parquet_cache.keys()) == [1, 2]
    ds.get_episode(1)
    assert list(ds._parquet_cache.keys()) == [2, 1]
    ds.close()
    assert len(ds._parquet_cache) == 0


@pytest.mark.slow
def test_torch_backend_with_video(tmp_path):
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    path = write_v3(tmp_path, data_files=[[0, 1], [2]], video_files=[[0], [1, 2]])
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=TB, device="cpu", video_backend="pyav")
    batch = ds.get_at(torch.tensor(PICKS))
    assert isinstance(batch[IMAGE_KEY], torch.Tensor) and batch[IMAGE_KEY].dtype == torch.uint8
    assert _state_ids(batch) == PICKS.tolist()
    _assert_frames(batch[IMAGE_KEY], PICKS)
    assert ds.single_space.contains(ds.get_at(3))


# ---------- caching of data files and video readers ----------

def _count_parquet_reads(monkeypatch):
    import unienv_data.integrations.lerobot as lerobot_module
    reads = []
    real = lerobot_module.pq.read_table

    def read_table(path, *args, **kwargs):
        reads.append((str(path), kwargs.get("columns")))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(lerobot_module.pq, "read_table", read_table)
    return reads


def test_v3_episodes_of_one_data_file_share_one_projected_read(tmp_path, monkeypatch):
    # Episodes packed into one v3 data file used to re-read (and re-materialize) the whole
    # file, every column included, once per episode.
    path = write_v3(tmp_path, data_files=[[0, 1], [2]])
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB, include_features=["action"])
    reads = _count_parquet_reads(monkeypatch)
    batch = ds.get_at(PICKS)
    np.testing.assert_array_equal(to_numpy(batch["action"]), np.stack([PICKS, -PICKS], -1).astype(np.float32))
    data_reads = [r for r in reads if "/data/" in r[0]]
    assert len(data_reads) == 2  # one per data file, not one per episode
    assert all(sorted(cols) == ["action", "index"] for _, cols in data_reads)
    # Episodes evicted from the episode LRU are served from the file cache.
    ds._parquet_cache.clear()
    assert to_numpy(ds.get_at(np.array([1, 3]))["action"])[:, 0].tolist() == [1.0, 3.0]
    assert len([r for r in reads if "/data/" in r[0]]) == 2


def test_parquet_file_cache_is_bounded(tmp_path):
    ds = LeRobotAsUniEnvDataset.from_local(str(write_v2(tmp_path, with_video=False)), backend=NB)
    ds._parquet_cache_max = 2
    for e in range(3):
        assert _state_ids(ds.get_episode(e)) == list(range(STARTS[e], STARTS[e] + EPISODE_LENGTHS[e]))
    assert len(ds._parquet_file_cache) == 2
    # The episode entry of the evicted file went with it.
    assert list(ds._parquet_cache.keys()) == [1, 2]
    ds.close()
    assert len(ds._parquet_file_cache) == 0


class _CountingReaders:
    """Patches the PyAV reader class to record the readers it opens and their kwargs."""

    def __init__(self, monkeypatch):
        from unienv_data.storages import video_storage
        self.opened = []
        counter = self

        class Reader(video_storage.PyAvVideoReader):
            def __init__(self, *args, **kwargs):
                counter.opened.append((self, kwargs))
                super().__init__(*args, **kwargs)
        monkeypatch.setattr(video_storage, "PyAvVideoReader", Reader)


@pytest.mark.slow
@pytest.mark.parametrize("hwaccel", ["auto", None])
def test_video_readers_are_reused_and_match_fresh_readers(tmp_path, monkeypatch, hwaccel):
    path = write_v3(tmp_path, data_files=[[0, 1, 2]], video_files=[[0, 1], [2]])
    rng = np.random.default_rng(0)
    picks = [rng.integers(0, TOTAL, size=int(rng.integers(1, 6))) for _ in range(12)]
    fresh = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB, video_backend="pyav")
    fresh._video_reader_cache_max = 0  # opens and closes a reader per call, as before
    expected = [to_numpy(fresh.get_at(idx)[IMAGE_KEY]) for idx in picks]
    readers = _CountingReaders(monkeypatch)
    kwargs = {} if hwaccel == "auto" else {"video_hwaccel": hwaccel}
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB, video_backend="pyav", **kwargs)
    for idx, images in zip(picks, expected):
        np.testing.assert_array_equal(to_numpy(ds.get_at(idx)[IMAGE_KEY]), images)
        _assert_frames(images, idx)
    assert len(readers.opened) == 2  # one per video file
    assert all(kw["hwaccel"] == hwaccel for _, kw in readers.opened)
    ds.close()
    assert all(reader.container is None for reader, _ in readers.opened)
    assert not ds._video_reader_cache


@pytest.mark.slow
def test_video_reader_cache_is_bounded_and_thread_safe(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    path = write_v3(tmp_path, data_files=[[0, 1, 2]], video_files=[[0], [1], [2]])
    readers = _CountingReaders(monkeypatch)
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB, video_backend="pyav")
    ds._video_reader_cache_max = 2
    for e in range(3):
        ds.get_episode(e)
    assert len(ds._video_reader_cache) == 2
    evicted = readers.opened[0][0]
    assert evicted.container is None
    rng = np.random.default_rng(1)
    picks = [rng.integers(0, TOTAL, size=4) for _ in range(32)]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda idx: to_numpy(ds.get_at(idx)[IMAGE_KEY]), picks))
    for idx, images in zip(picks, results):
        _assert_frames(images, idx)
    assert len(ds._video_reader_cache) <= 2
    ds.close()
    assert all(reader.container is None for reader, _ in readers.opened)


@pytest.mark.slow
def test_video_readers_are_not_shared_with_forked_or_unpickled_copies(tmp_path):
    import multiprocessing as mp
    import pickle
    path = write_v3(tmp_path, data_files=[[0, 1, 2]], video_files=[[0, 1, 2]])
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB, video_backend="pyav")
    _assert_frames(ds.get_at(PICKS)[IMAGE_KEY], PICKS)
    (parent_reader,) = ds._video_reader_cache.values()

    clone = pickle.loads(pickle.dumps(ds))
    assert not clone._video_reader_cache
    _assert_frames(clone.get_at(PICKS)[IMAGE_KEY], PICKS)
    assert next(iter(clone._video_reader_cache.values())) is not parent_reader
    clone.close()

    def child(queue):
        try:
            images = to_numpy(ds.get_at(PICKS)[IMAGE_KEY])
            (reader,) = ds._video_reader_cache.values()
            ok = reader is not parent_reader and ds._inherited_video_readers == [parent_reader]
            ds.close()  # must not close the parent's reader in the child
            queue.put((ok and parent_reader.container is not None, images))
        except BaseException as e:  # report instead of hanging the parent
            queue.put((repr(e), None))

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=child, args=(queue,))
    proc.start()
    ok, images = queue.get(timeout=60)
    proc.join(timeout=30)
    assert ok is True, ok
    _assert_frames(images, PICKS)
    # The parent's reader is still open and in use.
    assert ds._video_reader_cache[next(iter(ds._video_reader_cache))] is parent_reader
    _assert_frames(ds.get_at(PICKS)[IMAGE_KEY], PICKS)
    ds.close()
