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


def test_feature_spaces_carry_metadata(tmp_path):
    ds = LeRobotAsUniEnvDataset.from_local(str(write_v2(tmp_path)), backend=NB, video_backend="pyav")
    space = ds.single_space
    assert space[IMAGE_KEY].metadata == {
        "kind": "image", "format": "rgb", "layout": "HWC", "fps": FPS, "source": IMAGE_KEY,
    }
    assert space["observation.state"].metadata == {"kind": "proprio"}
    assert space["action"].metadata == {"kind": "action"}
    assert space["next.done"].metadata == {}
    assert space["episode_index"].metadata == {}
    # The described layout is the one actually produced
    assert to_numpy(ds.get_at(0)[IMAGE_KEY]).shape == (H, W, 3)


def test_image_feature_metadata_describes_the_normalized_layout():
    from unienv_data.integrations.lerobot import _feature_to_space
    from unienv_data.third_party.lerobot import LeRobotFeature

    for dtype in ("image", "video"):
        # Stored channel-first, produced channel-last
        feature = LeRobotFeature(name="observation.images.top", dtype=dtype, shape=(3, H, W))
        space = _feature_to_space(feature, NB, fps=30)
        assert space.shape == (H, W, 3)
        assert space.metadata == {"kind": "image", "format": "rgb", "layout": "HWC", "fps": 30, "source": "observation.images.top"}
    assert _feature_to_space(LeRobotFeature(name="action", dtype="float32", shape=(2,)), NB, fps=30).metadata == {"kind": "action"}
    assert _feature_to_space(LeRobotFeature(name="observation.state", dtype="float64", shape=(2,)), NB).metadata == {"kind": "proprio"}
    assert _feature_to_space(LeRobotFeature(name="reward", dtype="float32", shape=(1,)), NB).metadata == {}


# --------------------------------------------------------------------------- episode metadata

TASKS = ["stack the cubes", "open the drawer"]
EPISODE_TASKS = [0, 1, 0]


def _with_text_tasks_v3(ds: Path, index_name="task") -> Path:
    """The metadata LeRobot v3.0 writes: episodes list their tasks by text, ``meta/tasks.parquet`` is a pandas
    frame indexed by the task text (``index_name=None``: an unnamed index, stored as ``__index_level_0__``)."""
    pd = pytest.importorskip("pandas")
    path, = (ds / "meta" / "episodes").rglob("file-000.parquet")
    table = pq.read_table(path).drop(["task_index"])
    table = table.append_column("tasks", pa.array([[TASKS[t]] for t in EPISODE_TASKS], type=pa.list_(pa.string())))
    pq.write_table(table, path)
    frame = pd.DataFrame({"task_index": range(len(TASKS))}, index=pd.Index(TASKS, name=index_name))
    pq.write_table(pa.Table.from_pandas(frame), ds / "meta" / "tasks.parquet")
    return ds


def _with_text_tasks_v2(ds: Path) -> Path:
    with open(ds / "meta" / "episodes.jsonl", "w") as f:
        for e, n in enumerate(EPISODE_LENGTHS):
            f.write(json.dumps({"episode_index": e, "length": n, "tasks": [TASKS[EPISODE_TASKS[e]]]}) + "\n")
    with open(ds / "meta" / "tasks.jsonl", "w") as f:
        for i, task in enumerate(TASKS):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")
    return ds


def _episodes_of(global_indices):
    return (np.searchsorted(STARTS, global_indices, side="right") - 1).tolist()


def _check_episode_metadata(ds):
    data, metadata = ds.get_at_with_metadata(PICKS)
    episodes = _episodes_of(PICKS)
    assert to_numpy(metadata["episode_id"]).tolist() == episodes
    assert list(metadata["episode"]["task"]) == [TASKS[EPISODE_TASKS[e]] for e in episodes]
    assert to_numpy(metadata["episode"]["task_index"]).tolist() == [EPISODE_TASKS[e] for e in episodes]
    assert to_numpy(metadata["episode"]["length"]).tolist() == [EPISODE_LENGTHS[e] for e in episodes]
    assert _state_ids(data) == PICKS.tolist()

    _, single = ds.get_at_with_metadata(4)
    assert int(single["episode_id"]) == 1 and single["episode"]["task"] == TASKS[1]
    assert int(single["episode"]["length"]) == 2


@pytest.mark.parametrize("index_name", ["task", None])
def test_v3_frames_carry_their_episode(tmp_path, index_name):
    path = _with_text_tasks_v3(write_v3(tmp_path, data_files=[[0, 1, 2]], nested_episodes=True), index_name)
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB)
    _check_episode_metadata(ds)
    assert ds.get_episode_ids(PICKS).tolist() == _episodes_of(PICKS)
    assert ds.get_episode_ids(5) == 2
    assert ds.get_episode_data(1)["task"] == TASKS[1]
    records = ds.get_episode_data([2, 0])
    assert list(records["task"]) == [TASKS[0], TASKS[0]] and records["length"].tolist() == [4, 3]
    assert ds.get_episode_segments() == [(0, 3, 0), (3, 5, 1), (5, 9, 2)]
    with pytest.raises(IndexError, match=r"episode_index \[3\]"):
        ds.get_episode_data(3)


def test_v2_text_task_lists_map_to_their_task(tmp_path):
    ds = LeRobotAsUniEnvDataset.from_local(str(_with_text_tasks_v2(write_v2(tmp_path, with_video=False))), backend=NB)
    assert [e.task_index for e in ds.episode_metadata] == EPISODE_TASKS
    _check_episode_metadata(ds)


def test_episode_metadata_through_a_sampler_on_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend
    from unienv_data.samplers import StepSampler

    path = _with_text_tasks_v3(write_v3(tmp_path, data_files=[[0, 1], [2]]))
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=PyTorchComputeBackend)
    sampler = StepSampler(ds, batch_size=16, seed=0)
    data, metadata = sampler.sample_with_metadata()
    states = data["observation.state"].reshape(-1).long().tolist()
    assert isinstance(metadata["episode_id"], torch.Tensor)
    assert metadata["episode_id"].tolist() == _episodes_of(states)
    assert list(metadata["episode"]["task"]) == [TASKS[EPISODE_TASKS[e]] for e in _episodes_of(states)]
    assert sampler.sampled_metadata_space.contains(metadata)


def test_episode_ids_are_lerobot_episode_indices(tmp_path):
    path = write_v3(tmp_path, data_files=[[0, 1, 2]])
    episodes_path = path / "meta" / "episodes" / "file-000.parquet"
    table = pq.read_table(episodes_path)
    table = table.set_column(table.column_names.index("episode_index"), "episode_index", pa.array([0, 5, 7], pa.int64()))
    pq.write_table(table, episodes_path)
    ds = LeRobotAsUniEnvDataset.from_local(str(path), backend=NB)
    assert ds.get_episode_ids(PICKS).tolist() == [[0, 5, 7][e] for e in _episodes_of(PICKS)]
    assert _state_ids(ds.get_episode(5)) == [3, 4]
    assert int(ds.get_episode_data(7)["length"]) == 4
    assert ds.get_episode_segments() == [(0, 3, 0), (3, 5, 5), (5, 9, 7)]
    with pytest.raises(IndexError):
        ds.get_episode_data(1)


# ---------- column encodings LeRobot / HF `datasets` write (regressions) ----------

def _write_v2_single_feature(root: Path, name, feature, column, lengths=(3, 2)) -> Path:
    """A v2.1 dataset with an "index" column and one feature ``name`` stored as the arrow ``column``
    (one value per global frame)."""
    ds = root / f"lerobot_{name}"
    (ds / "meta").mkdir(parents=True)
    info = {"codebase_version": "v2.1", "robot_type": "t", "fps": FPS, "total_episodes": len(lengths),
            "total_frames": sum(lengths), "total_tasks": 1, "chunks_size": 1000,
            "features": {"index": {"dtype": "int64", "shape": [1]}, name: feature}}
    (ds / "meta" / "info.json").write_text(json.dumps(info))
    with open(ds / "meta" / "episodes.jsonl", "w") as f:
        for e, n in enumerate(lengths):
            f.write(json.dumps({"episode_index": e, "length": n, "tasks": ["t"]}) + "\n")
    start = 0
    for e, n in enumerate(lengths):
        path = ds / "data" / "chunk-000" / f"episode_{e:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"index": pa.array(range(start, start + n), pa.int64()),
                                 name: column.slice(start, n)}), path)
        start += n
    return ds


def _backends():
    from unienv_interface.backends.pytorch import PyTorchComputeBackend
    return [NB, PyTorchComputeBackend]


def _png(pixels):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("channels_first", [False, True])
def test_image_features_stored_in_parquet_are_decoded(tmp_path, channels_first):
    # Regression: dtype "image" features (HF Image(): struct<bytes, path> of PNG bytes) were
    # read as numeric columns and failed with "cannot reshape array of size 1 into shape (16,16,3)".
    frames = [np.full((H, W, 3), 20 * k, np.uint8) for k in range(5)]
    frames[3][0, 0] = (1, 2, 3)  # check the channel order too
    column = pa.array([{"bytes": _png(f), "path": None} for f in frames],
                      pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    shape = [3, H, W] if channels_first else [H, W, 3]
    ds_dir = _write_v2_single_feature(tmp_path, "img", {"dtype": "image", "shape": shape}, column)
    picks = np.array([4, 0, 3])
    for backend in _backends():
        ds = LeRobotAsUniEnvDataset.from_local(str(ds_dir), backend=backend)
        assert ds.single_space["img"].shape == (H, W, 3)
        batch = ds.get_at(backend.asarray(picks))
        np.testing.assert_array_equal(to_numpy(batch["img"]), np.stack([frames[k] for k in picks]))
        single = ds.get_at(3)
        np.testing.assert_array_equal(to_numpy(single["img"]), frames[3])
        assert ds.single_space.contains(single)


def test_image_features_given_by_path(tmp_path):
    frames = [np.full((H, W, 3), 30 * k, np.uint8) for k in range(5)]
    (tmp_path / "images").mkdir()
    for k, f in enumerate(frames):
        (tmp_path / "images" / f"{k}.png").write_bytes(_png(f))
    column = pa.array([{"bytes": None, "path": str(tmp_path / "images" / f"{k}.png")} for k in range(5)],
                      pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    ds_dir = _write_v2_single_feature(tmp_path, "img", {"dtype": "image", "shape": [H, W, 3]}, column)
    ds = LeRobotAsUniEnvDataset.from_local(str(ds_dir), backend=NB)
    np.testing.assert_array_equal(to_numpy(ds.get_at(np.array([2, 4]))["img"]), np.stack([frames[2], frames[4]]))


@pytest.mark.parametrize("encoding", ["nested-list", "variable-list", "fixed-size-list"])
def test_multi_dimensional_and_variable_length_list_columns(tmp_path, encoding):
    # Regression: only fixed-size lists were flattened; HF Array2D (list<list<float>> without
    # `datasets` imported) and variable-size lists failed with "cannot reshape array of size 1".
    values = np.arange(5 * 6, dtype=np.float32).reshape(5, 2, 3)
    if encoding == "nested-list":
        column, shape = pa.array(values.tolist(), pa.list_(pa.list_(pa.float32()))), [2, 3]
    elif encoding == "variable-list":
        values = values.reshape(5, 6)
        column, shape = pa.array(values.tolist(), pa.list_(pa.float32())), [6]
    else:
        column, shape = pa.array(values.tolist(), pa.list_(pa.list_(pa.float32(), 3), 2)), [2, 3]
    ds_dir = _write_v2_single_feature(tmp_path, "x", {"dtype": "float32", "shape": shape}, column)
    picks = np.array([4, 1, 3])
    for backend in _backends():
        ds = LeRobotAsUniEnvDataset.from_local(str(ds_dir), backend=backend)
        out = ds.get_at(backend.asarray(picks))["x"]
        np.testing.assert_array_equal(to_numpy(out), values[picks])
        assert ds.single_space.contains(ds.get_at(2))


def test_list_column_with_wrong_length_is_rejected(tmp_path):
    column = pa.array([[1.0, 2.0]] * 4 + [[1.0]], pa.list_(pa.float32()))
    ds_dir = _write_v2_single_feature(tmp_path, "x", {"dtype": "float32", "shape": [2]}, column)
    ds = LeRobotAsUniEnvDataset.from_local(str(ds_dir), backend=NB)
    with pytest.raises(ValueError, match="expected"):
        ds.get_at(4)


def test_column_dtype_differing_from_info_json_is_cast_on_every_backend(tmp_path):
    # Regression: the torch fast path raised "Index put requires the source and destination
    # dtypes match" where the numpy path cast float64 columns to the declared float32.
    column = pa.array([[k, -k] for k in np.arange(5, dtype=np.float64)], pa.list_(pa.float64(), 2))
    ds_dir = _write_v2_single_feature(tmp_path, "x", {"dtype": "float32", "shape": [2]}, column)
    for backend in _backends():
        ds = LeRobotAsUniEnvDataset.from_local(str(ds_dir), backend=backend)
        out = ds.get_at(backend.asarray(np.array([4, 0])))
        np.testing.assert_array_equal(to_numpy(out["x"]), np.array([[4, -4], [0, 0]], np.float32))
        assert ds.single_space.contains(ds.get_at(1))


def test_v3_data_files_without_index_column_use_episode_offsets(tmp_path):
    # Regression: without an "index" column the row offset was the *global* dataset_from_index,
    # so episodes in file-001 read out-of-range (or other episodes') rows.
    ds = LeRobotAsUniEnvDataset.from_local(
        str(write_v3(tmp_path, data_files=[[0], [1, 2]], with_index_column=False)), backend=NB)
    assert _state_ids(ds.get_at(...)) == list(range(TOTAL))
    assert _state_ids(ds.get_episode(2)) == [5, 6, 7, 8]
    assert _state_ids(ds.get_at(PICKS)) == PICKS.tolist()
