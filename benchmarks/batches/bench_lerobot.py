"""Random-access reads from a LeRobot v3.0 dataset (packed parquet files + mp4 videos).

    python benchmarks/batches/bench_lerobot.py [--quick] [--data-dir DIR] [--ref main]

Generates a small synthetic dataset (needs PyAV; kept in --data-dir if given, otherwise in a
temporary directory) and times ``LeRobotAsUniEnvDataset.get_at`` on random indices, tabular
features only and with video decoding (PyAV), with the numpy and torch backends.
"""
import argparse
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import _common

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_common.add_common_args(parser)
parser.add_argument("--data-dir", default=None, help="where to keep the generated dataset")
ARGV = sys.argv[1:]
ARGS = parser.parse_args(ARGV)

FPS = 10
KEY = "observation.image"


def build(ds: Path, episodes: int, ep_len: int, eps_per_file: int, eps_per_video: int, hw: int) -> None:
    import av
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    total = episodes * ep_len
    (ds / "meta" / "episodes").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0", "robot_type": "bench", "fps": FPS, "total_episodes": episodes,
        "total_frames": total, "total_tasks": 1, "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [7]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            KEY: {"dtype": "video", "shape": [hw, hw, 3], "video_info": {"video.fps": FPS, "video.codec": "mpeg4"}},
        },
    }
    (ds / "meta" / "info.json").write_text(json.dumps(info))
    rng = np.random.default_rng(0)
    eps = {k: [] for k in ("episode_index", "length", "task_index", "data/chunk_index", "data/file_index",
                           "dataset_from_index", "dataset_to_index")}
    for f, e0 in enumerate(range(0, episodes, eps_per_file)):
        es = range(e0, min(episodes, e0 + eps_per_file))
        g = np.concatenate([np.arange(e * ep_len, (e + 1) * ep_len) for e in es])
        path = ds / "data" / "chunk-000" / f"file-{f:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(rng.standard_normal(len(g) * 14, dtype=np.float32)), 14),
            "action": pa.FixedSizeListArray.from_arrays(pa.array(rng.standard_normal(len(g) * 7, dtype=np.float32)), 7),
            "next.done": pa.array((g % ep_len) == ep_len - 1),
            "episode_index": pa.array(g // ep_len, type=pa.int64()),
            "index": pa.array(g, type=pa.int64()),
        }), path)
        for e in es:
            eps["episode_index"].append(e)
            eps["length"].append(ep_len)
            eps["task_index"].append(0)
            eps["data/chunk_index"].append(0)
            eps["data/file_index"].append(f)
            eps["dataset_from_index"].append(e * ep_len)
            eps["dataset_to_index"].append((e + 1) * ep_len)
    video_cols = {f"videos/{KEY}/{c}": [] for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")}
    for f, e0 in enumerate(range(0, episodes, eps_per_video)):
        es = list(range(e0, min(episodes, e0 + eps_per_video)))
        path = ds / "videos" / KEY / "chunk-000" / f"file-{f:03d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=FPS)
            stream.width = stream.height = hw
            stream.pix_fmt = "yuv420p"
            for k in range(len(es) * ep_len):
                img = np.full((hw, hw, 3), (k * 7) % 256, np.uint8)
                img[:8, :8] = rng.integers(0, 255, (8, 8, 3))
                for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        for i, e in enumerate(es):
            video_cols[f"videos/{KEY}/chunk_index"].append(0)
            video_cols[f"videos/{KEY}/file_index"].append(f)
            video_cols[f"videos/{KEY}/from_timestamp"].append(i * ep_len / FPS)
            video_cols[f"videos/{KEY}/to_timestamp"].append((i + 1) * ep_len / FPS)
    eps.update(video_cols)
    pq.write_table(pa.table(eps), ds / "meta" / "episodes" / "file-000.parquet")


def run(rec):
    import numpy as np
    from unienv_interface.backends.numpy import NumpyComputeBackend as NB
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    from unienv_data.integrations.lerobot import LeRobotAsUniEnvDataset

    args = ARGS
    episodes, ep_len = (20, 50) if args.quick else (100, 200)
    tmp = None
    root = args.data_dir
    if root is None:
        tmp = tempfile.TemporaryDirectory(prefix="unienv_bench_lerobot_")
        root = tmp.name
    ds_dir = Path(root) / f"v3_{episodes}x{ep_len}"
    if not ds_dir.exists():
        build(ds_dir, episodes, ep_len, eps_per_file=50, eps_per_video=10, hw=96)
    total = episodes * ep_len
    has_hwaccel = "video_hwaccel" in inspect.signature(LeRobotAsUniEnvDataset.from_local).parameters
    rng = np.random.default_rng(1)
    n_iter = 2 if args.quick else 5

    for backend, bname in ((NB, "numpy"), (TB, "torch")):
        for video, bs in ((False, 256), (True, 16)):
            variants = [("", {})]
            if video and has_hwaccel:
                variants.append((" video_hwaccel=None", {"video_hwaccel": None}))
            for suffix, kwargs in variants:
                ds = LeRobotAsUniEnvDataset.from_local(
                    str(ds_dir), backend, decode_video=video, video_backend="pyav", **kwargs)
                picks = [rng.integers(0, total, bs) for _ in range(16)]
                it = iter(range(10 ** 9))
                what = "tabular+video" if video else "tabular"
                rec.time(f"LeRobot {bname} {what} get_at B={bs}{suffix}",
                         lambda: ds.get_at(picks[next(it) % len(picks)]), n_iter, warmup=1)
                ds.close()
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    _common.main(__file__, ARGS, ARGV, run)
