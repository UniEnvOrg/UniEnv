"""Growth of unbounded (capacity=None) file-backed replay buffers: per-step append and extend(64).

The cost of growing a storage can depend on how much it already holds (e.g. rewriting a growing file piece),
so the cumulative mean append time is also reported after 500 / 1000 / ... steps.

  python benchmarks/storages/bench_unbounded.py [--quick] [--backends parquet,hdf5] [--src CHECKOUT]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common  # noqa: E402

BACKENDS = ["parquet", "hdf5", "dict-video+hdf5"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bench_common.add_common_args(parser)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--steps", type=int, default=None, help="appended steps (default 2000, --quick 300)")
    parser.add_argument("--img", type=int, default=128)
    args = parser.parse_args()
    bench_common.import_src(args.src)

    import numpy as np
    from unienv_interface.backends.numpy import NumpyComputeBackend as NB
    from unienv_interface.space import BinarySpace, BoxSpace, DictSpace
    from unienv_data.replay_buffer import ReplayBuffer
    from unienv_data.storages.dict_storage import DictStorage
    from unienv_data.storages.hdf5 import HDF5Storage
    from unienv_data.storages.parquet import ParquetStorage

    steps = args.steps or (300 if args.quick else 2000)
    space = DictSpace(NB, {
        "joint_pos": BoxSpace(NB, -10.0, 10.0, np.float32, shape=(7,)),
        "image": BoxSpace(NB, 0, 255, np.uint8, shape=(args.img, args.img, 3)),
        "action": BoxSpace(NB, -1.0, 1.0, np.float32, shape=(7,)),
        "reward": BoxSpace(NB, -10.0, 10.0, np.float32, shape=()),
        "terminated": BinarySpace(NB, shape=(), dtype=np.bool_),
    })

    def create(name, path):
        if name == "parquet":
            return ReplayBuffer.create(ParquetStorage, space, cache_path=path, capacity=None)
        if name == "hdf5":
            return ReplayBuffer.create(HDF5Storage, space, cache_path=path, capacity=None)
        from unienv_data.storages.video_storage import VideoStorage
        return ReplayBuffer.create(
            DictStorage, space, storage_cls_map={"image": VideoStorage, "*": HDF5Storage}, cache_path=path,
            capacity=None, key_kwargs={"image": dict(codec="libx264", file_ext="mp4", hardware_acceleration=None,
                                                     decode_backend="pyav", encoder_options={"g": "15"})})

    frames = bench_common.make_frames(256, args.img)
    rng = np.random.default_rng(0)
    data = {
        "joint_pos": rng.normal(size=(steps, 7)).astype(np.float32), "image": frames[np.arange(steps) % 256],
        "action": rng.uniform(-1, 1, (steps, 7)).astype(np.float32), "reward": rng.normal(size=steps).astype(np.float32),
        "terminated": rng.random(steps) < 0.01,
    }
    rows = [{key: value[i] for key, value in data.items()} for i in range(steps)]
    data_dir = bench_common.DataDir(args.data_dir)
    try:
        for name in args.backends.split(","):
            result = {"bench": "unbounded", "backend": name, "steps": steps}
            try:
                rb = create(name, data_dir.fresh(f"{name}-append"))
                marks = []
                start = time.perf_counter()
                for i, row in enumerate(rows):
                    rb.append(row)
                    if (i + 1) % 100 == 0:
                        rb.mark_segment_end()
                    if (i + 1) % 500 == 0:
                        marks.append([i + 1, (time.perf_counter() - start) / (i + 1) * 1e6])
                rb.mark_segment_end()
                result["append_us"] = (time.perf_counter() - start) / steps * 1e6
                result["append_us_cumulative"] = marks
                start = time.perf_counter()
                rb.dumps(rb.cache_path)
                rb.storage.close()
                result["dump_close_s"] = time.perf_counter() - start
                data_dir.remove(f"{name}-append")

                rb = create(name, data_dir.fresh(f"{name}-extend"))
                start = time.perf_counter()
                for s in range(0, steps, 64):
                    rb.extend({key: value[s:s + 64] for key, value in data.items()})
                result["extend64_tps"] = steps / (time.perf_counter() - start)
                rb.storage.close()
                data_dir.remove(f"{name}-extend")
            except Exception as error:
                result["error"] = repr(error)
            bench_common.emit(result, args.out)
    finally:
        data_dir.close()


if __name__ == "__main__":
    main()
