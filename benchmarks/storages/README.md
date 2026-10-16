# Storage benchmarks

Throughput of the `unienv_data` storage backends behind a `ReplayBuffer`. Run from the repository root with the
dependencies of the backends installed (torch, h5py, pyarrow, Pillow, PyAV with libx264 for the video backend).

| script | measures |
|---|---|
| `bench_storages.py` | the backend × operation table: per-step append, extend(64), extend(1024), 256 random reads, sequential reads, dump + close, size on disk, random reads after a read-only reopen |
| `bench_rb_overhead.py` | ReplayBuffer bookkeeping: appends / extends into a wrapping buffer with and without segment metadata, `get_segments()`, lock-free reads of a `multiprocessing=True` buffer with sampler processes |
| `bench_unbounded.py` | growth of unbounded (`capacity=None`) file-backed buffers: per-step append (cumulative mean as the buffer grows) and extend(64) |

```bash
python benchmarks/storages/bench_storages.py --quick                    # smoke run, about a minute
python benchmarks/storages/bench_storages.py --out after.jsonl          # full table (capacity 10k, 128x128 images)
python benchmarks/storages/bench_storages.py --backends hdf5,parquet --capacity 20000 --img 96
python benchmarks/storages/bench_rb_overhead.py [--quick]
python benchmarks/storages/bench_unbounded.py [--quick] [--backends parquet,hdf5,dict-video+hdf5]
```

Common options: `--src DIR` benchmarks the `unienv_data` / `unienv_interface` of another checkout (e.g. an
export of an older commit) instead of this repository, `--out FILE` appends one JSON line per result,
`--data-dir DIR` keeps the storages somewhere specific (default: a temporary directory that is removed), and
`--threads N` sets `torch.set_num_threads` (default 4).

To compare two versions:

```bash
git archive <old-commit> | tar -x -C /tmp/unienv-old
python benchmarks/storages/bench_storages.py --src /tmp/unienv-old --out before.jsonl
python benchmarks/storages/bench_storages.py --out after.jsonl
python benchmarks/storages/bench_storages.py --compare before.jsonl after.jsonl   # markdown before → after table
```

Reads are warm-cache numbers (the page cache is not dropped), and results vary with the machine and its load:
compare runs made back to back on the same machine.
