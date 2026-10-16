# Batch view / sampler / integration benchmarks

Timings of typical training-loop data pipelines in `unienv_data`. Run them from the
repository root with the environment that has `unienv` installed:

```bash
python benchmarks/batches/bench_pipeline.py            # in-memory replay buffer pipelines
python benchmarks/batches/bench_lerobot.py             # LeRobot v3.0 parquet + mp4 reads (needs PyAV)
python benchmarks/batches/bench_pipeline.py --quick    # smoke run: small data, few iterations
```

Each script imports `unienv_*` from this checkout (not from wherever the package is
installed) and prints the time per call of each case.

| script | cases |
| --- | --- |
| `bench_pipeline.py` | `StepSampler.sample()` over frame/slice-stacked views, their `get_column` views and `CombinedBatch`es (with and without a thread pool); `MultiprocessingSampler` (fork, with and without `shared_memory`) next to in-process sampling; `ImageDecompressTransformation` on a batch of JPEGs |
| `bench_lerobot.py` | `LeRobotAsUniEnvDataset.get_at` on random indices of a generated dataset (packed parquet files, mp4 videos): tabular only and with video, numpy and torch backends, `video_hwaccel` default and `None` |

## Options

- `--quick`: small data and few iterations, to check that everything runs.
- `--only NAME,...`: only the cases whose name contains one of the substrings.
- `--ref REV`: compare with a git revision. The `unienv_*` packages of `REV` are exported
  with `git archive` to a temporary directory; the script then runs on them and on this
  checkout (or on `--new REV2`), alternating for `--rounds` rounds (default 2), and prints
  the best time per case with the speedup. Cases a revision does not support (e.g. a new
  option) are shown as n/a.
- `bench_pipeline.py`: `--n` transitions, `--batch` and `--mp-batch` batch sizes,
  `--workers`.
- `bench_lerobot.py`: `--data-dir DIR` keeps the generated dataset between runs
  (otherwise it is generated in a temporary directory on every run).

Example: `python benchmarks/batches/bench_pipeline.py --ref main --rounds 3 --only combined`.

## Notes

- Timings on a shared machine are noisy, especially for cases moving hundreds of MB per
  batch; use several `--rounds` and compare speedups rather than absolute numbers.
- The `MultiprocessingSampler` cases run first and the consumer reads one byte per page of
  every received tensor (pages received through shared memory are only mapped in on
  access). Revisions from before the fork deadlock fix get its workaround (single-threaded
  worker intra-op pools) patched in so that they can be measured at all.
