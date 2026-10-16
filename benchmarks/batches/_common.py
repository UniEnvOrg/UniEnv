"""Shared helpers for the ``benchmarks/batches`` scripts (see README.md).

Every script imports ``unienv_*`` from this checkout (or from ``$UNIENV_BENCH_SRC``), times
named cases and either prints them, prints them as JSON (``--json``), or, with ``--ref REF``,
runs the same cases on a ``git archive`` of ``REF`` and on this checkout, alternating the two
for ``--rounds`` rounds, and prints a before -> after table (best time per case).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.environ.get("UNIENV_BENCH_SRC") or REPO_ROOT)

PACKAGES = ("unienv_interface", "unienv_data", "unienv_utils")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action="store_true", help="small data and few iterations (smoke run)")
    parser.add_argument("--only", default="", help="comma-separated substrings: only run cases whose name contains one")
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ref", default=None, help="git revision to compare against (e.g. main or HEAD~3)")
    parser.add_argument("--new", default=None, help="with --ref: git revision to compare (default: this checkout)")
    parser.add_argument("--rounds", type=int, default=2, help="with --ref: alternating before/after runs")


class Recorder:
    """Collects ``name -> seconds per call`` and prints rows as they are measured."""

    def __init__(self, args):
        self.args = args
        self.results = {}

    def wanted(self, name: str) -> bool:
        only = [o for o in self.args.only.split(",") if o]
        return not only or any(o in name for o in only)

    def time(self, name: str, fn, n_iter: int, warmup: int = 2, repeats: int = 3) -> None:
        """Best (over ``repeats``) mean time of ``n_iter`` calls of ``fn``."""
        if not self.wanted(name):
            return
        for _ in range(warmup):
            fn()
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(n_iter):
                fn()
            best = min(best, (time.perf_counter() - t0) / n_iter)
        self.record(name, best)

    def time_for(self, name: str, fn, seconds: float, warmup: int = 3) -> None:
        """Mean time per call of ``fn`` over ``seconds`` of wall time (for pipelines with queues)."""
        if not self.wanted(name):
            return
        for _ in range(warmup):
            fn()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            fn()
            n += 1
        self.record(name, (time.perf_counter() - t0) / n)

    def record(self, name: str, seconds: float) -> None:
        self.results[name] = seconds
        if not self.args.json:
            print(f"{name:<64s} {seconds * 1e3:10.2f} ms", flush=True)

    def finish(self) -> None:
        if self.args.json:
            print("JSON:" + json.dumps(self.results), flush=True)


def _run_json(script: str, argv, src: str) -> dict:
    """Run ``script --json`` on the sources in ``src``; returns {} if it fails. Its whole process
    group is killed afterwards, so stuck worker processes cannot outlive it."""
    import signal
    env = dict(os.environ, UNIENV_BENCH_SRC=src)
    proc = subprocess.Popen(
        [sys.executable, script, *argv, "--json"], env=env, stdout=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    out = ""
    try:
        for line in proc.stdout:  # stops when the script exits (or its process group is killed)
            out += line
            if line.startswith("JSON:"):
                break
        proc.wait()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.stdout.close()
    lines = [l for l in out.splitlines() if l.startswith("JSON:")]
    if not lines:
        print(f"# run on {src} failed (exit code {proc.returncode})", file=sys.stderr, flush=True)
        return {}
    return json.loads(lines[-1][len("JSON:"):])


def _export(ref: str, dest: str) -> str:
    archive = subprocess.run(
        ["git", "-C", REPO_ROOT, "archive", "--format=tar", ref, *PACKAGES],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    subprocess.run(["tar", "-x", "-C", dest], input=archive, check=True)
    return dest


def compare_with_ref(script: str, args, argv) -> None:
    """Run ``script`` on ``args.ref`` and on this checkout (or ``args.new``), alternating, and
    print a before -> after table."""
    drop = ("--ref", "--new", "--rounds")
    argv = [a for i, a in enumerate(argv)
            if a not in drop and (i == 0 or argv[i - 1] not in drop) and not a.startswith(tuple(d + "=" for d in drop))]
    with tempfile.TemporaryDirectory(prefix="unienv_bench_ref_") as tmp:
        os.makedirs(os.path.join(tmp, "ref"))
        if args.new is not None:
            os.makedirs(os.path.join(tmp, "new"))
        ref_src = _export(args.ref, os.path.join(tmp, "ref"))
        new_src = REPO_ROOT if args.new is None else _export(args.new, os.path.join(tmp, "new"))
        before, after = {}, {}
        for r in range(args.rounds):
            print(f"# round {r + 1}/{args.rounds}", file=sys.stderr, flush=True)
            for src, acc in ((ref_src, before), (new_src, after)):
                for name, t in _run_json(script, argv, src).items():
                    acc[name] = min(t, acc.get(name, float("inf")))
    new_name = "this" if args.new is None else args.new
    print(f"{'case':<64s} {args.ref + ' (ms)':>14s} {new_name + ' (ms)':>12s} {'speedup':>8s}")
    for name in list(dict.fromkeys([*before, *after])):
        b, a = before.get(name), after.get(name)
        fb = f"{b * 1e3:14.2f}" if b is not None else f"{'n/a':>14s}"
        fa = f"{a * 1e3:12.2f}" if a is not None else f"{'n/a':>12s}"
        sp = f"{b / a:7.2f}x" if a and b else f"{'':>8s}"
        print(f"{name:<64s} {fb} {fa} {sp}", flush=True)


def main(script: str, args, argv, run) -> None:
    """Entry point shared by the scripts: ``run(recorder)`` times the cases."""
    if args.ref:
        compare_with_ref(script, args, argv)
        return
    rec = Recorder(args)
    run(rec)
    rec.finish()
