"""Per-call latency of space / space_utils / transformation operations.

How to run (from the repository root):

    python benchmarks/spaces/bench_spaces.py numpy            # or torch / jax
    python benchmarks/spaces/bench_spaces.py numpy --quick    # fewer cases, shorter timing
    python benchmarks/spaces/bench_spaces.py numpy --baseline /path/to/other/checkout

``--baseline DIR`` runs the same benchmark against the ``unienv_interface`` found in
``DIR`` (e.g. a ``git worktree`` or ``git archive`` export of ``main``) in a subprocess
and prints both timings side by side with the speedup. ``--src DIR`` selects which
checkout is imported (default: the repository containing this script).

Timings are the minimum over several repeats (robust to interference from other
processes) in microseconds per call; torch/jax run on CPU.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("backend", nargs="?", default="numpy", choices=["numpy", "torch", "jax"])
    p.add_argument("--quick", action="store_true", help="fewer batch sizes and shorter timing")
    p.add_argument("--only", default=None, help="only run ops whose name contains this substring")
    p.add_argument("--batches", default=None, help="comma separated batch sizes, 'none' = unbatched")
    p.add_argument("--src", default=REPO, help="checkout whose unienv_interface is imported")
    p.add_argument("--baseline", default=None, help="checkout to compare against (run in a subprocess)")
    p.add_argument("--json", default=None, help="write the results to this file")
    p.add_argument("--quiet", action="store_true", help="do not print rows while running")
    return p.parse_args(argv)


def get_backend(name):
    if name == "numpy":
        from unienv_interface.backends.numpy import NumpyComputeBackend as B
        return B, None
    if name == "torch":
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))
        from unienv_interface.backends.pytorch import PyTorchComputeBackend as B
        return B, torch.device("cpu")
    import jax
    jax.config.update("jax_platform_name", "cpu")
    from unienv_interface.backends.jax import JaxComputeBackend as B
    return B, jax.devices("cpu")[0]


def _block(x):
    import jax
    jax.block_until_ready(x)
    return x


def bench(fn, min_time, repeat=7, is_jax=False):
    """Minimum microseconds per call over ``repeat`` timing rounds."""
    f = (lambda: _block(fn())) if is_jax else fn
    f()  # warmup (also fills lazily built caches)
    n = 1
    while True:
        t = time.perf_counter()
        for _ in range(n):
            f()
        if time.perf_counter() - t > min_time / repeat or n > 1_000_000:
            break
        n *= 4
    res = []
    for _ in range(repeat):
        t = time.perf_counter()
        for _ in range(n):
            f()
        res.append((time.perf_counter() - t) / n)
    return min(res) * 1e6


def make_obs_space(B, device, batch=None):
    """Typical robot-learning observation: proprio vectors + 2 cameras (HWC uint8)."""
    from unienv_interface.space import BoxSpace, DictSpace
    pre = () if batch is None else (batch,)
    return DictSpace(B, {
        "joint_pos": BoxSpace(B, -3.14, 3.14, B.float32, device=device, shape=pre + (7,)),
        "joint_vel": BoxSpace(B, -10.0, 10.0, B.float32, device=device, shape=pre + (7,)),
        "ee_pos": BoxSpace(B, -1.0, 1.0, B.float32, device=device, shape=pre + (3,)),
        "gripper": BoxSpace(B, 0.0, 0.08, B.float32, device=device, shape=pre + (1,)),
        "cam_front": BoxSpace(B, 0, 255, B.uint8, device=device, shape=pre + (128, 128, 3)),
        "cam_wrist": BoxSpace(B, 0, 255, B.uint8, device=device, shape=pre + (128, 128, 3)),
    }, device=device)


def make_vec_space(B, device, batch=None):
    """Proprio-only dict space (no images): isolates the Python overhead."""
    from unienv_interface.space import BoxSpace, DictSpace
    pre = () if batch is None else (batch,)
    return DictSpace(B, {
        "joint_pos": BoxSpace(B, -3.14, 3.14, B.float32, device=device, shape=pre + (7,)),
        "joint_vel": BoxSpace(B, -10.0, 10.0, B.float32, device=device, shape=pre + (7,)),
        "ee_pos": BoxSpace(B, -1.0, 1.0, B.float32, device=device, shape=pre + (3,)),
        "gripper": BoxSpace(B, 0.0, 0.08, B.float32, device=device, shape=pre + (1,)),
    }, device=device)


def make_action_space(B, device, batch=None):
    from unienv_interface.space import BoxSpace
    pre = () if batch is None else (batch,)
    return BoxSpace(B, -1.0, 1.0, B.float32, device=device, shape=pre + (7,))


def run_benchmarks(args):
    """Run every op and return ``[(op, case, us), ...]``."""
    import numpy as np
    from unienv_interface.space import BoxSpace
    from unienv_interface.space.space_utils import batch_utils as sbu, flatten_utils as sfu
    from unienv_interface import transformations as T

    B, dev = get_backend(args.backend)
    is_jax = args.backend == "jax"
    min_time = 0.08 if args.quick else 0.3
    if args.batches is not None:
        batches = [None if b == "none" else int(b) for b in args.batches.split(",")]
    else:
        batches = [None, 64] if args.quick else [None, 1, 64, 1024]
    rows = []

    def run(op, case, fn):
        if args.only and args.only not in op:
            return
        us = bench(fn, min_time, is_jax=is_jax)
        rows.append((op, case, us))
        if not args.quiet:
            print(f"{args.backend:6s} {op:36s} {case:7s} {us:11.2f} us", flush=True)

    def copy_tree(x):
        if isinstance(x, dict):
            return {k: copy_tree(v) for k, v in x.items()}
        return x.copy() if hasattr(x, "copy") else x.clone()

    rng = B.random.random_number_generator(0, device=dev)
    for b in batches:
        case = "single" if b is None else f"B={b}"
        obs = make_obs_space(B, dev, b)
        vec = make_vec_space(B, dev, b)
        act = make_action_space(B, dev, b)
        img = obs["cam_front"]
        _, act_x = act.sample(rng)
        _, vec_x = vec.sample(rng)
        _, obs_x = obs.sample(rng)

        # --- construction
        shape = act.shape
        run("BoxSpace(scalar bounds)", case, lambda: BoxSpace(B, -1.0, 1.0, B.float32, device=dev, shape=shape))

        # --- core space ops
        run("Box.sample (act, float)", case, lambda: act.sample(rng))
        run("Box.sample (img, uint8)", case, lambda: img.sample(rng))
        run("Dict.sample (vec)", case, lambda: vec.sample(rng))
        run("Dict.sample (obs+img)", case, lambda: obs.sample(rng))
        run("Box.contains (act)", case, lambda: act.contains(act_x))
        run("Box.contains (img, uint8)", case, lambda: img.contains(obs_x["cam_front"]))
        run("Dict.contains (vec)", case, lambda: vec.contains(vec_x))
        run("Dict.contains (obs+img)", case, lambda: obs.contains(obs_x))
        run("Box.low", case, lambda: act.low)

        # --- flatten
        sd = 0 if b is None else 1
        flat_vec = sfu.flatten_data(vec, vec_x, start_dim=sd)
        run("flat_dim (vec)", case, lambda: sfu.flat_dim(vec, sd))
        run("flatten_data (vec)", case, lambda: sfu.flatten_data(vec, vec_x, start_dim=sd))
        run("unflatten_data (vec)", case, lambda: sfu.unflatten_data(vec, flat_vec, start_dim=sd))
        if b is None or b <= 64:
            flat_obs = sfu.flatten_data(obs, obs_x, start_dim=sd)
            run("unflatten_data (obs+img)", case, lambda: sfu.unflatten_data(obs, flat_obs, start_dim=sd))

        # --- batch utils
        if b is None:
            run("batch_space (obs, 64)", case, lambda: sbu.batch_space(obs, 64))
            items_v = [vec.sample(rng)[1] for _ in range(8)]
            run("concatenate x8 (vec)", case, lambda: sbu.concatenate(vec, items_v))
        else:
            if b <= 64:
                run("unbatch_spaces (vec)", case, lambda: list(sbu.unbatch_spaces(vec)))
            _, one = make_obs_space(B, dev, None).sample(rng)
            buf = obs_x if is_jax else copy_tree(obs_x)
            vec_buf = vec_x if is_jax else copy_tree(vec_x)
            one_vec = {k: one[k] for k in vec.keys()}
            run("get_at int (obs)", case, lambda: sbu.get_at(obs, obs_x, 0))
            run("set_at int (obs)", case, lambda: sbu.set_at(obs, buf, 0, one))
            run("set_at int (vec)", case, lambda: sbu.set_at(vec, vec_buf, 0, one_vec))

        # --- transformations
        resc = T.RescaleTransformation(-1.0, 1.0)
        run("Rescale.transform (act)", case, lambda: resc.transform(act, act_x))
        resc_arr = T.RescaleTransformation(B.full((7,), -1.0, dtype=B.float32, device=dev), 1.0)
        run("Rescale.transform (act, array tgt)", case, lambda: resc_arr.transform(act, act_x))
        run("Rescale.get_target_space (act)", case, lambda: resc.get_target_space_from_source(act))
        chain = T.ChainedTransformation([
            T.DictExcludeKeyTransformation(["cam_wrist"]),
            T.DictTransformation({
                "joint_pos": T.RescaleTransformation(-1.0, 1.0),
                "joint_vel": T.RescaleTransformation(-1.0, 1.0),
            }),
            T.FlattenDictTransformation(),
        ])
        run("Chained(excl,dict-rescale,flat)", case, lambda: chain.transform(obs, obs_x))

        # --- one vectorised env step worth of space-layer work
        if b is not None and b > 1:
            act_tf = T.RescaleTransformation(-1.0, 1.0)
            norm_act_space = act_tf.get_target_space_from_source(act)
            act_inv = act_tf.direction_inverse(act)
            _, nact = norm_act_space.sample(rng)
            cap = 4
            buf_space = sbu.batch_space(make_obs_space(B, dev, None), cap * b)
            ring = buf_space.create_empty()
            idx = B.asarray(np.arange(0, cap * b, 3)[:256], device=dev)
            state = {"t": 0}

            def step():
                assert obs.contains(obs_x)
                o = chain.transform(obs, obs_x)
                a = act_inv.transform(norm_act_space, nact)
                assert act.contains(a)
                t = state["t"]
                s = slice((t % cap) * b, (t % cap + 1) * b)
                new_ring = sbu.set_at(buf_space, ring, s, obs_x)
                state["t"] = t + 1
                return o, a, sbu.get_at(buf_space, new_ring, idx)

            run("pipeline step", case, step)
    return rows


def print_table(rows, base_rows=None):
    base = {(op, case): us for op, case, us in (base_rows or [])}
    if base_rows is None:
        print(f"\n{'op':36s} {'case':7s} {'us':>11s}")
        for op, case, us in rows:
            print(f"{op:36s} {case:7s} {us:11.2f}")
        return
    print(f"\n{'op':36s} {'case':7s} {'baseline us':>12s} {'this us':>10s} {'speedup':>8s}")
    speedups = []
    for op, case, us in rows:
        b = base.get((op, case))
        if b is None:
            print(f"{op:36s} {case:7s} {'-':>12s} {us:10.2f}")
            continue
        speedups.append(b / us)
        print(f"{op:36s} {case:7s} {b:12.2f} {us:10.2f} {b / us:7.2f}x")
    if speedups:
        print(f"geometric mean speedup: {statistics.geometric_mean(speedups):.2f}x")


def main():
    args = parse_args()
    src = os.path.abspath(args.src)
    sys.path.insert(0, src)
    import unienv_interface
    assert os.path.abspath(unienv_interface.__file__).startswith(src), unienv_interface.__file__

    base_rows = None
    if args.baseline is not None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "baseline.json")
            cmd = [sys.executable, os.path.abspath(__file__), args.backend, "--src", args.baseline, "--json", out, "--quiet"]
            for flag in ("quick",):
                if getattr(args, flag):
                    cmd.append(f"--{flag}")
            for opt in ("only", "batches"):
                if getattr(args, opt) is not None:
                    cmd += [f"--{opt}", getattr(args, opt)]
            print(f"running baseline ({args.baseline}) ...", flush=True)
            subprocess.run(cmd, check=True)
            with open(out) as f:
                base_rows = [tuple(r) for r in json.load(f)]
    rows = run_benchmarks(args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f)
    if args.baseline is not None or (args.quiet and not args.json):
        print_table(rows, base_rows)


if __name__ == "__main__":
    main()
