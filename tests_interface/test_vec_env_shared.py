"""Tests for ``AsyncVecEnv(shared_memory=True)`` (shared-memory observation transfer).

CPU backends (numpy / torch CPU / jax CPU) are fully tested. torch CUDA and
jax GPU tests carry the ``gpu`` marker and skip without a usable device (the jax
GPU tests need a CUDA-enabled jaxlib). Real TPUs cannot be tested; the
host-staging path they use is exercised with jax / torch on the CPU.
"""

import functools
import multiprocessing as mp
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from unienv_interface.env_base import _shared_memory as shm_mod
from unienv_interface.env_base import vec_env as vec_env_mod
from unienv_interface.env_base.vec_env import AsyncVecEnv
from tests_common.trees import assert_tree_equal, to_numpy
from tests_common.vecenv_envs import ArrayEnv, EnvVarProbeEnv, UnsupportedSpaceEnv, expected_obs

N = 3
IMAGE = (3, 8, 8)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = [pytest.mark.slow, pytest.mark.timeout(240)]


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _jax_gpu_available() -> bool:
    try:
        import jax
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


CPU_BACKENDS = [
    pytest.param(("numpy", None), id="numpy"),
    pytest.param(("pytorch", "cpu"), id="torch-cpu"),
    pytest.param(("jax", "cpu"), id="jax-cpu"),
]

GPU_BACKENDS = [
    pytest.param(("pytorch", "cuda"), id="torch-cuda", marks=[
        pytest.mark.gpu,
        pytest.mark.skipif(not _torch_cuda_available(), reason="needs a CUDA device and torch"),
    ]),
    pytest.param(("jax", "gpu"), id="jax-gpu", marks=[
        pytest.mark.gpu,
        pytest.mark.skipif(not (_torch_cuda_available() and _jax_gpu_available()),
                           reason="needs a CUDA-enabled jaxlib and torch with CUDA"),
    ]),
]


def _require(backend: str) -> None:
    pytest.importorskip({"numpy": "numpy", "pytorch": "torch", "jax": "jax"}[backend])


def _make(backend_device, n=N, *, shared=True, copy=True, ctx=None, **env_kwargs):
    backend, device = backend_device
    _require(backend)
    env_kwargs.setdefault("image_shape", IMAGE)
    factory = functools.partial(ArrayEnv, backend, device, **env_kwargs)
    return AsyncVecEnv([factory] * n, shared_memory=shared, copy=copy, ctx=ctx)


def _action(env):
    return env.backend.zeros((env.batch_size, 2), dtype=env.action_space.dtype, device=env.device)


def _expected_batch(seeds, t, **kwargs):
    """Batched numpy version of the observations of envs reset with ``seeds`` after ``t`` steps."""
    return _expected_rows([(seed, t) for seed in seeds], **kwargs)


def _expected_rows(seeds_and_steps, **kwargs):
    """Batched numpy observations, row ``i`` from env ``(seed, t)`` = ``seeds_and_steps[i]``."""
    per_env = [expected_obs(seed, t, image_shape=IMAGE, **kwargs) for seed, t in seeds_and_steps]

    def stack(items):
        first = items[0]
        if isinstance(first, dict):
            return {k: stack([item[k] for item in items]) for k in first}
        if isinstance(first, tuple):
            return tuple(stack([item[i] for item in items]) for i in range(len(first)))
        return np.stack([np.asarray(item) for item in items])
    return stack(per_env)


def _assert_backend_arrays(env, tree):
    for leaf in _leaves(tree):
        assert env.backend.is_backendarray(leaf), type(leaf)
        if env.backend.simplified_name == "pytorch":
            import torch
            expected = torch.empty(0, device=env.device).device  # resolves "cuda" to "cuda:<current>"
            assert leaf.device == expected
        elif env.backend.simplified_name == "jax":
            assert leaf.devices() == {env.device}


def _leaves(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _leaves(value)
    elif isinstance(tree, tuple):
        for value in tree:
            yield from _leaves(value)
    else:
        yield tree


def _leaf_index(env, path, group="obs"):
    return [leaf[0] for leaf in env._shared.layout[group]].index(path)


def _segment_path(env):
    name = env._shared.segment_name
    return None if name is None else os.path.join("/dev/shm", name.lstrip("/"))


# ---------------------------------------------------------------------------
# Results match the pipe transport
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_device", CPU_BACKENDS + GPU_BACKENDS)
@pytest.mark.parametrize("copy", [True, False], ids=["copy", "view"])
def test_shared_matches_pipe(backend_device, copy):
    pipe = _make(backend_device, shared=False, ctx=mp.get_context("spawn"))
    shared = _make(backend_device, copy=copy)
    try:
        assert shared.observation_space == pipe.observation_space
        assert shared.context_space == pipe.context_space
        assert shared.backend is pipe.backend and shared.device == pipe.device

        reset = shared.reset(seed=3)
        assert_tree_equal(reset, pipe.reset(seed=3))
        _assert_backend_arrays(shared, reset[0])
        _assert_backend_arrays(shared, reset[1])
        assert shared.observation_space.contains(reset[1])
        for _ in range(2):
            step = shared.step(_action(shared))
            assert_tree_equal(step, pipe.step(_action(pipe)))
            _assert_backend_arrays(shared, step[0])
        mask = np.array([True, False, True])
        masked = shared.reset(mask=mask, seed=9)
        assert_tree_equal(masked, pipe.reset(mask=mask, seed=9))
        assert masked[1]["vec"].shape == (2, 4)
        _assert_backend_arrays(shared, masked[1])
        assert_tree_equal(shared.step(_action(shared)), pipe.step(_action(pipe)))
    finally:
        shared.close()
        pipe.close()


@pytest.mark.parametrize("backend_device", CPU_BACKENDS + GPU_BACKENDS)
def test_box_observation_without_dict(backend_device):
    env = _make(backend_device, obs_kind="box")
    try:
        _, obs, _ = env.reset(seed=0)
        np.testing.assert_array_equal(to_numpy(obs), _expected_batch([0, 1, 2], 0, obs_kind="box"))
        obs, *_ = env.step(_action(env))
        np.testing.assert_array_equal(to_numpy(obs), _expected_batch([0, 1, 2], 1, obs_kind="box"))
    finally:
        env.close()


# ---------------------------------------------------------------------------
# copy semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_device", CPU_BACKENDS + GPU_BACKENDS)
def test_copy_true_results_never_change(backend_device):
    env = _make(backend_device, copy=True)
    try:
        context, obs0, _ = env.reset(seed=0)
        obs1, *_ = env.step(_action(env))
        _, masked, _ = env.reset(mask=np.array([False, True, False]), seed=5)
        obs2, *_ = env.step(_action(env))
        env.step(_action(env))
        assert_tree_equal(obs0, _expected_batch([0, 1, 2], 0))
        assert_tree_equal(obs1, _expected_batch([0, 1, 2], 1))
        assert_tree_equal(masked, _expected_batch([6], 0))
        assert_tree_equal(obs2, _expected_rows([(0, 2), (6, 1), (2, 2)]))
        np.testing.assert_array_equal(to_numpy(context), [[0, 0], [1, 1], [2, 2]])
    finally:
        env.close()


def _same_memory(a, b) -> bool:
    if isinstance(a, np.ndarray):
        return np.shares_memory(a, b)
    if hasattr(a, "data_ptr"):
        return a.data_ptr() == b.data_ptr()
    return a.unsafe_buffer_pointer() == b.unsafe_buffer_pointer()


@pytest.mark.parametrize("backend_device", CPU_BACKENDS + GPU_BACKENDS)
def test_copy_false_returns_views_valid_until_next_step(backend_device):
    env = _make(backend_device, copy=False)
    try:
        _, obs0, _ = env.reset(seed=0)
        obs1, *_ = env.step(_action(env))
        # views are zero-copy and overwritten in place by the next step
        assert _same_memory(obs0["image"], obs1["image"])
        assert_tree_equal(obs0, _expected_batch([0, 1, 2], 1))
        # masked resets gather fresh rows
        _, masked, _ = env.reset(mask=np.array([True, False, False]), seed=4)
        env.step(_action(env))
        assert_tree_equal(masked, _expected_batch([4], 0))
    finally:
        env.close()


@pytest.mark.parametrize("backend_device", [CPU_BACKENDS[0], CPU_BACKENDS[1]])
def test_copy_false_views_survive_close(backend_device):
    env = _make(backend_device, copy=False)
    path = _segment_path(env)
    _, obs, _ = env.reset(seed=1)
    env.close()
    assert path is None or not os.path.exists(path)
    assert_tree_equal(obs, _expected_batch([1, 2, 3], 0))
    del obs


# ---------------------------------------------------------------------------
# Strict space checking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_device", CPU_BACKENDS + GPU_BACKENDS)
@pytest.mark.parametrize("misbehave", ["shape", "dtype"])
def test_mismatching_observation_raises_from_worker(backend_device, misbehave):
    env = _make(backend_device, misbehave=misbehave)
    try:
        env.reset(seed=0)
        with pytest.raises(RuntimeError, match="worker") as excinfo:
            env.step(_action(env))
        message = repr(excinfo.value.__cause__) + str(excinfo.value)
        assert "does not match the declared space" in message
        assert "['vec']" in message
    finally:
        env.close()


def test_mismatching_observation_on_reset_raises():
    env = _make(("numpy", None), misbehave="dtype", misbehave_after=-1)
    try:
        with pytest.raises(RuntimeError, match="does not match the declared space"):
            env.reset(seed=0)
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["dynamic_box", "text", "batched", "union", "graph"])
@pytest.mark.parametrize("where", ["observation", "context"])
def test_unsupported_spaces_raise_at_construction(kind, where):
    before = set(p.pid for p in mp.active_children())
    with pytest.raises(ValueError, match=f"only supports fixed-size {where} spaces"):
        AsyncVecEnv([functools.partial(UnsupportedSpaceEnv, kind, where)] * 2, shared_memory=True)
    assert set(p.pid for p in mp.active_children()) <= before
    # the pipe transport still supports them
    AsyncVecEnv([functools.partial(UnsupportedSpaceEnv, kind, where)] * 2,
                ctx=mp.get_context("spawn")).close()


# ---------------------------------------------------------------------------
# Lifetime / cleanup
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="needs /dev/shm")
@pytest.mark.parametrize("backend_device", CPU_BACKENDS)
def test_close_unlinks_host_segment(backend_device):
    env = _make(backend_device)
    path = _segment_path(env)
    assert os.path.exists(path)
    env.reset(seed=0)
    env.close()
    assert not os.path.exists(path)
    env.close()  # idempotent


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="needs /dev/shm")
@pytest.mark.parametrize("misbehave", ["raise", "exit"])
def test_close_after_worker_crash(misbehave):
    env = _make(("numpy", None), misbehave=misbehave)
    path = _segment_path(env)
    env.reset(seed=0)
    with pytest.raises(RuntimeError, match="worker"):
        env.step(_action(env))
    env.close()
    assert env.processes == []
    assert not os.path.exists(path)


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="needs /dev/shm")
def test_unclosed_env_is_cleaned_up_at_exit():
    script = textwrap.dedent("""
        import functools
        from unienv_interface.env_base.vec_env import AsyncVecEnv
        from tests_common.vecenv_envs import ArrayEnv
        if __name__ == "__main__":
            env = AsyncVecEnv([functools.partial(ArrayEnv, "numpy")] * 2, shared_memory=True)
            env.reset(seed=0)
            print(env._shared.segment_name)
    """)
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    name = out.stdout.strip().splitlines()[-1]
    assert not os.path.exists(os.path.join("/dev/shm", name.lstrip("/")))
    assert "leaked" not in out.stderr, out.stderr


def test_constructor_error_after_workers_started_closes_them():
    with pytest.raises(ValueError):
        AsyncVecEnv([functools.partial(UnsupportedSpaceEnv, "text")] * 2, shared_memory=True)
    assert not vec_env_mod._LIVE_SHARED_ENVS


# ---------------------------------------------------------------------------
# Start methods, worker environment, mode selection
# ---------------------------------------------------------------------------

@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="needs fork")
def test_host_sharing_works_with_an_explicit_fork_context():
    env = _make(("numpy", None), ctx=mp.get_context("fork"))
    try:
        assert_tree_equal(env.reset(seed=2)[1], _expected_batch([2, 3, 4], 0))
    finally:
        env.close()


def test_shared_mode_defaults_to_spawn_and_sets_xla_preallocate(monkeypatch):
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    shared = AsyncVecEnv([EnvVarProbeEnv] * 2, shared_memory=True)
    pipe = AsyncVecEnv([EnvVarProbeEnv] * 2, ctx=mp.get_context("spawn"))
    try:
        assert all("spawn" in type(p._popen).__module__ for p in shared.processes)
        assert shared.reset()[2]["xla_preallocate"] == ["false", "false"]
        assert pipe.reset()[2]["xla_preallocate"] == ["None", "None"]
    finally:
        shared.close()
        pipe.close()
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    shared = AsyncVecEnv([EnvVarProbeEnv] * 2, shared_memory=True)
    try:
        assert shared.reset()[2]["xla_preallocate"] == ["true", "true"]
    finally:
        shared.close()


def test_select_mode():
    from unienv_interface.backends.numpy import NumpyComputeBackend
    assert shm_mod.select_mode(NumpyComputeBackend, None, "fork") == shm_mod.HOST


def test_select_mode_torch(monkeypatch):
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as TB
    obj = lambda d: ("object", torch.device(d))  # noqa: E731
    assert shm_mod.select_mode(TB, None, "spawn") == shm_mod.HOST
    assert shm_mod.select_mode(TB, obj("cpu"), "fork") == shm_mod.HOST
    assert shm_mod.select_mode(TB, obj("mps"), "spawn") == shm_mod.HOST_STAGED
    with pytest.raises(ValueError, match="'spawn' or 'forkserver'"):
        shm_mod.select_mode(TB, obj("cuda:0"), "fork")
    monkeypatch.setattr(sys, "platform", "linux")
    assert shm_mod.select_mode(TB, obj("cuda:1"), "forkserver") == shm_mod.CUDA_IPC
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(shm_mod, "_warned_no_cuda_ipc", False)
    with pytest.warns(RuntimeWarning, match="only supported on Linux"):
        assert shm_mod.select_mode(TB, obj("cuda:0"), "spawn") == shm_mod.HOST_STAGED
    assert shm_mod.select_mode(TB, obj("cuda:0"), "spawn") == shm_mod.HOST_STAGED  # warns once


def test_select_mode_jax(monkeypatch):
    pytest.importorskip("jax")
    from unienv_interface.backends.jax import JaxComputeBackend as JB
    assert shm_mod.select_mode(JB, ("str", "cpu:0"), "fork") == shm_mod.HOST
    assert shm_mod.select_mode(JB, ("str", "tpu:0"), "spawn") == shm_mod.HOST_STAGED
    assert shm_mod.select_mode(JB, None, "spawn") == shm_mod.HOST_STAGED
    monkeypatch.setattr(sys, "platform", "linux")
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        assert shm_mod.select_mode(JB, ("str", "gpu:0"), "spawn") == shm_mod.CUDA_IPC
        with pytest.raises(ValueError, match="'spawn' or 'forkserver'"):
            shm_mod.select_mode(JB, ("str", "gpu:0"), "fork")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="needs PyTorch"):
        shm_mod.select_mode(JB, ("str", "gpu:0"), "spawn")


@pytest.mark.parametrize("backend_device", [CPU_BACKENDS[1], CPU_BACKENDS[2]])
@pytest.mark.parametrize("copy", [True, False], ids=["copy", "view"])
def test_host_staged_path(backend_device, copy, monkeypatch):
    """The path used for TPUs / CUDA without IPC: workers copy to the host and the
    parent transfers every read to the env device (here the CPU)."""
    monkeypatch.setattr(vec_env_mod, "select_mode", lambda *args: shm_mod.HOST_STAGED)
    env = _make(backend_device, copy=copy)
    try:
        assert env._shared.mode == shm_mod.HOST_STAGED
        context, obs0, _ = env.reset(seed=0)
        obs1, *_ = env.step(_action(env))
        _, masked, _ = env.reset(mask=np.array([False, False, True]), seed=7)
        env.step(_action(env))
        _assert_backend_arrays(env, obs1)
        # staged reads are always fresh device arrays
        assert_tree_equal(obs0, _expected_batch([0, 1, 2], 0))
        assert_tree_equal(obs1, _expected_batch([0, 1, 2], 1))
        assert_tree_equal(masked, _expected_batch([9], 0))
        np.testing.assert_array_equal(to_numpy(context), [[0, 0], [1, 1], [2, 2]])
    finally:
        env.close()


def test_jax_cpu_view_is_zero_copy():
    pytest.importorskip("jax")
    env = _make(("jax", "cpu"), copy=False)
    try:
        _, obs, _ = env.reset(seed=0)
        obs1, *_ = env.step(_action(env))
        buffer = env._shared.buffers["obs"][_leaf_index(env, ("image",))]
        assert obs["image"].unsafe_buffer_pointer() == buffer.ctypes.data
        assert obs1["image"].unsafe_buffer_pointer() == buffer.ctypes.data
    finally:
        env.close()


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda_available(), reason="needs a CUDA device and torch")
@pytest.mark.parametrize("scenario", ["clean", "views_held", "not_closed", "worker_raises"])
def test_cuda_ipc_lifetimes_do_not_warn(scenario):
    """torch warns at exit ("Producer process has been terminated before all
    shared CUDA tensors released") when CUDA IPC lifetimes are mismanaged."""
    script = textwrap.dedent(f"""
        import functools
        import numpy as np
        import torch
        from unienv_interface.env_base.vec_env import AsyncVecEnv
        from tests_common.vecenv_envs import ArrayEnv
        if __name__ == "__main__":
            scenario = {scenario!r}
            misbehave = "raise" if scenario == "worker_raises" else None
            env = AsyncVecEnv([functools.partial(ArrayEnv, "pytorch", "cuda", misbehave=misbehave)] * 2,
                              shared_memory=True, copy=scenario != "views_held")
            assert env._shared.mode == "cuda_ipc"
            held = env.reset(seed=0)
            try:
                held = env.step(torch.zeros((2, 2), device="cuda"))
            except RuntimeError:
                assert scenario == "worker_raises"
            if scenario != "not_closed":
                env.close()
            print("done", flush=True)
    """)
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert "done" in out.stdout
    assert "Producer process has been terminated" not in out.stderr, out.stderr


@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda_available(), reason="needs a CUDA device and torch")
def test_torch_cuda_views_live_on_the_env_device():
    import torch
    env = _make(("pytorch", "cuda"), copy=False)
    try:
        assert env._shared.mode == shm_mod.CUDA_IPC
        _, obs, _ = env.reset(seed=0)
        assert obs["image"].device == torch.empty(0, device=env.device).device and obs["image"].is_cuda
        assert obs["image"].data_ptr() == env._shared.buffers["obs"][_leaf_index(env, ("image",))].data_ptr()
        # parent-side asynchronous consumers are synchronised before workers overwrite
        total = obs["image"].float().sum()
        obs1, *_ = env.step(_action(env))
        assert torch.equal(obs["image"], obs1["image"])
        assert total.item() == float(_expected_batch([0, 1, 2], 0)["image"].astype(np.float64).sum())
    finally:
        env.close()


@pytest.mark.gpu
@pytest.mark.skipif(not (_torch_cuda_available() and _jax_gpu_available()),
                    reason="needs a CUDA-enabled jaxlib and torch with CUDA")
def test_jax_gpu_uses_cuda_ipc_zero_copy():
    env = _make(("jax", "gpu"), copy=False)
    try:
        assert env._shared.mode == shm_mod.CUDA_IPC
        _, obs, _ = env.reset(seed=0)
        obs1, *_ = env.step(_action(env))
        assert obs["image"].unsafe_buffer_pointer() == obs1["image"].unsafe_buffer_pointer()
        assert list(obs1["image"].devices())[0].platform == "gpu"
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Buffers exercised in-process (no worker processes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_name", ["pytorch", "jax"])
@pytest.mark.parametrize("copy", [True, False], ids=["copy", "view"])
def test_bfloat16_leaves_in_process(backend_name, copy):
    """bfloat16 has no numpy dtype in torch; host copies fall back to torch ops."""
    _require(backend_name)
    from unienv_interface.space import BoxSpace, DictSpace
    if backend_name == "pytorch":
        import torch
        from unienv_interface.backends.pytorch import PyTorchComputeBackend as B
        bf16, device = torch.bfloat16, torch.device("cpu")
        make = lambda v: torch.full((2, 3), v, dtype=bf16)  # noqa: E731
    else:
        import jax
        import jax.numpy as jnp
        from unienv_interface.backends.jax import JaxComputeBackend as B
        bf16, device = jnp.bfloat16, jax.devices("cpu")[0]
        make = lambda v: jnp.full((2, 3), v, dtype=bf16)  # noqa: E731
    space = DictSpace(B, {"x": BoxSpace(B, -10.0, 10.0, bf16, shape=(2, 3), device=device),
                          "y": BoxSpace(B, -10.0, 10.0, B.default_floating_dtype, shape=(1,), device=device)})
    parent = shm_mod.ParentSharedBuffers(B, device, 2, {"obs": space}, shm_mod.HOST, copy)
    workers = [shm_mod.WorkerSharedBuffers(parent.worker_spec(), i, B) for i in range(2)]
    try:
        for i, worker in enumerate(workers):
            worker.write("obs", {"x": make(1.5 + i), "y": B.full((1,), float(i), dtype=B.default_floating_dtype)})
        out = parent.read("obs")
        assert out["x"].dtype == bf16 and tuple(out["x"].shape) == (2, 2, 3)
        np.testing.assert_array_equal(_as_f32(out["x"])[:, 0, 0], [1.5, 2.5])
        workers[0].write("obs", {"x": make(-3.0), "y": B.full((1,), 9.0, dtype=B.default_floating_dtype)})
        first = float(_as_f32(out["x"])[0, 0, 0])
        if copy:
            assert first == 1.5
        elif backend_name == "pytorch":
            assert first == -3.0
        # (jax may not import bfloat16 through DLPack and then returns a copy)
        masked = parent.read("obs", [1])
        assert tuple(masked["x"].shape) == (1, 2, 3)
        np.testing.assert_array_equal(_as_f32(masked["x"])[:, 0, 0], [2.5])
        del out, masked
    finally:
        for worker in workers:
            worker.close()
        parent.close()


def _as_f32(x):
    if hasattr(x, "float") and hasattr(x, "data_ptr"):  # torch
        return x.float().numpy()
    return np.asarray(x, dtype=np.float32)
