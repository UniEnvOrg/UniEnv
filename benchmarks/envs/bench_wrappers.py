"""Per-step overhead of common wrapper stacks over small numpy envs.

Usage (from the repository root)::

    python benchmarks/envs/bench_wrappers.py            # all cases
    python benchmarks/envs/bench_wrappers.py --quick    # short smoke run
    python benchmarks/envs/bench_wrappers.py frame_stack stack_torch   # selected cases
    UNIENV_ROOT=/path/to/main-checkout python benchmarks/envs/bench_wrappers.py  # A/B

Each line reports the best-of-5 wall time of one ``env.step(action)``.
"""
import _common
from _common import bench, report, cli_args, ImageEnv, QUICK, torch

import numpy as np

from tests_common.envs import StubEnv, StubFuncEnv
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.env_base.funcenv import FuncEnvBasedEnv
from unienv_interface.func_wrapper.frame_stack import FuncFrameStackWrapper
from unienv_interface.wrapper.action_rescale import ActionRescaleWrapper
from unienv_interface.wrapper.backend_compat import ToBackendOrDeviceWrapper
from unienv_interface.wrapper.frame_stack import FrameStackWrapper
from unienv_interface.wrapper.time_limit import TimeLimitWrapper


def make(kind, batch):
    env = StubEnv(batch, terminate_above=1e30)
    if kind == "bare":
        return env
    if kind == "time_limit":
        return TimeLimitWrapper(env, 10**9)
    if kind == "rescale":
        return ActionRescaleWrapper(env, -2.0, 2.0)
    if kind == "frame_stack":
        return FrameStackWrapper(env, obs_stack_size=4)
    if kind == "frame_stack_actions":
        action = np.zeros(env.action_space.shape[-1:], np.float32)
        return FrameStackWrapper(env, obs_stack_size=4, action_stack_size=4, action_default_value=action)
    if kind == "frame_stack_image":
        return FrameStackWrapper(ImageEnv(batch, (84, 84, 3)), obs_stack_size=3)
    if kind == "to_torch":
        return ToBackendOrDeviceWrapper(env, backend=PyTorchComputeBackend, device="cpu")
    if kind == "stack_numpy":
        return TimeLimitWrapper(FrameStackWrapper(ActionRescaleWrapper(env, -2.0, 2.0), obs_stack_size=4), 10**9)
    if kind == "stack_torch":
        env = ToBackendOrDeviceWrapper(env, backend=PyTorchComputeBackend, device="cpu")
        return TimeLimitWrapper(FrameStackWrapper(ActionRescaleWrapper(env, -2.0, 2.0), obs_stack_size=4), 10**9)
    if kind == "func_frame_stack":
        return FuncEnvBasedEnv(FuncFrameStackWrapper(StubFuncEnv(batch, terminate_above=1e30), obs_stack_size=4))
    raise KeyError(kind)


KINDS = ["bare", "time_limit", "rescale", "frame_stack", "frame_stack_actions", "frame_stack_image",
         "func_frame_stack", "to_torch", "stack_numpy", "stack_torch"]


def main():
    kinds = cli_args() or KINDS
    batches = (None, 64) if QUICK else (None, 1, 64, 1024)
    for kind in kinds:
        for batch in batches:
            if kind == "frame_stack_image" and batch is not None and batch > 64:
                continue
            env = make(kind, batch)
            env.reset(seed=0)
            action = np.zeros(env.action_space.shape, np.float32)
            if env.backend.simplified_name == "pytorch":
                action = torch.from_numpy(action)
            report(kind, bench(lambda: env.step(action)), batch)


if __name__ == "__main__":
    main()
