"""Run the shared env contract (``tests_common.env_contract``) over every env / wrapper."""

from __future__ import annotations

import multiprocessing as mp
import warnings

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.env_base.funcenv import FuncEnvBasedEnv
from unienv_interface.env_base.funcenv_wrapper import FuncEnvWrapper
from unienv_interface.env_base.vec_env import AsyncVecEnv, SyncVecEnv
from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.func_wrapper import FuncFrameStackWrapper, FuncTransformWrapper
from unienv_interface.space import BoxSpace
from unienv_interface.transformations.batch_and_unbatch import BatchifyTransformation
from unienv_interface.transformations.filter_dict import DictIncludeKeyTransformation
from unienv_interface.transformations.rescale import RescaleTransformation
from unienv_interface.world import FuncWorldEnv, WorldEnv
from unienv_interface.wrapper import (
    ActionRescaleWrapper,
    ActionTransformWrapper,
    BatchifyWrapper,
    ContextObservationTransformWrapper,
    ControlFrequencyLimitWrapper,
    EpisodeRenderStackWrapper,
    EpisodeVideoWrapper,
    FlattenActionWrapper,
    FlattenContextObservationWrapper,
    FrameStackWrapper,
    TimeLimitWrapper,
    ToBackendOrDeviceWrapper,
    UnBatchifyWrapper,
)
from unienv_interface.wrapper.gym_compat import FromGymnasiumEnv, ToGymnasiumEnv
from unienv_interface.wrapper.transformation import TransformWrapper
from tests_common import env_contract
from tests_common.backends import cpu_backend_params
from tests_common.env_gym import TinyGymEnv
from tests_common.env_stubs import ArrayStepsStubEnv as StubEnv
from tests_common.env_stubs import ArrayStepsStubFuncEnv as StubFuncEnv
from tests_common.env_worlds import CounterFuncNode, CounterNode, SimFuncWorld, SimWorld

ZERO = np.zeros(2, np.float32)
BATCH_SIZES = [None, 3]


class _Plain(Wrapper):
    pass


class _PlainFunc(FuncEnvWrapper):
    pass


def _action_shape(env):
    return env.action_space.shape


# Each factory takes the batch size (None or int) and returns a fresh env.
STATEFUL_FACTORIES = {
    "StubEnv": lambda bs: StubEnv(bs),
    "Wrapper": lambda bs: _Plain(_Plain(StubEnv(bs))),
    "TimeLimit": lambda bs: TimeLimitWrapper(StubEnv(bs), time_limit=2),
    "ControlFrequencyLimit": lambda bs: ControlFrequencyLimitWrapper(StubEnv(bs), control_dt=1e-4),
    "ActionRescale": lambda bs: ActionRescaleWrapper(StubEnv(bs, action_low=0.0, action_high=10.0)),
    "ActionTransform": lambda bs: ActionTransformWrapper(
        StubEnv(bs), RescaleTransformation(-1.0, 1.0),
        BoxSpace(NumpyComputeBackend, 0.0, 2.0, np.float32, shape=StubEnv(bs).action_space.shape)),
    "FlattenAction": lambda bs: FlattenActionWrapper(StubEnv(bs)),
    "FlattenContextObservation": lambda bs: FlattenContextObservationWrapper(StubEnv(bs), flatten_context=True),
    "FrameStackObs": lambda bs: FrameStackWrapper(StubEnv(bs), obs_stack_size=2),
    "FrameStackObsAction": lambda bs: FrameStackWrapper(
        StubEnv(bs), obs_stack_size=1, action_stack_size=2, action_default_value=ZERO),
    "TransformWrapper": lambda bs: TransformWrapper(
        StubEnv(bs), context_transformation=BatchifyTransformation() if bs is None else None,
        observation_transformation=DictIncludeKeyTransformation(["position"])),
    "ContextObservationTransform": lambda bs: ContextObservationTransformWrapper(
        StubEnv(bs), observation_transformation=DictIncludeKeyTransformation(["velocity"])),
    "EpisodeRenderStack": lambda bs: EpisodeRenderStackWrapper(StubEnv(bs), rendered_index=None if bs is None else 0),
    "FuncEnvBasedEnv": lambda bs: FuncEnvBasedEnv(StubFuncEnv(bs)),
    "FuncFrameStack": lambda bs: FuncEnvBasedEnv(FuncFrameStackWrapper(
        StubFuncEnv(bs), obs_stack_size=1, action_stack_size=1, action_default_value=ZERO)),
    "FuncTransform": lambda bs: FuncEnvBasedEnv(FuncTransformWrapper(
        StubFuncEnv(bs), observation_transformation=DictIncludeKeyTransformation(["position"]))),
    "FuncEnvWrapper": lambda bs: FuncEnvBasedEnv(_PlainFunc(_PlainFunc(StubFuncEnv(bs)))),
    "WorldEnv": lambda bs: (lambda w: WorldEnv(w, CounterNode(w, control_timestep=0.03, with_signals=True)))(
        SimWorld(0.01, batch_size=bs)),
    "FuncWorldEnv": lambda bs: (lambda w: FuncEnvBasedEnv(FuncWorldEnv(
        w, CounterFuncNode(w, control_timestep=0.02, with_signals=True))))(SimFuncWorld(0.01, batch_size=bs)),
}


@pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=["unbatched", "batched3"])
@pytest.mark.parametrize("name", sorted(STATEFUL_FACTORIES))
def test_env_contract(name, batch_size):
    env_contract.check_env_contract(lambda: STATEFUL_FACTORIES[name](batch_size))


@pytest.mark.parametrize("name", ["StubEnv", "TimeLimit", "FrameStackObsAction", "FuncEnvBasedEnv"])
def test_env_contract_batch_size_one(name):
    env_contract.check_env_contract(lambda: STATEFUL_FACTORIES[name](1))


def test_batchify_contract():
    env_contract.check_env_contract(lambda: BatchifyWrapper(StubEnv()))


def test_unbatchify_contract():
    env_contract.check_env_contract(lambda: UnBatchifyWrapper(StubEnv(1)))


def test_batchify_unbatchify_stack_contract():
    env_contract.check_env_contract(lambda: TimeLimitWrapper(UnBatchifyWrapper(BatchifyWrapper(StubEnv())), 5))


@pytest.mark.parametrize("backend,device", cpu_backend_params())
@pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=["unbatched", "batched3"])
def test_backend_compat_contract(backend, device, batch_size):
    env_contract.check_env_contract(
        lambda: ToBackendOrDeviceWrapper(StubEnv(batch_size), backend=backend, device=device))


@pytest.mark.parametrize("backend,device", cpu_backend_params())
def test_backend_compat_over_other_wrappers_contract(backend, device):
    env_contract.check_env_contract(lambda: FrameStackWrapper(
        TimeLimitWrapper(ToBackendOrDeviceWrapper(StubEnv(3), backend=backend, device=device), 4),
        obs_stack_size=1))


def test_from_gymnasium_contract():
    env_contract.check_env_contract(lambda: FromGymnasiumEnv(TinyGymEnv()))


def test_from_to_gymnasium_contract():
    class _NoContext(StubEnv):
        def __init__(self):
            super().__init__()
            self.context_space = None

        def reset(self, **kwargs):
            return (None, *super().reset(**kwargs)[1:])

    env_contract.check_env_contract(lambda: FromGymnasiumEnv(ToGymnasiumEnv(_NoContext())))


def test_sync_vec_env_contract():
    env_contract.check_env_contract(lambda: SyncVecEnv([StubEnv for _ in range(3)]))


def test_sync_vec_env_of_wrapped_envs_contract():
    env_contract.check_env_contract(
        lambda: SyncVecEnv([lambda: FrameStackWrapper(TimeLimitWrapper(StubEnv(), 2), obs_stack_size=1)
                            for _ in range(2)]))


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_async_vec_env_contract():
    env_contract.check_env_contract(
        lambda: AsyncVecEnv([StubEnv for _ in range(3)], ctx=mp.get_context("spawn")))


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_episode_video_contract(tmp_path):
    env_contract.check_env_contract(
        lambda: EpisodeVideoWrapper(StubEnv(), store_dir=tmp_path, format="mp4"), n_steps=2)


# ---------------------------------------------------------------------------
# The checker itself must catch contract violations
# ---------------------------------------------------------------------------

class _FullBatchMaskedReset(StubEnv):
    def reset(self, *, mask=None, seed=None, **kwargs):
        out = super().reset(mask=mask, seed=seed, **kwargs)
        return self.goal.copy(), self._obs(), out[2]


class _ArrayRewardUnbatched(StubEnv):
    def step(self, action):
        obs, reward, *rest = super().step(action)
        return (obs, np.array([reward], np.float32), *rest)


class _ScalarRewardBatched(StubEnv):
    def step(self, action):
        obs, reward, *rest = super().step(action)
        return (obs, float(reward.sum()), *rest)


class _NoDeterminism(StubEnv):
    def reset(self, *, mask=None, seed=None, **kwargs):
        return super().reset(mask=mask, seed=None, **kwargs)


class _NoClose(StubEnv):
    def close(self):
        pass


@pytest.mark.parametrize("make_env, expected", [
    (lambda: _FullBatchMaskedReset(3), "selected rows"),
    (lambda: _ArrayRewardUnbatched(), "scalar"),
    (lambda: _ScalarRewardBatched(3), "reward"),
    (lambda: _NoDeterminism(), None),
    (lambda: _NoClose(), "close"),
], ids=["masked-reset-full-batch", "unbatched-array-reward", "batched-scalar-reward", "non-deterministic", "no-close"])
def test_contract_detects_violations(make_env, expected):
    with pytest.raises(AssertionError, match=expected):
        env_contract.check_env_contract(make_env)
