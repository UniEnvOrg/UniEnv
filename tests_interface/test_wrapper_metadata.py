"""Environment wrappers keep (or deliberately drop) ``Space.metadata`` on the spaces they expose."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from unienv_interface.space import DictSpace
from unienv_interface.utils.dict_util import flatten_keys_in_mapping, unflatten_keys_in_mapping
from unienv_interface.wrapper import (
    ActionRescaleWrapper,
    BatchifyWrapper,
    FlattenActionWrapper,
    FlattenContextObservationWrapper,
    FrameStackWrapper,
    UnBatchifyWrapper,
)
from unienv_interface.wrapper.backend_compat import ToBackendOrDeviceWrapper
from tests_common.envs import StubEnv

ACTION_MD = {"kind": "action", "units": "m"}
POSITION_MD = {"kind": "proprio", "units": "m"}
OBS_MD = {"label": "observation"}


class AnnotatedEnv(StubEnv):
    def __init__(self, batch_size: Optional[int] = None) -> None:
        super().__init__(batch_size)
        self.action_space = self.action_space.with_metadata(ACTION_MD)
        self.observation_space = DictSpace(self.backend, {
            "position": self.observation_space["position"].with_metadata(POSITION_MD),
            "velocity": self.observation_space["velocity"],
        }, metadata=OBS_MD)
        self.context_space = self.context_space.with_metadata(label="goal")


def test_frame_stack_keeps_leaf_metadata_on_stacked_leaves():
    wrapper = FrameStackWrapper(AnnotatedEnv(), obs_stack_size=2, action_stack_size=2, action_default_value=np.zeros(2, np.float32))
    obs_space = wrapper.observation_space
    assert obs_space["position"].shape == (3, 2)
    assert obs_space["position"].metadata == POSITION_MD
    assert obs_space["velocity"].metadata == {}
    assert obs_space["past_actions"].metadata == ACTION_MD
    assert obs_space.metadata == OBS_MD
    assert wrapper.action_space.metadata == ACTION_MD


def test_batchify_and_unbatchify_wrappers_keep_metadata():
    batched = BatchifyWrapper(AnnotatedEnv())
    assert batched.action_space.shape == (1, 2)
    assert batched.action_space.metadata == ACTION_MD
    assert batched.observation_space.metadata == OBS_MD
    assert batched.observation_space["position"].metadata == POSITION_MD
    assert batched.context_space.metadata == {"label": "goal"}

    unbatched = UnBatchifyWrapper(AnnotatedEnv(batch_size=1))
    assert unbatched.action_space.shape == (2,)
    assert unbatched.action_space.metadata == ACTION_MD
    assert unbatched.observation_space["position"].metadata == POSITION_MD


def test_action_rescale_keeps_metadata():
    wrapper = ActionRescaleWrapper(AnnotatedEnv(), -0.5, 0.5)
    assert wrapper.action_space.metadata == ACTION_MD


def test_backend_compat_wrapper_keeps_metadata():
    pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend
    wrapper = ToBackendOrDeviceWrapper(AnnotatedEnv(), backend=PyTorchComputeBackend, device="cpu")
    assert wrapper.action_space.backend is PyTorchComputeBackend
    assert wrapper.action_space.metadata == ACTION_MD
    assert wrapper.observation_space.metadata == OBS_MD
    assert wrapper.observation_space["position"].metadata == POSITION_MD
    assert wrapper.context_space.metadata == {"label": "goal"}


def test_flatten_wrappers_drop_metadata():
    env = AnnotatedEnv()
    assert FlattenActionWrapper(env).action_space.metadata == {}
    flat = FlattenContextObservationWrapper(env, flatten_context=True)
    assert flat.observation_space.metadata == {}
    assert flat.context_space.metadata == {}


def test_dict_util_flatten_keys_keeps_metadata():
    space = DictSpace(StubEnv.backend, {"a": DictSpace(StubEnv.backend, {"b": AnnotatedEnv().action_space})}, metadata=OBS_MD)
    flat = flatten_keys_in_mapping(space)
    assert flat.metadata == OBS_MD and flat["a/b"].metadata == ACTION_MD
    nested = unflatten_keys_in_mapping(flat)
    assert nested.metadata == OBS_MD and nested["a"]["b"].metadata == ACTION_MD
