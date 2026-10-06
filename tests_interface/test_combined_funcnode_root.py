"""Tests for ``root_node`` support in ``CombinedFuncWorldNode`` / ``FlatCombinedFuncWorldNode``.

Mirrors ``test_combined_root_node.py`` for the functional API: a ``root_node`` is
passed separately from ``nodes`` and merges its data at the TOP LEVEL of the
aggregated dicts, while its *node state* lives under the reserved key
``ROOT_NODE_KEY`` (``""``) inside the persisted ``CombinedNodeStateT`` dict.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace, DictSpace, Space
from unienv_interface.world.funcnode import FuncWorldNode
from unienv_interface.world.funcworld import FuncWorld
from unienv_interface.world.funcenv_composer import FuncWorldEnv
from unienv_interface.world.funcnodes.combined_funcnode import (
    CombinedFuncWorldNode,
    ROOT_NODE_KEY,
)
from unienv_interface.world.funcnodes.flat_combined_funcnode import (
    FlatCombinedFuncWorldNode,
)


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def box_space(shape: Tuple[int, ...] = (1,), low: float = 0.0, high: float = 1e9) -> BoxSpace:
    return BoxSpace(NumpyComputeBackend, low=low, high=high, dtype=np.float32, shape=shape)


def dict_space(*keys: str) -> DictSpace:
    return DictSpace(NumpyComputeBackend, {key: box_space() for key in keys})


def action_summary(action: Any) -> Any:
    """Convert an action into a comparable, plain-Python summary.

    Used so node states stay free of numpy arrays and can be compared with ``==``
    when asserting state purity.
    """
    if isinstance(action, Mapping):
        return {
            key: round(float(np.asarray(value, dtype=np.float64).ravel()[0]), 6)
            for key, value in action.items()
        }
    return tuple(round(float(x), 6) for x in np.asarray(action, dtype=np.float64).ravel())


class StubFuncWorld(FuncWorld):
    """Trivial FuncWorld whose state is a plain dict carrying an event log."""

    def __init__(self, world_timestep: float = 0.01):
        self.backend = NumpyComputeBackend
        self.device = None
        self.world_timestep = world_timestep
        self.world_subtimestep = None
        self.batch_size = None

    def initial(self, *, seed=None, **kwargs):
        return {'log': []}

    def reset(self, state, *, seed=None, mask=None, **kwargs):
        return state

    def step(self, state):
        return state, self.world_timestep

    def close(self, state):
        pass

    def is_control_timestep_compatible(self, timestep) -> bool:
        return True


class StubFuncNode(FuncWorldNode):
    """Recording stub FuncWorldNode with configurable spaces / priorities / signals."""

    def __init__(
        self,
        world,
        name: str,
        *,
        obs_keys: Sequence[str] = (),
        obs_space: Optional[Space] = None,
        action_space: Optional[Space] = None,
        update_timestep: Optional[float] = 0.01,
        control_timestep: Optional[float] = 0.01,
        priorities: Set[int] = frozenset(),
        reward: Optional[float] = None,
        terminate: bool = False,
        truncate: bool = False,
        info_value: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ):
        self.name = name
        self.world = world
        self.control_timestep = control_timestep
        self.update_timestep = update_timestep

        if obs_space is None and obs_keys:
            obs_space = dict_space(*obs_keys)
        self.observation_space = obs_space
        self.context_space = None
        self.action_space = action_space

        self.has_reward = reward is not None
        self._reward = reward
        self.has_termination_signal = terminate
        self.has_truncation_signal = truncate
        self._terminate = terminate
        self._truncate = truncate
        self._info_value = info_value
        self.render_mode = render_mode

        prios = set(priorities)
        self.initial_priorities = prios
        self.reset_priorities = prios
        self.reload_priorities = prios
        self.after_reset_priorities = prios
        self.after_reload_priorities = prios
        self.pre_environment_step_priorities = prios
        self.post_environment_step_priorities = prios

        self.close_count = 0

    # ---- state -----------------------------------------------------------
    def _initial_state(self) -> Dict[str, Any]:
        return {'value': 0.0, 'pre_count': 0, 'post_count': 0, 'actions': []}

    def initial(self, world_state, *, priority=0, seed=None, **kwargs):
        world_state.setdefault('log', []).append((self.name, 'initial', priority))
        return world_state, self._initial_state()

    def reload(self, world_state, *, priority=0, seed=None, **kwargs):
        world_state.setdefault('log', []).append((self.name, 'reload', priority))
        return world_state, self._initial_state()

    def reset(self, world_state, node_state, *, priority=0, seed=None, mask=None, **kwargs):
        world_state.setdefault('log', []).append((self.name, 'reset', priority))
        node_state = dict(node_state)
        node_state['value'] = 0.0
        return world_state, node_state

    def after_reset(self, world_state, node_state, *, priority=0, mask=None):
        world_state.setdefault('log', []).append((self.name, 'after_reset', priority))
        return world_state, node_state

    def after_reload(self, world_state, node_state, *, priority=0, mask=None):
        world_state.setdefault('log', []).append((self.name, 'after_reload', priority))
        return world_state, node_state

    def pre_environment_step(self, world_state, node_state, dt, *, priority=0):
        world_state.setdefault('log', []).append((self.name, 'pre', priority))
        node_state = dict(node_state)
        node_state['pre_count'] = node_state['pre_count'] + 1
        return world_state, node_state

    def post_environment_step(self, world_state, node_state, dt, *, priority=0):
        world_state.setdefault('log', []).append((self.name, 'post', priority))
        node_state = dict(node_state)
        node_state['post_count'] = node_state['post_count'] + 1
        node_state['value'] = node_state['value'] + 1.0
        return world_state, node_state

    def close(self, world_state, node_state):
        self.close_count += 1
        world_state.setdefault('log', []).append((self.name, 'close', 0))
        return world_state

    # ---- data / signals --------------------------------------------------
    def get_observation(self, world_state, node_state):
        assert self.observation_space is not None
        value = node_state['value']
        if isinstance(self.observation_space, DictSpace):
            return {
                key: np.full(space.shape, value, dtype=np.float32)
                for key, space in self.observation_space.spaces.items()
            }
        return np.full(self.observation_space.shape, value, dtype=np.float32)

    def set_next_action(self, world_state, node_state, action):
        node_state = dict(node_state)
        node_state['actions'] = node_state['actions'] + [action_summary(action)]
        return world_state, node_state

    def get_reward(self, world_state, node_state):
        assert self._reward is not None
        return self._reward

    def get_termination(self, world_state, node_state):
        return self._terminate

    def get_truncation(self, world_state, node_state):
        return self._truncate

    def get_info(self, world_state, node_state):
        return self._info_value

    def render(self, world_state, node_state):
        if not self.can_render:
            return None
        return np.zeros((2, 2, 3), dtype=np.uint8)


def initial_state(combined: CombinedFuncWorldNode) -> Dict[str, Any]:
    """Mimic ``FuncWorldEnv.initial``: merge per-priority states into one dict."""
    world_state = {'log': []}
    node_state: Optional[Dict[str, Any]] = None
    for priority in sorted(combined.initial_priorities, reverse=True):
        world_state, ns_p = combined.initial(world_state, priority=priority, seed=0)
        if isinstance(node_state, dict):
            node_state.update(ns_p)
        else:
            node_state = ns_p
    assert node_state is not None
    return {'world_state': world_state, 'node_state': node_state}


# ---------------------------------------------------------------------------
# 1. State purity
# ---------------------------------------------------------------------------

def test_func_root_state_purity_no_input_mutation():
    """Every functional call returns fresh state and never mutates its inputs."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'child', obs_keys=('c0',), action_space=dict_space('j1'), priorities={0})
    root = StubFuncNode(world, 'root', obs_keys=('grip',), action_space=dict_space('grip_act'), priorities={0})
    combined = CombinedFuncWorldNode('combined', [child], root_node=root)

    state = initial_state(combined)
    world_state, node_state = state['world_state'], state['node_state']
    action = {'child': {'j1': np.array([0.5], dtype=np.float32)},
              'grip_act': np.array([0.25], dtype=np.float32)}

    snapshot = copy.deepcopy(node_state)
    ws_out, ns_out = combined.set_next_action(world_state, node_state, action)

    assert ws_out is world_state                     # world state object untouched
    assert node_state == snapshot                    # input state dict unchanged
    assert ns_out is not node_state                  # a new dict is returned
    assert ns_out['child'] is not node_state['child']
    assert ns_out[ROOT_NODE_KEY] is not node_state[ROOT_NODE_KEY]
    # ... while the returned state does carry the dispatched actions.
    assert ns_out['child']['actions'] == [{'j1': 0.5}]
    assert ns_out[ROOT_NODE_KEY]['actions'] == [{'grip_act': 0.25}]

    # Reading data must not mutate anything either.
    snapshot = copy.deepcopy(ns_out)
    combined.get_observation(world_state, ns_out)
    combined.get_info(world_state, ns_out)
    assert ns_out == snapshot


# ---------------------------------------------------------------------------
# 2. Root state under the reserved key across the lifecycle
# ---------------------------------------------------------------------------

def test_func_root_state_key_across_lifecycle():
    """Root state lives under the reserved key through initial -> reset -> step -> close."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'child', obs_keys=('c0',), action_space=dict_space('j1'), priorities={0})
    root = StubFuncNode(world, 'root', obs_keys=('grip',), action_space=dict_space('grip_act'), priorities={0})
    combined = CombinedFuncWorldNode('combined', [child], root_node=root)

    assert combined.nodes == [child]
    assert combined.root_node is root
    assert combined.get_nodes_by_fn(lambda node: node is root) == [root]
    assert combined.get_node('root') is None

    state = initial_state(combined)
    world_state, node_state = state['world_state'], state['node_state']
    assert set(node_state.keys()) == {'child', ROOT_NODE_KEY, '__pre_substeps', '__action_substeps', '__cached_actions'}
    assert all(isinstance(key, str) for key in node_state.keys())   # serialization friendly

    world_state, node_state = combined.reset(world_state, node_state, priority=0)
    assert node_state['child']['value'] == 0.0
    assert node_state[ROOT_NODE_KEY]['value'] == 0.0

    world_state, node_state = combined.set_next_action(
        world_state, node_state,
        {'child': {'j1': np.array([1.0], dtype=np.float32)},
         'grip_act': np.array([2.0], dtype=np.float32)},
    )
    assert node_state['child']['actions'] == [{'j1': 1.0}]
    assert node_state[ROOT_NODE_KEY]['actions'] == [{'grip_act': 2.0}]
    assert set(node_state['__cached_actions'].keys()) == {'child', ROOT_NODE_KEY}

    world_state, node_state = combined.pre_environment_step(world_state, node_state, 0.01, priority=0)
    world_state, node_state = combined.post_environment_step(world_state, node_state, 0.01, priority=0)
    assert node_state[ROOT_NODE_KEY]['post_count'] == 1
    assert node_state['child']['post_count'] == 1

    world_state = combined.close(world_state, node_state)
    assert child.close_count == 1 and root.close_count == 1

    # Routing state (the root's cached action) resets with after_reset too.
    world_state, node_state = combined.after_reset(
        world_state, node_state, priority=combined._INTERNAL_RESET_PRIORITY
    )
    assert node_state['__cached_actions'] == {}
    assert node_state['__action_substeps'] == 0


# ---------------------------------------------------------------------------
# 3. Remainder routing + sole-root passthrough
# ---------------------------------------------------------------------------

def test_func_action_remainder_routes_to_root():
    """``{child_name: x, root_key: y}``: child gets exactly x, root only {root_key: y}."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'arm', action_space=dict_space('j1'), priorities={0})
    root = StubFuncNode(world, 'hand', action_space=dict_space('grip'), priorities={0})
    combined = CombinedFuncWorldNode('combined', [child], root_node=root)
    state = initial_state(combined)
    world_state, node_state = state['world_state'], state['node_state']

    assert isinstance(combined.action_space, DictSpace)
    assert set(combined.action_space.spaces.keys()) == {'arm', 'grip'}

    world_state, node_state = combined.set_next_action(
        world_state, node_state,
        {'arm': {'j1': np.array([0.5], dtype=np.float32)},
         'grip': np.array([0.25], dtype=np.float32)},
    )
    assert node_state['arm']['actions'] == [{'j1': 0.5}]
    assert node_state[ROOT_NODE_KEY]['actions'] == [{'grip': 0.25}]

    # Hold-last in both directions.
    world_state, node_state = combined.set_next_action(
        world_state, node_state, {'arm': {'j1': np.array([0.75], dtype=np.float32)}}
    )
    assert node_state['arm']['actions'] == [{'j1': 0.5}, {'j1': 0.75}]
    assert node_state[ROOT_NODE_KEY]['actions'] == [{'grip': 0.25}, {'grip': 0.25}]

    world_state, node_state = combined.set_next_action(
        world_state, node_state, {'grip': np.array([0.1], dtype=np.float32)}
    )
    assert node_state['arm']['actions'][-1] == {'j1': 0.75}
    assert node_state[ROOT_NODE_KEY]['actions'][-1] == {'grip': 0.1}


def test_func_sole_root_action_passthrough():
    """Sole-root action provider in both ``direct_return`` modes."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'obs_only_child', obs_keys=('c0',), priorities={0})
    root = StubFuncNode(world, 'policy', action_space=box_space(shape=(2,), low=-1.0, high=1.0),
                        priorities={0})

    direct = CombinedFuncWorldNode('combined', [child], root_node=root, direct_return=True)
    assert direct.action_space is root.action_space
    assert direct._action_node_name_direct == ROOT_NODE_KEY
    state = initial_state(direct)
    world_state, node_state = state['world_state'], state['node_state']
    world_state, node_state = direct.set_next_action(
        world_state, node_state, np.array([0.7, -0.7], dtype=np.float32)
    )
    assert node_state[ROOT_NODE_KEY]['actions'] == [(0.7, -0.7)]

    wrapped = CombinedFuncWorldNode('combined', [child], root_node=root, direct_return=False)
    assert isinstance(wrapped.action_space, DictSpace)
    assert set(wrapped.action_space.spaces.keys()) == {ROOT_NODE_KEY}
    state = initial_state(wrapped)
    world_state, node_state = state['world_state'], state['node_state']
    world_state, node_state = wrapped.set_next_action(
        world_state, node_state, {ROOT_NODE_KEY: np.array([0.7, -0.7], dtype=np.float32)}
    )
    # The reserved key only wraps the *action space*: the root receives the
    # unwrapped action value, exactly like a single DictSpace child does.
    assert node_state[ROOT_NODE_KEY]['actions'] == [(0.7, -0.7)]


# ---------------------------------------------------------------------------
# 4. Mixed DictSpace merging + collision errors
# ---------------------------------------------------------------------------

def test_func_mixed_space_merge_and_collisions():
    """Merged DictSpace holds child names + root keys; collisions raise ValueError."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'arm', obs_keys=('arm_obs',))
    root = StubFuncNode(world, 'hand', obs_keys=('grip',))
    combined = CombinedFuncWorldNode('combined', [child], root_node=root)

    assert isinstance(combined.observation_space, DictSpace)
    assert set(combined.observation_space.spaces.keys()) == {'arm', 'grip'}

    bad_root = StubFuncNode(world, 'bad_root', obs_keys=('arm',))
    with pytest.raises(ValueError) as err:
        CombinedFuncWorldNode('combined', [child], root_node=bad_root)
    message = str(err.value)
    assert 'bad_root' in message and "'arm'" in message

    box_root = StubFuncNode(world, 'box_root', obs_space=box_space(shape=(4,)))
    with pytest.raises(AssertionError):
        CombinedFuncWorldNode('combined', [child], root_node=box_root)

    # Construction invariants.
    with pytest.raises(AssertionError):
        CombinedFuncWorldNode('combined', [child, root], root_node=root)
    with pytest.raises(AssertionError):
        CombinedFuncWorldNode('combined', [child], root_node=StubFuncNode(StubFuncWorld(), 'foreign'))


def test_func_root_obs_data_merge_and_direct_return():
    """Data merging mirrors the space rule, including sole-root direct_return."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'arm', obs_keys=('arm_obs',), priorities={0})
    root = StubFuncNode(world, 'hand', obs_keys=('grip',), priorities={0})
    combined = CombinedFuncWorldNode('combined', [child], root_node=root)
    state = initial_state(combined)
    world_state, node_state = state['world_state'], state['node_state']
    node_state['arm']['value'] = 3.0
    node_state[ROOT_NODE_KEY]['value'] = 5.0

    obs = combined.get_observation(world_state, node_state)
    assert set(obs.keys()) == {'arm', 'grip'}
    np.testing.assert_allclose(obs['arm']['arm_obs'], [3.0])
    np.testing.assert_allclose(obs['grip'], [5.0])
    assert 'hand' not in obs

    # Sole root contributor, direct_return True -> unwrapped; False -> reserved key.
    obs_only_child = StubFuncNode(world, 'obs_only_child', priorities={0})
    box_root = StubFuncNode(world, 'box_root', obs_space=box_space(shape=(3,)), priorities={0})
    direct = CombinedFuncWorldNode('combined', [obs_only_child], root_node=box_root, direct_return=True)
    assert direct.observation_space is box_root.observation_space
    state = initial_state(direct)
    world_state, node_state = state['world_state'], state['node_state']
    node_state[ROOT_NODE_KEY]['value'] = 4.0
    np.testing.assert_allclose(direct.get_observation(world_state, node_state), [4.0, 4.0, 4.0])

    wrapped = CombinedFuncWorldNode('combined', [obs_only_child], root_node=box_root, direct_return=False)
    assert set(wrapped.observation_space.spaces.keys()) == {ROOT_NODE_KEY}
    state = initial_state(wrapped)
    world_state, node_state = state['world_state'], state['node_state']
    node_state[ROOT_NODE_KEY]['value'] = 4.0
    obs = wrapped.get_observation(world_state, node_state)
    assert set(obs.keys()) == {ROOT_NODE_KEY}
    np.testing.assert_allclose(obs[ROOT_NODE_KEY], [4.0, 4.0, 4.0])


# ---------------------------------------------------------------------------
# 5. Flat funcnode
# ---------------------------------------------------------------------------

def test_flat_func_unclaimed_action_keys_route_to_root():
    """Flat func routing sends keys no child claims to the root; obs keys merge flat."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'child', obs_keys=('value',), action_space=dict_space('j1'), priorities={0})
    root = StubFuncNode(world, 'root', obs_keys=('grip',), action_space=dict_space('g'), priorities={0})
    combined = FlatCombinedFuncWorldNode('flat', [child], root_node=root)

    assert set(combined.action_space.spaces.keys()) == {'j1', 'g'}
    assert set(combined.observation_space.spaces.keys()) == {'value', 'grip'}

    state = initial_state(combined)
    world_state, node_state = state['world_state'], state['node_state']
    node_state['child']['value'] = 1.0
    node_state[ROOT_NODE_KEY]['value'] = 2.0
    obs = combined.get_observation(world_state, node_state)
    assert set(obs.keys()) == {'value', 'grip'}
    np.testing.assert_allclose(obs['value'], [1.0])
    np.testing.assert_allclose(obs['grip'], [2.0])

    world_state, node_state = combined.set_next_action(
        world_state, node_state,
        {'j1': np.array([0.4], dtype=np.float32), 'g': np.array([0.6], dtype=np.float32)},
    )
    assert node_state['child']['actions'] == [{'j1': 0.4}]
    assert node_state[ROOT_NODE_KEY]['actions'] == [{'g': 0.6}]

    # Keys outside the aggregate action space (claimed by no child) also go to the root.
    world_state, node_state = combined.set_next_action(
        world_state, node_state,
        {'j1': np.array([0.4], dtype=np.float32), 'unmapped': np.array([0.9], dtype=np.float32)},
    )
    assert node_state[ROOT_NODE_KEY]['actions'][-1] == {'unmapped': 0.9}


def test_flat_func_unclaimed_action_key_without_root_raises():
    """Without a root node, unclaimed flat action keys are still rejected."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'child', obs_keys=('value',), action_space=dict_space('j1'), priorities={0})
    combined = FlatCombinedFuncWorldNode('flat', [child])
    state = initial_state(combined)
    world_state, node_state = state['world_state'], state['node_state']

    world_state, node_state = combined.set_next_action(
        world_state, node_state, {'j1': np.array([0.4], dtype=np.float32)}
    )
    assert node_state['child']['actions'] == [{'j1': 0.4}]

    with pytest.raises(ValueError, match="not claimed"):
        combined.set_next_action(
            world_state, node_state,
            {'j1': np.array([0.4], dtype=np.float32), 'junk': np.array([0.1], dtype=np.float32)},
        )


# ---------------------------------------------------------------------------
# 6. Multi-rate ratios
# ---------------------------------------------------------------------------

def test_func_multi_rate_root_and_child_dispatch_counts():
    """Child at 2x the root control rate: independent ratio ticks, hold-last."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'fast', action_space=dict_space('j1'),
                         control_timestep=0.01, update_timestep=0.01, priorities={0})
    root = StubFuncNode(world, 'slow', action_space=dict_space('grip'),
                        control_timestep=0.02, update_timestep=0.01, priorities={0})
    combined = CombinedFuncWorldNode('combined', [child], root_node=root)

    assert combined._action_ratios['fast'] == 1
    assert combined._action_ratios[ROOT_NODE_KEY] == 2
    assert combined._action_period == 2

    state = initial_state(combined)
    world_state, node_state = state['world_state'], state['node_state']
    for i in range(4):
        world_state, node_state = combined.set_next_action(
            world_state, node_state,
            {'fast': {'j1': np.array([float(i)], dtype=np.float32)},
             'grip': np.array([float(i)], dtype=np.float32)},
        )

    assert len(node_state['fast']['actions']) == 4
    assert len(node_state[ROOT_NODE_KEY]['actions']) == 2
    assert node_state[ROOT_NODE_KEY]['actions'] == [{'grip': 0.0}, {'grip': 2.0}]


def test_func_multi_rate_update_step_dispatch_includes_root():
    """pre/post step ratios include the root's update timestep."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'fast', update_timestep=0.01, control_timestep=None, priorities={0})
    root = StubFuncNode(world, 'slow', update_timestep=0.02, control_timestep=None, priorities={0})
    combined = CombinedFuncWorldNode('combined', [child], root_node=root)

    assert combined._update_ratios['fast'] == 1
    assert combined._update_ratios[ROOT_NODE_KEY] == 2
    assert combined.update_timestep == 0.01

    state = initial_state(combined)
    world_state, node_state = state['world_state'], state['node_state']
    for _ in range(4):
        world_state, node_state = combined.pre_environment_step(world_state, node_state, 0.01, priority=0)
        world_state, node_state = combined.post_environment_step(world_state, node_state, 0.01, priority=0)

    assert node_state['fast']['pre_count'] == 4 and node_state['fast']['post_count'] == 4
    assert node_state[ROOT_NODE_KEY]['pre_count'] == 2
    assert node_state[ROOT_NODE_KEY]['post_count'] == 2


# ---------------------------------------------------------------------------
# 7. Lifecycle threading
# ---------------------------------------------------------------------------

def test_func_root_lifecycle_threads_world_and_node_state():
    """Root hooks receive / return (world_state, node_state) at their own priorities."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'child', priorities={0, 10})
    root = StubFuncNode(world, 'root', priorities={5})
    combined = CombinedFuncWorldNode('combined', [child], root_node=root)

    assert combined.initial_priorities == {0, 5, 10}
    assert combined.reset_priorities == {0, 5, 10}
    assert combined.after_reload_priorities == {0, 5, 10, combined._INTERNAL_RESET_PRIORITY}

    # World state threaded through: the stub nodes append their events to it.
    state = initial_state(combined)
    world_state, node_state = state['world_state'], state['node_state']
    assert set(node_state.keys()) >= {'child', ROOT_NODE_KEY}
    assert [(entry[0], entry[1], entry[2]) for entry in world_state['log']] == [
        ('child', 'initial', 10), ('root', 'initial', 5), ('child', 'initial', 0),
    ]

    # reload (world_state only, per-priority states merged by the composer).
    world_state = {'log': []}
    merged: Optional[Dict[str, Any]] = None
    for priority in sorted(combined.reload_priorities, reverse=True):
        world_state, ns_p = combined.reload(world_state, priority=priority)
        merged = ns_p if merged is None else {**merged, **ns_p}
    assert merged is not None
    assert 'child' in merged and ROOT_NODE_KEY in merged
    assert [(entry[0], entry[1], entry[2]) for entry in world_state['log']] == [
        ('child', 'reload', 10), ('root', 'reload', 5), ('child', 'reload', 0),
    ]

    for phase in ('reset', 'after_reset', 'after_reload'):
        world_state['log'].clear()
        for priority in sorted(getattr(combined, f'{phase}_priorities'), reverse=True):
            world_state, node_state = getattr(combined, phase)(
                world_state, node_state, priority=priority
            )
        assert [(entry[0], entry[1], entry[2]) for entry in world_state['log']] == [
            ('child', phase, 10), ('root', phase, 5), ('child', phase, 0),
        ]

    world_state['log'].clear()
    for priority in sorted(combined.pre_environment_step_priorities, reverse=True):
        world_state, node_state = combined.pre_environment_step(world_state, node_state, 0.01, priority=priority)
    for priority in sorted(combined.post_environment_step_priorities, reverse=True):
        world_state, node_state = combined.post_environment_step(world_state, node_state, 0.01, priority=priority)
    assert [(entry[0], entry[1], entry[2]) for entry in world_state['log'] if entry[1] == 'pre'] == [
        ('child', 'pre', 10), ('root', 'pre', 5), ('child', 'pre', 0),
    ]
    assert [(entry[0], entry[1], entry[2]) for entry in world_state['log'] if entry[1] == 'post'] == [
        ('child', 'post', 10), ('root', 'post', 5), ('child', 'post', 0),
    ]


def test_func_root_env_integration():
    """End-to-end FuncWorldEnv: initial / step with a root node in the composition."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'arm', obs_keys=('arm_obs',), action_space=dict_space('j1'),
                         priorities={0}, control_timestep=0.01, update_timestep=0.01,
                         reward=1.0)
    root = StubFuncNode(world, 'hand', obs_keys=('grip',), action_space=dict_space('grip_act'),
                        priorities={0}, control_timestep=0.01, update_timestep=0.01,
                        reward=0.5, terminate=False)
    env = FuncWorldEnv(world, CombinedFuncWorldNode('combined', [child], root_node=root))

    state, context, obs, info = env.initial()
    assert set(obs.keys()) == {'arm', 'grip'}
    assert state.node_state[ROOT_NODE_KEY]['post_count'] == 0

    for _ in range(3):
        state, obs, reward, terminated, truncated, info = env.step(state, {
            'arm': {'j1': np.array([0.5], dtype=np.float32)},
            'grip_act': np.array([0.25], dtype=np.float32),
        })

    assert state.node_state['arm']['post_count'] == 3
    assert state.node_state[ROOT_NODE_KEY]['post_count'] == 3
    assert state.node_state[ROOT_NODE_KEY]['actions'][-1] == {'grip_act': 0.25}
    assert reward == 1.5
    assert terminated is False


# ---------------------------------------------------------------------------
# 8. Reward / info / render aggregation
# ---------------------------------------------------------------------------

def test_func_root_reward_signals_info_and_render():
    """Root reward/signals are aggregated; info/render use the root's display name."""
    world = StubFuncWorld()
    child = StubFuncNode(world, 'child', priorities={0}, reward=1.0, info_value={'x': 1},
                         render_mode='rgb_array')
    root = StubFuncNode(world, 'root', priorities={0}, reward=0.5, terminate=True, truncate=True,
                        info_value={'y': 2}, render_mode='rgb_array')
    combined = CombinedFuncWorldNode('combined', [child], root_node=root, render_mode='dict')

    state = initial_state(combined)
    world_state, node_state = state['world_state'], state['node_state']

    assert combined.has_reward and combined.has_termination_signal and combined.has_truncation_signal
    assert combined.get_reward(world_state, node_state) == 1.5
    assert combined.get_termination(world_state, node_state) is True
    assert combined.get_truncation(world_state, node_state) is True

    info = combined.get_info(world_state, node_state)
    assert info == {'child': {'x': 1}, 'root': {'y': 2}}
    assert ROOT_NODE_KEY not in info

    rendered = combined.render(world_state, node_state)
    assert set(rendered.keys()) == {'child', 'root'}
    assert ROOT_NODE_KEY not in rendered
