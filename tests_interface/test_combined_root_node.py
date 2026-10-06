"""Tests for ``root_node`` support in ``CombinedWorldNode`` / ``FlatCombinedWorldNode``.

A ``root_node`` is passed separately from ``nodes`` and merges its data at the
TOP LEVEL of the aggregated dicts (children keep nesting under their names). It
rides all existing child machinery under the reserved internal key
``ROOT_NODE_KEY`` (``""``). These tests cover construction invariants, space /
data merging, action remainder routing + hold-last + multi-rate dispatch,
``direct_return`` interaction for a sole root contributor, lifecycle dispatch,
flat routing and reward / signal aggregation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace, DictSpace, Space
from unienv_interface.world import RealWorld, WorldEnv
from unienv_interface.world.node import WorldNode
from unienv_interface.world.nodes.combined_node import CombinedWorldNode, ROOT_NODE_KEY
from unienv_interface.world.nodes.flat_combined_node import FlatCombinedWorldNode


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def box_space(shape: Tuple[int, ...] = (1,), low: float = 0.0, high: float = 1e9) -> BoxSpace:
    return BoxSpace(NumpyComputeBackend, low=low, high=high, dtype=np.float32, shape=shape)


def dict_space(*keys: str) -> DictSpace:
    return DictSpace(NumpyComputeBackend, {key: box_space() for key in keys})


class StubNode(WorldNode):
    """Recording stub node with configurable spaces, priorities and signals.

    Every lifecycle dispatch is appended to ``log`` as ``(name, method, priority)``
    so priority ordering / interleaving with the root node can be asserted.
    """

    def __init__(
        self,
        world,
        name: str,
        *,
        obs_keys: Sequence[str] = (),
        obs_space: Optional[Space] = None,
        context_keys: Sequence[str] = (),
        action_space: Optional[Space] = None,
        update_timestep: Optional[float] = 0.01,
        control_timestep: Optional[float] = 0.01,
        priorities: Set[int] = frozenset(),
        log: Optional[List[Tuple[str, str, int]]] = None,
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
        self.context_space = dict_space(*context_keys) if context_keys else None
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
        self.reset_priorities = prios
        self.reload_priorities = prios
        self.after_reset_priorities = prios
        self.after_reload_priorities = prios
        self.pre_environment_step_priorities = prios
        self.post_environment_step_priorities = prios

        self.log = log if log is not None else []
        self.value = 0.0
        self.pre_count = 0
        self.post_count = 0
        self.actions: List[Any] = []
        self.close_count = 0

    # ---- lifecycle -------------------------------------------------------
    def reset(self, *, priority: int = 0, seed=None, mask=None, **kwargs) -> None:
        self.log.append((self.name, 'reset', priority))

    def reload(self, *, priority: int = 0, seed=None, mask=None, **kwargs) -> None:
        self.log.append((self.name, 'reload', priority))

    def after_reset(self, *, priority: int = 0, mask=None) -> None:
        self.log.append((self.name, 'after_reset', priority))

    def after_reload(self, *, priority: int = 0, mask=None) -> None:
        self.log.append((self.name, 'after_reload', priority))

    def pre_environment_step(self, dt, *, priority: int = 0) -> None:
        self.log.append((self.name, 'pre', priority))
        self.pre_count += 1

    def post_environment_step(self, dt, *, priority: int = 0) -> None:
        self.log.append((self.name, 'post', priority))
        self.post_count += 1
        self.value += 1.0

    def close(self) -> None:
        self.close_count += 1

    # ---- data / signals --------------------------------------------------
    def get_context(self):
        return {key: np.full(1, self.value, dtype=np.float32) for key in self.context_space.spaces}

    def get_observation(self):
        assert self.observation_space is not None
        if isinstance(self.observation_space, DictSpace):
            return {
                key: np.full(space.shape, self.value, dtype=np.float32)
                for key, space in self.observation_space.spaces.items()
            }
        return np.full(self.observation_space.shape, self.value, dtype=np.float32)

    def set_next_action(self, action) -> None:
        self.actions.append(action)

    def get_reward(self) -> float:
        assert self._reward is not None
        return self._reward

    def get_termination(self) -> bool:
        return self._terminate

    def get_truncation(self) -> bool:
        return self._truncate

    def get_info(self):
        return self._info_value

    def render(self):
        if not self.can_render:
            return None
        return np.zeros((2, 2, 3), dtype=np.uint8)


def make_world() -> RealWorld:
    return RealWorld(NumpyComputeBackend, world_timestep=0.01)


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------

def test_root_construction_invariants():
    """The root node stays out of ``nodes`` / ``get_node`` but joins the subtree."""
    world = make_world()
    child = StubNode(world, 'child', obs_keys=('c',))
    root = StubNode(world, 'root', obs_keys=('r',))
    combined = CombinedWorldNode('combined', [child], root_node=root)

    assert combined.root_node is root
    assert combined.nodes == [child]                 # public children list unchanged
    assert combined.get_node('child') is child
    assert combined.get_node('root') is None         # reserved-key path is not a user path
    assert combined.get_nodes_by_fn(lambda node: node is root) == [root]
    assert list(combined._all_nodes) == [child, root]

    # The root must not also be listed in ``nodes``.
    with pytest.raises(AssertionError):
        CombinedWorldNode('combined', [child, root], root_node=root)

    # The root must share the world.
    foreign_root = StubNode(make_world(), 'foreign', obs_keys=('f',))
    with pytest.raises(AssertionError):
        CombinedWorldNode('combined', [child], root_node=foreign_root)

    # The reserved key must not be reachable as a child name.
    nameless = StubNode(world, '', obs_keys=('x',))
    with pytest.raises(AssertionError):
        CombinedWorldNode('combined', [nameless], root_node=root)


def test_combined_close_includes_root():
    """``close`` reaches the root node as well as the children."""
    world = make_world()
    child = StubNode(world, 'child', obs_keys=('c',))
    root = StubNode(world, 'root', obs_keys=('r',))
    combined = CombinedWorldNode('combined', [child], root_node=root)
    combined.close()
    assert child.close_count >= 1
    assert root.close_count >= 1


# ---------------------------------------------------------------------------
# 1. Observation merge
# ---------------------------------------------------------------------------

def test_root_obs_keys_merge_at_top_level():
    """Root observation keys sit next to the nested child names, values/shapes intact."""
    world = make_world()
    a = StubNode(world, 'a', obs_keys=('a0',))
    b = StubNode(world, 'b', obs_keys=('b0',))
    root = StubNode(world, 'root', obs_keys=('grip',), obs_space=DictSpace(
        NumpyComputeBackend, {'grip': box_space(shape=(2,))}
    ))
    combined = CombinedWorldNode('combined', [a, b], root_node=root)

    a.value, b.value, root.value = 1.0, 2.0, 3.0
    obs = combined.get_observation()

    assert set(obs.keys()) == {'a', 'b', 'grip'}
    assert set(obs['a'].keys()) == {'a0'} and set(obs['b'].keys()) == {'b0'}
    assert obs['a']['a0'].shape == (1,) and obs['b']['b0'].shape == (1,)
    assert obs['grip'].shape == (2,)
    np.testing.assert_allclose(obs['a']['a0'], [1.0])
    np.testing.assert_allclose(obs['b']['b0'], [2.0])
    np.testing.assert_allclose(obs['grip'], [3.0, 3.0])
    # The root's own name is NOT a nesting key in the observation channel.
    assert 'root' not in obs


def test_root_context_merge_at_top_level():
    """Context follows the same top-level merge rule as observations."""
    world = make_world()
    child = StubNode(world, 'child', context_keys=('c0',))
    root = StubNode(world, 'root', context_keys=('mode',))
    combined = CombinedWorldNode('combined', [child], root_node=root)

    child.value, root.value = 4.0, 5.0
    context = combined.get_context()
    assert set(context.keys()) == {'child', 'mode'}
    np.testing.assert_allclose(context['child']['c0'], [4.0])
    np.testing.assert_allclose(context['mode'], [5.0])


# ---------------------------------------------------------------------------
# 2. Space merge + collisions
# ---------------------------------------------------------------------------

def test_root_space_merge_and_collision_errors():
    """Merged DictSpace holds child names + root keys; collisions raise ValueError."""
    world = make_world()
    child = StubNode(world, 'arm', obs_keys=('arm_obs',), action_space=dict_space('j1'))
    root = StubNode(world, 'hand', obs_keys=('grip',), action_space=dict_space('grip_act'))
    combined = CombinedWorldNode('combined', [child], root_node=root)

    assert isinstance(combined.observation_space, DictSpace)
    assert set(combined.observation_space.spaces.keys()) == {'arm', 'grip'}
    assert isinstance(combined.action_space, DictSpace)
    # Nested routing: children are addressed by node name, root keys sit inline.
    assert set(combined.action_space.spaces.keys()) == {'arm', 'grip_act'}
    assert combined._action_node_name_direct is None

    # Root observation key colliding with a child node name.
    bad_root = StubNode(world, 'bad_root', obs_keys=('arm',))
    with pytest.raises(ValueError) as err:
        CombinedWorldNode('combined', [child], root_node=bad_root)
    message = str(err.value)
    assert 'bad_root' in message and "'arm'" in message

    # Root action key colliding with an action child name.
    bad_action_root = StubNode(world, 'bad_action_root', action_space=dict_space('arm'))
    with pytest.raises(ValueError):
        CombinedWorldNode('combined', [child], root_node=bad_action_root)

    # A non-DictSpace root space is only allowed when the root is the sole contributor.
    box_root = StubNode(world, 'box_root', obs_space=box_space(shape=(4,)))
    with pytest.raises(AssertionError):
        CombinedWorldNode('combined', [child], root_node=box_root)


# ---------------------------------------------------------------------------
# 3. Action remainder routing
# ---------------------------------------------------------------------------

def test_action_remainder_routes_to_root():
    """``{child_name: x, root_key: y}``: child gets exactly x, root only {root_key: y}."""
    world = make_world()
    child = StubNode(world, 'arm', action_space=dict_space('j1'))
    root = StubNode(world, 'hand', action_space=dict_space('grip'))
    combined = CombinedWorldNode('combined', [child], root_node=root)

    child_action = {'j1': np.array([0.5], dtype=np.float32)}
    root_action = np.array([0.25], dtype=np.float32)
    combined.set_next_action({'arm': child_action, 'grip': root_action})

    assert len(child.actions) == 1
    assert set(child.actions[0].keys()) == {'j1'}
    np.testing.assert_allclose(child.actions[0]['j1'], [0.5])

    assert len(root.actions) == 1
    assert set(root.actions[0].keys()) == {'grip'}   # no child action leaked into the root
    np.testing.assert_allclose(root.actions[0]['grip'], [0.25])


def test_action_child_only_step_keeps_root_state_untouched():
    """A child-only action does not fabricate a root action."""
    world = make_world()
    child = StubNode(world, 'arm', action_space=dict_space('j1'))
    root = StubNode(world, 'hand', action_space=dict_space('grip'))
    combined = CombinedWorldNode('combined', [child], root_node=root)

    combined.set_next_action({'arm': {'j1': np.array([1.0], dtype=np.float32)}})
    assert len(root.actions) == 0          # root never received an action yet
    assert ROOT_NODE_KEY not in combined._cached_actions


# ---------------------------------------------------------------------------
# 4. Hold-last semantics
# ---------------------------------------------------------------------------

def test_hold_last_semantics_both_directions():
    """Each consumer re-receives its previous action when only the other is updated."""
    world = make_world()
    child = StubNode(world, 'arm', action_space=dict_space('j1'))
    root = StubNode(world, 'hand', action_space=dict_space('grip'))
    combined = CombinedWorldNode('combined', [child], root_node=root)

    combined.set_next_action({
        'arm': {'j1': np.array([0.1], dtype=np.float32)},
        'grip': np.array([0.2], dtype=np.float32),
    })
    assert len(child.actions) == 1 and len(root.actions) == 1

    # Child-only step: the root re-receives its previous action at its tick.
    combined.set_next_action({'arm': {'j1': np.array([0.3], dtype=np.float32)}})
    assert len(child.actions) == 2 and len(root.actions) == 2
    np.testing.assert_allclose(root.actions[-1]['grip'], [0.2])

    # Root-only step: the child re-receives its previous action at its tick.
    combined.set_next_action({'grip': np.array([0.4], dtype=np.float32)})
    assert len(child.actions) == 3 and len(root.actions) == 3
    np.testing.assert_allclose(child.actions[-1]['j1'], [0.3])
    np.testing.assert_allclose(root.actions[-1]['grip'], [0.4])
    assert set(child.actions[-1].keys()) == {'j1'}


# ---------------------------------------------------------------------------
# 5. Multi-rate dispatch
# ---------------------------------------------------------------------------

def test_multi_rate_root_and_child_dispatch_counts():
    """Child at 2x the root control rate: ratio ticks route independently."""
    world = make_world()
    child = StubNode(world, 'fast', action_space=dict_space('j1'), control_timestep=0.01)
    root = StubNode(world, 'slow', action_space=dict_space('grip'), control_timestep=0.02)
    combined = CombinedWorldNode('combined', [child], root_node=root)

    assert combined._action_ratios['fast'] == 1
    assert combined._action_ratios[ROOT_NODE_KEY] == 2
    assert combined._action_period == 2

    for i in range(4):
        combined.set_next_action({
            'fast': {'j1': np.array([float(i)], dtype=np.float32)},
            'grip': np.array([float(i)], dtype=np.float32),
        })

    # Fast child dispatches on every tick, the root only on odd substeps.
    assert len(child.actions) == 4
    assert len(root.actions) == 2
    np.testing.assert_allclose(root.actions[0]['grip'], [0.0])
    np.testing.assert_allclose(root.actions[1]['grip'], [2.0])


def test_multi_rate_update_step_dispatch_includes_root():
    """pre/post step ratios include the root's update timestep."""
    world = make_world()
    child = StubNode(world, 'fast', priorities={0}, update_timestep=0.01, control_timestep=None)
    root = StubNode(world, 'slow', priorities={0}, update_timestep=0.02, control_timestep=None)
    combined = CombinedWorldNode('combined', [child], root_node=root)

    assert combined._update_ratios['fast'] == 1
    assert combined._update_ratios[ROOT_NODE_KEY] == 2
    assert combined.update_timestep == 0.01

    for _ in range(4):
        combined.pre_environment_step(0.01, priority=0)
        combined.post_environment_step(0.01, priority=0)

    assert child.pre_count == 4 and child.post_count == 4
    assert root.pre_count == 2 and root.post_count == 2


def test_world_env_integration_with_root_node():
    """End-to-end: WorldEnv steps route actions/observations through the root node."""
    world = make_world()
    child = StubNode(world, 'arm', obs_keys=('arm_obs',), priorities={0},
                     action_space=dict_space('j1'), control_timestep=0.01, update_timestep=0.01)
    root = StubNode(world, 'hand', obs_keys=('grip',), priorities={0},
                    action_space=dict_space('grip_act'), control_timestep=0.01, update_timestep=0.01)
    env = WorldEnv(world, CombinedWorldNode('combined', [child], root_node=root))

    env.reset()
    child.post_count = 0
    root.post_count = 0

    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step({
            'arm': {'j1': np.array([0.5], dtype=np.float32)},
            'grip_act': np.array([0.25], dtype=np.float32),
        })

    assert child.post_count == 3 and root.post_count == 3
    assert len(root.actions) == 3 and len(child.actions) == 3
    np.testing.assert_allclose(root.actions[-1]['grip_act'], [0.25])
    assert set(obs.keys()) == {'arm', 'grip'}


# ---------------------------------------------------------------------------
# 6. Direct-return interaction
# ---------------------------------------------------------------------------

def test_sole_root_obs_provider_direct_return_true():
    """Sole root contributor + direct_return=True: unwrapped, any Space type."""
    world = make_world()
    child = StubNode(world, 'obs_only_child')          # no spaces at all
    root = StubNode(world, 'box_root', obs_space=box_space(shape=(3,)))
    combined = CombinedWorldNode('combined', [child], root_node=root, direct_return=True)

    assert combined.observation_space is root.observation_space
    assert isinstance(combined.observation_space, BoxSpace)   # non-Dict space works when sole
    root.value = 7.0
    obs = combined.get_observation()
    assert obs.shape == (3,)
    np.testing.assert_allclose(obs, [7.0, 7.0, 7.0])


def test_sole_root_obs_provider_direct_return_false():
    """Sole root contributor + direct_return=False: wrapped under the reserved key."""
    world = make_world()
    child = StubNode(world, 'obs_only_child')
    root = StubNode(world, 'box_root', obs_space=box_space(shape=(3,)))
    combined = CombinedWorldNode('combined', [child], root_node=root, direct_return=False)

    assert isinstance(combined.observation_space, DictSpace)
    assert set(combined.observation_space.spaces.keys()) == {ROOT_NODE_KEY}
    assert combined.observation_space.spaces[ROOT_NODE_KEY] is root.observation_space
    root.value = 2.0
    obs = combined.get_observation()
    assert set(obs.keys()) == {ROOT_NODE_KEY}
    np.testing.assert_allclose(obs[ROOT_NODE_KEY], [2.0, 2.0, 2.0])


def test_single_child_direct_return_regression():
    """No root: a single child keeps the existing direct-return behaviour."""
    world = make_world()
    child = StubNode(world, 'only_child', obs_space=box_space(shape=(2,)))
    direct = CombinedWorldNode('combined', [child], direct_return=True)
    assert direct.observation_space is child.observation_space
    child.value = 3.0
    np.testing.assert_allclose(direct.get_observation(), [3.0, 3.0])

    wrapped = CombinedWorldNode('combined', [child], direct_return=False)
    assert set(wrapped.observation_space.spaces.keys()) == {'only_child'}
    assert set(wrapped.get_observation().keys()) == {'only_child'}


# ---------------------------------------------------------------------------
# 7. Sole-root action passthrough
# ---------------------------------------------------------------------------

def test_sole_root_box_action_passthrough():
    """Sole-root BoxSpace action: the whole array is forwarded to the root."""
    world = make_world()
    child = StubNode(world, 'obs_only_child')
    root = StubNode(world, 'policy', action_space=box_space(shape=(2,), low=-1.0, high=1.0))
    combined = CombinedWorldNode('combined', [child], root_node=root, direct_return=True)

    assert combined.action_space is root.action_space            # passthrough, not DictSpace
    assert combined._action_node_name_direct == ROOT_NODE_KEY
    assert combined._action_ratios[ROOT_NODE_KEY] == 1

    action = np.array([0.7, -0.7], dtype=np.float32)
    combined.set_next_action(action)
    assert len(root.actions) == 1
    np.testing.assert_allclose(root.actions[0], action)

    # Hold-last: a second call with no action changes still re-delivers at the tick.
    combined.set_next_action(action)
    assert len(root.actions) == 2
    np.testing.assert_allclose(root.actions[1], action)


def test_sole_root_step_none_is_noop_without_action_space():
    """A root without an action space leaves the action channel empty."""
    world = make_world()
    child = StubNode(world, 'obs_only_child')
    root = StubNode(world, 'obs_root', obs_space=box_space(shape=(1,)))
    combined = CombinedWorldNode('combined', [child], root_node=root)
    assert combined.action_space is None
    combined.set_next_action(None)                       # no-op
    with pytest.raises(AssertionError):
        combined.set_next_action(np.zeros(2))


# ---------------------------------------------------------------------------
# 8. Lifecycle dispatch
# ---------------------------------------------------------------------------

def test_root_lifecycle_dispatch_interleaves_by_priority():
    """Root hooks run at their own priorities, interleaved with the children's."""
    world = make_world()
    log: List[Tuple[str, str, int]] = []
    child = StubNode(world, 'child', priorities={0, 10}, log=log)
    root = StubNode(world, 'root', priorities={5}, log=log)
    combined = CombinedWorldNode('combined', [child], root_node=root)

    assert combined.reset_priorities == {0, 5, 10}
    assert combined.get_priority_order('reset') == [10, 5, 0]
    assert combined.get_priority_order('pre_step') == [10, 5, 0]

    for phase in ('reset', 'reload'):
        log.clear()
        for priority in combined.get_priority_order(phase):
            getattr(combined, phase)(priority=priority)
        assert [entry for entry in log if entry[1] == phase] == [
            ('child', phase, 10),
            ('root', phase, 5),
            ('child', phase, 0),
        ]

    log.clear()
    for priority in combined.get_priority_order('after_reload'):
        combined.after_reload(priority=priority)
    assert [entry for entry in log if entry[1] == 'after_reload'] == [
        ('child', 'after_reload', 10),
        ('root', 'after_reload', 5),
        ('child', 'after_reload', 0),
    ]

    log.clear()
    for priority in combined.get_priority_order('after_reset'):
        combined.after_reset(priority=priority)
    assert [entry for entry in log if entry[1] == 'after_reset'] == [
        ('child', 'after_reset', 10),
        ('root', 'after_reset', 5),
        ('child', 'after_reset', 0),
    ]

    log.clear()
    for priority in combined.get_priority_order('pre_step'):
        combined.pre_environment_step(0.01, priority=priority)
    for priority in combined.get_priority_order('post_step'):
        combined.post_environment_step(0.01, priority=priority)
    assert [entry for entry in log if entry[1] == 'pre'] == [
        ('child', 'pre', 10),
        ('root', 'pre', 5),
        ('child', 'pre', 0),
    ]
    assert [entry for entry in log if entry[1] == 'post'] == [
        ('child', 'post', 10),
        ('root', 'post', 5),
        ('child', 'post', 0),
    ]

    # Routing state reset reaches the root's action cache too.
    combined._cached_actions[ROOT_NODE_KEY] = {'grip': np.zeros(1)}
    combined.after_reset(priority=combined._INTERNAL_RESET_PRIORITY)
    assert combined._cached_actions == {}


def test_root_reset_clears_root_action_cache_via_world_env():
    """env.reset() re-arms the root's cached action (hold-last does not leak across resets)."""
    world = make_world()
    child = StubNode(world, 'arm', action_space=dict_space('j1'), priorities={0})
    root = StubNode(world, 'hand', action_space=dict_space('grip'), priorities={0})
    combined = CombinedWorldNode('combined', [child], root_node=root)
    env = WorldEnv(world, combined)

    env.reset()
    combined.set_next_action({'arm': {'j1': np.array([1.0], dtype=np.float32)},
                              'grip': np.array([2.0], dtype=np.float32)})
    assert len(root.actions) == 1

    env.reset()
    assert combined._cached_actions == {}
    assert combined._action_substeps == 0


# ---------------------------------------------------------------------------
# 9. Flat variant
# ---------------------------------------------------------------------------

def test_flat_unclaimed_action_keys_route_to_root():
    """Flat routing sends keys no child claims to the root, obs keys merge flat."""
    world = make_world()
    child = StubNode(world, 'child', obs_keys=('value',), action_space=dict_space('j1'))
    root = StubNode(world, 'root', obs_keys=('grip',), action_space=dict_space('g'))
    combined = FlatCombinedWorldNode('flat', [child], root_node=root)

    assert set(combined.action_space.spaces.keys()) == {'j1', 'g'}
    assert set(combined.observation_space.spaces.keys()) == {'value', 'grip'}

    child.value, root.value = 1.0, 2.0
    obs = combined.get_observation()
    assert set(obs.keys()) == {'value', 'grip'}      # flattened, no nesting / reserved key
    np.testing.assert_allclose(obs['value'], [1.0])
    np.testing.assert_allclose(obs['grip'], [2.0])

    combined.set_next_action({'j1': np.array([0.4], dtype=np.float32),
                              'g': np.array([0.6], dtype=np.float32)})
    assert len(child.actions) == 1 and set(child.actions[0].keys()) == {'j1'}
    assert len(root.actions) == 1 and set(root.actions[0].keys()) == {'g'}
    np.testing.assert_allclose(root.actions[0]['g'], [0.6])

    # A key that is not even part of the aggregate action space still belongs to
    # the root (it is claimed by no child).
    combined.set_next_action({'j1': np.array([0.4], dtype=np.float32),
                              'unmapped': np.array([0.9], dtype=np.float32)})
    assert set(root.actions[-1].keys()) == {'unmapped'}
    np.testing.assert_allclose(root.actions[-1]['unmapped'], [0.9])


def test_flat_unclaimed_action_key_without_root_raises():
    """Without a root node, unclaimed flat action keys are still rejected."""
    world = make_world()
    child = StubNode(world, 'child', obs_keys=('value',), action_space=dict_space('j1'))
    combined = FlatCombinedWorldNode('flat', [child])

    # Claimed keys keep working.
    combined.set_next_action({'j1': np.array([0.4], dtype=np.float32)})
    assert len(child.actions) == 1

    with pytest.raises(ValueError, match="not claimed"):
        combined.set_next_action({'j1': np.array([0.4], dtype=np.float32),
                                  'junk': np.array([0.1], dtype=np.float32)})


def test_flat_root_space_collision_raises():
    """Flat root keys overlapping a child key raise the existing overlap error."""
    world = make_world()
    child = StubNode(world, 'child', obs_keys=('value',))
    root = StubNode(world, 'root', obs_keys=('value',))
    with pytest.raises(ValueError, match="Overlapping key 'value'"):
        FlatCombinedWorldNode('flat', [child], root_node=root)


def test_flat_sole_root_space_and_data_unwrapped():
    """A flat root that is the only contributor returns its space/data unwrapped."""
    world = make_world()
    child = StubNode(world, 'child')                       # no spaces
    root = StubNode(world, 'root', obs_space=box_space(shape=(2,)))
    combined = FlatCombinedWorldNode('flat', [child], root_node=root)
    assert combined.observation_space is root.observation_space
    root.value = 6.0
    np.testing.assert_allclose(combined.get_observation(), [6.0, 6.0])


def test_flat_root_reward_and_lifecycle_included():
    """Flat variant: root joins lifecycles, rewards and signals."""
    world = make_world()
    log: List[Tuple[str, str, int]] = []
    child = StubNode(world, 'child', priorities={0}, log=log, reward=1.0)
    root = StubNode(world, 'root', priorities={0}, log=log, reward=0.5, terminate=True, truncate=True)
    combined = FlatCombinedWorldNode('flat', [child], root_node=root)

    for priority in combined.get_priority_order('reset'):
        combined.reset(priority=priority)
    assert [entry for entry in log if entry[1] == 'reset'] == [
        ('child', 'reset', 0),
        ('root', 'reset', 0),
    ]

    assert combined.has_reward and combined.has_termination_signal and combined.has_truncation_signal
    assert combined.get_reward() == 1.5
    assert combined.get_termination() is True
    assert combined.get_truncation() is True


# ---------------------------------------------------------------------------
# 10. Reward / signals / info / render aggregation
# ---------------------------------------------------------------------------

def test_root_reward_termination_truncation_aggregation():
    """Root reward is summed; root termination/truncation flip the combined flags."""
    world = make_world()
    child = StubNode(world, 'child', reward=1.5)
    root = StubNode(world, 'root', reward=2.5, terminate=True)
    combined = CombinedWorldNode('combined', [child], root_node=root)

    assert combined.has_reward and combined.has_termination_signal
    assert combined.get_reward() == 4.0
    assert combined.get_termination() is True

    assert not combined.has_truncation_signal
    truncating_root = StubNode(world, 'trunc_root', truncate=True)
    combined_trunc = CombinedWorldNode('combined', [child], root_node=truncating_root)
    assert combined_trunc.has_truncation_signal
    assert combined_trunc.get_truncation() is True

    # A root without a reward must not change the aggregate behaviour.
    plain_root = StubNode(world, 'plain_root')
    combined_plain = CombinedWorldNode('combined', [child], root_node=plain_root)
    assert combined_plain.get_reward() == 1.5


def test_root_info_and_render_use_display_name():
    """Info / render dicts key the root by its own name, never by the reserved key."""
    world = make_world()
    child = StubNode(world, 'child', info_value={'x': 1}, render_mode='rgb_array')
    root = StubNode(world, 'root', info_value={'y': 2}, render_mode='rgb_array')
    combined = CombinedWorldNode('combined', [child], root_node=root, render_mode='dict')

    info = combined.get_info()
    assert info == {'child': {'x': 1}, 'root': {'y': 2}}
    assert ROOT_NODE_KEY not in info

    rendered = combined.render()
    assert set(rendered.keys()) == {'child', 'root'}
    assert ROOT_NODE_KEY not in rendered
    assert set(combined._renderable_nodes) == {child, root}


def test_sole_renderable_root_uses_its_own_render():
    """With a single renderable node (the root), the root's render is passed through."""
    world = make_world()
    child = StubNode(world, 'child')
    root = StubNode(world, 'root', render_mode='rgb_array')
    combined = CombinedWorldNode('combined', [child], root_node=root, render_mode='auto')
    assert combined.can_render
    frame = combined.render()
    assert isinstance(frame, np.ndarray) and frame.shape == (2, 2, 3)
