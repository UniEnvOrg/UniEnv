"""Reusable behavioural contract for :class:`unienv_interface.env_base.env.Env` implementations.

``check_env_contract(make_env)`` builds fresh environments with ``make_env``
and asserts the invariants every ``Env`` (batched or not, wrapped or not)
must uphold:

* reset/step outputs are contained in the declared spaces, which for batched
  envs already carry the leading batch axis (the batched-space invariant);
* reward / terminated / truncated are scalars (``float``/``bool``) for
  unbatched envs and backend arrays of shape ``(batch_size,)`` for batched
  envs (boolean dtype for the flags);
* a masked reset of a batched env returns only the selected rows, and
  ``update_observation_post_reset`` / ``update_context_post_reset`` merge
  them back into the full batch;
* the same seed gives the same reset/step outputs;
* ``sample_action`` returns actions in ``action_space`` and advances ``rng``;
* ``unwrapped`` / ``prev_wrapper_layer`` form a chain ending in the base env;
* ``get/has/set_wrapper_attr`` reach the base env through every layer and
  ``get_wrapper_attr`` raises ``AttributeError`` for unknown names;
* the context manager closes the env.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from unienv_interface.env_base.env import Env
from unienv_interface.space import Space
from unienv_interface.space.space_utils import batch_utils as sbu
from tests_common.trees import assert_tree_equal, to_numpy

MakeEnv = Callable[[], Env]


def _leaves(data: Any) -> List[Any]:
    if isinstance(data, dict):
        return [leaf for value in data.values() for leaf in _leaves(value)]
    if isinstance(data, (tuple, list)):
        return [leaf for value in data for leaf in _leaves(value)]
    return [data]


def _assert_in_space(space: Optional[Space], value: Any, what: str) -> None:
    if space is None:
        assert value is None, f"{what}: space is None but got {type(value).__name__}"
        return
    assert space.contains(value), f"{what} is not contained in its space {space}"


def _assert_space_batch(space: Optional[Space], batch_size: Optional[int], what: str) -> None:
    if space is None or batch_size is None:
        return
    space_batch = sbu.batch_size(space)
    assert space_batch == batch_size, (
        f"{what} must already include the batch axis ({batch_size}), got leading size {space_batch}")


def _assert_step_signals(env: Env, reward: Any, terminated: Any, truncated: Any) -> None:
    if env.batch_size is None:
        assert np.ndim(to_numpy(reward)) == 0, f"unbatched reward must be a scalar, got {reward!r}"
        float(reward)
        for name, flag in (("terminated", terminated), ("truncated", truncated)):
            assert isinstance(flag, (bool, np.bool_)), f"unbatched {name} must be a bool, got {type(flag)}"
        return
    backend = env.backend
    for name, value in (("reward", reward), ("terminated", terminated), ("truncated", truncated)):
        assert backend.is_backendarray(value), f"batched {name} must be a {backend.simplified_name} array"
        assert tuple(value.shape) == (env.batch_size,), f"batched {name} shape {tuple(value.shape)}"
    for name, flag in (("terminated", terminated), ("truncated", truncated)):
        assert flag.dtype == backend.default_boolean_dtype, f"batched {name} dtype {flag.dtype}"


def _closed_probe(env: Env) -> Optional[Callable[[], bool]]:
    """Return a callable telling whether the stub(s) underneath ``env`` were closed, if detectable."""
    from unienv_interface.env_base.funcenv import FuncEnvBasedEnv
    from unienv_interface.env_base.vec_env import SyncVecEnv

    base = env.unwrapped
    if isinstance(base, SyncVecEnv):
        subs = list(base.envs)
        probes = [_closed_probe(sub) for sub in subs]
        if all(p is not None for p in probes):
            return lambda: all(p() for p in probes)
        return None
    if hasattr(base, "closed") and isinstance(getattr(base, "closed"), int):
        start = base.closed
        return lambda: base.closed > start
    if isinstance(base, FuncEnvBasedEnv) and hasattr(base.func_env.unwrapped, "calls"):
        calls = base.func_env.unwrapped.calls
        start = len(calls)
        return lambda: ("close",) in calls[start:]
    return None


def _mask_for(env: Env) -> Any:
    selected = [i % 2 == 0 for i in range(env.batch_size)]
    if env.batch_size > 1 and all(selected):
        selected[-1] = False
    return env.backend.asarray(selected, dtype=env.backend.default_boolean_dtype, device=env.device)


def _rows(env: Env, space: Space, data: Any, mask: Any) -> Any:
    return sbu.get_at(space, data, mask)


def check_spaces(env: Env) -> None:
    assert env.action_space is not None and env.observation_space is not None
    _assert_space_batch(env.action_space, env.batch_size, "action_space")
    _assert_space_batch(env.observation_space, env.batch_size, "observation_space")
    _assert_space_batch(env.context_space, env.batch_size, "context_space")


def check_reset_step(env: Env, *, seed: int = 0, n_steps: int = 3) -> None:
    context, obs, info = env.reset(seed=seed)
    _assert_in_space(env.context_space, context, "reset context")
    _assert_in_space(env.observation_space, obs, "reset observation")
    assert isinstance(info, dict), f"reset info must be a dict, got {type(info)}"
    for _ in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(env.sample_action())
        _assert_in_space(env.observation_space, obs, "step observation")
        _assert_step_signals(env, reward, terminated, truncated)
        assert isinstance(info, dict), f"step info must be a dict, got {type(info)}"
    # resetting again after stepping must work
    context, obs, info = env.reset(seed=seed + 1)
    _assert_in_space(env.observation_space, obs, "second reset observation")


def check_masked_reset(env: Env, *, seed: int = 0) -> None:
    if env.batch_size is None:
        return
    context, obs, _ = env.reset(seed=seed)
    obs, *_ = env.step(env.sample_action())
    mask = _mask_for(env)
    n_selected = int(np.asarray(to_numpy(mask)).sum())
    new_context, new_obs, info = env.reset(mask=mask, seed=seed + 3)
    assert isinstance(info, dict)
    for leaf in _leaves(new_obs):
        assert tuple(leaf.shape)[0] == n_selected, (
            f"masked reset must return only the {n_selected} selected rows, got leading size {leaf.shape[0]}")
    merged = env.update_observation_post_reset(obs, new_obs, mask)
    _assert_in_space(env.observation_space, merged, "merged observation")
    assert_tree_equal(_rows(env, env.observation_space, merged, mask), new_obs)
    not_mask = env.backend.logical_not(mask)
    assert_tree_equal(_rows(env, env.observation_space, merged, not_mask),
                      _rows(env, env.observation_space, obs, not_mask))
    if env.context_space is not None:
        for leaf in _leaves(new_context):
            assert tuple(leaf.shape)[0] == n_selected
        merged_context = env.update_context_post_reset(context, new_context, mask)
        _assert_in_space(env.context_space, merged_context, "merged context")
        assert_tree_equal(_rows(env, env.context_space, merged_context, mask), new_context)
    # the env keeps working after a partial reset
    obs, reward, terminated, truncated, _ = env.step(env.sample_action())
    _assert_in_space(env.observation_space, obs, "step observation after masked reset")
    _assert_step_signals(env, reward, terminated, truncated)


def check_determinism(make_env: MakeEnv, *, seed: int = 0, n_steps: int = 3) -> None:
    env_a, env_b = make_env(), make_env()
    try:
        assert_tree_equal(env_a.reset(seed=seed), env_b.reset(seed=seed))
        for _ in range(n_steps):
            action = env_a.sample_action()
            out_a = env_a.step(action)
            out_b = env_b.step(action)
            assert_tree_equal(out_a[:4], out_b[:4])
    finally:
        env_a.close()
        env_b.close()


def check_sample_action(env: Env) -> None:
    actions = [env.sample_action() for _ in range(3)]
    for action in actions:
        _assert_in_space(env.action_space, action, "sampled action")
    flat = [np.asarray(to_numpy(leaf), dtype=np.float64).ravel() for a in actions for leaf in _leaves(a)]
    assert not all(np.array_equal(flat[0], f) for f in flat[1:]), "sample_action must advance the rng"


def check_wrapper_layers(env: Env) -> None:
    base = env.unwrapped
    assert isinstance(base, Env)
    assert base.unwrapped is base
    assert base.prev_wrapper_layer is None
    layer, depth = env, 0
    while layer.prev_wrapper_layer is not None:
        assert layer.unwrapped is base
        layer = layer.prev_wrapper_layer
        depth += 1
        assert depth < 100
    assert layer is base

    marker = "_env_contract_marker"
    assert not env.has_wrapper_attr(marker)
    try:
        env.get_wrapper_attr(marker)
    except AttributeError:
        pass
    else:
        raise AssertionError("get_wrapper_attr must raise AttributeError for unknown names")
    setattr(base, marker, 1)
    try:
        assert env.has_wrapper_attr(marker)
        assert env.get_wrapper_attr(marker) == 1
        env.set_wrapper_attr(marker, 2)
        assert getattr(base, marker) == 2, "set_wrapper_attr must update the layer that defines the attribute"
        assert env.get_wrapper_attr(marker) == 2
    finally:
        delattr(base, marker)


def check_context_manager_closes(env: Env) -> None:
    probe = _closed_probe(env)
    with env as entered:
        assert entered is env
    if probe is not None:
        assert probe(), "leaving the context manager must close the environment"


def check_env_contract(
    make_env: MakeEnv,
    *,
    seed: int = 0,
    n_steps: int = 3,
    determinism: bool = True,
    wrapper_attrs: bool = True,
) -> None:
    """Run every contract check on fresh instances created by ``make_env``."""
    env = make_env()
    try:
        check_spaces(env)
        check_reset_step(env, seed=seed, n_steps=n_steps)
        check_masked_reset(env, seed=seed)
        check_sample_action(env)
        if wrapper_attrs:
            check_wrapper_layers(env)
    finally:
        env.close()
    if determinism:
        check_determinism(make_env, seed=seed, n_steps=n_steps)
    check_context_manager_closes(make_env())
