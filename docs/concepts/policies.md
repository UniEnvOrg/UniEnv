# Policies

`unienv_interface.policy` is the agent side of the loop. Like environments, policies come in two styles:

- `Policy`: a stateful object that keeps its own memory (RNG, recurrent state, history)
- `FuncPolicy`: a functional interface that takes its state in and returns it explicitly

A policy declares its own spaces, the same kind an environment does (`observation_space`, `action_space`, optional `context_space`, plus `backend`, `device` and `batch_size`): what it reads and what it outputs. It is defined independently of any one environment and runs with every source whose data fits those spaces: a simulator, a real robot, or a dataset replayed through the same schema.

## `Policy`

A policy's state changes only when it is told what happened: `reset` starts episodes, and `observe_transition` feeds it each executed action and the observation that followed. `get_action` reads that state and does not change it, so it takes no observation. Subclass `Policy` and implement the three of them:

| Method | Called | Default |
| --- | --- | --- |
| `reset(context, observation, *, seed=None, mask=None)` | after `env.reset`; with `mask`, only the masked batch entries start over | **required** |
| `get_action()` | whenever an action is needed; reads the state | **required** |
| `observe_transition(action_taken, new_observation)` | after `env.step` | **required** |
| `visualize_action(environment_render, observation, action)` | to draw the action on a render frame | returns the frame |
| `close()` | when done | no-op |

The context, as for environments, only comes with `reset`: it is fixed for an episode, so a policy that needs it keeps it from there.

`sample_action()`, `sample_observation()` and `sample_context()` draw from the declared spaces and advance `policy.rng`.

```python
import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace
from unienv_interface.policy import Policy


class RandomPolicy(Policy):
    """Uniform random actions from its own action space."""

    backend = B
    device = None
    observation_space = BoxSpace(B, -10.0, 10.0, np.float32, shape=(3,))
    action_space = BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,))

    def __init__(self, seed=0):
        self.rng = B.random.random_number_generator(seed)

    def reset(self, context, observation, *, seed=None, mask=None, **kwargs):
        pass  # nothing to remember

    def observe_transition(self, action_taken, new_observation):
        pass

    def get_action(self, **kwargs):
        return self.sample_action()


policy = RandomPolicy()
policy.reset(None, np.zeros(3, np.float32))
policy.get_action()  # array of shape (2,), within [-1, 1]
```

The spaces belong to the policy: they describe its inputs and outputs, not a particular environment.

## The Control Loop

The same loop drives a policy against any `Env`:

```python
context, obs, info = env.reset(seed=0)
policy.reset(context, obs, seed=0)
for _ in range(100):
    action = policy.get_action()
    obs, reward, terminated, truncated, info = env.step(action)
    policy.observe_transition(action, obs)
```

Batching follows the environment's convention: with `batch_size` set, the spaces include the leading batch dimension and every context, observation and action is a full batch. To restart only the finished entries of a batched environment, merge what `env.reset(mask=done)` returns back into the full batch and give the policy that full batch with the same mask; it starts over only the masked entries and keeps its state for the others:

```python
done = terminated | truncated
if done.any():
    new_context, new_obs, info = env.reset(mask=done)  # only the finished entries
    context = env.update_context_post_reset(context, new_context, done)
    obs = env.update_observation_post_reset(obs, new_obs, done)
    policy.reset(context, obs, mask=done)
```

## `FuncPolicy`

A `FuncPolicy` keeps no state on the object. Each method takes the state and returns the next one, which fits `jax.jit`, vectorized rollouts, and checkpointing. Its batching is set by a `batch_size` attribute, as on `FuncEnv`.

- `initial(context, observation, *, seed=None) -> state` (**required**)
- `get_action(state) -> (state, action)` (**required**): the returned state may only differ in bookkeeping, such as an advanced RNG key
- `observe_transition(state, action_taken, new_observation) -> state` (**required**)
- `reset(state, context, observation, *, seed=None, mask=None) -> state` (**required**): full batch in, as for `Policy`
- `close(state)`

```python
from unienv_interface.policy import FuncPolicy


class SmoothedPolicy(FuncPolicy):
    """Moves the action a fraction of the way from the last executed action toward a target from the observation."""

    backend = B
    device = None
    observation_space = BoxSpace(B, -10.0, 10.0, np.float32, shape=(3,))
    action_space = BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,))

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def initial(self, context, observation, *, seed=None, **kwargs):
        return {"last": np.zeros(2, np.float32), "target": np.clip(observation[:2], -1.0, 1.0)}

    def reset(self, state, context, observation, *, seed=None, mask=None, **kwargs):
        return self.initial(context, observation)  # unbatched: no mask

    def observe_transition(self, state, action_taken, new_observation):
        return {"last": action_taken, "target": np.clip(new_observation[:2], -1.0, 1.0)}

    def get_action(self, state, **kwargs):
        return state, state["last"] + self.alpha * (state["target"] - state["last"])


func_policy = SmoothedPolicy()
obs = np.array([1.0, -1.0, 0.0], np.float32)
state = func_policy.initial(None, obs)
state, action = func_policy.get_action(state)  # [0.5, -0.5]
state = func_policy.observe_transition(state, action, obs)
state, action = func_policy.get_action(state)  # [0.75, -0.75]
```

## Running A `FuncPolicy` As A `Policy`

`FuncPolicyBasedPolicy` holds the state for you, so a functional policy plugs into the stateful loop above (with the functional policy's batch size and spaces):

```python
from unienv_interface.policy import FuncPolicyBasedPolicy

policy = FuncPolicyBasedPolicy(func_policy, seed=0)
policy.reset(None, obs)          # calls func_policy.initial the first time
policy.get_action()              # [0.5, -0.5]
```

## Wrappers

`PolicyWrapper` and `FuncPolicyWrapper` forward every call and attribute to the wrapped policy, the same way environment wrappers do. Override only what you change:

```python
from unienv_interface.policy import PolicyWrapper


class ScaledActions(PolicyWrapper):
    def __init__(self, policy, scale):
        super().__init__(policy)
        self.scale = scale

    def get_action(self, **kwargs):
        return self.scale * self.policy.get_action(**kwargs)


slow = ScaledActions(FuncPolicyBasedPolicy(SmoothedPolicy()), 0.5)
slow.reset(None, obs)
slow.get_action()                # [0.25, -0.25]
slow.unwrapped                   # the FuncPolicyBasedPolicy underneath
```

`unwrapped`, `prev_wrapper_layer` and `get_wrapper_attr` / `set_wrapper_attr` walk the wrapper stack.
