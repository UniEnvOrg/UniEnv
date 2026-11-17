# Policies

`unienv_policy` is the agent side of the loop. Like environments, policies come in two styles:

- `Policy`: a stateful object that keeps its own memory (RNG, recurrent state, history)
- `FuncPolicy`: a functional interface that takes its state in and returns it explicitly

A policy declares its own spaces, the same kind an environment does (`observation_space`, `action_space`, optional `context_space`, plus `backend`, `device` and `batch_size`): what it reads and what it outputs. It is defined independently of any one environment and runs with every source whose data fits those spaces: a simulator, a real robot, or a dataset replayed through the same schema.

## `Policy`

Subclass `Policy` and implement `compute_action`. The other methods have do-nothing defaults:

| Method | Called | Default |
| --- | --- | --- |
| `reset(context, observation, *, seed=None, mask=None)` | after `env.reset`; with `mask`, only the masked batch entries were reset | no-op |
| `compute_action(observation)` | every step | **required** |
| `observe_transition(action_taken, new_observation)` | after `env.step`, for policies that track history | no-op |
| `visualize_action(environment_render, observation, action)` | to draw the action on a render frame | returns the frame |
| `close()` | when done | no-op |

`sample_action()`, `sample_observation()` and `sample_context()` draw from the declared spaces and advance `policy.rng`.

```python
import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace
from unienv_policy import Policy


class RandomPolicy(Policy):
    """Uniform random actions from its own action space."""

    backend = B
    device = None
    observation_space = BoxSpace(B, -10.0, 10.0, np.float32, shape=(3,))
    action_space = BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,))

    def __init__(self, seed=0):
        self.rng = B.random.random_number_generator(seed)

    def compute_action(self, observation, **kwargs):
        return self.sample_action()


policy = RandomPolicy()
policy.compute_action(np.zeros(3, np.float32))  # array of shape (2,), within [-1, 1]
```

The spaces belong to the policy: they describe its inputs and outputs, not a particular environment.

## The Control Loop

The same loop drives a policy against any `Env`:

```python
context, obs, info = env.reset(seed=0)
policy.reset(context, obs, seed=0)
for _ in range(100):
    action = policy.compute_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    policy.observe_transition(action, obs)
```

With a batched environment, reset only the finished entries: `env.reset(mask=done)` returns their context and observation, and `policy.reset(context, obs, mask=done)` resets the same entries.

## `FuncPolicy`

A `FuncPolicy` keeps no state on the object. Each method takes the state and returns the next one, which fits `jax.jit`, vectorized rollouts, and checkpointing:

- `initial(context, observation, *, batch_size, seed=None) -> state` (**required**)
- `compute_action(state, observation) -> (state, action)` (**required**)
- `reset(state, context, observation, *, seed=None, mask=None) -> state`
- `observe_transition(state, action_taken, new_observation) -> state`
- `close(state)`

```python
from unienv_policy import FuncPolicy


class SmoothedPolicy(FuncPolicy):
    """Moves the action a fraction of the way toward a target each step; the state is the last action."""

    backend = B
    device = None
    observation_space = BoxSpace(B, -10.0, 10.0, np.float32, shape=(3,))
    action_space = BoxSpace(B, -1.0, 1.0, np.float32, shape=(2,))

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def initial(self, context, observation, *, batch_size, seed=None, **kwargs):
        return np.zeros(2, np.float32)

    def compute_action(self, state, observation, **kwargs):
        target = np.clip(observation[:2], -1.0, 1.0)
        action = state + self.alpha * (target - state)
        return action, action


func_policy = SmoothedPolicy()
obs = np.array([1.0, -1.0, 0.0], np.float32)
state = func_policy.initial(None, obs, batch_size=None)
state, action = func_policy.compute_action(state, obs)  # [0.5, -0.5]
state, action = func_policy.compute_action(state, obs)  # [0.75, -0.75]
```

## Running A `FuncPolicy` As A `Policy`

`FuncPolicyBasedPolicy` holds the state for you, so a functional policy plugs into the stateful loop above:

```python
from unienv_policy import FuncPolicyBasedPolicy

policy = FuncPolicyBasedPolicy(func_policy, seed=0)
policy.reset(None, obs)          # calls func_policy.initial the first time
policy.compute_action(obs)       # [0.5, -0.5]
```

## Wrappers

`PolicyWrapper` and `FuncPolicyWrapper` forward every call and attribute to the wrapped policy, the same way environment wrappers do. Override only what you change:

```python
from unienv_policy import PolicyWrapper


class ScaledActions(PolicyWrapper):
    def __init__(self, policy, scale):
        super().__init__(policy)
        self.scale = scale

    def compute_action(self, observation, **kwargs):
        return self.scale * self.policy.compute_action(observation, **kwargs)


slow = ScaledActions(FuncPolicyBasedPolicy(SmoothedPolicy()), 0.5)
slow.reset(None, obs)
slow.compute_action(obs)         # [0.25, -0.25]
slow.unwrapped                   # the FuncPolicyBasedPolicy underneath
```

`unwrapped`, `prev_wrapper_layer` and `get_wrapper_attr` / `set_wrapper_attr` walk the wrapper stack.
