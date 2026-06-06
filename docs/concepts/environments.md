# Environments

UniEnv exposes two environment styles:

- `Env`: a stateful object that owns its runtime state internally
- `FuncEnv`: a functional interface that passes state explicitly

Both carry the same high-level contract: typed action, observation, and optional context spaces plus the standard `reset`, `step`, and `render` lifecycle.

## `Env`

Use `Env` when you want the familiar environment shape:

- `reset(...) -> context, observation, info`
- `step(action) -> observation, reward, terminated, truncated, info`
- `render()`

This is the most direct fit for interactive control loops, evaluation harnesses, and compatibility layers that expect a stateful object.

Important properties exposed by `Env` implementations:

- `action_space`
- `observation_space`
- `context_space`
- `backend`
- `device`
- `batch_size`

`Env` also includes convenience helpers such as `sample_action()` and `sample_observation()`, which draw from the declared spaces while updating the environment RNG.

## `FuncEnv`

Use `FuncEnv` when explicit state passing is more important than object-owned mutable state.

The core methods are:

- `initial(...) -> state, context, observation, info`
- `reset(state, ...) -> state, context, observation, info`
- `step(state, action) -> state, observation, reward, terminated, truncated, info`

This style is useful when you want:

- JAX-friendly execution patterns
- easier functional composition
- tighter control over state snapshots, rollouts, or transformations

## Bridging The Two

`FuncEnvBasedEnv` adapts a `FuncEnv` into a stateful `Env`.

That means you can:

- implement the hard logic once in functional form
- expose a familiar object-oriented environment API to downstream code
- keep wrappers and tooling that expect `Env` unchanged

## Batched Environments

Both environment styles can represent batched execution through `batch_size`.

When `batch_size` is set:

- actions, observations, rewards, and done flags are expected to carry a batch dimension
- `reset(mask=...)` can reset only selected batch elements
- helper methods such as `update_observation_post_reset` merge masked resets back into a full batch

### Space invariant for batched environments

A key contract that all `Env` and `FuncEnv` implementations share:

> **If an environment is batched, then its `observation_space`, `action_space`,
> and `context_space` (when present) already describe batched values.**

Concretely:

| `batch_size` | Meaning | Spaces |
|---|---|---|
| `None` | Unbatched – single instance, no leading batch axis. | Describe single-instance values. |
| `N` (including `1`) | Batched – leading batch dimension of size `N`. | **Already include** the leading batch axis of size `N`. |

`batch_size == 1` is **still a batched environment**. Its spaces carry a leading dimension of size 1; they are *not* the same as the unbatched spaces.

#### Why this matters

Code that builds source/target spaces from the env – replay-buffer builders, data transformations, wrappers that unbatch per-slot data – must use the env-side spaces **as-is**. Applying an additional `batch_space(...)` on top of an already-batched env space will double-batch and produce incorrect shapes.

Common patterns:

- **Correct**: `source_space = env.observation_space` (works for both batched and unbatched envs).
- **Incorrect**: `source_space = batch_space(env.observation_space, env.batch_size)` when the env is already batched – this adds a second batch dimension.
- **Unbatching per-slot data**: use `get_at(env.observation_space, data, i)` to slice slot `i` out of the batched data; the resulting per-slot space is the single-instance space obtained by *unbatching* `env.observation_space`, not by re-batching it.

## Wrappers

UniEnv wrappers work at the `Env` layer. They can change:

- the action interface
- the observation and context interface
- backend placement
- episode length
- rendering and video export behavior

See [Wrappers and Transformations](../guides/wrappers-and-transformations.md) for the main wrapper stack.

## When To Use Which

Choose `Env` if:

- you want the most familiar interface
- your simulator already manages mutable runtime state
- you are wrapping an existing imperative system

Choose `FuncEnv` if:

- you want explicit state passing
- you care about purely functional rollout logic
- you want the same core logic to be easier to test, checkpoint, or stage

Use `FuncEnvBasedEnv` if you want both.
