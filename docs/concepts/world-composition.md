# World Composition

UniEnv can build an environment from smaller pieces instead of forcing everything into one monolithic class.

The relevant abstractions are:

- `World`: the shared simulator or runtime
- `WorldNode`: one subsystem layered on top of that world
- `WorldEnv`: an `Env` that composes a world and one or more nodes

## `World`

`World` owns the underlying runtime state.

Typical responsibilities:

- stepping the simulator or real-time clock
- resetting or reloading the runtime
- exposing a backend, device, and optional batch size
- coordinating timestep information

UniEnv includes `RealWorld`, which measures wall-clock elapsed time and is useful for real-time control loops.

## `WorldNode`

A `WorldNode` represents one logical concern inside the world, such as:

- a robot controller
- a sensor
- a reward model
- a termination rule
- an object or scene component

Nodes expose some combination of:

- `action_space`
- `observation_space`
- `context_space`
- reward, termination, and truncation signals
- rendering output

## Lifecycle

`WorldEnv` drives a staged lifecycle around the world and its nodes.

Reset flow:

1. `World.reset()`
2. node `reset(...)`
3. `World.after_reset()`
4. node `after_reset(...)`
5. read context, observation, and info

Reload flow:

1. `World.reload()`
2. node `reload(...)`
3. `World.after_reload()`
4. node `after_reload(...)`
5. read context, observation, and info

Step flow:

1. node receives the next action
2. node `pre_environment_step(...)`
3. `World.step()`
4. node `post_environment_step(...)`
5. read observation, reward, done flags, and info

## Priorities

Nodes opt into lifecycle callbacks through priority sets such as:

- `reset_priorities`
- `reload_priorities`
- `after_reset_priorities`
- `after_reload_priorities`
- `pre_environment_step_priorities`
- `post_environment_step_priorities`

Higher priorities run earlier.

This lets you coordinate scene setup order, controller updates, reward computation, and sensor refreshes without hard-coding everything into one call site.

## Combined Nodes

If you pass multiple nodes to `WorldEnv`, UniEnv wraps them in a `CombinedWorldNode`.

That gives you:

- one environment surface
- aggregated spaces and signals
- nested node lookup helpers such as `get_node(...)`

## When To Use World Composition

Use the world/node stack when:

- your environment is naturally composed from reusable subsystems
- you need clear ordering around scene construction and runtime updates
- you want one simulator world to power multiple observation or control components

If your environment is simple and self-contained, a direct `Env` or `FuncEnv` can still be the better fit.
