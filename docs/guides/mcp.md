# MCP Servers

`unienv_mcp` serves UniEnv environments and policies over the [Model Context Protocol](https://modelcontextprotocol.io),
so LLM agents (Claude, IDE assistants, any MCP host) can look at an environment, act in it, and run policies as skills.

```bash
pip install "unienv[mcp]"
```

## Serving

```python
from unienv_mcp import UniEnvMCPServer

server = UniEnvMCPServer("lab")
server.add_env("sim", make_env, dt=0.05)          # a factory: one env per opened instance
server.add_policy("pick", make_pick_policy)       # a policy, run closed-loop as a skill
server.run("stdio")                               # or server.run("http", port=8000)
```

Or from the command line, with `module:attribute` targets (an `Env`, a function building one, or a whole
`UniEnvMCPServer` configured in Python):

```bash
unienv-mcp serve mypkg.envs:make_env --name sim --policy mypkg.policies:make_pick=pick
unienv-mcp serve mypkg.envs:make_env --transport http --port 8000
```

A local MCP host (e.g. Claude Code or Claude Desktop) starts the stdio server itself: register the command above
in its MCP configuration. HTTP servers bind `127.0.0.1` by default.

## Tools

Each environment `E` gets:

| Tool | |
|---|---|
| `E.describe` | the action as a JSON Schema, the observation and context as results show them, the camera names, the step duration, the policies that run on it, and the open instances |
| `E.open` | an instance id (a new env for factories; the shared instance for objects) |
| `E.reset` | start an episode: the context (e.g. the task) and the first observation |
| `E.step` | apply an action, `repeat` times (stopping when the episode ends): rewards, `terminated` / `truncated`, the last observation |
| `E.observe` | the current state again without acting: the context, the latest observation, the episode's progress, any run in progress |
| `E.render` | the env's render as an image |
| `E.close` | close an instance |

Each policy `P` gets `P.run` and `P.suggest`, and with any policy registered there are `runs.status` and
`runs.cancel`:

- **`P.run`** runs the policy closed-loop on an env instance, from its current state, until the episode ends or
  `max_steps`. This is how an agent uses policies as skills: it plans (`"pick up the bowl"`, then `"open the
  drawer"`), and each step of the plan is a run. `task` replaces the env's context for the policy, which is how
  language-conditioned policies (e.g. [unienv-lerobot](https://github.com/UniEnvOrg/unienv-lerobot)'s VLA
  policies) get their instruction. The result has the outcome (steps, success, reward), keyframes and the final
  observation. Runs report progress while they run and stop when the call is cancelled; with `wait=false` the
  call returns a run id at once, for `runs.status` and `runs.cancel`.
- **`P.suggest`** returns the action the policy would take now, without executing it; the agent may apply it with
  `E.step`, possibly edited.

The step duration passed to policies (`observe_transition`'s `dt`) is `add_env`'s `dt`, or the env's
`control_timestep`.

An agent whose context was compacted recovers with `E.describe`, which lists the open instances (with their episode
progress and any running policy), and `E.observe`, which shows an instance's latest observation again. `observe`
does not call the environment, so it is safe on a real robot and during a policy run.

## Instances

MCP requests may be stateless, so state lives in **instances** addressed by id:

- An env registered as a **factory** is built per `E.open`, at most `max_instances` at a time, and closed by
  `E.close` or after `idle_timeout` seconds unused (default 10 minutes). One is built at registration, to read
  the spaces for the tool schemas, and handed to the first `open`.
- An env registered as an **object** is one shared instance whose id is its name; every client acts on it, and
  the server never closes it.
- Each instance runs on its own thread, like hosted objects in [remote execution](remote-execution.md), so
  simulators that need thread affinity always see the same thread. A policy factory is built once per instance;
  a shared policy object runs one run at a time.

There is no access control between clients: anyone connected may call any tool.

## Observations and actions

Actions are JSON, checked against tool schemas generated from the action space
(`unienv_interface.space.space_utils.space_to_json_schema`):

- vectors whose space metadata has `names` are objects keyed by those names (`{"dx": 0.5, "dy": 0.0}`), with the
  bounds, and `units` in the descriptions;
- an action outside the bounds is a tool error listing the violations, which the model can correct; with
  `clip_actions=True` it is clipped instead.

Observations come back as structured JSON plus images:

- leaves whose metadata says `kind="image"` (or unannotated `uint8` boxes shaped like pictures) are PNG images,
  upright and in RGB (following the `layout`, `format` and `flipped` metadata), downscaled to at most 512 pixels a
  side; the JSON refers to them by number;
- `kind="depth"` leaves are gray images, with the depth range in the JSON;
- arrays with more than 256 entries are summarized (shape, min, max, mean); other values are listed, floats
  rounded to 4 decimals.

Images are what fills a model's context fastest, so their number is bounded:

- `reset`, `step`, `observe` and `run` take `images`: `"all"`, `"none"`, or a list of camera paths
  (`["front", "cameras/wrist"]`; `describe` lists them). Images not shown are still listed in the JSON, with their
  size and why they were omitted;
- `ContentOptions` sets the defaults: `cameras` (shown when a call doesn't say; default all), `max_images` per
  result (default 8; a run's keyframes count too), `batch_images` (`"first"`: only the first env of a batched
  environment, the default; `"all"`), `max_image_side`, `max_values` and `digits`:

```python
UniEnvMCPServer("lab", content=ContentOptions(cameras=["front"], max_images=4))
```

## Real systems

Pass `physical=True` to `add_env` for a real robot: its acting tools (`open`, `reset`, `step`, and policies'
`run`) are annotated as destructive and open-world, so MCP hosts ask before calling them. Action bounds always
apply; `max_steps_per_call` bounds `step`'s `repeat`, and `add_policy`'s `max_steps` bounds runs.

## Hosting instances for other processes

`E.host(instance, address?)` serves an open instance with UniEnv's [remote execution](remote-execution.md), so other
processes can use exactly the environment the model set up: a training script, a viewer, a policy on another
machine. The result says how to connect, e.g. `RemoteClient.connect("tcp://127.0.0.1:41233").env("sim-1")`.

- `address` is any address the remote layer serves (`tcp://host:port`, `unix:///path`, `ws://host:port`); the
  default is `tcp://127.0.0.1` on a free port. Calling `host` again with another address adds it.
- Remote calls run on the instance's thread, like the model's, so the two interleave one at a time; there is no
  ownership. Remote resets and steps update what `E.observe` reports.
- Remote **subscribers** (`client.subscribe("sim-1")`) see every reset and step of the instance: the model's own,
  every step of policy runs, and other clients'. All of them are published from the instance's thread, so the
  stream's order is the order the steps happened in.
- `P.host(instance)` serves the policy object paired with the instance (the one its runs use), as
  `<policy>-<instance>`, on the same addresses: `RemoteClient.connect(address).policy("pick-sim-1")`.
- A hosted instance is never closed for being idle. `E.unhost` stops hosting (remote clients are disconnected), as
  do `E.close` and closing the server. `describe` and `observe` show where an instance is hosted.
- Environments and policies of a bridged remote stack are already hosted: `host` returns their original address.

Hosting is on by default; `UniEnvMCPServer(hosting=False)` removes the tools. The remote protocol has no
authentication: anyone who can reach a hosted address can control the instance, so a model that hosts on
`0.0.0.0` exposes it to the network.

## Serving a remote stack

When the simulator or robot runs in its own process (or Python environment) under a UniEnv `RemoteServer`, bridge
it instead of importing it:

```bash
unienv-mcp bridge tcp://sim-box:7000 --physical robot
```

Hosted environments and policies become shared instances (worlds, nodes and transformations are not exposed);
objects the remote server builds on demand are opened by the bridge and stay open while it runs.
