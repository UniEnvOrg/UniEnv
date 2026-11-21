# Remote execution

`unienv_interface.remote` is a small RPC layer: a `RemoteServer` hosts existing
`Env`, `World`, `WorldNode`, `DataTransformation` and `Policy` objects, and a `RemoteClient` returns proxies that
implement the same interfaces. Each proxy method is one remote call.

Transports: TCP and Unix domain sockets (standard library), WebSocket
(`pip install 'unienv[network]'`), and an in-memory pair for tests.

## Hosting an environment

```python
from unienv_interface.remote import RemoteServer

with RemoteServer() as server:
    server.register("robot", make_env())     # an Env, World or WorldNode
    server.serve("tcp://0.0.0.0:7000")      # blocks until Ctrl-C
```

`listen(address)` starts the same listener in the background and returns the bound
address (use port `0` for a free port). Addresses are `tcp://host:port`,
`unix:///path/to.sock` or `ws://host:port`; a server can listen on several.

```python
from unienv_interface.remote import RemoteClient

with RemoteClient.connect("tcp://robot-pc:7000") as client:
    env = client.env("robot")
    context, observation, info = env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(env.sample_action())
```

`RemoteEnv` has the hosted env's spaces, `batch_size`, `render_mode`, `render_fps`,
`metadata` and timings. Results are converted to the client's backend and device
(`RemoteClient.connect(address, backend=PyTorchComputeBackend, device="cuda")`);
arguments are converted to the hosted object's backend and device on the server.
`server.connect()` returns an in-process client (same wire format, no sockets).

The protocol has no authentication. Bind to loopback or a trusted network, or
tunnel it (SSH, VPN).

## Hosting a data transformation

A `DataTransformation` can be served too, e.g. one whose dependencies live in
another Python environment. It has no backend of its own: register it with the
backend and device its calls should run on, and the source space and data of
each call are rebuilt there.

```python
from unienv_interface.backends.pytorch import PyTorchComputeBackend

server.register("grasps", make_transformation(), backend=PyTorchComputeBackend, device="cuda")
```

```python
transformation = client.transformation("grasps")   # a RemoteDataTransformation
target_space = transformation.get_target_space_from_source(source_space)
result = transformation.transform(source_space, data)
```

Only `transform` and `get_target_space_from_source` are remote. The proxy has
no inverse and cannot be serialized (it is bound to its connection): a storage
that serializes its transformations needs the hosted one.

## Hosting a policy

A `Policy` is hosted like an env. Its state lives on the server, and the client
drives it with the usual `reset` / `get_action` / `observe_transition` protocol,
e.g. a GPU machine runs the policy while the robot computer runs the env:

```python
server.register("pi", make_policy())       # any Policy, e.g. PyTorch on cuda
server.serve("tcp://0.0.0.0:7001")
```

```python
with RemoteClient.connect("tcp://gpu-box:7001") as client:
    policy = client.policy("pi")           # a RemotePolicy: a regular Policy
    context, obs, info = env.reset(seed=0)
    policy.reset(context, obs)
    for _ in range(1000):
        action = policy.get_action()
        obs, reward, terminated, truncated, info = env.step(action)
        policy.observe_transition(action, obs)
```

Only `reset`, `observe_transition`, `get_action` and `visualize_action` are
remote. The contexts and observations the client sends are rebuilt on the
hosted policy's backend and device (torch tensors on the GPU for a PyTorch
policy), and actions come back on the client's. The proxy has the hosted
policy's spaces, `batch_size` and `metadata`.

A `FuncPolicy` passes its state through every call, so host it as
`FuncPolicyBasedPolicy(func_policy)`; the server rejects a bare `FuncPolicy`.

Each control step is two round trips (`observe_transition`, then `get_action`).
Observations travel to the server uncompressed: the `compress` option below
applies only to results the server sends back.

## Building objects on demand

`server.register_factory(name, factory)` hosts an object that exists only while
clients use it. `factory()` runs on the object's own worker thread when a client
first opens `name`, and the server closes the object there once the last client
that opened it is gone:

```python
server.register_factory("sim", lambda: make_env(), release_after=60)
```

```python
with RemoteClient.connect("tcp://sim-box:7000") as client:
    env = client.env("sim")        # builds the env on the server (if not built yet)
    env.reset(seed=0)              # a newly built env starts unreset, like a local one
    ...
    env.close()                    # releases it (so does closing the client or losing the connection)
```

- Clients open a factory-hosted object by getting its proxy (`client.env(name)`,
  `client.resource(name)`, ...), calling it (`client.call`) or subscribing to it;
  its spaces and attributes are read then. Until a client opens it, `client.resources`
  lists it as `{"kind": None, "factory": True}`.
- Every client that opened it uses the same object. It is released by
  `client.release(name)`, `close()` on its proxy (a `with env:` block, a wrapper's
  `close()`), closing the client or losing the connection.
- Once no client uses it, the object is closed after `release_after` seconds
  (default 60; `0`: at once; `None`: never, which only makes building lazy).
  Opening it again within that time gets the same object, state included;
  afterwards, a newly built one.
- A released proxy stops working (`RemoteError` wrapping `PermissionError`);
  get the proxy again to reopen.
- `backend` / `device` are passed for a factory of `DataTransformation`s, as in
  `register`. There is no recursive variant: only the built object is hosted.
- Clients of earlier versions do not see factory-hosted objects; everything
  else works between versions as before.

## Threads and concurrency

Every registered root object gets one worker thread. All calls on it, and on the
world and nodes it exposes, run on that thread one at a time, in arrival order.
Simulators that need thread affinity therefore always see the same thread. Any
number of clients may connect and call; their calls simply queue. A client is
thread-safe and may be shared by several threads. Different roots run in parallel.

When the server closes, it drops all connections and closes the objects registered
with `close=True` (the default), and every object a factory built, on their worker
threads. Proxies' `close()` does nothing for objects hosted with `register`: clients
never close those; it releases objects hosted with `register_factory`.

## Errors, timeouts and disconnects

- An exception raised by the hosted object is re-raised as `RemoteError`, with the
  remote exception's class name (`err.type`), message and traceback
  (`err.remote_traceback`, also included in `str(err)`). The connection stays usable.
- Only the methods of the standard interfaces can be called (`reset`, `step`,
  `render`, ... see `unienv_interface.remote.protocol.METHODS`); anything else is
  rejected. There is no generic attribute access.
- If the connection is lost, every pending and later call raises
  `ConnectionError`. **A call interrupted by a disconnect may or may not have run on
  the server**: nothing is retried, so check the state (or reset) after reconnecting.
- `RemoteClient.connect(address, timeout=5.0)` bounds every call; a timed-out call
  raises `TimeoutError` but keeps running on the server (the late reply is dropped).

## Spaces that change on reset

Spaces are sent once, in the `hello` handshake. After each `reset` / `reload` /
`after_reset` / `after_reload`, the server checks whether any space of that root
(including its world's nodes) changed and, if so, attaches the new spaces to the
next response for every client. Proxies update their `observation_space`,
`action_space` and `context_space` in place.

## Components mode

`server.register(name, obj)` hosts only `obj` itself. To also host what an env is
built from, use `server.register_recursive(name, env)`: for an env built on a
`WorldEnv` it exposes its world as `<name>/world`, its root node as `<name>/node`
and every child node as `<node name>/<child name>` (for example
`robot/node/camera`). You can also register individual worlds and nodes yourself
with `register`; objects that share a world always run on the same worker.
Compose them with a plain local `WorldEnv` (or a local combination of remote child
nodes); each lifecycle hook is then one remote call:

```python
from unienv_interface.world import WorldEnv

server.register_recursive("robot", make_env())   # on the server

world = client.world("robot/world")
env = WorldEnv(world, client.node("robot/node"))   # the whole node tree
env = WorldEnv(world, [client.node("robot/node/arm"), client.node("robot/node/camera")])
```

`client.resources` lists every hosted resource with its descriptor (kind, callable
methods, and links to its world, root node and children when those are hosted too;
a node proxy's `world`, `nodes` and `get_node` only see hosted objects). A node whose `world` is `None` runs on its own
worker; it must declare `backend` / `device` for argument conversion.

## Observers

```python
with observer.subscribe("robot") as stream:
    for event in stream:                    # or stream.next(timeout=1.0)
        print(event.kind, event.sequence, event.value)
```

A subscription yields the result of every `reset`, `reload` and `step` of a hosted
env, whichever client called it, as `SubscriptionEvent(kind, sequence, value)`.
Delivery is latest-wins: a slow subscriber skips results (visible as gaps in
`sequence`) and never slows the server or other clients down. Closing the
subscription unsubscribes; a lost connection ends it with `ConnectionError`.

## Image compression

Over a slow network, a client can ask for lossy compression of camera images:

```python
client = RemoteClient.connect(address, compress={"image": "jpeg:90"})  # quality 1-100
```

The server then JPEG-encodes every uint8 leaf of a `reset` / `step` result (and of
node `get_observation` / `get_context`) whose space has `metadata["kind"] ==
"image"`, laid out `HWC` or `CHW` (`metadata["layout"]`, default `HWC`) with 1 or 3
channels (`metadata["format"]`: `rgb`, `bgr` or `gray`); leading dimensions are
batch dimensions. Everything else (depth maps, float images, RGBA, arrays without
image metadata) and everything the client sends stays raw. Each client chooses for
itself; the default is raw. Encoding uses OpenCV when installed, otherwise Pillow.

An 84x84 RGB image shrinks about 11x and costs ~20 µs to encode plus ~20 µs to
decode, so it pays off once the link is slower than a few hundred Mbit/s; on
loopback raw is faster. Other codecs can be added with
`unienv_interface.remote.wire.register_codec` (on both sides).

## Performance

Arrays travel as raw attachments next to a small JSON header, with no pickling and
no copies on the sending side; stream sockets send each frame with one `sendmsg`.
On one machine a step with a small observation takes about 40-50 µs over TCP or
Unix sockets, an 84x84x3 image about 50 µs and a batch of 64 such images about
140-160 µs; WebSocket adds roughly 50 µs per call. Run
`python benchmarks/envs/bench_remote.py` to measure your setup.

## Runnable examples

```bash
python examples/remote/environment.py
python examples/remote/components.py
```

See the [protocol reference](../reference/remote-protocol.md) for the wire format.
