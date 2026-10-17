# Remote execution

`unienv_interface.remote` is a small RPC layer: a `RemoteServer` hosts existing
`Env`, `World` and `WorldNode` objects, and a `RemoteClient` returns proxies that
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

## Threads and concurrency

Every registered root object gets one worker thread. All calls on it, and on the
world and nodes it exposes, run on that thread one at a time, in arrival order.
Simulators that need thread affinity therefore always see the same thread. Any
number of clients may connect and call; their calls simply queue. A client is
thread-safe and may be shared by several threads. Different roots run in parallel.

When the server closes, it drops all connections and closes the objects registered
with `close=True` (the default) on their worker threads. Proxies' `close()` does
nothing: clients never close hosted objects.

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

Registering an env built on a `WorldEnv` also exposes its world as `<name>/world`,
its root node as `<name>/node` and every child node as `<node name>/<child name>`
(for example `robot/node/camera`). Compose them with a plain local `WorldEnv` (or
a local combination of remote child nodes); each lifecycle hook is then one remote
call:

```python
from unienv_interface.world import WorldEnv

world = client.world("robot/world")
env = WorldEnv(world, client.node("robot/node"))   # the whole node tree
env = WorldEnv(world, [client.node("robot/node/arm"), client.node("robot/node/camera")])
```

`client.resources` lists every hosted resource with its descriptor (kind, callable
methods, links to its world, root node and children). A `WorldNode` or `World`
can also be registered directly. A node whose `world` is `None` runs on its own
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
