# Remote execution and observation

`unienv_interface.remote` exports existing environments and world/node trees over
an ordered binary message transport. Python clients use familiar synchronous
UniEnv interfaces. Additional clients can subscribe to completed-operation
snapshots. Execution remains caller-driven.

Install the WebSocket adapter with `pip install 'unienv[network]'`. The in-memory
adapter and protocol core need no networking dependency.

## Hosting an environment

```python
from unienv_interface.remote import RemoteServer, RemoteClient

with RemoteServer() as server:
    server.register("robot", my_env)  # an already constructed Env or WorldEnv
    endpoint = server.listen()       # ws://127.0.0.1:<allocated port>
    with RemoteClient.connect(endpoint) as client:
        env = client.env("robot")
        context, observation, info = env.reset(seed=42)
        observation, reward, terminated, truncated, info = env.step(env.sample_action())
```

The server and clients can run in different processes. Configure `listen(host,
port)` and pass the resulting endpoint to the clients. V1 is for explicitly
configured trusted networks; it provides no application authentication. The
server binds to loopback by default.

Register every resource before the first connection or listener starts.
`register(id, obj)` accepts `Env`, `World`, and `WorldNode`. Registering a
`WorldEnv` also exports its world, root node, and descendants. The return value is
the canonical resource ID; registering the same object again returns that ID.
Use `discover()` or `describe(id)` to find relationships instead of constructing
child IDs yourself. Attached nodes must retain their server-side world; detached
nodes must remain worldless (see below).
Topology is frozen after registration: do not add, remove, rename, or reparent
children, or replace an environment's world/root node, even during reset/reload.
The server does not detect such changes or rebuild captured relationships; use a
new server registry when the resource graph changes.

For embedded use, replace `listen()` and `RemoteClient.connect(...)` with
`server.connect()`. It exercises the same codec and protocol over bounded
in-memory queues.

## Detached nodes and independent devices

An arm and a hand controlled through independent device SDKs need not declare a
shared simulator world. Register each as a root `WorldNode` with `world = None`:

```python
from unienv_interface.backends import NumpyComputeBackend
from unienv_interface.world import WorldNode

class ArmNode(WorldNode):
    world = None
    backend = NumpyComputeBackend
    device = None
    # Implement the node's spaces, priorities, and device lifecycle methods.

server.register("arm", arm_node)
server.register("hand", lambda: make_hand_node())
```

Each detached root owns a separate worker, controller claim, fault state, revision,
and boundary sequence. Different sessions can control arm and hand concurrently;
a blocked call or fault on one does not block or fault the other. The same node
still permits only one controller. Instance registrations remain eager and are
never idle-evicted; factories retain lazy instantiation, eviction, and revival.

Every detached node must provide its own `backend`, normally a class attribute or
an overridden property. `WorldNode`'s inherited backend/device properties delegate
to `world` and cannot supply a backend when it is `None`. Registration rejects a
missing or `None` backend before committing the tree. An undeclared device defaults
to `None` for RPC argument conversion; declare `device = None` explicitly if your
node implementation also reads `self.device`. Client proxies use their client's
configured local backend/device.

No synthetic `/world` resource is exported. `client.node("arm").world` is `None`,
and the descriptor's `world_id` is null. Detached nodes are standalone components,
not inputs to `RemoteWorldEnv`; that composer requires a real `RemoteWorld`.

```python
arm = client.node("arm")
client.acquire("arm")  # components mode
with client.operation("arm", "reset"):
    arm.reset()
    arm.after_reset()

with client.operation("arm", "step"):
    arm.set_next_action(action)
    arm.post_environment_step(dt)  # use the hooks required by this device SDK

observation = arm.get_observation()  # completed-boundary cache
client.release("arm")
```

Reset/reload completion and fault recovery require a successful reset/reload of
the detached **root node** instead of a world. Priorities and additional lifecycle
hooks remain the caller's responsibility. Step brackets invoke node methods; the
remote layer does not invent a `World.step()` or a new node `step()` method.
Subscriptions publish at completion as usual, and controller read caching has the
same boundary and invalidation semantics as for attached nodes.

A detached root may expose child nodes through its `nodes` tree. Every child must
also have `world = None` and a declared backend. Children share the root's domain,
use URL-quoted child-name IDs, and have the root resource ID as `domain_id`. An
operation can target a child ID, but root reset/reload is still required for
recovery. Registering an already exported child returns its canonical ID; a child
cannot belong to two independent detached domains. Topology must remain fixed,
including across factory revival. Shutdown/eviction closes only the node tree,
using its root close hook to close children, with no world close call.

**Trusted construction:** the server cannot detect two distinct detached nodes
that secretly operate the same physical device or share an SDK connection. There
is no underlying-world ownership check for detached nodes. Do not register such
nodes as independent roots; represent shared ownership explicitly in one node
tree or an attached world domain.

## Lazy registration and idle eviction

Pass a zero-argument factory instead of an instance to defer construction:

```python
with RemoteServer(idle_timeout=60.0) as server:
    server.register("robot", lambda: make_robot_env())
    endpoint = server.listen()
```

Factories are accepted only as explicit root registrations and must return an
`Env`, `World`, or `WorldNode`. The factory and the automatic world/node tree walk
run on the domain's single worker, on first access requiring the object:
`describe`, `call`, `begin`, or `subscribe` (and control acquisition). `hello`
alone does not construct anything. **Discovery instantiates pending domains** to
obtain descriptors; the Python client discovers automatically when connecting.
Use a suitable client timeout for expensive construction.

Root IDs are validated immediately; the returned object and tree are validated
on first access, before registry changes are committed. Factory exceptions and
invalid trees become request errors, leave the domain pending, and allow a later
request to retry. Each factory must create its own world or detached node tree,
without sharing objects with another registration. It must reproduce the same
resource kinds, child names,
and relationships on every invocation. IDs use the same `/world`, `/node`, and
URL-quoted child-name derivation as eager registration. Spaces and metadata may
change across invocations.

Only factory-registered domains are evictable. The default `idle_timeout` is 60
seconds; `None` disables eviction entirely. A timeout must otherwise be positive
and finite. A domain is idle only when it has no RPC activity, no active
subscriptions, and no client claim/control. Activity, including discovery and
failed requests routed to the domain, resets the clock. Time is measured after
requests finish; releasing control, unsubscribing, or detaching a controlling or
subscribing session also starts a fresh idle interval. Claims persist between
calls, so release control when finished. Open subscriptions prevent eviction even
when no snapshots are being produced.

A daemon reaper checks periodically, so disposal occurs after the idle timeout,
not at an exact deadline. Disposal uses the normal resource-tree cleanup on the
domain worker, serialized with client requests. Resource IDs and relationships
remain registered, but simulator state and cached snapshots are discarded. The
next access rebuilds the tree on the same worker and increments the descriptor
revision; existing proxies refresh from returned descriptors. Reset the revived
simulator as required by its normal lifecycle; eviction does not preserve state.
Cleanup errors are logged and the domain still returns to pending state.

Eager instance registrations are never evicted. Closing a proxy releases control
without immediately disposing either kind of registration. Server shutdown stops
and joins the reaper, disposes instantiated domains, and drops pending factories
without calling them.

## Choosing who coordinates lifecycle calls

`client.env(id)` returns a `RemoteEnv`. Its reset and step calls execute the entire
environment lifecycle on the server, with one request per call.

For client-side composition:

```python
from unienv_interface.remote import RemoteWorldEnv

description = client.describe("robot")
world = client.world(description["world_id"])
root = client.node(description["node_id"])
env = RemoteWorldEnv(world, root)
env.reset()
result = env.step(env.sample_action())
```

You can also pass a list of remote child nodes to `RemoteWorldEnv`. All proxies
must share the same client and world, following `WorldEnv`'s existing invariant.
The local composer preserves node priorities and control/update/world substeps;
actual node methods still run on the server beside their simulator references.
This path incurs network round trips within the lifecycle.

Use `RemoteWorldEnv` for automatic observation boundaries. Direct component calls
require explicit contexts:

```python
with client.operation(world.resource_id, "reset"):
    world.reset(seed=42)
    # Run your node reset and world/node after_reset hooks here, in priority order.

with client.operation(world.resource_id, "step"):
    # Submit node actions and pre-step hooks here.
    dt = world.step()
    # Run post-step hooks and read results here.
```

The context marks completion; it does not invent missing lifecycle calls. Manual
coordinators are responsible for executing the complete lifecycle. Contexts on
different clients may coordinate different servers, but cannot roll back an
already executed operation on any server.

One client session owns control of a shared world at a time, including all of its
exported nodes and environments. Ownership is acquired on the first operation
or explicitly through `client.acquire(id, mode="env" | "components")`. Switching
modes requires `client.release(id)`. Observer subscriptions do not claim control.

## Read caching and transactions

After a successful `RemoteWorldEnv.reset()`, `reload()`, or `step()`, or a completed
`client.operation()`, the controlling client receives boundary snapshots for the
domain's nodes and world. Server-side `RemoteEnv` mutations also deliver the
associated node snapshots and the environment's returned fields. Outside an
explicit operation, node `get_observation()`, `get_context()`, `get_reward()`,
`get_termination()`, `get_truncation()`, and `get_info()` use those cached fields
without another request. `RemoteEnv` has no equivalent field-getter API: its reset
and step return tuples remain the way to read environment results.

With no controller subscription for a resource, caching uses the default
subscription fields, excluding `render`. If the controller subscribes to that
resource, its selected fields are combined for caching, still excluding `render`.
`render()` is always a live RPC. Missing fields, failed snapshot capture, or a
snapshot that does not fit the negotiated response limit cause a live-getter
fallback, not failure of an otherwise successful mutation. Arrays are codec
copies detached from server memory. Each cached getter returns a deep copy before
local backend conversion, so modifying a returned NumPy array, CPU Torch tensor,
or nested container cannot poison later cached reads. This costs a local copy
proportional to the returned value's size, but no RPC; it is not a zero-copy API.

For env-mode mutations, the server invokes the method, freezes its returned value
and descriptors, and only then performs optional controller capture. Env snapshots
are built from that frozen return tuple. A later node getter that refreshes a
reusable buffer cannot change either the mandatory result or its Env snapshot.
For component completion, getter RPC results have already been frozen individually;
the completion descriptors are frozen before boundary capture.

**Proactive capture requires cheap, idempotent getters.** By default it runs for
nodes at successful boundaries even without subscriptions. Each directly captured
resource/field is evaluated at most once per boundary and shared between controller
and observer delivery, including overlapping selections. Failed captures are also
remembered for that boundary rather than retried. Lifecycle reads inside an Env or
explicit operation remain separate live calls; capture does not intercept those
calls or arbitrary reads made inside aggregate getters.

For a camera, one-shot counter, or other consumptive getter, acquire data in a
lifecycle hook and have getters return the latest reading, or restrict proactive
capture on the hosted node:

```python
sensor.remote_snapshot_fields = ()                  # no proactive field reads
sensor.remote_snapshot_fields = ("context", "info")  # only known-safe fields
```

This server-side setting intersects the controller's normal field selection; it
does not suppress explicit getter RPCs or explicitly requested observer fields.
Restrict aggregate nodes as well if their getters consume readings from children.
Without this restriction, a stateful getter may advance sensor state during
capture and its node snapshot may differ from the earlier lifecycle result.
Freezing protects returned values, not the simulator from arbitrary getter side
effects. Missing cached fields continue to use live RPCs.

**Inside `client.operation()` and `RemoteWorldEnv` lifecycle boundaries, getters
always perform live RPCs.** The cache is not a transaction-local working copy.
The no-read-your-writes gotcha applies to boundary-snapshot reads: a cached value
does not predict or incorporate simulator changes that have not been captured at
a successful boundary. For example, setting an action does not itself advance
the simulation or promise a new observation:

```python
with client.operation(world.resource_id, "step"):
    node.set_next_action(action)
    before_step = node.get_observation()  # live RPC, never an old cached field
    world.step()
    # Run the application's normal pre/post-step hooks as required.
    after_step = node.get_observation()   # live RPC

published = node.get_observation()        # completed-boundary snapshot
```

Bare calls are separate serialized units, not an implicit multi-call transaction.
An environment reset/step is one server-worker unit with a publication boundary;
a bare cache-miss getter is one live RPC. Component mutations such as
`set_next_action()` still require an explicit operation; two adjacent bare calls
do not create one. Use `client.call(node.resource_id, "get_observation")` to request
a live read explicitly outside an operation, subject to normal control/mode rules.

Starting a mutation or operation drops cached fields for that domain. Reset,
reload, aborts, request errors/faults, release/reacquisition, and newer observed
descriptor revisions or boundary sequences also invalidate them. A successful
boundary replaces available fields; omitted fields never survive from an older
boundary. Session close or detach drops all cached fields. Factory revival bumps
the revision and refreshes descriptors; release has already cleared the cache
before an idle domain can be evicted.

Another client cannot mutate a world while the first retains control. Once control
is released, the former controller has no cached fields to read: it can use its
observer stream, or reacquire control for live reads and complete a new boundary
to refill its cache. Observer messages invalidate older controller cache versions
when received, but do not populate the controller cache or claim ownership.
Discovery/description can refresh revisions and sequences, but do not refill
field values. Changes made directly to the hosted simulator outside this protocol
require explicit live reads or a new operation boundary; no background polling
is added.

## Subscriptions and spaces

```python
with observer.subscribe("robot", ["observation", "reward"]) as snapshots:
    snapshot = snapshots.next(timeout=5)
    print(snapshot["sequence"], snapshot["data"])
```

Subscriptions are closeable iterators. `next(timeout=...)` raises `TimeoutError`
if no event arrives; normal iteration waits indefinitely. They publish future
completed operations, with no historical replay or initial snapshot. Rendering
requires selecting the `render` field explicitly.

Each subscription retains its latest pending snapshot. Slow readers can see
sequence gaps, while control responses and subscription errors remain reliable.
Arrays are detached from simulator memory. Sequence numbers are shared by the
whole world domain and never reset during the server lifetime. Operation IDs
correlate snapshots of different nodes at the same completion boundary.

In component mode, subscribe to node/world resources. No snapshot is attributed
to a registered server environment when the client's composition may differ from
that environment. A bare world's output is `dt` from its most recent world step;
node outputs are captured after all coordinated updates. Whole-environment
subscriptions expose the exact fields returned by the completed reset or step.
Fields absent from that result are omitted, rather than retaining stale values.

Descriptors include spaces, relationships, priorities, timing, supported methods,
and revisions. Proxies refresh descriptors before returning method results,
including after reload hooks change spaces. Snapshots carry their own descriptor
and revision, so observer code can interpret a new shape without relying on an
earlier discovery response. Batch dimensions are never added by the remote layer;
masked reset results retain UniEnv's smaller selected batch dimension.
Spaces may be `None` until the first reset initializes the hosted resource. Reset
before sampling actions or inspecting those spaces; the call refreshes proxies
automatically, so an extra `client.describe(id)` is not required. Unchanged space
descriptors preserve existing proxy space objects; changed spaces replace them.

## Backends, errors, and lifetime

`RemoteClient.connect(endpoint, backend=..., device=...)` selects local array and
space reconstruction. NumPy on CPU is the default. Unsupported dtype conversions
raise an error instead of silently reducing precision. Payloads support primitive
values, nested string-keyed dictionaries, lists, tuples, bytes, and numeric arrays.
Built-in `GraphInstance` values and `BatchedSpace` object containers use explicit
portable container tags; every element must itself be a supported value.
Custom Python objects in `info`, kwargs, or render outputs require application
conversion to these types. No pickle, dynamic imports, or uploaded code are used.

Requests default to a 30-second send/response deadline; configure `timeout=None`
for indefinite waits or choose a larger limit for expensive scene construction.
Timeouts close the connection. Mutations are never automatically retried:
`UncertainOutcomeError` means the remote action may already have executed.
`RemoteError` exposes `code` and `uncertain`; a lifecycle or result serialization
failure after execution begins also has an uncertain outcome.
Timeouts cannot interrupt a simulator method already running on the server;
recovery must wait for that worker to finish.

An interrupted component operation faults its world domain. Reconnect and perform
an explicit complete reset/reload before stepping again. No rollback or session
resume is provided. A failed whole-environment mutation similarly requires reset
or reload. Observation errors terminate the affected subscription without undoing
an otherwise successful control operation.

Closing a proxy or `RemoteWorldEnv` releases control of its world domain but keeps
that client's subscriptions active; it doesn't destroy simulator objects or
close sibling proxies. They can reacquire control later. `client.release(id)` also
retains subscriptions; use `client.release(id, close_subscriptions=True)` to close
all of this client's streams in that domain, or close individual subscriptions.
Closing `RemoteClient`
disconnects the whole session. Server shutdown disposes the registered resource
graph on its world workers. Node garbage collection doesn't issue network calls.

All lifecycle calls, discovery, snapshot reads, and disposal for a world run on
one dedicated worker thread. Constructed simulators must permit use on that
worker; eager constructors are not moved to it, while factories run there. Getters used for subscriptions should
be side-effect-free. V1 makes no hard real-time or device-memory sharing promise.
Remote type queries operate on proxy types, and arbitrary wrapped attributes are
not exported.

## Examples and protocol

From a checkout with `unienv[network]` installed, run:

```bash
python examples/remote/environment.py
python examples/remote/components.py
```

Both examples start a loopback WebSocket server, a controller, and an observer.
See the [v1 wire specification](../reference/remote-protocol.md) for non-Python
clients and transport implementations.
