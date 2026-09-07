# UniEnv remote protocol v1

This document specifies the wire contract implemented by
`unienv_interface.remote`. It is independent of WebSocket and Python object
serialization. V1 has one exact protocol version, `1`; there is no downgrade.

## Transport and envelope

A transport carries reliable, ordered, complete binary messages in both
directions. One reader and one writer may run concurrently. `close()` must unblock
both. Implementations bound their queues and raise connection errors on failure.
The WebSocket binding uses one binary WebSocket message per protocol envelope;
text messages close the connection. Compression is disabled. The reference
in-memory transport has 16-message bounded queues.

Each envelope contains:

1. Four bytes: unsigned, big-endian length of the JSON header in bytes.
2. That many UTF-8 bytes: the tagged value tree described below.
3. Zero or more raw binary attachments, in tree traversal order.

Attachment offsets are relative to the beginning of part 3. Attachments must be
contiguous, nonoverlapping, and exhaust part 3, with no gaps or trailing data.
The total message limit defaults to 64 MiB and is configurable on both peers.
`hello` may include `max_message_size`; the connection uses the smaller of that
limit and the server limit, returned in the hello response.
An invalid envelope closes the connection. A valid envelope with an invalid
request produces a structured error when possible.

The JSON header is **not** the plain request dictionary. All dictionaries and
containers, including the outer request, use these tags:

| Value | JSON header node |
| --- | --- |
| Null, boolean, string, integer, finite float | JSON primitive |
| Nonfinite float | `{"t":"float","v":"nan"}`; `v` may also be `inf` or `-inf` |
| String-keyed dictionary | `{"t":"dict","v":[["key",VALUE],...]}` |
| List | `{"t":"list","v":[VALUE,...]}` |
| Tuple | `{"t":"tuple","v":[VALUE,...]}` |
| Bytes | `{"t":"bytes","offset":0,"length":3}` |
| Numeric array | `{"t":"array","dtype":"<f4","shape":[2,3],"offset":3,"length":24}` |
| Structured batch container | `{"t":"object_array","shape":[2],"v":[VALUE,VALUE]}` |
| UniEnv graph instance | `{"t":"graph","v":TAGGED_DICTIONARY}` |

Dictionary keys must be unique strings. Nesting is limited to 64 levels. Array
shapes have at most 32 nonnegative dimensions; zero-sized arrays and scalar shape
`[]` are valid. Array bytes use contiguous C order. Dtypes use NumPy-style endian,
kind, and item-size strings: boolean `b1`; signed/unsigned integers `i1`–`i8` /
`u1`–`u8` with widths 1, 2, 4, 8; floats `f2`, `f4`, `f8`; complex `c8`, `c16`.
Endian markers are `<`, `>`, or `|` for byte-order-independent one-byte values.
Complex values store real then imaginary components. Structured, object, string,
and platform-specific extended numeric **binary** dtypes are unsupported. NumPy scalars are
encoded as scalar-shaped arrays to retain their dtype.

An `object_array` is a shaped container of recursively encoded portable values,
in C order, with exactly `product(shape)` elements. It never contains raw object
pointers or pickled bytes; unsupported elements fail encoding. The reference
implementation also bounds the container's pointer-array allocation by the
message size limit. It preserves UniEnv `BatchedSpace` structured values.
The `graph` dictionary has exactly the fields `n_nodes`, `n_edges`,
`nodes_features`, `edges_features`, and `edges`, matching UniEnv `GraphInstance`.
These two fixed container tags do not permit arbitrary class reconstruction.

Validate bounds, dtype, and `product(shape) * itemsize == length` before allocating
arrays. Binary floating-point arrays may contain IEEE nonfinite values unchanged.
JSON's nonstandard `NaN`/`Infinity` literals are forbidden. Implementations must
parse JSON integer tokens losslessly, including nanosecond timestamps; clients
using JavaScript numbers should use a lossless JSON parser where necessary.

## Requests and responses

Examples below show **decoded** dictionaries, before applying the codec.

```json
{"version":1,"type":"request","request_id":"r1","session_id":null,"op":"hello","resource_id":null}
```

The response is:

```json
{"version":1,"type":"response","request_id":"r1","session_id":"SERVER_SESSION_ID","result":{"session_id":"SERVER_SESSION_ID","max_message_size":67108864,"server":"unienv","server_version":"0.0.1b13","protocol_versions":[1]}}
```

Clients use fresh string request IDs and the returned session ID on subsequent
requests. The response echoes its request ID. IDs correlate messages; they do not
provide replay deduplication. Clients must not retry uncertain mutations.
Clients must ignore unknown result fields for forward compatibility. The hello
capabilities identify the server and its installed package version (`"unknown"`
when package metadata is unavailable), and list supported protocol versions.

All operations except `hello` and `discover` require `resource_id`:

| `op` | Additional request fields | Result |
| --- | --- | --- |
| `hello` | Optional `max_message_size` | Session ID, negotiated message limit, `server`, `server_version`, `protocol_versions` |
| `discover` | None | All resource descriptors |
| `describe` | None | Descriptors for the resource's entire world domain |
| `acquire` | `mode`: `env` or `components` | Null |
| `release` | None | Null; releases ownership and faults an unfinished operation |
| `begin` | `operation_id`, `kind`: `reset`, `reload`, or `step` | Null |
| `complete` | `operation_id` | Refreshed domain descriptors; publishes snapshots |
| `abort` | `operation_id` | Null; faults the domain |
| `call` | `method`, `args` sequence, `kwargs` dictionary, `operation_id` | `{"value":VALUE,"descriptors":[...]}` |
| `subscribe` | Client-chosen `subscription_id`, optional `fields` list | Subscription ID |
| `unsubscribe` | `subscription_id` | Null |

The reference Python client's `release()` retains subscriptions by default.
`release(id, close_subscriptions=True)` explicitly unsubscribes that client's
domain subscriptions. Wire-level `release` only changes control ownership.

`describe` may instantiate a pending factory-registered domain on its world
worker. `discover` instantiates all pending domains to obtain descriptors; `hello`
does not. Calls, component operations, subscriptions, and control acquisition also
instantiate their target domain when needed. Factory or tree-validation failures
return an operation error and leave the domain pending for a later retry.

Resource descriptors contain `id`, `kind` (`env`, `world`, `node`), `world_id`,
`node_id`, `children` (name-to-ID dictionary), `revision`, and `operations` (exported
method names). Relationships not applicable to a resource are null or empty.
In particular, a detached node has `kind: "node"` and `world_id: null`; there is
no corresponding synthetic world resource. Its `domain_id` identifies its detached
root, and all worldless children share that domain ID. `domain_id`, rather than a
non-null `world_id`, is the authoritative scope for control and cache invalidation.
Descriptors include available name, timing, batch, render, metadata, signal flags,
space descriptions, and node lifecycle priority sets represented as lists.
Spaces use UniEnv's existing `space_to_json` schema inside the tagged value tree,
including nonfinite bounds. Revisions are monotonic within a world domain and may
advance at each reset/reload lifecycle phase, even when a shape stays unchanged.
Revisions also advance on revival after idle eviction. Clients must refresh cached
descriptors from responses, including for existing resource IDs; new instances may
have different spaces or metadata. Revival preserves IDs and topology, not simulator
state. The reference server validates the rebuilt topology before committing it.
Proxy space objects retain their identity while their serialized form is unchanged.
Space fields may be null (`None` in Python) until the first reset initializes the
hosted resource. Reset call responses and completed component operations include
refreshed descriptors; the Python client updates proxies before returning, so no
additional `describe` request is needed. Snapshots also carry refreshed descriptors
for observers. Spaces that remain unavailable continue to be null.

### Controller boundary fields

Descriptors additionally carry `domain_id` (the canonical execution-domain resource
ID, including for wrappers around an otherwise unexported Env) and `sequence`
(the domain's latest completed boundary). These are additive result fields.

In a successful `complete` result or the `descriptors` list of an env-mutation
`call` result, a descriptor may also contain this optional field:

```json
{"snapshot":{"sequence":12,"revision":3,"fields":["observation","reward"],"data":{"observation":[1,2],"reward":3.0}}}
```

The example shows decoded data, before applying the codec. `snapshot.data` uses
the same field names and value rules as observer snapshots. `fields` is the sorted
capture selection; unavailable fields may be absent from `data`. The sequence and
revision match the completed boundary and its descriptor. Without a controlling
session subscription for that resource, selection defaults to all supported
fields except `render`; otherwise it is the union of that controller's subscription
selections, excluding `render`. Render is never served from this cache.

The Python host additionally intersects a node's proactive selection with its
`remote_snapshot_fields` attribute when supplied. An empty iterable disables
proactive reads for that node. Without this attribute, the existing default field
selection applies. This is a trusted server-side configuration, not a new wire
operation; explicit getter calls and observer subscription selections are unchanged.

Capture ordering for an env mutation is: invoke the lifecycle method, freeze its
mandatory `value` and descriptors on the domain worker, then capture optional
controller fields and attach them to the already-frozen result. Env snapshot data
comes from the frozen return tuple, not live buffers retained by that tuple.
Optional getter evaluation must not rewrite the mandatory result. At component
completion, prior call values are already frozen and the completion descriptors
are frozen before boundary getter capture and publication.

Proactive node getters must be cheap and idempotent. Boundary capture is performed
without subscription demand by default; each directly captured `(resource_id,
field)` is read at most once for controller/observer delivery at that boundary.
Results and capture errors are shared across overlapping field selections. Normal
lifecycle reads and reads made internally by aggregate getters are not intercepted
or deduplicated. Consumptive sensors must return a latest-reading value from their
getter or restrict `remote_snapshot_fields`; explicit subscriptions still request
live boundary reads. A stateful node snapshot can reflect a later read than the
mandatory Env result. Freezing prevents result-buffer corruption, not getter side
effects on future simulator state.

The existing result shapes are unchanged: `complete` still returns a descriptor
list, and `call` still returns `{"value":VALUE,"descriptors":[...]}`. As with other
forward-compatible result additions, clients must ignore unknown fields. Snapshot
fields are boundary-only data, not part of persistent descriptor schema. Normal
`describe`/`discover` and non-boundary call responses do not include field values.
Capture failures omit the affected resource's optional snapshot. If cache data
would exceed a response limit, snapshots are omitted; the original response and
its existing error/uncertainty semantics remain authoritative.

Controller snapshots cover node/world resources at component completion, and the
mutated Env plus node/world resources after an env-mode mutation. They do not
attribute component results to an unrelated registered Env. This addition does
not change observer publication rules, subscription selection, error delivery,
or lossy/latest-wins queues. Captures are shared per resource/field for delivery.

The reference client caches fields by resource ID and boundary sequence/revision.
Its six node field getters use the cache only outside explicit operations; inside
an operation they always issue live calls. Missing cache fields and explicit
`client.call()` reads also use live RPCs. It discards affected domain fields before
mutations and operation starts, on reset/reload, abort, release/acquire, request
errors or faults, and when a newer revision or boundary sequence is observed.
Revival revisions invalidate older state. Session close/detach clears all fields.
Successful boundaries replace, rather than merge, prior fields. Older responses
or queued subscription descriptors cannot restore an older cache version.
Cached getter results are deep-copied locally before backend conversion. Caller
mutation cannot alter internal cache arrays or nested containers, including when
CPU Torch conversion shares the copied NumPy buffer. Copy-on-read adds local work
proportional to the returned data size, with no additional network request.

Controller ownership prevents another session from mutating the domain until
release/disconnect has invalidated the controller cache. Former controllers must
use observer streams or reacquire control and perform live reads/a new boundary;
observer pushes invalidate older cache versions but never populate a controller
cache. No polling or cross-session cache ownership is introduced.

World/node topology is frozen after registration (first instantiation for factories): child membership and names,
node-to-world links, and an environment's world/root node must not change, including
during reset/reload. The server captures relationships at registration and does not
detect later structural mutation; descriptor revisions refresh spaces and metadata,
not topology. Construct a new server registry to export a changed resource graph.
Detached node trees must remain worldless; attached and detached children cannot
be mixed within one tree.

## Execution rules

| Resource kind | Exported methods |
| --- | --- |
| `env` | `reset`, `step`, `render`, `reload` (when implemented) |
| `world` | `step`, `reset`, `reload`, `after_reset`, `after_reload` |
| `node` | `set_next_action`, `pre_environment_step`, `post_environment_step`, `reset`, `reload`, `after_reset`, `after_reload`, `get_context`, `get_observation`, `get_reward`, `get_termination`, `get_truncation`, `get_info`, `render` |

Method arguments and returns follow the corresponding UniEnv Python interface;
custom values must fit the portable codec. No generic attribute or code RPC exists.

Before invocation, the server recursively converts every array leaf in `args` and
`kwargs` to the hosted resource's backend and device, regardless of the originating
client backend/device. Dtypes must be representable without precision loss;
unsupported or dtype-changing conversions produce `serialization_error` before
the method is invoked, with `uncertain: false`.

Every world domain executes on one worker. The first control call claims ownership
for its session and mode. A competing owner or different mode gets
`ownership_conflict`. Read methods through proxies also belong to this control
session; observer clients use subscriptions instead. Ownership lasts until
release/disconnect, not just until the current request completes.

Detached roots each define a node-scoped execution domain with an independent
worker, owner, fault state, revision, and sequence. Only `components` mode applies.
`describe`, `acquire`/`release`, `begin`/`complete`/`abort`, allowed node calls, and
subscriptions use that domain unchanged. Children belong to the detached root's
domain even when an operation targets a child resource ID. The Python host keys
eager detached domains by a tagged root-node identity; factory domains retain their
registered root-ID key and worker across eviction/revival.

Detached hosted nodes must declare their own backend; there is no world from
which to inherit it. RPC device coercion defaults to `None` if no device is
declared. A missing backend is a registration error (an execution error on first
factory access). Reusing the same exported object does not create a second domain,
but distinct detached nodes have no protection against SDK/device resources
secretly shared behind their interfaces. Construction remains trusted.

Environment calls provide an operation ID and execute the whole method.
Component mutations require a matching `begin`/`complete` operation around them.
Component reads during an operation also carry its ID. One operation may span
world and node calls sharing the domain. Reset/reload lifecycle methods cannot
run in a step operation, and stepping/action hooks cannot run in a reset/reload
operation. Priorities and lifecycle ordering remain the coordinator's job.

Completion is a publication boundary, not an atomic transaction. Aborting,
disconnecting mid-operation, or encountering a lifecycle failure faults the
domain. Faulted domains reject stepping until successful reset/reload recovery.
A recovery operation must actually reset or reload the world before completion.
For a detached domain, a successful reset/reload of its root node satisfies this
requirement instead. Resetting only a child or running only after-hooks does not
recover the domain. Detached step operations invoke the existing node-method
allowlist; no world or additional node step method is introduced.
No state rollback, automatic replay, persistence, or reconnect resume is implied.

## Observation messages

```json
{"version":1,"type":"snapshot","session_id":"s","subscription_id":"sub","resource_id":"env/node","operation_id":"op","kind":"step","sequence":12,"timestamp_ns":1800000000000000000,"descriptor_revision":3,"descriptor":{},"data":{"observation":[]}}
```

The illustrative empty `descriptor` and observation above stand for the actual
resource descriptor and codec-supported values. `timestamp_ns` is server wall
clock time, not simulation time. `sequence` counts successful operation boundaries
within the shared world, starting at 1; gaps are permitted. No snapshot is emitted
until an operation completes. Subscriptions only receive future events.

Environment and node fields are `context`, `observation`, `reward`, `terminated`,
`truncated`, `info`, and `render`. World fields are `dt`. Defaults include all
applicable fields except `render`; unavailable fields are omitted. Environment
fields come from that operation's return tuple. Node getters run after the
completion boundary. World `dt` is the final world substep's elapsed time, not
necessarily the total environment control time; `WorldEnv` supplies total control
time in its existing info output. `dt` is present only when the coordinator
explicitly called the remote world's `step` in this operation. Environment calls
don't expose their internal last-substep `dt`; their world snapshots omit it.

Component operations publish node/world resources only, because a client's
composition isn't necessarily the registered server environment's composition.
Environment operations publish their environment and associated node/world views.

Pending snapshots are replaceable per subscription. Reliable responses take
priority over them. The reference server bounds pending reliable messages to 64
per connection and disconnects a peer that overruns that queue. Subscription
capture errors use the reliable queue, end that subscription, and have type
`subscription_error`, the same correlation fields, and
`error: {"code":"snapshot_error","message":"..."}` instead of `data`.

## Errors and closure

Request errors use `type: "error"`, echo the request/session IDs, and replace
`result` with `error: {"code":"...","message":"...","uncertain":false}`.
Stable categories are `invalid_request`, `version_mismatch`, `invalid_session`,
`not_found`, `unsupported_operation`, `ownership_conflict`,
`operation_in_progress`, `invalid_operation`, `faulted`, `serialization_error`,
and `execution_error`. Messages are diagnostic text, not machine-readable policy.
`uncertain: true` marks an execution/result failure after mutation started.
Clients must use the server's `uncertain` flag when an error response is available.
A transport failure without a response also leaves a mutation's outcome uncertain.

Closing a connection releases its controller and subscriptions. It does not directly
close the hosted simulator. Factory-registered domains may be disposed after an
idle timeout with no RPC activity, subscriptions, or control claim, and rebuilt on
next access. Eager instance registrations are never idle-evicted. Server shutdown
disposes all instantiated domains and drops pending factories. The protocol
has no remote-destruction operation. Transports and hosting applications are
responsible for deployment security outside v1's trusted-network scope.
