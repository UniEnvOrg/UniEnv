# UniEnv remote protocol (version 2)

The wire contract of `unienv_interface.remote`, for writing clients or servers in
other languages. The Python implementation is in `wire.py` (frames and values),
`protocol.py` (message layout and the method allowlist) and `server.py`.

## Frames

A transport carries ordered, reliable frames. TCP and Unix sockets send frames
back to back; WebSocket sends one frame per binary message.

```
u32 buffer count N | u32 JSON length L | N x u64 buffer length | L bytes UTF-8 JSON | N buffers
```

All integers are little-endian. The JSON is strict (no `NaN` / `Infinity`
literals). Receivers reject frames over their size limit (256 MiB by default),
with more than 65536 buffers, with lengths that do not add up, or whose buffers
are not each referenced exactly once.

## Values

`null`, booleans, integers, strings, lists and objects with string keys are plain
JSON. Other values are objects with the reserved key `"__t"`:

| `__t` | Fields | Value |
| --- | --- | --- |
| `tuple` | `v`: list | tuple |
| `dict` | `v`: list of `[key, value]` | dict with non-string keys, or with a `"__t"` key |
| `float` | `v`: `"nan"`, `"inf"`, `"-inf"` | non-finite float |
| `bytes` | `b`: buffer index | bytes |
| `nd` | `d`: dtype, `s`: shape, `b`: buffer index | array, C order |
| `nd` | `d`, `s`, `c`: codec, `o`: codec options, `b`: list of buffer indices | compressed array (see below) |
| `space` | `v`: `space_to_json` document | UniEnv space |
| `graph` | `v`: `n_nodes`, `n_edges`, `nodes_features`, `edges_features`, `edges` | `GraphInstance` |

Dtypes are NumPy type strings with an optional byte-order prefix: `b1`, `i1`-`i8`,
`u1`-`u8`, `f2`, `f4`, `f8`, `c8`, `c16` (e.g. `<f4`, `|u1`). A shape is a list of
at most 32 non-negative integers; `product(shape) * itemsize` must equal the buffer
length. Scalars of a dtype are 0-d arrays. Nesting is limited to 64 levels. Other
values (objects, strings arrays, structured dtypes, ...) cannot be sent.

## Messages

| Message | Layout |
| --- | --- |
| Request | `[id, target, method, args, kwargs]` |
| Success | `[id, 0, value]` or `[id, 0, value, updates]` |
| Error | `[id, 1, [exception type name, message, formatted traceback]]` |
| Event | `[resource name, 2, [kind, sequence, value]]` |

`id` is a client-chosen integer; responses may arrive in any order. `target` is a
resource name, or `""` for the protocol methods below. `args` is a list, `kwargs`
an object. Only the methods in `protocol.METHODS` can be called:

| Kind | Methods |
| --- | --- |
| `env` | `reset`, `reload` (if defined), `step`, `render` |
| `world` | `step`, `reset`, `reload`, `after_reset`, `after_reload` |
| `node` | `reset`, `reload`, `after_reset`, `after_reload`, `pre_environment_step`, `post_environment_step`, `set_next_action`, `get_context`, `get_observation`, `get_reward`, `get_termination`, `get_truncation`, `get_info`, `render` |

Calls on one registered root (and the world and nodes it exposes) execute one at a
time, in arrival order.

## Protocol methods

- `hello(version=2, compress=null)` must be the first request. It returns
  `{"version": 2, "resources": {name: descriptor}}`. A descriptor has `kind`
  (`env`, `world`, `node`), `methods`, `attributes` (`name`, `batch_size`, timings,
  `render_mode`, `render_fps`, `supported_render_modes`, `has_*` signals,
  `metadata`; only those present), `spaces` (`[version, {observation_space,
  action_space, context_space}]`), node `priorities`, and links: `world` and `node`
  (resource names) and node `children` (`{child name: resource name}`).
- `subscribe([env name])` / `unsubscribe([env name])`: start or stop events for
  that env. After every `reset`, `reload` and `step`, whoever called it, the server
  sends `[name, 2, [method, sequence, result]]`. Unsent events are replaced by newer
  ones (latest wins).

`updates` (`{name: [version, spaces]}`) carries spaces that changed since the
client last saw them; the server checks after `reset`, `reload`, `after_reset` and
`after_reload` calls. Clients ignore updates with a version not newer than theirs.

## Compression

`hello(compress={"image": "jpeg:90"})` asks the server to JPEG-encode (quality 90)
the uint8 array leaves of results whose space has `metadata["kind"] == "image"`, an
`HWC` or `CHW` `metadata["layout"]` (default `HWC`) and 1 or 3 channels. Such a leaf
is sent as `{"__t": "nd", "d": "|u1", "s": shape, "c": "jpeg", "o": {"quality": 90, "layout": ...,
"format": ...}, "b": [one buffer per image]}` (leading dimensions are batch
dimensions). Clients never compress what they send.
