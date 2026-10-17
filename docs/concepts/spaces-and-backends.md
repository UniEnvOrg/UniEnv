# Spaces And Backends

`Space` is the core data contract in UniEnv.

A space does more than describe shape:

- it declares what values are valid
- it knows which backend and device the data belongs to
- it can sample data
- it can create empty values
- it can convert both the space definition and concrete data across backends

## Why Spaces Matter

The rest of the package depends on spaces:

- environments expose `action_space`, `observation_space`, and `context_space`
- wrappers transform spaces when they transform runtime data
- replay buffers and storages know what one sample should look like
- flattening, batching, serialization, and validation all start from the space definition

## Main Space Types

The most important built-in spaces are:

- `BoxSpace`: continuous or integer tensor-like ranges
- `BinarySpace`: boolean tensors
- `DictSpace`: named structured collections of subspaces
- `TupleSpace`: positional structured collections
- `TextSpace`: string-valued data
- `GraphSpace`: graph-structured data
- `UnionSpace`: one-of-many alternatives
- `BatchedSpace`: a batched view over another space
- `DynamicBoxSpace`: shape or bounds that are determined dynamically

## Backends

Each space is tied to a `ComputeBackend`, such as:

- NumPy
- PyTorch
- JAX

Because the backend is part of the space definition, UniEnv can move both metadata and data in a consistent way:

```python
target_space = source_space.to(target_backend, target_device)
target_data = source_space.data_to(data, target_backend, target_device)
```

This is the mechanism used by backend-conversion wrappers and many data utilities.

## Structured Data Without Losing Structure

UniEnv does not force you to flatten observations or actions at the boundary.

You can keep rich structures such as:

- nested dictionaries for multi-sensor observations
- separate action components for different effectors
- mixed image, state, and text metadata in one sample

When a downstream system needs dense arrays, use the helpers in:

- `unienv_interface.space.space_utils.flatten_utils`
- `unienv_interface.space.space_utils.batch_utils`

Those utilities are also used internally by wrappers, batches, and replay buffers.

## Subspaces And Clipping

`a.is_subspaceeq(b)` checks `a ⊆ b`: every valid value of `a` is also valid in `b` (equality allowed). `a.is_subspace(b)` is the strict `a ⊂ b`. The check is structural: same space type, backend and dtype, with tighter bounds, shape / length / count ranges or charsets, and for `DictSpace` a subset of the keys. A controller can therefore declare the observation space it requires and check it with `required.is_subspaceeq(env.observation_space)`, even when the environment exposes extra keys.

`clip` goes the other way. Called on the subspace, it projects any value of the larger space onto it:

```python
if required.is_subspaceeq(env.observation_space):
    obs = required.clip(env_obs)   # required.contains(obs) holds
```

`clip` leaves values that already belong to the space unchanged, never modifies its input, and returns data with the space's backend and dtype. When a value cannot be projected without inventing data, `clip` raises `ValueError`:

| Space | `clip` | `ValueError` when |
| --- | --- | --- |
| `BoxSpace` | clamps to the bounds (`±inf` bounds keep values), casts to the box dtype | never |
| `BinarySpace` | returns the value (only an identical space is a subspace) | never |
| `DynamicBoxSpace` | trims axes longer than `shape_high`, clamps the values | an axis is shorter than `shape_low` |
| `DictSpace` | drops extra keys, clips each value | a key of the space is missing |
| `TupleSpace` / `BatchedSpace` | clips each element | an element cannot be clipped |
| `UnionSpace` | clips the value with the alternative at the same index | the value cannot be clipped |
| `TextSpace` | removes characters outside the charset, truncates to `max_length` | fewer than `min_length` characters remain |
| `GraphSpace` | keeps the first `max_nodes` nodes (dropping their edges), then the first `max_edges` edges, clamps the features; per graph in a batch | fewer than `min_nodes` nodes or `min_edges` edges remain |

## Space Metadata

Shape, dtype and bounds say what values are valid, not what they mean. Every space therefore takes an optional, read-only `metadata` mapping of JSON values:

```python
from unienv_interface.space import BoxSpace, DictSpace

obs_space = DictSpace(backend, {
    "front": BoxSpace(backend, 0, 255, np.uint8, shape=(3, 480, 640),
                      metadata={"kind": "image", "format": "rgb", "layout": "CHW", "source": "front_cam", "fps": 30}),
    "joints": BoxSpace(backend, -3.14, 3.14, np.float32, shape=(7,),
                       metadata={"kind": "proprio", "units": "rad"}),
})

obs_space["front"].metadata["layout"]         # "CHW"
wrist = obs_space["front"].with_metadata(source="wrist_cam")  # a new space; the original is unchanged
```

Nested spaces carry their own metadata, so each camera in a `DictSpace` is described separately. The keys below form a shared vocabulary. None is required and none is enforced when a space is built; any other keys (and kinds) are allowed. "Per element" means a list with one entry per element of the space's last axis.

| Key | Type | Meaning | Values / examples |
| --- | --- | --- | --- |
| `kind` | str | what the data is | recommended: `"image"`, `"depth"`, `"segmentation"`, `"normal"`, `"pointcloud"`, `"proprio"`, `"action"`, `"pose"`, `"force"`, `"tactile"`, `"teleop"`, `"keypoints"`, `"mask"`, `"intrinsics"`, `"extrinsics"`, `"privileged"`, `"task"` (goals / targets), `"state"`; others allowed |
| `format` | str | pixel format of an image | `"rgb"`, `"bgr"`, `"rgba"`, `"bgra"`, `"gray"` |
| `layout` | str | axis order of the trailing image dims (leading dims are batch dims) | `"HWC"`, `"CHW"`, `"HW"` (single channel, no channel axis) |
| `units` | str, or per element list of str | physical units | `"m"`, `"mm"`, `"rad"`, `"rad/s"`, `"m/s"`, `"m/s^2"`, `"N"`, `"N*m"`, `"A"`, `"V"`, `"degC"`, `"px"`, `"s"`; `"normalized"` for unitless normalized data |
| `frame` | str | reference frame | `"world"`, `"base"`, `"camera_cv"` (OpenCV: x right, y down, z forward), `"camera_gl"`, `"link_local"`, `"wrist"` |
| `rotation` | str | rotation representation inside a pose / action vector | `"quat_wxyz"`, `"quat_xyzw"`, `"euler_<axes>"` with [transforms3d](https://matthew-brett.github.io/transforms3d/reference/transforms3d.euler.html) axes (`s` static / extrinsic, `r` rotating / intrinsic, e.g. `"euler_sxyz"`, `"euler_rxyz"`), `"rot6d"`, `"matrix"`, `"axis_angle"` |
| `control` | str | what an action commands | `"joint_position"`, `"joint_velocity"`, `"joint_torque"`, `"eef_pose"`, `"eef_velocity"`, `"gripper"` |
| `mode` | str | how an action commands it | `"absolute"`, `"delta"`, `"velocity"` |
| `normalized` | bool | the values are normalized | `True` |
| `range` | `[low, high]`, scalars or per element lists | the physical range normalized values map to, when known | `[-0.05, 0.05]`, `[[-3.14, -2.0], [3.14, 2.0]]` |
| `names` | per element list of str | names of the elements | `["shoulder", "elbow", "wrist"]` |
| `source` | str | camera or sensor name | `"front_cam"` |
| `fps` | positive number | capture rate | `30` |
| `label` | str | human readable name | `"gripper camera"` |

A vector that packs several quantities, e.g. an end-effector command of position (3) + quaternion (4) + gripper (1), is described per element with `names` and / or a `units` list, and `rotation` names the representation of its rotation part:

```python
from unienv_interface.space.space_utils.metadata_utils import action_metadata

action_space = BoxSpace(backend, -1.0, 1.0, np.float32, shape=(8,), metadata=action_metadata(
    control="eef_pose", mode="delta", frame="base", rotation="quat_wxyz",
    names=["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"],
    units=["m", "m", "m", "normalized", "normalized", "normalized", "normalized", "normalized"],
))
```

Metadata describes the data but is not part of the data domain: `==`, `is_subspace` and `is_subspaceeq` ignore it, so compatibility checks between environments, wrappers and replay buffers are unaffected. Use `space_utils.metadata_utils.spaces_equal_with_metadata` to compare it too, and `find_spaces_by_metadata(space, kind="image")` to locate the described leaves.

Metadata follows the space where its meaning survives: `space.to(...)`, the batching utilities, serialization, frame stacking and the transformations that keep the meaning of the data (rescale, crop, image resize, batchify, dict filtering and flattening). It is dropped where the meaning is lost, e.g. by `flatten_space`, and spaces built from data (`construct_space_from_data`, `from_gym_space`) have none.

When combined world nodes merge their children's spaces, a child's space nested under the child's name keeps its metadata, the dict a `CombinedWorldNode` merges its root node's keys into keeps the root's metadata, and the dict a `FlatCombinedWorldNode` merges several dicts into keeps their metadata only if all of them have the same.

UniEnv uses it to:

- record the image leaves of observations with the video wrappers, converting `CHW` / `bgr` images for writing
- route image leaves to image storages with `dict_storage_kwargs_from_space`
- describe the features of LeRobot datasets

### Metadata Helpers

`unienv_interface.space.space_utils.metadata_utils` has helpers so that environments and adaptors describe their spaces the same way.

The builders return plain dicts with the vocabulary keys, dropping arguments left at `None`, so keys cannot be misspelled: `image_metadata(*, format="rgb", layout="HWC", source=None, fps=None, **extra)`, `depth_metadata(*, units="m", layout="HW", source=None, fps=None, **extra)`, `proprio_metadata(*, units=None, names=None, frame=None, rotation=None, **extra)`, `action_metadata(*, control=None, mode=None, units=None, normalized=None, range=None, names=None, frame=None, rotation=None, **extra)` and, for any other kind, `metadata(kind, **fields)`:

```python
from unienv_interface.space.space_utils.metadata_utils import image_metadata, depth_metadata, metadata

camera_space = DictSpace(backend, {
    "rgb": BoxSpace(backend, 0, 255, np.uint8, shape=(480, 640, 3),
                    metadata=image_metadata(source="front_cam", fps=30)),
    "depth": BoxSpace(backend, 0.0, 10.0, np.float32, shape=(480, 640),
                      metadata=depth_metadata(source="front_cam", fps=30)),
    "intrinsics": BoxSpace(backend, -np.inf, np.inf, np.float32, shape=(3, 3),
                           metadata=metadata("intrinsics", source="front_cam")),
})
```

`annotate(space, mapping, *, replace=False, strict=True)` attaches metadata to an existing (e.g. converted or inferred) space. Each key of `mapping` is a path through nested `DictSpace` and `TupleSpace` spaces, a `"/"`-separated string or a tuple of dict keys and tuple positions, with `""` / `()` for the space itself. Its metadata is merged into that space's (replaced with `replace=True`); everything else, including metadata already present, is kept. Unknown paths raise a `KeyError` listing all of them unless `strict=False`:

```python
from unienv_interface.space.space_utils.gym_utils import from_gym_space
from unienv_interface.space.space_utils.metadata_utils import annotate, image_metadata, proprio_metadata

obs_space = annotate(from_gym_space(env.observation_space, backend), {
    "pixels": image_metadata(source="front_cam"),
    "agent/qpos": proprio_metadata(units="rad", names=joint_names),
    ("agent", "tcp_pose"): proprio_metadata(frame="base", rotation="quat_wxyz"),
})
```

`metadata_issues(space)` lists the problems with the metadata of a space and all its nested spaces, and `validate_metadata(space)` raises a `ValueError` listing them. They check the types of the vocabulary keys, the values of `format`, `layout` and `rotation`, that per element lists match the last axis of array spaces, and that image and depth spaces fit their `layout` and `format` (`rgb` / `bgr` 3 channels, `rgba` / `bgra` 4, `gray` 1, depth 1; leading batch dims allowed). They are lenient: unknown keys and kinds pass, and what cannot be checked is not. Use them in an adaptor's tests:

```python
from unienv_interface.space.space_utils.metadata_utils import validate_metadata

def test_spaces_are_described():
    env = make_env()
    validate_metadata(env.observation_space)
    validate_metadata(env.action_space)
```

`common_metadata(spaces)` returns the metadata all `spaces` share (empty if they differ); it is the rule used where several spaces become one.

## Serialization

Spaces can be serialized and restored through the space serialization helpers used across the package.

That matters when:

- a replay buffer needs to persist its schema
- a storage backend needs to reconstruct structured data on load
- a transformation or wrapper needs to preserve its target-space definition

Serialized spaces include their (nested) metadata when there is any; payloads written before metadata existed still load.

## Practical Guidance

Use a `DictSpace` when the names of fields matter to downstream code.

Use a flattened representation only at the boundary where a learner, policy, or storage format truly needs it.

Keep the original structured space around as long as possible. That is where most of UniEnv's portability comes from.
