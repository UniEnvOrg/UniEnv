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

Nested spaces carry their own metadata, so each camera in a `DictSpace` is described separately. These keys form a shared vocabulary; none is required and any other keys are allowed:

| Key | Meaning | Examples |
| --- | --- | --- |
| `kind` | what the data is | `"image"`, `"depth"`, `"proprio"`, `"action"` |
| `format` | pixel format of an image | `"rgb"`, `"bgr"`, `"gray"`, `"rgba"` |
| `layout` | axis order of the trailing image dims | `"HWC"`, `"CHW"` |
| `units` | physical units | `"m"`, `"rad"` |
| `source` | camera or sensor name | `"front_cam"` |
| `fps` | capture rate | `30` |
| `label` | human readable name | `"gripper camera"` |

Metadata describes the data but is not part of the data domain: `==`, `is_subspace` and `is_subspaceeq` ignore it, so compatibility checks between environments, wrappers and replay buffers are unaffected. Use `space_utils.metadata_utils.spaces_equal_with_metadata` to compare it too, and `find_spaces_by_metadata(space, kind="image")` to locate the described leaves.

Metadata follows the space where its meaning survives: `space.to(...)`, the batching utilities, serialization, frame stacking and the transformations that keep the meaning of the data (rescale, crop, image resize, batchify, dict filtering and flattening). It is dropped where the meaning is lost, e.g. by `flatten_space`, and spaces built from data (`construct_space_from_data`, `from_gym_space`) have none.

UniEnv uses it to:

- record the image leaves of observations with the video wrappers, converting `CHW` / `bgr` images for writing
- route image leaves to image storages with `dict_storage_kwargs_from_space`
- describe the features of LeRobot datasets

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
