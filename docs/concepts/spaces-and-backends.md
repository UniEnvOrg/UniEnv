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

## Serialization

Spaces can be serialized and restored through the space serialization helpers used across the package.

That matters when:

- a replay buffer needs to persist its schema
- a storage backend needs to reconstruct structured data on load
- a transformation or wrapper needs to preserve its target-space definition

## Practical Guidance

Use a `DictSpace` when the names of fields matter to downstream code.

Use a flattened representation only at the boundary where a learner, policy, or storage format truly needs it.

Keep the original structured space around as long as possible. That is where most of UniEnv's portability comes from.
