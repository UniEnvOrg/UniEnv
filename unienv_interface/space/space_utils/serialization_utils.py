import typing
from copy import deepcopy
from functools import singledispatch
from typing import Optional, Any, Iterable, Iterator
import numpy as np
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.backends import serialization as bsu
from unienv_interface.space.spaces import *

__all__ = [
    "space_to_json",
    "json_to_space",
]

@singledispatch
def space_to_json(space : Space) -> typing.Dict[str, Any]:
    """
    Serialize a Space to a JSON-compatible dictionary.
    
    Args:
        space (Space): The space to serialize.
    
    Returns:
        dict: A JSON-compatible representation of the space.
    """
    raise NotImplementedError(f"Serialization for {type(space)} is not implemented.")

@singledispatch
def json_to_space(json_data: typing.Dict[str, Any], map_backend: ComputeBackend, map_device: Optional[BDeviceType]) -> Space:
    """
    Deserialize a JSON-compatible dictionary to a Space.
    
    Args:
        json_data (dict): The JSON-compatible representation of the space.
        backend (ComputeBackend): The backend to use for creating the space.
    
    Returns:
        Space: The deserialized space.
    """
    type_map = {
        "BoxSpace": BoxSpace,
        "BinarySpace": BinarySpace,
        "DynamicBoxSpace": DynamicBoxSpace,
        "TupleSpace": TupleSpace,
        "DictSpace": DictSpace,
        "TextSpace": TextSpace,
        "GraphSpace": GraphSpace,
        "UnionSpace": UnionSpace,
    }
    type = json_data.get("type")
    if type not in type_map:
        raise ValueError(f"Unknown space type: {type}")
    space_class = type_map[type]
    if space_class not in json_to_space.registry:
        raise NotImplementedError(f"Deserialization for {space_class} is not implemented.")
    return json_to_space.dispatch(space_class)(json_data, map_backend, map_device)

@space_to_json.register(BoxSpace)
def _box_space_to_json(space: BoxSpace) -> typing.Dict[str, Any]:
    return {
        "type": "BoxSpace",
        "shape": space.shape,
        "low": space.backend.to_numpy(space._low).tolist(),
        "high": space.backend.to_numpy(space._high).tolist(),
        "dtype": bsu.serialize_dtype(space.backend, space.dtype),
        "shape": space.shape,
    }

@json_to_space.register(BoxSpace)
def _json_to_box_space(json_data: typing.Dict[str, Any], map_backend: ComputeBackend, map_device: Optional[BDeviceType]) -> BoxSpace:
    np_dtype = bsu.deserialize_dtype(NumpyComputeBackend, json_data["dtype"])
    new_dtype = bsu.deserialize_dtype(map_backend, json_data["dtype"])

    low = map_backend.from_numpy(np.array(json_data["low"], dtype=np_dtype), dtype=new_dtype, device=map_device)
    high = map_backend.from_numpy(np.array(json_data["high"], dtype=np_dtype), dtype=new_dtype, device=map_device)
    return BoxSpace(
        map_backend,
        low=low,
        high=high,
        shape=json_data["shape"],
        dtype=bsu.deserialize_dtype(map_backend, json_data["dtype"]),
        device=map_device
    )

@space_to_json.register(BinarySpace)
def _binary_space_to_json(space: BinarySpace) -> typing.Dict[str, Any]:
    return {
        "type": "BinarySpace",
        "shape": space.shape,
        "dtype": bsu.serialize_dtype(space.backend, space.dtype),
    }

@json_to_space.register(BinarySpace)
def _json_to_binary_space(json_data: typing.Dict[str, Any], map_backend: ComputeBackend, map_device: Optional[BDeviceType]) -> BinarySpace:
    return BinarySpace(
        map_backend,
        shape=json_data["shape"],
        dtype=bsu.deserialize_dtype(map_backend, json_data["dtype"]),
        device=map_device
    )

@space_to_json.register(DynamicBoxSpace)
def _dynamic_box_space_to_json(space: DynamicBoxSpace) -> typing.Dict[str, Any]:
    return {
        "type": "DynamicBoxSpace",
        "low": space.backend.to_numpy(space._low).tolist(),
        "high": space.backend.to_numpy(space._high).tolist(),
        "shape_low": space.shape_low,
        "shape_high": space.shape_high,
        "dtype": bsu.serialize_dtype(space.backend, space.dtype),
        "fill_value": space.fill_value
    }

@json_to_space.register(DynamicBoxSpace)
def _json_to_dynamic_box_space(json_data: typing.Dict[str, Any], map_backend: ComputeBackend, map_device: Optional[BDeviceType]) -> DynamicBoxSpace:
    np_dtype = bsu.deserialize_dtype(NumpyComputeBackend, json_data["dtype"])
    new_dtype = bsu.deserialize_dtype(map_backend, json_data["dtype"])

    low = map_backend.from_numpy(np.array(json_data["low"], dtype=np_dtype), dtype=new_dtype, device=map_device)
    high = map_backend.from_numpy(np.array(json_data["high"], dtype=new_dtype), device=map_device)
    return DynamicBoxSpace(
        map_backend,
        low=low,
        high=high,
        shape_low=json_data["shape_low"],
        shape_high=json_data["shape_high"],
        dtype=bsu.deserialize_dtype(map_backend, json_data["dtype"]),
        device=map_device,
        fill_value=json_data["fill_value"]
    )

@space_to_json.register(TupleSpace)
def _tuple_space_to_json(space: TupleSpace) -> typing.Dict[str, Any]:
    return {
        "type": "TupleSpace",
        "spaces": [space_to_json(s) for s in space.spaces],
    }

@json_to_space.register(TupleSpace)
def _json_to_tuple_space(json_data: typing.Dict[str, Any], map_backend: ComputeBackend, map_device: Optional[BDeviceType]) -> TupleSpace:
    spaces = [json_to_space(s, map_backend, map_device) for s in json_data["spaces"]]
    return TupleSpace(map_backend, spaces, device=map_device)

@space_to_json.register(DictSpace)
def _dict_space_to_json(space: DictSpace) -> typing.Dict[str, Any]:
    return {
        "type": "DictSpace",
        "spaces": {k: space_to_json(v) for k, v in space.spaces.items()},
    }

@json_to_space.register(DictSpace)
def _json_to_dict_space(json_data: typing.Dict[str, Any], map_backend: ComputeBackend, map_device: Optional[BDeviceType]) -> DictSpace:
    spaces = {k: json_to_space(v, map_backend, map_device) for k, v in json_data["spaces"].items()}
    return DictSpace(map_backend, spaces, device=map_device)

@space_to_json.register(TextSpace)
def _text_space_to_json(space: TextSpace) -> typing.Dict[str, Any]:
    return {
        "type": "TextSpace",
        "min_length": space.min_length,
        "max_length": space.max_length,
        "charset": "".join(space.charset) if space.charset is not None else None,
    }

@json_to_space.register(TextSpace)
def _json_to_text_space(json_data: typing.Dict[str, Any], map_backend: ComputeBackend, map_device: Optional[BDeviceType]) -> TextSpace:
    return TextSpace(
        map_backend,
        max_length=json_data['max_length'],
        min_length=json_data['min_length'],
        charset=json_data.get("charset", None),
        device=map_device
    )

@space_to_json.register(GraphSpace)
def _graph_space_to_json(space: GraphSpace) -> typing.Dict[str, Any]:
    return {
        "type": "GraphSpace",
        "max_nodes": space.max_nodes,
        "min_nodes": space.min_nodes,
        "max_edges": space.max_edges,
        "min_edges": space.min_edges,
        "is_edge": space.is_edge,
        "node_feature_space": space_to_json(space.node_feature_space),
        "edge_feature_space": space_to_json(space.edge_feature_space) if space.edge_feature_space is not None else None,
        "batch_shape": space.batch_shape,
    }

@json_to_space.register(GraphSpace)
def _json_to_graph_space(json_data: typing.Dict[str, Any], map_backend: ComputeBackend, map_device: Optional[BDeviceType]) -> GraphSpace:
    node_feature_space = json_to_space(json_data["node_feature_space"], map_backend, map_device)
    edge_feature_space = None
    if json_data["edge_feature_space"] is not None:
        edge_feature_space = json_to_space(json_data["edge_feature_space"], map_backend, map_device)

    return GraphSpace(
        map_backend,
        node_feature_space=node_feature_space,
        edge_feature_space=edge_feature_space,
        is_edge=json_data['is_edge'],
        min_nodes=json_data['min_nodes'],
        max_nodes=json_data.get('max_nodes', None),
        min_edges=json_data['min_edges'],
        max_edges=json_data.get('max_edges', None),
        batch_shape=json_data.get("batch_shape", ()),
        device=map_device
    )

@space_to_json.register(UnionSpace)
def _union_space_to_json(space: UnionSpace) -> typing.Dict[str, Any]:
    return {
        "type": "UnionSpace",
        "spaces": [space_to_json(s) for s in space.spaces],
    }

@json_to_space.register(UnionSpace)
def _json_to_union_space(json_data: typing.Dict[str, Any], map_backend: ComputeBackend, map_device: Optional[BDeviceType]) -> UnionSpace:
    spaces = [json_to_space(s, map_backend, map_device) for s in json_data["spaces"]]
    return UnionSpace(map_backend, spaces, device=map_device)

@space_to_json.register(BatchedSpace)
def _batched_space_to_json(space: BatchedSpace) -> typing.Dict[str, Any]:
    return {
        "type": "BatchedSpace",
        "single_space": space_to_json(space.single_space),
        "batch_shape": space.batch_shape,
    }

@json_to_space.register(BatchedSpace)
def _json_to_batched_space(json_data: typing.Dict[str, Any], map_backend: ComputeBackend, map_device: Optional[BDeviceType]) -> BatchedSpace:
    single_space = json_to_space(json_data["single_space"], map_backend, map_device)
    return BatchedSpace(
        single_space,
        batch_shape=json_data["batch_shape"]
    )