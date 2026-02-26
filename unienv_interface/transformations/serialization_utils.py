"""
Serialization utilities for DataTransformations.

Provides functions to serialize/deserialize DataTransformation objects to/from JSON-compatible dictionaries.
"""
from typing import Dict, Any, Type, Optional

from unienv_interface.space import Space
from unienv_interface.utils.symbol_util import get_full_class_name, get_class_from_full_name
from .transformation import DataTransformation

__all__ = [
    "transformation_to_json",
    "json_to_transformation",
]


def transformation_to_json(
    transformation: DataTransformation,
    source_space: Optional[Space] = None,
) -> Dict[str, Any]:
    """
    Serialize a DataTransformation to a JSON-compatible dictionary.

    Args:
        transformation: The transformation to serialize.
        source_space: Optional source space context used by transforms that need
            backend/device information during serialization.

    Returns:
        dict: A JSON-compatible representation of the transformation.
    """
    json_data = dict(transformation.serialize(source_space=source_space))
    json_data["type"] = get_full_class_name(type(transformation))
    return json_data


def json_to_transformation(
    json_data: Dict[str, Any],
    source_space: Optional[Space] = None,
) -> DataTransformation:
    """
    Deserialize a DataTransformation from a JSON-compatible dictionary.

    Args:
        json_data: The dictionary containing the transformation data.
        source_space: Optional source space context used by transforms that need
            backend/device information during deserialization.

    Returns:
        DataTransformation: The deserialized transformation.
    """
    type_name = json_data.get("type")
    if type_name is None:
        raise ValueError(f"JSON data must contain 'type' field: {json_data}")

    transformation_class: Type[DataTransformation] = get_class_from_full_name(type_name)
    payload = dict(json_data)
    payload.pop("type", None)
    return transformation_class.deserialize_from(payload, source_space=source_space)
