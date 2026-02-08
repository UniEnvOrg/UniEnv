"""
Serialization utilities for DataTransformations.

Provides functions to serialize/deserialize DataTransformation objects to/from JSON-compatible dictionaries.
"""
from typing import Dict, Any, Type

from unienv_interface.utils.symbol_util import get_class_from_full_name
from .transformation import DataTransformation

__all__ = [
    "transformation_to_json",
    "json_to_transformation",
]


def transformation_to_json(transformation: DataTransformation) -> Dict[str, Any]:
    """
    Serialize a DataTransformation to a JSON-compatible dictionary.

    Args:
        transformation: The transformation to serialize.

    Returns:
        dict: A JSON-compatible representation of the transformation.
    """
    return transformation.serialize()


def json_to_transformation(json_data: Dict[str, Any]) -> DataTransformation:
    """
    Deserialize a DataTransformation from a JSON-compatible dictionary.

    Args:
        json_data: The dictionary containing the transformation data.

    Returns:
        DataTransformation: The deserialized transformation.
    """
    type_name = json_data.get("type")
    if type_name is None:
        raise ValueError(f"JSON data must contain 'type' field: {json_data}")

    transformation_class: Type[DataTransformation] = get_class_from_full_name(type_name)
    return transformation_class.deserialize_from(json_data)
