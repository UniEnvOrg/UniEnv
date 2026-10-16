import base64
import importlib
from typing import Type, Callable, Any, Dict, Mapping

__all__ = [
    "get_full_class_name",
    "get_class_from_full_name",
    "get_full_function_name",
    "get_function_from_full_name",
    "serialize_function",
    "deserialize_function",
]

REMAP = {
    "unienv_data.storages.common.FlattenedStorage": "unienv_data.storages.flattened.FlattenedStorage",
}

def get_full_class_name(cls : Type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"

def _resolve_full_name(full_name : str) -> Any:
    """Resolve ``module.path.Qual.Name`` where the qualified name may be nested (e.g. ``Cls.method``).

    The longest importable module prefix is used and the remaining parts are
    looked up as attributes.
    """
    parts = full_name.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            # Only skip this prefix if it is the prefix itself that is not a module;
            # re-raise import errors coming from inside an existing module.
            if e.name is not None and (module_name == e.name or module_name.startswith(e.name + ".")):
                continue
            raise
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"Cannot resolve {full_name!r}: no importable module prefix")

def get_class_from_full_name(full_name : str) -> Type:
    if full_name in REMAP:
        full_name = REMAP[full_name]
    return _resolve_full_name(full_name)


def get_full_function_name(fn: Callable) -> str:
    """Get the full name of a function for serialization."""
    return f"{fn.__module__}.{fn.__qualname__}"


def get_function_from_full_name(full_name: str) -> Callable:
    """Get a function from its full name (``module.qualname``, qualname may be nested)."""
    return _resolve_full_name(full_name)


def serialize_function(fn: Callable) -> Dict[str, Any]:
    """
    Serialize a function to a JSON-compatible dictionary.
    
    Tries to serialize by full name first. If the function doesn't have
    a proper module name (e.g., lambda, local function), uses cloudpickle + base64.
    
    Returns:
        dict: Contains either:
            - {"mode": "name", "value": "full.module.name"}
            - {"mode": "pickle", "value": "base64_encoded_pickle"}
    """
    # Try to get full name - if it's a proper function with a module
    if hasattr(fn, "__module__") and hasattr(fn, "__qualname__") and \
       fn.__module__ is not None and fn.__module__ != "__main__" and \
       fn.__qualname__ is not None and "<" not in fn.__qualname__:
        try:
            # Verify we can retrieve it
            full_name = get_full_function_name(fn)
            resolved = get_function_from_full_name(full_name)
            # Bound methods resolve to the plain function; those must be pickled.
            if resolved is fn or resolved == fn:
                return {"mode": "name", "value": full_name}
        except (ImportError, AttributeError):
            pass
    
    # Fall back to cloudpickle
    import cloudpickle
    pickled = cloudpickle.dumps(fn)
    encoded = base64.b64encode(pickled).decode("utf-8")
    return {"mode": "pickle", "value": encoded}


def deserialize_function(data: Mapping[str, Any]) -> Callable:
    """
    Deserialize a function from a JSON-compatible dictionary.
    
    Args:
        data: Mapping containing the serialized function with "mode" and "value" keys.
        
    Returns:
        Callable: The deserialized function.
    """
    mode = data.get("mode")
    value = data.get("value")
    
    if mode == "name":
        return get_function_from_full_name(value)
    elif mode == "pickle":
        import cloudpickle
        pickled = base64.b64decode(value.encode("utf-8"))
        return cloudpickle.loads(pickled)
    else:
        raise ValueError(f"Unknown serialization mode: {mode}")