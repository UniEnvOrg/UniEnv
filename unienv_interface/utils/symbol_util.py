from typing import Type

__all__ = [
    "get_full_class_name",
    "get_class_from_full_name",
]

def get_full_class_name(cls : Type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"

def get_class_from_full_name(full_name : str) -> Type:
    module_name, class_name = full_name.rsplit(".", 1)
    return getattr(__import__(module_name, fromlist=[class_name]), class_name)
