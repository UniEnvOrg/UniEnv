"""Build a ``DictStorage`` layout from the ``Space.metadata`` of a ``DictSpace``.

Camera images are best stored by an image / video storage, everything else by a
tensor storage. Instead of listing the keys by hand::

    from unienv_data.replay_buffer import ReplayBuffer
    from unienv_data.storages import DictStorage, dict_storage_kwargs_from_space

    rb = ReplayBuffer.create(
        DictStorage, obs_space,
        cache_path="cache/rb", capacity=None,
        **dict_storage_kwargs_from_space(obs_space),
    )

routes every ``kind == "image"`` uint8 leaf of ``obs_space`` to an
``ImageStorage`` and the remaining leaves to one ``HDF5Storage``.
"""
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from unienv_interface.space import BoxSpace, DictSpace, Space
from unienv_data.base import SpaceStorage

from .dict_storage import DictStorage

__all__ = [
    "dict_storage_kwargs_from_space",
    "create_dict_storage_from_space",
]

def _default_kind_storage_cls() -> Dict[str, Optional[Type[SpaceStorage]]]:
    from .image_storage import ImageStorage
    return {"image": ImageStorage}

def _default_storage_cls() -> Type[SpaceStorage]:
    try:
        from .hdf5 import HDF5Storage
    except ImportError as e:  # h5py is optional
        raise ImportError(
            "The default storage for non-image data is HDF5Storage, which needs h5py; "
            "install h5py or pass `default_storage_cls`"
        ) from e
    return HDF5Storage

def _is_hwc_rgb_uint8(space : Space) -> bool:
    """A single (H, W, 3) uint8 image whose metadata does not declare another layout."""
    if not isinstance(space, BoxSpace) or len(space.shape) != 3 or space.shape[-1] != 3:
        return False
    if space.backend.to_numpy(space.backend.zeros((1,), dtype=space.dtype)).dtype.name != "uint8":
        return False
    layout = space.metadata.get("layout")
    return layout is None or str(layout).upper() == "HWC"

def _accepts(storage_cls : Type[SpaceStorage], kind : str, space : Space) -> bool:
    """Whether a leaf of ``kind`` can go to ``storage_cls``.

    ``ImageStorage`` and ``VideoStorage`` (and ``kind == "image"`` in general) only
    take (H, W, 3) uint8 images; other storages and kinds are taken as configured.
    """
    from .image_storage import ImageStorage
    image_storage_classes : Tuple[type, ...] = (ImageStorage,)
    try:
        from .video_storage import VideoStorage
        image_storage_classes += (VideoStorage,)
    except ImportError:
        pass
    if kind == "image" or issubclass(storage_cls, image_storage_classes):
        return _is_hwc_rgb_uint8(space)
    return True

def _iter_dict_leaves(space : DictSpace, separator : str, prefix : str = ""):
    for key, subspace in space.spaces.items():
        path = prefix + key
        if isinstance(subspace, DictSpace):
            yield from _iter_dict_leaves(subspace, separator, path + separator)
        else:
            yield path, subspace

def dict_storage_kwargs_from_space(
    space : DictSpace,
    *,
    kind_storage_cls : Optional[Mapping[str, Optional[Type[SpaceStorage]]]] = None,
    key_storage_cls : Optional[Mapping[str, Optional[Type[SpaceStorage]]]] = None,
    default_storage_cls : Optional[Type[SpaceStorage]] = None,
    kind_kwargs : Optional[Mapping[str, Mapping[str, Any]]] = None,
    key_kwargs : Optional[Mapping[str, Mapping[str, Any]]] = None,
    default_kwargs : Optional[Mapping[str, Any]] = None,
    nested_separator : str = "/",
) -> Dict[str, Any]:
    """``DictStorage.create`` keyword arguments that route each leaf of ``space`` by its metadata.

    Every leaf of the (nested) ``DictSpace`` is assigned, in order of precedence:

    1. ``key_storage_cls[key]`` if its key (nested keys joined by ``nested_separator``)
       is listed (``None`` sends it to the default storage);
    2. ``kind_storage_cls[leaf.metadata["kind"]]`` if its kind is listed and the
       storage can take it. The default is ``{"image": ImageStorage}``, i.e. JPEG
       files (``ImageStorage``'s defaults; use ``kind_kwargs={"image": {"format": "PNG"}}``
       for lossless files or map ``"image"`` to ``VideoStorage``). Image leaves must be
       (H, W, 3) uint8 with an ``"HWC"`` (or undeclared) layout, others (e.g. CHW or
       float images) go to the default storage;
    3. otherwise the default storage, ``default_storage_cls`` (``HDF5Storage`` if not
       given, which needs h5py and a NumPy-backend space), which receives all remaining
       leaves as one ``DictSpace`` (key ``"*"``).

    Keyword arguments for the sub-storages are merged from ``kind_kwargs[kind]``
    and ``key_kwargs[key]`` (``default_kwargs`` / ``key_kwargs["*"]`` for the default
    storage). A ``VideoStorage`` leaf whose metadata declares ``fps`` gets it as the
    ``fps`` argument unless one is given.

    Returns ``{"storage_cls_map", "key_kwargs", "nested_separator"}``, to be passed as
    ``ReplayBuffer.create(DictStorage, space, **kwargs)`` or ``DictStorage.create(space, **kwargs)``.
    Image and video storages write files, so a ``cache_path`` is required when any leaf
    is routed to one.
    """
    if not isinstance(space, DictSpace):
        raise TypeError(f"dict_storage_kwargs_from_space expects a DictSpace, got {type(space).__name__}")
    kind_storage_cls = _default_kind_storage_cls() if kind_storage_cls is None else dict(kind_storage_cls)
    key_storage_cls = dict(key_storage_cls or {})
    default_storage_cls = default_storage_cls or _default_storage_cls()
    kind_kwargs = kind_kwargs or {}
    user_key_kwargs = dict(key_kwargs or {})

    try:
        from .video_storage import VideoStorage
    except ImportError:
        VideoStorage = None

    storage_cls_map : Dict[str, Type[SpaceStorage]] = {}
    out_key_kwargs : Dict[str, Dict[str, Any]] = {}
    for key, leaf in _iter_dict_leaves(space, nested_separator):
        kind = leaf.metadata.get("kind")
        if key in key_storage_cls:
            storage_cls = key_storage_cls[key]
        elif kind in kind_storage_cls and kind_storage_cls[kind] is not None and _accepts(kind_storage_cls[kind], kind, leaf):
            storage_cls = kind_storage_cls[kind]
        else:
            storage_cls = None
        if storage_cls is None:
            continue  # stored by the default storage
        leaf_kwargs = dict(kind_kwargs.get(kind, {})) if kind is not None else {}
        leaf_kwargs.update(user_key_kwargs.get(key, {}))
        if (
            VideoStorage is not None and issubclass(storage_cls, VideoStorage)
            and "fps" not in leaf_kwargs and "fps" in leaf.metadata
        ):
            leaf_kwargs["fps"] = int(round(leaf.metadata["fps"]))
        storage_cls_map[key] = storage_cls
        if len(leaf_kwargs) > 0:
            out_key_kwargs[key] = leaf_kwargs

    storage_cls_map["*"] = default_storage_cls
    residual_kwargs = dict(default_kwargs or {})
    residual_kwargs.update(user_key_kwargs.get("*", {}))
    if len(residual_kwargs) > 0:
        out_key_kwargs["*"] = residual_kwargs
    return {
        "storage_cls_map": storage_cls_map,
        "key_kwargs": out_key_kwargs,
        "nested_separator": nested_separator,
    }

def create_dict_storage_from_space(
    space : DictSpace,
    *,
    capacity : Optional[int] = None,
    cache_path : Optional[str] = None,
    multiprocessing : bool = False,
    **routing_kwargs : Any,
) -> DictStorage:
    """``DictStorage.create`` with the layout of :func:`dict_storage_kwargs_from_space` (which takes ``routing_kwargs``)."""
    return DictStorage.create(
        space,
        capacity=capacity,
        cache_path=cache_path,
        multiprocessing=multiprocessing,
        **dict_storage_kwargs_from_space(space, **routing_kwargs),
    )
