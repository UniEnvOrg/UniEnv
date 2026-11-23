"""UniEnv data as MCP tool results: structured JSON plus images."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import mcp_types as types
from unienv_interface.space import BoxSpace, DictSpace, DynamicBoxSpace, Space, TupleSpace
from unienv_interface.space.space_utils.json_utils import describe_metadata, to_json

from .images import encode_png, image_kind, prepare_image, split_batch

__all__ = ["ContentOptions", "Rendered", "render_observation", "render_image", "to_result", "error_result",
           "json_safe", "image_paths", "result_schema", "with_images"]


@dataclass
class ContentOptions:
    """How observations are shown to the model."""

    #: Decimal places of floats in structured output
    digits: int = 4
    #: Arrays with more entries are summarized (shape, min, max, mean) instead of listed
    max_values: int = 256
    #: Images are downscaled (by an integer stride) to at most this many pixels per side
    max_image_side: int = 512
    #: Which image leaves are shown, by path (e.g. ``["front", "cameras/wrist"]``); ``None``: all.
    #: The others are listed in the JSON without pixels. Tools' ``images`` argument overrides it per call.
    cameras: Optional[Sequence[str]] = None
    #: At most this many images per result (``None``: no limit); later ones are listed without pixels
    max_images: Optional[int] = 8
    #: Batched environments: ``"first"`` shows the first env's images only, ``"all"`` every env's
    batch_images: str = "first"


@dataclass
class Rendered:
    """Structured JSON and images for one result; images are referenced from the JSON by number."""

    structured: Dict[str, Any] = field(default_factory=dict)
    images: List[Tuple[str, bytes]] = field(default_factory=list)  # (caption, PNG bytes)

    def add_image(self, caption: str, png: bytes) -> int:
        self.images.append((caption, png))
        return len(self.images)

    def full(self, options: "ContentOptions") -> bool:
        return options.max_images is not None and len(self.images) >= options.max_images


def with_images(options: ContentOptions, images: Union[None, str, Sequence[str]]) -> ContentOptions:
    """``options`` with a tool call's ``images`` argument applied: ``"all"``, ``"none"`` or a list of paths."""
    if images is None:
        return options
    if images == "all":
        return replace(options, cameras=None)
    if images == "none":
        return replace(options, cameras=())
    if isinstance(images, (list, tuple)) and all(isinstance(path, str) for path in images):
        return replace(options, cameras=tuple(images))
    raise ValueError('images must be "all", "none" or a list of camera paths')


def _shown(path: str, options: ContentOptions) -> bool:
    return options.cameras is None or any(path == c or path.startswith(c.rstrip("/") + "/") for c in options.cameras)


def image_paths(space: Optional[Space], batched: bool, path: str = "") -> List[str]:
    """The paths of the image leaves of ``space`` (the names the ``images`` argument takes)."""
    if space is None:
        return []
    if isinstance(space, DictSpace):
        return [p for key, sub in space.spaces.items() for p in image_paths(sub, batched, f"{path}/{key}".lstrip("/"))]
    if isinstance(space, TupleSpace):
        return [p for i, sub in enumerate(space.spaces) for p in image_paths(sub, batched, f"{path}[{i}]")]
    return [path or "observation"] if image_kind(space, batched) is not None else []


def _summary(array: np.ndarray, digits: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size and (np.issubdtype(array.dtype, np.number) or array.dtype == bool):
        values = array.astype(np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            summary.update(min=round(float(finite.min()), digits), max=round(float(finite.max()), digits),
                           mean=round(float(finite.mean()), digits))
    summary["note"] = "too large to list; summarized"
    return summary


def _leaf(space: Space, data: Any, path: str, batch_ndim: int, out: Rendered, options: ContentOptions) -> Any:
    kind = image_kind(space, batch_ndim > 0)
    if kind is not None:
        array = np.asarray(space.backend.to_numpy(data))
        refs = []
        for number, (index, image) in enumerate(split_batch(array, batch_ndim)):
            height, width = image.shape[1:3] if str((space.metadata or {}).get("layout", "")).upper() == "CHW" \
                else image.shape[:2]
            if not _shown(path, options):
                refs.append({"image": None, "height": int(height), "width": int(width),
                             "omitted": "not requested (images argument)"})
                continue
            if batch_ndim and options.batch_images == "first" and number > 0:
                refs.append({"image": None, "omitted": "only the first env's images are shown"})
                continue
            if out.full(options):
                refs.append({"image": None, "height": int(height), "width": int(width),
                             "omitted": f"image limit ({options.max_images}) reached"})
                continue
            prepared, notes = prepare_image(space, image, kind, max_side=options.max_image_side)
            caption = path + ("" if not index else f" (env {', '.join(str(i) for i in index)})")
            number = out.add_image(caption, encode_png(prepared))
            refs.append({"image": number, "height": int(prepared.shape[0]), "width": int(prepared.shape[1]), **notes})
        return refs[0] if batch_ndim == 0 else refs
    if isinstance(space, (BoxSpace, DynamicBoxSpace)):
        array = np.asarray(space.backend.to_numpy(data))
        if array.size > options.max_values:
            return _summary(array, options.digits)
    try:
        return to_json(space, data, digits=options.digits)
    except TypeError:
        return {"note": f"{type(space).__name__} data is not shown"}


def _walk(space: Space, data: Any, path: str, batch_ndim: int, out: Rendered, options: ContentOptions) -> Any:
    if isinstance(space, DictSpace):
        return {str(key): _walk(sub, data[key], f"{path}/{key}".lstrip("/"), batch_ndim, out, options)
                for key, sub in space.spaces.items()}
    if isinstance(space, TupleSpace):
        return [_walk(sub, item, f"{path}[{i}]", batch_ndim, out, options)
                for i, (sub, item) in enumerate(zip(space.spaces, data))]
    return _leaf(space, data, path or "observation", batch_ndim, out, options)


def render_observation(space: Optional[Space], data: Any, out: Rendered, key: str, *, batched: bool,
                       options: ContentOptions) -> None:
    """Add ``data`` of ``space`` to ``out.structured[key]`` (images go to ``out.images``)."""
    if space is None or data is None:
        out.structured[key] = None
        return
    out.structured[key] = _walk(space, data, key if key != "observation" else "", 1 if batched else 0, out, options)


def render_image(frame: Any, caption: str, out: Rendered, options: ContentOptions) -> Optional[int]:
    """Add a rendered frame (``H x W x C`` array) as an image; ``None`` if it is not one or the limit is reached."""
    array = np.asarray(frame) if not hasattr(frame, "cpu") else frame.detach().cpu().numpy()
    if array.ndim not in (2, 3) or out.full(options):
        return None
    dummy = BoxSpace.__new__(BoxSpace)  # prepare_image only reads metadata
    object.__setattr__(dummy, "_metadata", {})
    prepared, _ = prepare_image(dummy, array, "image", max_side=options.max_image_side)
    return out.add_image(caption, encode_png(prepared))


def json_safe(value: Any, digits: int = 4) -> Any:
    """``value`` (e.g. an ``info`` dict) with arrays as lists; anything else unserializable as its repr."""
    if isinstance(value, dict):
        return {str(k): json_safe(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, digits) for v in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "__array__") and not isinstance(value, (str, bytes)):
        array = np.asarray(value)
        if array.dtype == object:
            return [json_safe(v, digits) for v in array.reshape(-1)]
        if array.size > 256:
            return _summary(array, digits)
        value = array.tolist()
        return _round(value, digits)
    if isinstance(value, float):
        return round(value, digits) if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _round(value: Any, digits: int) -> Any:
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits) if np.isfinite(value) else None
    return value


def to_result(out: Rendered) -> types.CallToolResult:
    """The MCP result: the JSON as text and structured content, then each image after its caption."""
    content: List[Any] = [types.TextContent(type="text", text=json.dumps(out.structured, separators=(",", ":")))]
    for number, (caption, png) in enumerate(out.images, start=1):
        content.append(types.TextContent(type="text", text=f"image {number}: {caption}"))
        content.append(types.ImageContent(type="image", data=base64.b64encode(png).decode("ascii"),
                                          mime_type="image/png"))
    return types.CallToolResult(content=content, structured_content=out.structured)


def error_result(message: str) -> types.CallToolResult:
    """A tool error the model can read and react to (bad arguments, unknown instance, ...)."""
    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], is_error=True)


def space_summary(space: Optional[Space]) -> Optional[str]:
    return None if space is None else describe_metadata(space)


def result_schema(space: Optional[Space], batched: bool, options: ContentOptions, path: str = "") -> Any:
    """The JSON Schema of how ``space``'s data appears in results: image leaves as image references,
    large arrays as summaries, the rest as their JSON form."""
    from unienv_interface.space.space_utils.json_utils import space_to_json_schema
    if space is None:
        return None
    if isinstance(space, DictSpace):
        return {"type": "object", "properties": {
            str(key): result_schema(sub, batched, options, f"{path}/{key}".lstrip("/")) for key, sub in space.spaces.items()
        }}
    if isinstance(space, TupleSpace):
        return {"type": "array", "prefixItems": [result_schema(sub, batched, options, f"{path}[{i}]")
                                                 for i, sub in enumerate(space.spaces)]}
    kind = image_kind(space, batched)
    if kind is not None:
        reference = {"type": "object", "description": (
            f"{'a depth image (gray: near is dark), with depth_range' if kind == 'depth' else 'an image'}: returned "
            "as image content after the JSON; `image` is its number, null when omitted (see `omitted`)"),
            "properties": {"image": {"type": ["integer", "null"]}, "height": {"type": "integer"},
                           "width": {"type": "integer"}, "omitted": {"type": "string"}}}
        return {"type": "array", "items": reference, "description": "one per env"} if batched else reference
    try:
        schema = space_to_json_schema(space)
    except TypeError as exc:
        return {"description": str(exc)}
    size = int(np.prod(tuple(getattr(space, "shape", None) or ())))
    if isinstance(space, (BoxSpace, DynamicBoxSpace)) and size > options.max_values:
        return {"description": f"summarized ({size} values): shape, dtype, min, max, mean", "type": "object"}
    return schema
