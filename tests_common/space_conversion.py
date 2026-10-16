"""Which cross-backend/device data conversions the installed ``xbarray`` can perform here.

``Space.to`` / ``Space.data_to`` delegate to ``backend.from_other_backend``, which
xbarray implements via DLPack. Two limitations of that (external) path matter
for tests on some machines:

* DLPack keeps the source device, so data on a CUDA device cannot be imported
  into a jax installation without CUDA support ("Unknown backend cuda").
* numpy arrays obtained from jax arrays are read-only, and read-only arrays
  cannot be exported through DLPack, so they cannot be imported back into jax.
"""

from __future__ import annotations

from typing import Any, Optional


def is_gpu_device(device: Optional[Any]) -> bool:
    return device is not None and ("cuda" in str(device).lower() or "gpu" in str(device).lower())


def _jax_has_gpu() -> bool:
    try:
        import jax
    except ImportError:
        return False
    return any(d.platform == "gpu" for d in jax.devices())


def can_convert(src_backend: type, src_device: Optional[Any], dst_backend: type,
                origin_backend: Optional[type] = None) -> bool:
    """Whether data living on ``(src_backend, src_device)`` can be moved to ``dst_backend``.

    ``origin_backend`` is the backend the data was created on before it reached
    ``src_backend`` (if different), since that affects whether it is read-only.
    """
    if dst_backend.simplified_name == "jax" and src_backend.simplified_name != "jax":
        if is_gpu_device(src_device) and not _jax_has_gpu():
            return False
        if src_backend.simplified_name == "numpy" and origin_backend is not None \
                and origin_backend.simplified_name == "jax":
            return False
    return True
