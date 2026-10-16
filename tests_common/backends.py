"""Backend/device parametrisation that only offers devices present on this machine."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest

from unienv_interface.backends import ComputeBackend
from unienv_interface.backends.numpy import NumpyComputeBackend


def _torch_devices() -> List[str]:
    try:
        import torch
    except ImportError:
        return []
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _jax_devices() -> List[Any]:
    try:
        import jax
    except ImportError:
        return []
    return list(jax.devices())


def backends_and_devices() -> List[Tuple[type, Optional[Any]]]:
    """Every (backend, device) pair usable here; numpy always uses ``device=None``."""
    pairs: List[Tuple[type, Optional[Any]]] = [(NumpyComputeBackend, None)]
    torch_devices = _torch_devices()
    if torch_devices:
        from unienv_interface.backends.pytorch import PyTorchComputeBackend
        pairs += [(PyTorchComputeBackend, d) for d in torch_devices]
    jax_devices = _jax_devices()
    if jax_devices:
        from unienv_interface.backends.jax import JaxComputeBackend
        pairs += [(JaxComputeBackend, d) for d in jax_devices]
    return pairs


def _param_id(backend: type, device: Optional[Any]) -> str:
    name = backend.__name__.replace("ComputeBackend", "").lower()
    return name if device is None else f"{name}-{device}"


def backend_params() -> list:
    """``pytest.param`` list for ``@pytest.mark.parametrize("backend,device", ...)``.

    CUDA/GPU devices carry the ``gpu`` marker so they can be deselected.
    """
    params = []
    for backend, device in backends_and_devices():
        is_gpu = device is not None and ("cuda" in str(device).lower() or "gpu" in str(device).lower())
        params.append(pytest.param(
            backend, device,
            id=_param_id(backend, device),
            marks=[pytest.mark.gpu] if is_gpu else [],
        ))
    return params


def cpu_backend_params() -> list:
    """Like :func:`backend_params` but one CPU entry per backend (cheaper)."""
    seen, params = set(), []
    for p in backend_params():
        backend, device = p.values
        if p.marks or backend in seen:
            continue
        seen.add(backend)
        params.append(p)
    return params
