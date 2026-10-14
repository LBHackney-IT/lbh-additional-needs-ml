"""Runtime device selection for inference services."""

from __future__ import annotations

import torch


def resolve_device(requested: torch.device | None = None) -> torch.device:
    """Return the requested device or the fastest available local device."""
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
