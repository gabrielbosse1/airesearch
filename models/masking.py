"""Validation helpers for raw-byte edit masks and requested values."""
from __future__ import annotations

import torch


def binary_edit_mask(mask: torch.Tensor, shape: tuple[int, int], device: torch.device,
                     *, name: str = "edit_mask") -> torch.Tensor:
    if not torch.is_tensor(mask) or tuple(mask.shape) != shape:
        raise ValueError(f"{name} must be a tensor with shape {shape}")
    if mask.dtype == torch.bool:
        return mask.to(device=device)
    if mask.is_complex() or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return mask.to(device=device, dtype=torch.bool)


def requested_bytes(ids: torch.Tensor, shape: tuple[int, int], device: torch.device,
                    *, name: str = "requested_byte_ids") -> torch.Tensor:
    if not torch.is_tensor(ids) or tuple(ids.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"{name} must be integer raw bytes")
    result = ids.to(device=device, dtype=torch.long)
    if bool(((result < 0) | (result > 255)).any()):
        raise ValueError(f"{name} must contain values in [0, 255]")
    return result
