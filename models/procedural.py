"""Structured procedural memories and safe, deterministic replay results.

A stored skill is data, not executable Python. Exact-match replay cannot call
arbitrary code; an optional host callback may consume the saved latent steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Procedure:
    slot: int
    input_pattern: bytes
    output_pattern: bytes
    latent_trace: torch.Tensor
    explanation_bytes: bytes
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProcedureExecution:
    slot: int
    output: bytes
    latent_trace: torch.Tensor
    explanation_bytes: bytes
    metadata: dict[str, Any]


def raw_bytes(value: bytes | bytearray, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes or bytearray")
    return bytes(value)


def byte_pattern_embedding(data: bytes, d_model: int, *, device: torch.device) -> torch.Tensor:
    """Deterministic order-sensitive sketch for retrieval, not an exact matcher."""
    if d_model < 1:
        raise ValueError("d_model must be positive")
    out = torch.zeros(d_model, device=device)
    if not data:
        return out
    values = torch.tensor(list(data), device=device, dtype=out.dtype)
    positions = torch.arange(len(data), device=device)
    # Two signed positional projections preserve byte value 0 and distinguish
    # permutations better than a bag-of-bytes mean. Exact replay compares bytes.
    out.index_add_(0, positions.remainder(d_model), (values + 1.0) / 256.0)
    out.index_add_(0, (positions * 7 + 3).remainder(d_model), (values - 127.5) / 255.0)
    return out / max(len(data), 1) ** 0.5
