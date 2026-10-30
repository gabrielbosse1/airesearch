"""Differentiable, length-aware soft dynamic time warping in embedding space.

This objective supports raw-byte sequences after embedding. It provides no
new learned parameters and does not itself insert or remove output bytes.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def soft_dtw_alignment(
    prediction: torch.Tensor, target: torch.Tensor, *,
    prediction_mask: torch.Tensor | None = None, target_mask: torch.Tensor | None = None,
    gamma: float = 0.02, max_cells: int = 2_000_000,
) -> torch.Tensor:
    """Mean length-normalized soft-DTW cost using squared embedding distances.

    Masks may contain holes; valid positions are compacted in order. Each
    sequence must have at least one valid position. Soft-DTW can be negative
    because it is an entropy-smoothed cost, not a metric; do not interpret the
    raw value as a distance or compare it across different gamma values.
    The anti-diagonal recurrence remains fully differentiable and bounds memory.
    """
    if prediction.ndim != 3 or target.ndim != 3 or prediction.shape[0] < 1 or prediction.shape[0] != target.shape[0] or prediction.shape[2] != target.shape[2] or prediction.shape[1] < 1 or target.shape[1] < 1:
        raise ValueError("prediction and target must have shapes (B,L,D), (B,M,D), with B,L,M,D positive")
    if prediction.shape[2] < 1 or prediction.device != target.device:
        raise ValueError("embedding dimensions must be positive and share a device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("alignment inputs must be floating point embeddings")
    if not math.isfinite(gamma) or gamma <= 0 or max_cells < 1:
        raise ValueError("gamma and max_cells must be positive")
    device = prediction.device
    b = prediction.shape[0]
    def compact(data: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            lengths = torch.full((b,), data.shape[1], device=device, dtype=torch.long)
            return data, lengths
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.shape != data.shape[:2]:
            raise ValueError("alignment masks must match sequence dimensions")
        lengths = mask.sum(dim=1)
        if bool((lengths == 0).any()):
            raise ValueError("each alignment sequence must include a valid byte")
        max_length = int(lengths.max().item())
        packed = torch.stack([F.pad(data[i][mask[i]], (0, 0, 0, max_length - int(lengths[i].item()))) for i in range(b)])
        return packed, lengths
    prediction, pred_lens = compact(prediction, prediction_mask)
    target, target_lens = compact(target, target_mask)
    if prediction.dtype in (torch.float16, torch.bfloat16):
        prediction = prediction.float()
    if target.dtype in (torch.float16, torch.bfloat16):
        target = target.float()
    target = target.to(dtype=prediction.dtype)
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise ValueError("valid embedding positions must be finite")
    n, m, d = prediction.shape[1], target.shape[1], prediction.shape[2]
    if b * n * m > max_cells:
        raise ValueError("soft-DTW matrix exceeds max_cells; reduce batch or sequence length")
    distances = (prediction.square().sum(-1, keepdim=True) + target.square().sum(-1).unsqueeze(1)
                 - 2 * prediction @ target.transpose(1, 2)).clamp_min(0) / d
    # Finite sentinel avoids NaN gradients from logsumexp([-inf,-inf,-inf]).
    high = min(1e6 * max(1.0, gamma), torch.finfo(prediction.dtype).max / 16)
    prevprev = prediction.new_full((b, n + 1), high)
    prev = torch.cat((prediction.new_zeros((b, 1)), prediction.new_full((b, n), high)), dim=1)
    result = prediction.new_zeros((b,))
    i = torch.arange(1, n + 1, device=device)
    for diagonal in range(1, n + m + 1):
        j = diagonal - i
        valid = (j >= 1) & (j <= m)
        cost = distances[:, i - 1, (j - 1).clamp(0, m - 1)]
        alternatives = torch.stack((prev[:, 1:], prev[:, :-1], prevprev[:, :-1]), dim=-1)
        minimum = alternatives.min(dim=-1).values
        best = minimum - gamma * torch.logsumexp(-(alternatives - minimum.unsqueeze(-1)) / gamma, dim=-1)
        cells = torch.where(valid.unsqueeze(0), cost + best, torch.full_like(cost, high))
        current = torch.cat((prediction.new_full((b, 1), high), cells), dim=1)
        terminal = current.gather(1, pred_lens.unsqueeze(1)).squeeze(1)
        result = torch.where(pred_lens + target_lens == diagonal, terminal, result)
        prevprev, prev = prev, current
    return (result / torch.maximum(pred_lens, target_lens).to(result.dtype)).mean()
