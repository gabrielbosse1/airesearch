"""Supervised learning for the temporal graph consistency scorer.

Labels must come from a trusted dataset or host. They are not inferred from
unstructured text, and the resulting learned scores are not guarantees.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.memory_model import MemoryModel


def graph_consistency_loss(
    memory: MemoryModel,
    query_embedding: torch.Tensor,
    node_values: torch.Tensor,
    adjacency: torch.Tensor,
    ages: torch.Tensor,
    consistency_targets: torch.Tensor,
    *,
    hops: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train a single graph's per-node consistency scores with explicit labels.

    ``query_embedding`` is in the shared input space, ``node_values`` contains
    persistent value vectors, and targets are in [0, 1]. Returns (loss, scores)
    with gradients through the query/value encoder and graph reasoner.
    """
    if query_embedding.numel() != memory.d_model:
        raise ValueError("query_embedding must contain exactly d_model values")
    if node_values.ndim != 2 or node_values.shape[0] < 1 or node_values.shape[1] != memory.d_model:
        raise ValueError("node_values must have shape (positive N, d_model)")
    if consistency_targets.shape != (node_values.shape[0],):
        raise ValueError("one consistency target is required per node")
    if not bool(torch.isfinite(consistency_targets).all()) or bool(((consistency_targets < 0) | (consistency_targets > 1)).any()):
        raise ValueError("consistency targets must be finite and in [0, 1]")
    query = memory.value_encoder(query_embedding.reshape(1, memory.d_model))[0]
    scores = memory.score_graph(query, node_values, adjacency, ages, hops=hops)
    targets = consistency_targets.to(device=scores.device, dtype=scores.dtype)
    loss = F.binary_cross_entropy(scores.clamp(1e-7, 1 - 1e-7), targets)
    return loss, scores
