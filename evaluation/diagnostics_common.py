"""Shared, side-effect-free held-out diagnostics for the persistent byte model."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from data.dataset import ByteMultimodalDataset, collate
from models.generator_conditioned import GeneratorConditioned, GeneratorState
from models.memory_model import MemoryModel


def load_memory(path: str, device: str = "cpu") -> MemoryModel:
    state = torch.load(path, map_location=device, weights_only=True)
    model = MemoryModel(d_model=int(state["d_model"]), capacity=int(state["capacity"]),
                        tier_capacities=(int(state["epi_cap"]), int(state["sem_cap"]), int(state["proc_cap"])),
                        n_exemplars=int(state["n_exemplars"]), max_trace_steps=int(state["max_trace_steps"]),
                        max_pattern_bytes=int(state["max_pattern_bytes"]),
                        max_explanation_bytes=int(state["max_explanation_bytes"])).to(device)
    model.load(path, map_location=device)
    return model.eval()


def test_batches(config: str = "configs/default.yaml", limit: int = 128, batch_size: int = 16,
                 device: str = "cpu", split: str = "test") -> list[dict[str, Any]]:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))["data"]
    data = ByteMultimodalDataset(str(Path(cfg["processed_dir"]) / f"{split}.jsonl"), max_len=cfg["max_seq_len"])
    if len(data) == 0 or limit < 1 or batch_size < 1:
        raise ValueError("diagnostics need a nonempty dataset and positive limits")
    loader = DataLoader(Subset(data, range(min(limit, len(data)))), batch_size=batch_size,
                        shuffle=False, collate_fn=collate)
    return [{**batch, "input_ids": batch["input_ids"].to(device)} for batch in loader]


@torch.no_grad()
def chunk_context(generator: GeneratorConditioned, memory: MemoryModel, ids: torch.Tensor,
                  allowed_tiers: list[str] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal previous-chunk retrieval, without mutating access/recency counters."""
    b, length = ids.shape; k = generator.memory_top_k; d = generator.d_model; size = generator.retro_chunk_size
    chunks = math.ceil(length / size)
    token_emb = generator.byte_emb(ids.clamp(0, 255))
    if bool((ids >= 256).any()):
        controls = (ids - 256).clamp(0, generator.control_emb.num_embeddings - 1)
        token_emb = token_emb + generator.control_emb(controls) * (ids >= 256).unsqueeze(-1)
    padded = F.pad(token_emb, (0, 0, 0, chunks * size - length))
    tokens = token_emb.new_zeros((b, chunks, k, d)); mask = torch.zeros((b, chunks, k), dtype=torch.bool, device=ids.device)
    if chunks > 1:
        queries = padded.view(b, chunks, size, d)[:, :-1].mean(dim=2).reshape(-1, d)
        values, _, valid = memory.retrieve_differentiable(queries, top_k=k, tier_filter=allowed_tiers)
        actual_k = min(k, values.shape[1])
        tokens[:, 1:, :actual_k] = values[:, :actual_k].reshape(b, chunks - 1, actual_k, d)
        mask[:, 1:, :actual_k] = valid[:, :actual_k].reshape(b, chunks - 1, actual_k)
    return tokens, mask


def contexts(generator: GeneratorConditioned, memory: MemoryModel, batches: list[dict[str, Any]],
             allowed_tiers: list[str] | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [chunk_context(generator, memory, b["input_ids"][:, :-1], allowed_tiers) for b in batches]


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if confidence.size == 0:
        return 0.0
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    return float(sum(np.mean(indices == i) * abs(float(confidence[indices == i].mean()) - float(correct[indices == i].mean()))
                     for i in range(bins) if np.any(indices == i)))


@torch.no_grad()
def score_generator(generator: GeneratorConditioned, batches: list[dict[str, Any]],
                    memory_contexts: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
                    *, collect_attention: bool = False) -> dict[str, Any]:
    """Real-byte teacher-forced loss and accuracy grouped by modality."""
    generator.eval()
    per: dict[str, dict[str, Any]] = {}
    confs, corrects = [], []
    attention: dict[str, list[list[np.ndarray]]] = {}
    for bidx, batch in enumerate(batches):
        ids = batch["input_ids"]; inputs, target = ids[:, :-1], ids[:, 1:]
        kwargs: dict[str, Any] = {}
        if memory_contexts is not None:
            tokens, mask = memory_contexts[bidx]
            kwargs = {"memory_context": tokens, "memory_mask": mask, "memory_chunk_size": generator.retro_chunk_size}
        state = generator(inputs, return_state=True, **kwargs)
        assert isinstance(state, GeneratorState)
        logits = state.output.clone(); logits[..., generator.LOAD_SPAN:] = -torch.inf
        losses = F.cross_entropy(logits.reshape(-1, generator.VOCAB_SIZE), target.reshape(-1), reduction="none").reshape_as(target)
        correct = logits.argmax(dim=-1) == target
        for i, (modality, length) in enumerate(zip(batch["modalities"], batch["real_lens"])):
            n = max(length - 1, 0)
            if not n:
                continue
            row = per.setdefault(modality, {"count": 0, "correct": 0, "nll_sum": 0.0})
            row["count"] += n; row["correct"] += int(correct[i, :n].sum())
            row["nll_sum"] += float(losses[i, :n].sum())
            confs.append(state.confidence[i, :n].cpu().numpy()); corrects.append(correct[i, :n].cpu().numpy())
            if collect_attention:
                maps = attention.setdefault(modality, [[] for _ in state.memory_attention])
                for layer, tensor in enumerate(state.memory_attention):
                    # (chunks, heads, chunk_len, K) -> (heads, positions).
                    if tensor.ndim == 5:
                        per_head = tensor[i].permute(1, 0, 2, 3).reshape(tensor.shape[2], -1, tensor.shape[-1]).sum(-1)[:, :n]
                    else:
                        per_head = tensor[i, :, :n].sum(-1)
                    maps[layer].append(per_head.cpu().numpy())
    total = sum(row["count"] for row in per.values())
    per = {key: {"count": row["count"], "accuracy": row["correct"] / row["count"],
                 "nll": row["nll_sum"] / row["count"]} for key, row in per.items()}
    confidence = np.concatenate(confs); correct = np.concatenate(corrects)
    result: dict[str, Any] = {"count": total, "accuracy": sum(v["accuracy"]*v["count"] for v in per.values()) / max(total, 1),
                              "nll": sum(v["nll"]*v["count"] for v in per.values()) / max(total, 1),
                              "ece": ece(confidence, correct), "brier": float(np.mean((confidence - correct)**2)),
                              "mean_confidence": float(confidence.mean()), "by_modality": per}
    if collect_attention:
        result["mean_memory_attention_mass"] = {modality: [np.concatenate(arrays, axis=1).mean(axis=1).tolist()
                                                             for arrays in layers] for modality, layers in attention.items()}
    return result
