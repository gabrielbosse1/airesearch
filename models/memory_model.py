"""MemoryModel: Hierarchical persistent long-term memory outside the context window.

Hierarchical Memory System across three specialized tiers in shared d_model space:
      1. Episodic Tier   -- Time-decaying biographical/session event store with
                            cross-modal continuity (e.g., text -> voice).
      2. Semantic Tier   -- Multi-prototype Mixture of Exemplars per concept/fact
                            slot (eliminates "mushy centroid" collapse via LogSumExp
                            scoring and posterior responsibility readout).
      3. Procedural Tier -- Multi-step skill & execution trace store (triggers ->
                            latent trace trajectories).
  * Differentiable importance scores govern retrieval, eviction and consolidation.
  * Exact raw-byte procedure replay with persistent latent traces and explanations.
  * Trainable temporal graph scoring and bounded, explicit multi-hop evidence paths.

Core Mission Constraints:
  * Raw Bytes Only (0-255) at system boundaries; NO external tokenizers ever.
  * Differentiable encoders + gradient-free fast writes + non-blurring consolidation.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contradiction_graph import (
    ContradictionReport, Fact, FactRecord, TemporalGraphReasoner,
    build_adjacency, fact_from_metadata, find_evidence,
)
from .procedural import Procedure, ProcedureExecution, byte_pattern_embedding, raw_bytes
from .holdout import HoldoutGuard, HoldoutViolation, reject_heldout_split

TIER_EPISODIC: str = "episodic"
TIER_SEMANTIC: str = "semantic"
TIER_PROCEDURAL: str = "procedural"
ALL_TIERS: tuple[str, str, str] = (TIER_EPISODIC, TIER_SEMANTIC, TIER_PROCEDURAL)


def _validated_metadata_filter(filters: Mapping[str, Any] | None) -> dict[str, str | int | bool]:
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise TypeError("metadata_filter must be a mapping")
    result = dict(filters)
    if len(result) > 16 or any(not isinstance(key, str) or not key or
                               not isinstance(value, (str, int, bool)) for key, value in result.items()):
        raise ValueError("metadata filters must use bounded string keys and primitive exact values")
    return result


def _metadata_matches(metadata: Mapping[str, Any], filters: Mapping[str, str | int | bool]) -> bool:
    # Exact primitive comparisons only; arbitrary tensor/list metadata must not
    # trigger ambiguous truth values or match bools to integer slots.
    return all(type(metadata.get(key)) is type(value) and metadata.get(key) == value for key, value in filters.items())


class ImportanceScorer(nn.Module):
    """Differentiable memory importance scorer: prediction error + access frequency.

    Maps per-slot statistics (prediction_error, access_count, age_ticks, value_norm)
    to a bounded scalar importance score in (0, 1). Monotonic positive base weights
    (via softplus) guarantee out-of-the-box preference for high-surprise and
    frequently accessed memories, while a zero-initialized residual MLP learns
    nonlinear synergies during co-training.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 16,
        recency_scale: float = 64.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.recency_scale = max(float(recency_scale), 1.0)

        # Positive monotonic sensitivities via softplus(raw_w).
        self.raw_w_error = nn.Parameter(torch.tensor(1.2))
        self.raw_w_freq = nn.Parameter(torch.tensor(1.0))
        self.raw_w_recency = nn.Parameter(torch.tensor(-0.5))
        self.base_bias = nn.Parameter(torch.tensor(-0.9))

        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.trunc_normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        pred_errors: torch.Tensor,
        access_counts: torch.Tensor,
        age_ticks: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance scores in (0, 1) for N memory slots."""
        if pred_errors.numel() == 0:
            return torch.empty(0, device=values.device, dtype=values.dtype)

        err_feat = torch.tanh(pred_errors.float().clamp_min(0.0))
        freq_feat = torch.tanh(torch.log1p(access_counts.float().clamp_min(0.0)) / 3.0)
        rec_feat = torch.exp(-age_ticks.float().clamp_min(0.0) / self.recency_scale)
        norm_feat = torch.tanh(
            values.float().norm(dim=-1) / math.sqrt(max(self.d_model, 1))
        )

        feats = torch.stack([err_feat, freq_feat, rec_feat, norm_feat], dim=-1)
        linear_drive = (
            F.softplus(self.raw_w_error) * err_feat
            + F.softplus(self.raw_w_freq) * freq_feat
            + F.softplus(self.raw_w_recency) * rec_feat
            + self.base_bias
        )
        residual = self.mlp(feats).squeeze(-1)
        return torch.sigmoid(linear_drive + residual)


class _GuardedMemoryBank(nn.Module):
    """Prevent public sub-bank state loading from bypassing a parent guard."""

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> Any:
        if getattr(self, "_guard_enabled", False):
            raise HoldoutViolation("load the complete bank through audited MemoryModel.load")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class EpisodicMemoryBank(_GuardedMemoryBank):
    """Time-decaying episodic memory bank for session events & interactions."""

    def __init__(
        self,
        capacity: int,
        d_model: int,
        time_scale: float = 50.0,
    ) -> None:
        super().__init__()
        self.capacity = max(int(capacity), 1)
        self.d_model = d_model
        self.time_scale = max(float(time_scale), 1.0)

        self.register_buffer("keys", torch.zeros(self.capacity, d_model))
        self.register_buffer("values", torch.zeros(self.capacity, d_model))
        self.register_buffer("used", torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer("written_at", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("last_access", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("access_counts", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("pred_errors", torch.zeros(self.capacity, dtype=torch.float))

        self.raw_decay = nn.Parameter(torch.tensor(-1.8))
        self.metadata: list[dict[str, Any]] = [{} for _ in range(self.capacity)]

    @property
    def decay_rate(self) -> torch.Tensor:
        return F.softplus(self.raw_decay)

    @torch.no_grad()
    def allocate_slot(
        self,
        current_tick: int,
        importance_scorer: ImportanceScorer,
    ) -> int:
        free = (~self.used).nonzero(as_tuple=False).squeeze(-1)
        if free.numel() > 0:
            return int(free[0].item())

        used_idx = self.used.nonzero(as_tuple=False).squeeze(-1)
        ages = float(current_tick) - self.written_at[used_idx].float()
        imp = importance_scorer(
            self.pred_errors[used_idx],
            self.access_counts[used_idx],
            float(current_tick) - self.last_access[used_idx].float(),
            self.values[used_idx],
        )
        retention_score = imp - self.decay_rate * (ages.clamp_min(0.0) / self.time_scale)
        victim_local = int(torch.argmin(retention_score).item())
        return int(used_idx[victim_local].item())

    @torch.no_grad()
    def write_slot(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: dict[str, Any],
        current_tick: int,
        prediction_error: float,
        importance_scorer: ImportanceScorer,
        *, _guard_checked: bool = False,
    ) -> int:
        if getattr(self, "_guard_enabled", False) and not _guard_checked:
            raise HoldoutViolation("use MemoryModel.write with raw provenance, not a low-level bank write")
        idx = self.allocate_slot(current_tick, importance_scorer)
        self.keys[idx] = key
        self.values[idx] = value
        self.used[idx] = True
        self.written_at[idx] = current_tick
        self.last_access[idx] = current_tick
        self.access_counts[idx] = 0
        self.pred_errors[idx] = max(float(prediction_error), 0.0)
        md = dict(metadata)
        md["written_at"] = current_tick
        md["tier"] = TIER_EPISODIC
        self.metadata[idx] = md
        return idx

    def score_active(
        self,
        queries: torch.Tensor,
        current_tick: int,
        importance_scorer: ImportanceScorer,
        importance_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = queries.device
        used_idx = self.used.nonzero(as_tuple=False).squeeze(-1)
        m = used_idx.numel()
        b = queries.shape[0]
        if m == 0:
            return (
                used_idx,
                torch.empty(b, 0, device=device),
                torch.empty(b, 0, self.d_model, device=device),
                torch.empty(0, device=device),
            )

        k_used = self.keys[used_idx]
        v_used = self.values[used_idx]
        write_age = (float(current_tick) - self.written_at[used_idx].float()).clamp_min(0.0)
        access_age = (float(current_tick) - self.last_access[used_idx].float()).clamp_min(0.0)

        imp = importance_scorer(
            self.pred_errors[used_idx],
            self.access_counts[used_idx],
            access_age,
            v_used,
        )
        sim = (queries @ k_used.T) / math.sqrt(self.d_model)
        decay_penalty = self.decay_rate * (write_age / self.time_scale)
        logits = sim - decay_penalty.unsqueeze(0) + importance_weight * imp.unsqueeze(0)
        vals = v_used.unsqueeze(0).expand(b, m, self.d_model)
        return used_idx, logits, vals, imp


class SemanticExemplarBank(_GuardedMemoryBank):
    """Semantic memory storing Mixtures of Exemplars (not mushy centroids).

    Each semantic slot i holds up to ``n_exemplars`` distinct prototype pairs
    (k_{i,m}, v_{i,m}) and mixture weights pi_{i,m}. Retrieval scores slots via
    exact mixture LogSumExp and reads out query-conditioned posterior values
    sum_m r_{i,m}(q) v_{i,m}, preserving sharp multimodal concepts.
    """

    def __init__(
        self,
        capacity: int,
        d_model: int,
        n_exemplars: int = 4,
        novelty_threshold: float = 0.88,
    ) -> None:
        super().__init__()
        self.capacity = max(int(capacity), 1)
        self.d_model = d_model
        self.n_exemplars = max(int(n_exemplars), 1)
        self.novelty_threshold = float(novelty_threshold)

        self.register_buffer(
            "keys", torch.zeros(self.capacity, self.n_exemplars, d_model)
        )
        self.register_buffer(
            "values", torch.zeros(self.capacity, self.n_exemplars, d_model)
        )
        self.register_buffer(
            "mixture_weights", torch.zeros(self.capacity, self.n_exemplars)
        )
        self.register_buffer(
            "exemplar_counts", torch.zeros(self.capacity, self.n_exemplars)
        )
        self.register_buffer(
            "exemplar_used",
            torch.zeros(self.capacity, self.n_exemplars, dtype=torch.bool),
        )
        self.register_buffer("used", torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer("written_at", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("last_access", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("access_counts", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("pred_errors", torch.zeros(self.capacity, dtype=torch.float))

        self.metadata: list[dict[str, Any]] = [{} for _ in range(self.capacity)]
        self.concept_to_slot: dict[str, int] = {}

    def _representative_values(self, slot_indices: torch.Tensor) -> torch.Tensor:
        w = self.mixture_weights[slot_indices].unsqueeze(-1)
        return (self.values[slot_indices] * w).sum(dim=1)

    def _renormalize_slot(self, idx: int) -> None:
        active = self.exemplar_used[idx]
        if not bool(active.any()):
            self.mixture_weights[idx].zero_()
            return
        counts = self.exemplar_counts[idx].clamp_min(0.0) * active.float()
        total = counts.sum()
        if total <= 0:
            self.mixture_weights[idx] = active.float() / active.float().sum()
        else:
            self.mixture_weights[idx] = counts / total

    @torch.no_grad()
    def allocate_slot(
        self,
        current_tick: int,
        importance_scorer: ImportanceScorer,
    ) -> int:
        free = (~self.used).nonzero(as_tuple=False).squeeze(-1)
        if free.numel() > 0:
            return int(free[0].item())

        used_idx = self.used.nonzero(as_tuple=False).squeeze(-1)
        access_age = (float(current_tick) - self.last_access[used_idx].float()).clamp_min(0.0)
        rep_vals = self._representative_values(used_idx)
        imp = importance_scorer(
            self.pred_errors[used_idx],
            self.access_counts[used_idx],
            access_age,
            rep_vals,
        )
        victim_local = int(torch.argmin(imp).item())
        victim_idx = int(used_idx[victim_local].item())
        old_concept = self.metadata[victim_idx].get("concept_cluster")
        if (
            old_concept in self.concept_to_slot
            and self.concept_to_slot[old_concept] == victim_idx
        ):
            del self.concept_to_slot[old_concept]
        return victim_idx

    @torch.no_grad()
    def write_or_merge(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: dict[str, Any],
        current_tick: int,
        prediction_error: float,
        importance_scorer: ImportanceScorer,
        *,
        cluster_key: str | None = None,
        ema_rate: float = 0.25,
        _guard_checked: bool = False,
    ) -> int:
        if getattr(self, "_guard_enabled", False) and not _guard_checked:
            raise HoldoutViolation("use MemoryModel.write with raw provenance, not a low-level bank write")
        if cluster_key is not None and cluster_key in self.concept_to_slot:
            idx = self.concept_to_slot[cluster_key]
            if bool(self.used[idx]):
                active_mask = self.exemplar_used[idx]
                active_pos = active_mask.nonzero(as_tuple=False).squeeze(-1)
                if active_pos.numel() > 0:
                    active_vals = F.normalize(self.values[idx, active_pos], dim=-1)
                    v_norm = F.normalize(value.unsqueeze(0), dim=-1)
                    sims = (v_norm @ active_vals.T).squeeze(0)
                    max_sim, best_rel = torch.max(sims, dim=0)
                    best_m = int(active_pos[int(best_rel.item())].item())

                    free_ex = (~active_mask).nonzero(as_tuple=False).squeeze(-1)
                    if (
                        float(max_sim.item()) < self.novelty_threshold
                        and free_ex.numel() > 0
                    ):
                        new_m = int(free_ex[0].item())
                        self.keys[idx, new_m] = key
                        self.values[idx, new_m] = value
                        self.exemplar_used[idx, new_m] = True
                        self.exemplar_counts[idx, new_m] = 1.0
                    else:
                        self.values[idx, best_m] = (
                            (1.0 - ema_rate) * self.values[idx, best_m] + ema_rate * value
                        )
                        self.keys[idx, best_m] = (
                            (1.0 - ema_rate) * self.keys[idx, best_m] + ema_rate * key
                        )
                        self.exemplar_counts[idx, best_m] += 1.0

                    self._renormalize_slot(idx)
                    self.last_access[idx] = current_tick
                    self.pred_errors[idx] = 0.5 * self.pred_errors[idx] + 0.5 * max(
                        float(prediction_error), 0.0
                    )
                    exemplar_history = list(self.metadata[idx].get("exemplars", []))
                    exemplar_history.append(dict(metadata))
                    self.metadata[idx]["exemplars"] = exemplar_history[
                        -self.n_exemplars * 2 :
                    ]
                    return idx

        idx = self.allocate_slot(current_tick, importance_scorer)
        self.keys[idx].zero_()
        self.values[idx].zero_()
        self.mixture_weights[idx].zero_()
        self.exemplar_counts[idx].zero_()
        self.exemplar_used[idx].fill_(False)

        self.keys[idx, 0] = key
        self.values[idx, 0] = value
        self.exemplar_used[idx, 0] = True
        self.exemplar_counts[idx, 0] = 1.0
        self.mixture_weights[idx, 0] = 1.0
        self.used[idx] = True
        self.written_at[idx] = current_tick
        self.last_access[idx] = current_tick
        self.access_counts[idx] = 0
        self.pred_errors[idx] = max(float(prediction_error), 0.0)

        md = dict(metadata)
        md["written_at"] = current_tick
        md["tier"] = TIER_SEMANTIC
        if cluster_key is not None:
            md["concept_cluster"] = cluster_key
            self.concept_to_slot[cluster_key] = idx
        self.metadata[idx] = md
        return idx

    def score_active(
        self,
        queries: torch.Tensor,
        current_tick: int,
        importance_scorer: ImportanceScorer,
        importance_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = queries.device
        used_idx = self.used.nonzero(as_tuple=False).squeeze(-1)
        m = used_idx.numel()
        b = queries.shape[0]
        if m == 0:
            return (
                used_idx,
                torch.empty(b, 0, device=device),
                torch.empty(b, 0, self.d_model, device=device),
                torch.empty(0, device=device),
            )

        k_mix = self.keys[used_idx]
        v_mix = self.values[used_idx]
        pi_mix = self.mixture_weights[used_idx]
        ex_mask = self.exemplar_used[used_idx]

        sim = torch.einsum("bd,med->bme", queries, k_mix) / math.sqrt(self.d_model)
        log_pi = torch.log(pi_mix.clamp_min(1e-8)).unsqueeze(0)
        comp_logits = (sim + log_pi).masked_fill(~ex_mask.unsqueeze(0), float("-inf"))

        slot_logits = torch.logsumexp(comp_logits, dim=-1)
        resp = F.softmax(comp_logits, dim=-1)
        resp = torch.nan_to_num(resp, nan=0.0)
        slot_vals = torch.einsum("bme,med->bmd", resp, v_mix)

        access_age = (float(current_tick) - self.last_access[used_idx].float()).clamp_min(0.0)
        rep_vals = self._representative_values(used_idx)
        imp = importance_scorer(
            self.pred_errors[used_idx],
            self.access_counts[used_idx],
            access_age,
            rep_vals,
        )

        logits = slot_logits + importance_weight * imp.unsqueeze(0)
        return used_idx, logits, slot_vals, imp

    @torch.no_grad()
    def prune_near_duplicate_exemplars(
        self,
        key_encoder: nn.Module,
        duplicate_cos: float = 0.985,
    ) -> int:
        merged = 0
        used_idx = self.used.nonzero(as_tuple=False).squeeze(-1)
        for idx_t in used_idx:
            idx = int(idx_t.item())
            active_pos = (
                self.exemplar_used[idx].nonzero(as_tuple=False).squeeze(-1).tolist()
            )
            if len(active_pos) <= 1:
                continue
            for i_pos in range(len(active_pos)):
                m1 = active_pos[i_pos]
                if not bool(self.exemplar_used[idx, m1]):
                    continue
                for j_pos in range(i_pos + 1, len(active_pos)):
                    m2 = active_pos[j_pos]
                    if not bool(self.exemplar_used[idx, m2]):
                        continue
                    cos = F.cosine_similarity(
                        self.values[idx, m1].unsqueeze(0),
                        self.values[idx, m2].unsqueeze(0),
                    ).item()
                    if cos >= duplicate_cos:
                        c1 = float(self.exemplar_counts[idx, m1].item())
                        c2 = float(self.exemplar_counts[idx, m2].item())
                        tot = max(c1 + c2, 1.0)
                        self.values[idx, m1] = (
                            c1 * self.values[idx, m1] + c2 * self.values[idx, m2]
                        ) / tot
                        self.keys[idx, m1] = key_encoder(
                            self.values[idx, m1].unsqueeze(0)
                        )[0]
                        self.exemplar_counts[idx, m1] = tot
                        self.exemplar_used[idx, m2] = False
                        self.exemplar_counts[idx, m2] = 0.0
                        self.keys[idx, m2].zero_()
                        self.values[idx, m2].zero_()
                        merged += 1
            self._renormalize_slot(idx)
        return merged


class ProceduralSkillBank(_GuardedMemoryBank):
    """Persistent skill triggers, exact raw-byte patterns and latent traces."""

    def __init__(
        self,
        capacity: int,
        d_model: int,
        max_trace_steps: int = 4,
        max_pattern_bytes: int = 512,
        max_explanation_bytes: int = 1024,
    ) -> None:
        super().__init__()
        if max_trace_steps < 1 or max_pattern_bytes < 1 or max_explanation_bytes < 1:
            raise ValueError("procedural storage limits must be positive")
        self.capacity = max(int(capacity), 1)
        self.d_model = d_model
        self.max_trace_steps = int(max_trace_steps)
        self.max_pattern_bytes = int(max_pattern_bytes)
        self.max_explanation_bytes = int(max_explanation_bytes)

        self.register_buffer("keys", torch.zeros(self.capacity, d_model))
        self.register_buffer("values", torch.zeros(self.capacity, d_model))
        self.register_buffer(
            "trace_steps",
            torch.zeros(self.capacity, self.max_trace_steps, d_model),
        )
        self.register_buffer("trace_lens", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("used", torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer("written_at", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("last_access", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("access_counts", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("pred_errors", torch.zeros(self.capacity, dtype=torch.float))
        self.register_buffer("utility_scores", torch.ones(self.capacity, dtype=torch.float))
        # Byte lengths and a separate flag distinguish an empty pattern from an
        # old unstructured trace. Buffers are saved with the memory checkpoint.
        self.register_buffer("structured", torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer("input_bytes", torch.zeros(self.capacity, self.max_pattern_bytes, dtype=torch.uint8))
        self.register_buffer("output_bytes", torch.zeros(self.capacity, self.max_pattern_bytes, dtype=torch.uint8))
        self.register_buffer("explanation_bytes", torch.zeros(self.capacity, self.max_explanation_bytes, dtype=torch.uint8))
        self.register_buffer("input_lens", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("output_lens", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("explanation_lens", torch.zeros(self.capacity, dtype=torch.long))

        self.metadata: list[dict[str, Any]] = [{} for _ in range(self.capacity)]

    @torch.no_grad()
    def allocate_slot(
        self,
        current_tick: int,
        importance_scorer: ImportanceScorer,
    ) -> int:
        free = (~self.used).nonzero(as_tuple=False).squeeze(-1)
        if free.numel() > 0:
            return int(free[0].item())

        used_idx = self.used.nonzero(as_tuple=False).squeeze(-1)
        access_age = (float(current_tick) - self.last_access[used_idx].float()).clamp_min(0.0)
        imp = importance_scorer(
            self.pred_errors[used_idx],
            self.access_counts[used_idx],
            access_age,
            self.values[used_idx],
        )
        score = imp + 0.2 * torch.tanh(self.utility_scores[used_idx])
        victim_local = int(torch.argmin(score).item())
        return int(used_idx[victim_local].item())

    @torch.no_grad()
    def write_skill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        trace: torch.Tensor,
        trace_len: int,
        metadata: dict[str, Any],
        current_tick: int,
        prediction_error: float,
        importance_scorer: ImportanceScorer,
        *, _guard_checked: bool = False,
    ) -> int:
        if getattr(self, "_guard_enabled", False) and not _guard_checked:
            raise HoldoutViolation("use MemoryModel.write with raw provenance, not a low-level bank write")
        idx = self.allocate_slot(current_tick, importance_scorer)
        self.keys[idx] = key
        self.values[idx] = value
        self.trace_steps[idx].zero_()
        steps_to_copy = min(trace.shape[0], self.max_trace_steps, max(int(trace_len), 0))
        self.trace_steps[idx, :steps_to_copy] = trace[:steps_to_copy]
        self.trace_lens[idx] = steps_to_copy
        # A slot may be evicted and reused for an unstructured write. Clear all
        # structured payloads first to prevent replay of the evicted skill.
        self.structured[idx] = False
        self.input_bytes[idx].zero_()
        self.output_bytes[idx].zero_()
        self.explanation_bytes[idx].zero_()
        self.input_lens[idx] = 0
        self.output_lens[idx] = 0
        self.explanation_lens[idx] = 0
        self.used[idx] = True
        self.written_at[idx] = current_tick
        self.last_access[idx] = current_tick
        self.access_counts[idx] = 0
        self.pred_errors[idx] = max(float(prediction_error), 0.0)
        self.utility_scores[idx] = float(metadata.get("utility", 1.0))

        md = dict(metadata)
        md["written_at"] = current_tick
        md["tier"] = TIER_PROCEDURAL
        self.metadata[idx] = md
        return idx

    def score_active(
        self,
        queries: torch.Tensor,
        current_tick: int,
        importance_scorer: ImportanceScorer,
        importance_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = queries.device
        used_idx = self.used.nonzero(as_tuple=False).squeeze(-1)
        m = used_idx.numel()
        b = queries.shape[0]
        if m == 0:
            return (
                used_idx,
                torch.empty(b, 0, device=device),
                torch.empty(b, 0, self.d_model, device=device),
                torch.empty(0, device=device),
            )

        k_used = self.keys[used_idx]
        v_used = self.values[used_idx]
        access_age = (float(current_tick) - self.last_access[used_idx].float()).clamp_min(0.0)
        imp = importance_scorer(
            self.pred_errors[used_idx],
            self.access_counts[used_idx],
            access_age,
            v_used,
        )
        util_bonus = 0.15 * torch.tanh(self.utility_scores[used_idx])
        sim = (queries @ k_used.T) / math.sqrt(self.d_model)
        logits = sim + importance_weight * imp.unsqueeze(0) + util_bonus.unsqueeze(0)
        vals = v_used.unsqueeze(0).expand(b, m, self.d_model)
        return used_idx, logits, vals, imp


class MemoryModel(nn.Module):
    """Hierarchical long-term memory (Episodic + Semantic Exemplars + Procedural)."""

    SCHEMA_VERSION: int = 4

    def __init__(
        self,
        d_model: int = 64,
        capacity: int = 512,
        n_heads: int = 4,
        n_exemplars: int = 4,
        max_trace_steps: int = 4,
        importance_weight: float = 0.35,
        promotion_threshold: float = 0.62,
        tier_capacities: tuple[int, int, int] | None = None,
        max_pattern_bytes: int = 512,
        max_explanation_bytes: int = 1024,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.capacity = max(int(capacity), 3)
        self.n_heads = n_heads
        self.n_exemplars = n_exemplars
        self.max_trace_steps = max_trace_steps
        self.max_pattern_bytes = int(max_pattern_bytes)
        self.max_explanation_bytes = int(max_explanation_bytes)
        self.importance_weight = float(importance_weight)
        self.promotion_threshold = float(promotion_threshold)

        if tier_capacities is not None:
            epi_cap, sem_cap, proc_cap = (max(int(c), 1) for c in tier_capacities)
            self.capacity = epi_cap + sem_cap + proc_cap
        else:
            epi_cap = max(self.capacity // 2, 1)
            proc_cap = max(self.capacity // 4, 1)
            sem_cap = max(self.capacity - epi_cap - proc_cap, 1)
            self.capacity = epi_cap + sem_cap + proc_cap

        self.epi_cap = epi_cap
        self.sem_cap = sem_cap
        self.proc_cap = proc_cap

        self.key_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.proc_key_adapter = nn.Linear(d_model, d_model)
        self.proc_trace_adapter = nn.Linear(d_model, d_model)
        self.trace_step_weights = nn.Linear(d_model, 1)
        self.tier_router = nn.Linear(d_model, 3)
        self.context_proj = nn.Linear(d_model * 2, d_model)

        self.contradiction_scorer = TemporalGraphReasoner(d_model)
        self.person_linker = nn.Linear(d_model, self.capacity)

        self.apply(self._init_weights)
        nn.init.zeros_(self.proc_key_adapter.weight)
        nn.init.zeros_(self.proc_key_adapter.bias)
        nn.init.zeros_(self.proc_trace_adapter.weight)
        nn.init.zeros_(self.proc_trace_adapter.bias)
        nn.init.zeros_(self.tier_router.weight)
        nn.init.zeros_(self.tier_router.bias)

        self.importance_scorer = ImportanceScorer(d_model=d_model)
        self.episodic = EpisodicMemoryBank(capacity=epi_cap, d_model=d_model)
        self.semantic = SemanticExemplarBank(
            capacity=sem_cap, d_model=d_model, n_exemplars=n_exemplars
        )
        self.procedural = ProceduralSkillBank(
            capacity=proc_cap, d_model=d_model, max_trace_steps=max_trace_steps,
            max_pattern_bytes=max_pattern_bytes, max_explanation_bytes=max_explanation_bytes,
        )

        self.session_tick: int = 0
        self.heldout_guard: HoldoutGuard | None = None
        self._loading_audited_state = False
        self._read_only_depth = 0

    @contextmanager
    def read_only(self):
        """Freeze persistent writes and access statistics during evaluation."""
        self._read_only_depth += 1
        try:
            yield self
        finally:
            self._read_only_depth -= 1

    def _assert_writable(self) -> None:
        if self._read_only_depth:
            raise HoldoutViolation("memory is read-only during evaluation")

    def _audit_payload(self) -> dict[str, Any]:
        return {"module_state": self.state_dict(), "episodic_metadata": self.episodic.metadata,
                "semantic_metadata": self.semantic.metadata, "procedural_metadata": self.procedural.metadata}

    def set_holdout_guard(self, guard: HoldoutGuard) -> dict[str, Any]:
        """Attach a registry only after auditing all existing slots.

        Legacy opaque embeddings have no trustworthy provenance and are
        rejected. Build guarded evaluation memories from an empty instance.
        """
        self._assert_writable()
        if not isinstance(guard, HoldoutGuard):
            raise TypeError("guard must be a HoldoutGuard")
        if self.heldout_guard is not None:
            self.heldout_guard.assert_compatible(guard)
        audit = guard.audit_checkpoint(self._audit_payload())
        if audit["failed_slots"]:
            raise HoldoutViolation(f"memory fails holdout audit: {audit['failures'][:3]}")
        self.heldout_guard = guard
        for bank in (self.episodic, self.semantic, self.procedural):
            bank._guard_enabled = True
        return audit

    def audit_holdouts(self) -> dict[str, Any]:
        if self.heldout_guard is None:
            raise HoldoutViolation("no holdout registry is attached")
        return self.heldout_guard.audit_checkpoint(self._audit_payload())

    def validate_holdout_write(self, source: bytes, target: bytes | None = None, *,
                               metadata: Mapping[str, Any] | None = None,
                               extra: tuple[bytes, ...] = ()) -> None:
        reject_heldout_split(metadata)
        if self.heldout_guard is not None:
            self.heldout_guard.validate(source, target, metadata=metadata, extra=extra)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Migrate Phase-1 hierarchical states without hiding unrelated errors."""
        if self.heldout_guard is not None and not self._loading_audited_state:
            raise HoldoutViolation("direct state loading bypasses provenance; use audited MemoryModel.load")
        migrated = OrderedDict(state_dict)
        old_graph = "contradiction_scorer.0.weight" in migrated
        old_procedures = "procedural.trace_steps" in migrated and "procedural.structured" not in migrated
        if old_graph:
            for key in list(migrated):
                if key.startswith("contradiction_scorer."):
                    suffix = key[len("contradiction_scorer."):]
                    if suffix.startswith(("0.", "2.")):
                        migrated["contradiction_scorer.pair_scorer." + suffix] = migrated.pop(key)
        if old_graph or old_procedures:
            defaults = self.state_dict()
            new_graph_prefixes = (
                "contradiction_scorer.time_mlp.", "contradiction_scorer.message_mlp.",
                "contradiction_scorer.time_gate", "contradiction_scorer.message_gate",
            )
            new_proc_names = {
                "procedural.structured", "procedural.input_bytes", "procedural.output_bytes",
                "procedural.explanation_bytes", "procedural.input_lens", "procedural.output_lens",
                "procedural.explanation_lens",
            }
            for key, value in defaults.items():
                if (old_graph and key.startswith(new_graph_prefixes)) or (old_procedures and key in new_proc_names):
                    migrated.setdefault(key, value)
        return super().load_state_dict(migrated, strict=strict, assign=assign)

    def _fit_dim(self, emb: torch.Tensor) -> torch.Tensor:
        orig_1d = emb.dim() == 1
        x = emb.float()
        if orig_1d:
            x = x.unsqueeze(0)
        d = x.shape[-1]
        if d < self.d_model:
            x = F.pad(x, (0, self.d_model - d))
        elif d > self.d_model:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), self.d_model).squeeze(1)
        return x.squeeze(0) if orig_1d else x

    def _to_global_slot(self, tier: str, local_slot: int) -> int:
        if tier == TIER_EPISODIC:
            return local_slot
        if tier == TIER_SEMANTIC:
            return self.epi_cap + local_slot
        if tier == TIER_PROCEDURAL:
            return self.epi_cap + self.sem_cap + local_slot
        raise ValueError(f"Unknown memory tier: {tier!r}")

    def _from_global_slot(self, global_slot: int) -> tuple[str, int]:
        if 0 <= global_slot < self.epi_cap:
            return TIER_EPISODIC, global_slot
        if self.epi_cap <= global_slot < self.epi_cap + self.sem_cap:
            return TIER_SEMANTIC, global_slot - self.epi_cap
        if self.epi_cap + self.sem_cap <= global_slot < self.capacity:
            return TIER_PROCEDURAL, global_slot - (self.epi_cap + self.sem_cap)
        raise IndexError(f"Global slot {global_slot} out of [0, {self.capacity})")

    @staticmethod
    def infer_tier(metadata: dict[str, Any]) -> str:
        explicit = metadata.get("tier")
        if explicit in ALL_TIERS:
            return str(explicit)
        mtype = str(metadata.get("type", "")).lower()
        if (
            mtype.startswith("skill")
            or mtype in ("procedural", "example_input", "trace", "tool", "procedure")
            or "skill" in metadata
        ):
            return TIER_PROCEDURAL
        if mtype in ("fact", "semantic", "concept", "definition", "rule", "knowledge"):
            return TIER_SEMANTIC
        return TIER_EPISODIC

    def encode(self, content_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aligned = self._fit_dim(content_emb)
        key = self.key_encoder(aligned)
        value = self.value_encoder(aligned)
        return key, value

    @torch.no_grad()
    def compute_novelty_error(
        self,
        key: torch.Tensor,
        tier: str | None = None,
    ) -> float:
        active_keys: list[torch.Tensor] = []
        tiers = [tier] if tier in ALL_TIERS else list(ALL_TIERS)
        if TIER_EPISODIC in tiers and bool(self.episodic.used.any()):
            active_keys.append(self.episodic.keys[self.episodic.used])
        if TIER_SEMANTIC in tiers and bool(self.semantic.used.any()):
            ex_mask = self.semantic.exemplar_used & self.semantic.used.unsqueeze(-1)
            if bool(ex_mask.any()):
                active_keys.append(self.semantic.keys[ex_mask])
        if TIER_PROCEDURAL in tiers and bool(self.procedural.used.any()):
            active_keys.append(self.procedural.keys[self.procedural.used])

        if not active_keys:
            return 1.0

        bank_k = F.normalize(torch.cat(active_keys, dim=0), dim=-1)
        q_k = F.normalize(key.reshape(1, -1), dim=-1)
        max_cos = float((q_k @ bank_k.T).max().item())
        return float(max(0.0, 1.0 - max_cos))

    @torch.no_grad()
    def update_prediction_error(
        self,
        global_slot: int,
        error: float | torch.Tensor,
        ema: float = 0.5,
    ) -> None:
        err_val = float(error.item() if torch.is_tensor(error) else error)
        err_val = max(err_val, 0.0)
        tier, local = self._from_global_slot(global_slot)
        bank = getattr(self, tier)
        bank.pred_errors[local] = (1.0 - ema) * bank.pred_errors[local] + ema * err_val

    @torch.no_grad()
    def score_slot_importance(self, global_slot: int) -> float:
        tier, local = self._from_global_slot(global_slot)
        bank = getattr(self, tier)
        if not bool(bank.used[local]):
            return 0.0
        idx_t = torch.tensor([local], device=bank.used.device, dtype=torch.long)
        age = (float(self.session_tick) - bank.last_access[idx_t].float()).clamp_min(0.0)
        val = (
            self.semantic._representative_values(idx_t)
            if tier == TIER_SEMANTIC
            else bank.values[idx_t]
        )
        score = self.importance_scorer(
            bank.pred_errors[idx_t],
            bank.access_counts[idx_t],
            age,
            val,
        )
        return float(score[0].item())

    @torch.no_grad()
    def write(
        self,
        content_emb: torch.Tensor,
        metadata: dict[str, Any],
        *,
        tier: str | None = None,
        prediction_error: float | None = None,
        trace_emb: torch.Tensor | None = None,
        trace_steps: torch.Tensor | None = None,
        merge_concept: bool = False,
        raw_input: bytes | None = None,
        raw_output: bytes | None = None,
        value_override: torch.Tensor | None = None,
    ) -> int:
        """Write a memory slot, optionally storing an explicit episodic vector.

        The optional value is detached on write and requires both raw input
        and output provenance; normal callers retain learned value encoding.
        Guards cannot prove that an opaque vector matches its declared source.
        """
        self._assert_writable()
        reject_heldout_split(metadata)
        if self.heldout_guard is not None:
            metadata = self.heldout_guard.authorize(raw_input, raw_output, metadata=metadata)
        device = self.episodic.keys.device
        aligned = self._fit_dim(content_emb.detach().to(device).reshape(-1))
        target_tier = tier if tier in ALL_TIERS else self.infer_tier(metadata)
        if value_override is not None:
            if target_tier != TIER_EPISODIC or not isinstance(raw_input, bytes) or not isinstance(raw_output, bytes):
                raise ValueError("explicit values are episodic-only and require raw input and output provenance")
            if (not torch.is_tensor(value_override) or value_override.shape != (self.d_model,)
                    or not value_override.is_floating_point() or not bool(torch.isfinite(value_override).all())):
                raise ValueError("explicit episodic value must be one finite d_model vector")

        base_k = self.key_encoder(aligned.unsqueeze(0))[0]
        base_v = (self.value_encoder(aligned.unsqueeze(0))[0] if value_override is None
                  else value_override.detach().to(device=device, dtype=self.episodic.values.dtype))
        surprise = (
            self.compute_novelty_error(base_k, target_tier)
            if prediction_error is None
            else max(float(prediction_error), 0.0)
        )

        if target_tier == TIER_EPISODIC:
            local_idx = self.episodic.write_slot(
                key=base_k,
                value=base_v,
                metadata=metadata,
                current_tick=self.session_tick,
                prediction_error=surprise,
                importance_scorer=self.importance_scorer,
                _guard_checked=True,
            )
        elif target_tier == TIER_SEMANTIC:
            cluster_key: str | None = None
            if merge_concept or "concept_cluster" in metadata:
                cluster_key = str(
                    metadata.get("concept_cluster")
                    or metadata.get("concept")
                    or metadata.get("person", "default")
                )
            local_idx = self.semantic.write_or_merge(
                key=base_k,
                value=base_v,
                metadata=metadata,
                current_tick=self.session_tick,
                prediction_error=surprise,
                importance_scorer=self.importance_scorer,
                cluster_key=cluster_key,
                _guard_checked=True,
            )
        else:
            proc_k = base_k + self.proc_key_adapter(aligned.unsqueeze(0))[0]
            if trace_steps is not None:
                if trace_steps.ndim not in (1, 2) or trace_steps.numel() == 0:
                    raise ValueError("trace_steps must be a non-empty vector or matrix")
                if trace_steps.ndim == 2 and trace_steps.shape[0] > self.procedural.max_trace_steps:
                    raise ValueError("trace_steps exceed max_trace_steps; no truncation is performed")
                steps_aligned = self._fit_dim(trace_steps.detach().to(device))
                if steps_aligned.dim() == 1:
                    steps_aligned = steps_aligned.unsqueeze(0)
                encoded_steps = self.value_encoder(steps_aligned) + self.proc_trace_adapter(
                    steps_aligned
                )
                step_w = F.softmax(
                    self.trace_step_weights(encoded_steps).squeeze(-1), dim=0
                )
                proc_v = (encoded_steps * step_w.unsqueeze(-1)).sum(dim=0)
                t_len = int(encoded_steps.shape[0])
            elif trace_emb is not None:
                tr_aligned = self._fit_dim(trace_emb.detach().to(device).reshape(-1))
                proc_v = (
                    self.value_encoder(tr_aligned.unsqueeze(0))[0]
                    + self.proc_trace_adapter(tr_aligned.unsqueeze(0))[0]
                )
                encoded_steps = torch.stack([base_v, proc_v], dim=0) if self.procedural.max_trace_steps >= 2 else proc_v.unsqueeze(0)
                t_len = int(encoded_steps.shape[0])
            else:
                proc_v = base_v + self.proc_trace_adapter(aligned.unsqueeze(0))[0]
                encoded_steps = proc_v.unsqueeze(0)
                t_len = 1

            local_idx = self.procedural.write_skill(
                key=proc_k,
                value=proc_v,
                trace=encoded_steps,
                trace_len=t_len,
                metadata=metadata,
                current_tick=self.session_tick,
                prediction_error=surprise,
                importance_scorer=self.importance_scorer,
                _guard_checked=True,
            )

        self.session_tick += 1
        return self._to_global_slot(target_tier, local_idx)

    @torch.no_grad()
    def write_fact(
        self,
        fact: Fact | Mapping[str, Any],
        *,
        content_emb: torch.Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Write an explicit graph fact without inferring structure from prose."""
        structured = fact if isinstance(fact, Fact) else Fact.from_mapping(fact)
        data = structured.subject + b"\0" + structured.predicate + b"\0" + structured.object
        if len(data) > 16384:
            raise ValueError("fact identifiers exceed the maximum total length")
        device = self.episodic.keys.device
        emb = byte_pattern_embedding(data, self.d_model, device=device) if content_emb is None else content_emb
        md = dict(metadata or {})
        md.setdefault("type", "fact")
        md["structured_fact"] = structured.as_metadata()
        # A separate slot per structured fact avoids losing time/provenance in a
        # semantic exemplar cluster. Direct graph edges always reflect live slots.
        return self.write(emb, md, tier=TIER_SEMANTIC, raw_input=data)

    @torch.no_grad()
    def write_procedure(
        self,
        input_pattern: bytes | bytearray,
        output_pattern: bytes | bytearray,
        latent_trace: torch.Tensor,
        explanation_bytes: bytes | bytearray = b"",
        *,
        metadata: Mapping[str, Any] | None = None,
        content_emb: torch.Tensor | None = None,
    ) -> int:
        """Directly store (input, output, latent trace, explanation) as data.

        Inputs are exact raw bytes; no code is evaluated. Oversized data and
        traces are rejected rather than silently truncated.
        """
        self._assert_writable()
        inp = raw_bytes(input_pattern, "input_pattern")
        out = raw_bytes(output_pattern, "output_pattern")
        explanation = raw_bytes(explanation_bytes, "explanation_bytes")
        md = dict(metadata or {})
        md.setdefault("type", "procedure")
        reject_heldout_split(md)
        if self.heldout_guard is not None:
            md = self.heldout_guard.authorize(inp, out, metadata=md, extra=(explanation,))
        bank = self.procedural
        if len(inp) > bank.max_pattern_bytes or len(out) > bank.max_pattern_bytes:
            raise ValueError("procedure pattern exceeds max_pattern_bytes")
        if len(explanation) > bank.max_explanation_bytes:
            raise ValueError("procedure explanation exceeds max_explanation_bytes")
        if not torch.is_tensor(latent_trace) or latent_trace.ndim != 2 or latent_trace.shape[1] != self.d_model:
            raise ValueError("latent_trace must have shape (steps, d_model)")
        if latent_trace.shape[0] > bank.max_trace_steps:
            raise ValueError("latent_trace exceeds max_trace_steps")
        if not bool(torch.isfinite(latent_trace).all()):
            raise ValueError("latent_trace must be finite")
        device = bank.keys.device
        trace = latent_trace.detach().to(device=device, dtype=bank.trace_steps.dtype)
        inp_emb = byte_pattern_embedding(inp, self.d_model, device=device) if content_emb is None else self._fit_dim(content_emb.detach().to(device).reshape(-1))
        out_emb = byte_pattern_embedding(out, self.d_model, device=device)
        key = self.key_encoder(inp_emb.unsqueeze(0))[0] + self.proc_key_adapter(inp_emb.unsqueeze(0))[0]
        value = self.value_encoder(out_emb.unsqueeze(0))[0]
        surprise = self.compute_novelty_error(key, TIER_PROCEDURAL)
        idx = bank.write_skill(key, value, trace, int(trace.shape[0]), md, self.session_tick, surprise,
                               self.importance_scorer, _guard_checked=True)
        for target, data in ((bank.input_bytes, inp), (bank.output_bytes, out), (bank.explanation_bytes, explanation)):
            if data:
                target[idx, :len(data)] = torch.tensor(list(data), dtype=torch.uint8, device=device)
        bank.input_lens[idx] = len(inp)
        bank.output_lens[idx] = len(out)
        bank.explanation_lens[idx] = len(explanation)
        bank.structured[idx] = True
        self.session_tick += 1
        return self._to_global_slot(TIER_PROCEDURAL, idx)

    @torch.no_grad()
    def delete_episodic_entries(self, *, person: str, event_type: str,
                                limit: int | None = None,
                                metadata_filter: Mapping[str, Any] | None = None) -> int:
        """Delete oldest exact owner/type episodic entries, or all when unbounded.

        Deletion clears the live bank only; old checkpoints, backups and disk
        remnants are outside this API. The caller must authorize ``person``.
        """
        self._assert_writable()
        if not isinstance(person, str) or not person or not isinstance(event_type, str) or not event_type:
            raise ValueError("person and event_type must be nonempty strings")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError("limit must be a positive integer or None")
        filters = _validated_metadata_filter(metadata_filter)
        bank = self.episodic
        matches = [idx for idx in bank.used.nonzero(as_tuple=False).flatten().tolist()
                   if bank.metadata[idx].get("person") == person and bank.metadata[idx].get("type") == event_type
                   and _metadata_matches(bank.metadata[idx], filters)]
        matches.sort(key=lambda idx: (int(bank.written_at[idx]), idx))
        if limit is not None:
            matches = matches[:limit]
        for idx in matches:
            for name in ("keys", "values", "used", "written_at", "last_access", "access_counts", "pred_errors"):
                getattr(bank, name)[idx].zero_()
            bank.metadata[idx] = {}
        if matches:
            self.session_tick += 1
        return len(matches)

    @torch.no_grad()
    def delete_procedures(self, input_pattern: bytes | bytearray) -> int:
        """Remove every exact matching structured procedure from the live bank.

        All matching versions and their payload/trace buffers are zeroed; this
        prevents an older value from resurfacing after a forget operation. This
        is not forensic erasure of prior checkpoints, filesystem copies or OS
        caches. Read-only memory remains read-only, including for deletion.
        """
        self._assert_writable()
        inp = raw_bytes(input_pattern, "input_pattern")
        bank = self.procedural
        matches: list[int] = []
        for idx in (bank.used & bank.structured).nonzero(as_tuple=False).flatten().tolist():
            if int(bank.input_lens[idx].item()) == len(inp):
                if bytes(bank.input_bytes[idx, :len(inp)].cpu().tolist()) == inp:
                    matches.append(idx)
        for idx in matches:
            for name in ("keys", "values", "trace_steps", "trace_lens", "used", "written_at",
                         "last_access", "access_counts", "pred_errors", "structured", "input_bytes",
                         "output_bytes", "explanation_bytes", "input_lens", "output_lens", "explanation_lens"):
                getattr(bank, name)[idx].zero_()
            bank.utility_scores[idx] = 1.0
            bank.metadata[idx] = {}
        if matches:
            self.session_tick += 1
        return len(matches)

    @torch.no_grad()
    def read_procedure(self, slot: int) -> Procedure:
        """Read a structured procedure by its current global slot."""
        tier, idx = self._from_global_slot(slot)
        bank = self.procedural
        if tier != TIER_PROCEDURAL or not bool(bank.used[idx]) or not bool(bank.structured[idx]):
            raise LookupError("slot does not contain a structured procedure")

        def unpack(buffer: torch.Tensor, length: torch.Tensor) -> bytes:
            return bytes(buffer[idx, :int(length[idx].item())].cpu().tolist())

        n_steps = int(bank.trace_lens[idx].item())
        return Procedure(
            slot=slot,
            input_pattern=unpack(bank.input_bytes, bank.input_lens),
            output_pattern=unpack(bank.output_bytes, bank.output_lens),
            latent_trace=bank.trace_steps[idx, :n_steps].clone(),
            explanation_bytes=unpack(bank.explanation_bytes, bank.explanation_lens),
            metadata=dict(bank.metadata[idx]),
        )

    @torch.no_grad()
    def execute_procedure(
        self,
        input_pattern: bytes | bytearray,
        *,
        slot: int | None = None,
        latent_step_handler: Callable[[torch.Tensor], Any] | None = None,
    ) -> ProcedureExecution:
        """Replay the newest exact matching skill; never evaluate stored code.

        A host may explicitly supply a callback to consume each latent step in
        order. Approximate/fuzzy execution is deliberately not implicit.
        """
        inp = raw_bytes(input_pattern, "input_pattern")
        bank = self.procedural
        if slot is None:
            matching: list[tuple[int, int]] = []
            for idx in (bank.used & bank.structured).nonzero(as_tuple=False).flatten().tolist():
                if int(bank.input_lens[idx].item()) == len(inp):
                    stored = bytes(bank.input_bytes[idx, :len(inp)].cpu().tolist())
                    if stored == inp:
                        matching.append((int(bank.written_at[idx].item()), idx))
            if not matching:
                raise LookupError("no exact matching procedure")
            _, idx = max(matching)
            slot = self._to_global_slot(TIER_PROCEDURAL, idx)
        procedure = self.read_procedure(slot)
        if procedure.input_pattern != inp:
            raise LookupError("procedure input does not match the requested bytes")
        if latent_step_handler is not None:
            for step in procedure.latent_trace:
                latent_step_handler(step.clone())
        _, idx = self._from_global_slot(slot)
        if not self._read_only_depth:
            bank.access_counts[idx] += 1
            bank.last_access[idx] = self.session_tick
        return ProcedureExecution(
            slot=slot, output=procedure.output_pattern,
            latent_trace=procedure.latent_trace.clone(),
            explanation_bytes=procedure.explanation_bytes,
            metadata=dict(procedure.metadata),
        )

    def _gather_candidates(
        self,
        queries: torch.Tensor,
        tier_filter: str | list[str] | tuple[str, ...] | None = None,
    ) -> tuple[
        list[tuple[str, int, int, dict[str, Any]]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        device = queries.device
        b = queries.shape[0]
        if tier_filter is None:
            allowed_tiers = set(ALL_TIERS)
        elif isinstance(tier_filter, str):
            allowed_tiers = {tier_filter}
        else:
            allowed_tiers = set(tier_filter)

        tier_bias = F.log_softmax(self.tier_router(queries), dim=-1) + math.log(3.0)

        meta_records: list[tuple[str, int, int, dict[str, Any]]] = []
        logit_chunks: list[torch.Tensor] = []
        val_chunks: list[torch.Tensor] = []
        imp_chunks: list[torch.Tensor] = []

        tier_specs = [
            (0, TIER_EPISODIC, self.episodic),
            (1, TIER_SEMANTIC, self.semantic),
            (2, TIER_PROCEDURAL, self.procedural),
        ]
        for t_idx, t_name, bank in tier_specs:
            if t_name not in allowed_tiers:
                continue
            used_idx, t_logits, t_vals, t_imp = bank.score_active(
                queries,
                self.session_tick,
                self.importance_scorer,
                self.importance_weight,
            )
            if used_idx.numel() == 0:
                continue
            logit_chunks.append(t_logits + tier_bias[:, t_idx : t_idx + 1])
            val_chunks.append(t_vals)
            imp_chunks.append(t_imp)
            for local_i in used_idx.tolist():
                li = int(local_i)
                gi = self._to_global_slot(t_name, li)
                meta_records.append((t_name, li, gi, bank.metadata[li]))

        if not logit_chunks:
            return (
                [],
                torch.empty(b, 0, device=device),
                torch.empty(b, 0, self.d_model, device=device),
                torch.empty(0, device=device),
            )

        return (
            meta_records,
            torch.cat(logit_chunks, dim=1),
            torch.cat(val_chunks, dim=1),
            torch.cat(imp_chunks, dim=0),
        )

    @torch.no_grad()
    def retrieve(
        self,
        query_emb: torch.Tensor,
        top_k: int = 5,
        modality_filter: str | None = None,
        person_filter: str | None = None,
        tier_filter: str | list[str] | tuple[str, ...] | None = None,
        pool: bool = False,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], torch.Tensor]:
        """Retrieve top-K memory entries across hierarchical tiers.

        Returns (entries, memory_tokens) where ``memory_tokens`` has shape
        ``(K_eff, d_model)`` by default (replacing the rank-1 bottleneck), or
        ``(d_model,)`` if ``pool=True``.
        """
        filters = _validated_metadata_filter(metadata_filter)
        device = self.episodic.keys.device
        q_in = self._fit_dim(query_emb.detach().to(device).reshape(-1)).unsqueeze(0)
        q = self.query_encoder(q_in)

        records, logits_bm, vals_bmd, imps = self._gather_candidates(
            q, tier_filter=tier_filter
        )
        if not records:
            empty = (
                torch.zeros(self.d_model, device=device)
                if pool
                else torch.zeros(0, self.d_model, device=device)
            )
            return [], empty

        logits = logits_bm.squeeze(0)
        vals = vals_bmd.squeeze(0)

        mask = torch.ones_like(logits, dtype=torch.bool)
        for i, (_, _, _, md) in enumerate(records):
            if modality_filter is not None and md.get("modality") != modality_filter:
                mask[i] = False
            if person_filter is not None and md.get("person") != person_filter:
                mask[i] = False
            if not _metadata_matches(md, filters):
                mask[i] = False

        valid_count = int(mask.sum().item())
        if valid_count == 0 or top_k <= 0:
            empty = (
                torch.zeros(self.d_model, device=device)
                if pool
                else torch.zeros(0, self.d_model, device=device)
            )
            return [], empty

        masked_logits = logits.masked_fill(~mask, float("-inf"))
        k = min(int(top_k), valid_count)
        top_logits, top_pos = torch.topk(masked_logits, k=k)
        top_weights = F.softmax(top_logits, dim=-1)

        entries: list[dict[str, Any]] = []
        selected_vals: list[torch.Tensor] = []
        for pos, w in zip(top_pos.tolist(), top_weights.tolist()):
            t_name, local_i, global_i, md = records[pos]
            v_slot = vals[pos].clone()
            imp_val = float(imps[pos].item())
            entries.append(
                {
                    "slot": global_i,
                    "tier": t_name,
                    "tier_slot": local_i,
                    "weight": float(w),
                    "importance": imp_val,
                    "metadata": md,
                    "value": v_slot,
                }
            )
            selected_vals.append(v_slot)
            if not self._read_only_depth:
                bank = getattr(self, t_name)
                bank.access_counts[local_i] += 1
                bank.last_access[local_i] = self.session_tick

        multi_vec = torch.stack(selected_vals, dim=0)
        if pool:
            pooled = (multi_vec * top_weights.unsqueeze(-1)).sum(dim=0)
            return entries, pooled
        return entries, multi_vec

    def retrieve_differentiable(
        self,
        query_emb: torch.Tensor,
        top_k: int = 4,
        tier_filter: str | list[str] | tuple[str, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable batched top-K multi-vector retrieval for co-training."""
        single = query_emb.dim() == 1
        q_in = self._fit_dim(query_emb)
        if single:
            q_in = q_in.unsqueeze(0)
        b = q_in.shape[0]
        device = q_in.device

        q = self.query_encoder(q_in)
        records, logits, vals, _ = self._gather_candidates(q, tier_filter=tier_filter)
        m = len(records)
        if m == 0 or top_k <= 0:
            return (
                torch.zeros(b, 1, self.d_model, device=device, dtype=q.dtype),
                torch.zeros(b, 1, device=device, dtype=q.dtype),
                torch.zeros(b, 1, device=device, dtype=torch.bool),
            )

        k = min(int(top_k), m)
        full_attn = F.softmax(logits, dim=-1)  # (B, M)
        soft_read = torch.einsum("bm,bmd->bd", full_attn, vals)  # (B, d)

        top_logits, top_idx = torch.topk(logits, k=k, dim=-1)
        top_weights = F.softmax(top_logits, dim=-1)

        gather_idx = top_idx.unsqueeze(-1).expand(b, k, self.d_model)
        picked_vals = torch.gather(vals, dim=1, index=gather_idx)

        # Tangential straight-through estimator: exact unblurred top-K vectors in
        # forward pass; full directional attention Jacobian (survives LayerNorm!) in backward.
        tangential_ste = (soft_read - soft_read.detach()).unsqueeze(1)
        mem_tokens = picked_vals + tangential_ste
        mem_mask = torch.ones(b, k, device=device, dtype=torch.bool)
        return mem_tokens, top_weights, mem_mask

    @torch.no_grad()
    def _graph_records(self, person: str | None = None) -> list[FactRecord]:
        records: list[FactRecord] = []
        for tier, bank in ((TIER_EPISODIC, self.episodic), (TIER_SEMANTIC, self.semantic)):
            for idx in bank.used.nonzero(as_tuple=False).flatten().tolist():
                md = bank.metadata[idx]
                if person is not None and md.get("person") not in (None, person):
                    continue
                if tier == TIER_SEMANTIC:
                    indices = torch.tensor([idx], device=bank.used.device, dtype=torch.long)
                    value = self.semantic._representative_values(indices)[0].clone()
                else:
                    value = bank.values[idx].clone()
                entry = {
                    "slot": self._to_global_slot(tier, idx), "tier": tier,
                    "tier_slot": idx, "metadata": md, "value": value,
                }
                records.append(FactRecord(entry=entry, fact=fact_from_metadata(md)))
        return records

    def score_graph(
        self,
        query_value: torch.Tensor,
        node_values: torch.Tensor,
        adjacency: torch.Tensor,
        ages: torch.Tensor,
        *,
        hops: int = 2,
    ) -> torch.Tensor:
        """Differentiable graph scores, usable with supervised training losses."""
        return self.contradiction_scorer(query_value, node_values, adjacency, ages, hops=hops)

    @torch.no_grad()
    def explain_contradiction(
        self,
        statement: Fact | Mapping[str, Any],
        *,
        new_statement_emb: torch.Tensor | None = None,
        person: str | None = None,
        as_of: float | None = None,
        max_hops: int = 3,
        max_states: int = 4096,
    ) -> ContradictionReport:
        """Return explicit, time-valid graph paths supporting/conflicting a fact.

        Unknown does not mean true. Relations are exclusive or transitive only
        when explicitly declared. ``as_of`` and all validity endpoints must be
        in the same caller-supplied numeric time scale.
        """
        claim = statement if isinstance(statement, Fact) else Fact.from_mapping(statement)
        if not 1 <= max_hops <= 16 or max_states < 1:
            raise ValueError("max_hops must be 1..16 and max_states must be positive")
        records = self._graph_records(person)
        device = self.episodic.keys.device
        if not records:
            # Validate as_of even for an empty graph.
            find_evidence([], claim, torch.empty(0, device=device), as_of=as_of, max_hops=max_hops, max_states=max_states)
            return ContradictionReport("unknown", 1.0, ())
        if new_statement_emb is None:
            payload = claim.subject + b"\0" + claim.predicate + b"\0" + claim.object
            if len(payload) > 16384:
                raise ValueError("fact identifiers exceed the maximum total length")
            aligned = byte_pattern_embedding(payload, self.d_model, device=device)
        else:
            aligned = self._fit_dim(new_statement_emb.detach().to(device).reshape(-1))
        query = self.value_encoder(aligned.unsqueeze(0))[0]
        values = torch.stack([r.entry["value"] for r in records])
        adjacency = build_adjacency(records, device=device)
        ages = torch.tensor([
            max(0, self.session_tick - int(r.entry["metadata"].get("written_at", self.session_tick)))
            for r in records
        ], device=device, dtype=values.dtype)
        learned = self.score_graph(query, values, adjacency, ages, hops=max_hops)
        evidence, truncated = find_evidence(
            records, claim, learned, as_of=as_of, max_hops=max_hops, max_states=max_states
        )
        kinds = {e.kind for e in evidence}
        status = "mixed" if kinds == {"support", "conflict"} else (next(iter(kinds)) if kinds else "unknown")
        # Logical contradictions are exact for the declared schema, while the
        # learned score is exposed separately on each item for inspection.
        consistency = 0.0 if "conflict" in kinds else 1.0
        if not self._read_only_depth:
            for slot in {s for ev in evidence for s in ev.path_slots}:
                tier, idx = self._from_global_slot(slot)
                bank = getattr(self, tier)
                bank.access_counts[idx] += 1
                bank.last_access[idx] = self.session_tick
        return ContradictionReport(status, consistency, tuple(evidence), truncated)

    @torch.no_grad()
    def check_contradiction(
        self,
        new_statement_emb: torch.Tensor,
        person: str | None = None,
        threshold: float = 0.5,
        *,
        statement: Fact | Mapping[str, Any] | None = None,
        as_of: float | None = None,
        max_hops: int = 3,
        max_states: int = 4096,
    ) -> tuple[float, list[tuple[dict[str, Any], float]]]:
        """Compatibility interface; structured claims use graph path reasoning.

        Without structured metadata this returns a learned/cosine *similarity*
        heuristic, not a reliable factual contradiction or lie detector.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if statement is not None:
            report = self.explain_contradiction(
                statement, new_statement_emb=new_statement_emb, person=person,
                as_of=as_of, max_hops=max_hops, max_states=max_states,
            )
            if report.truncated:
                warnings.warn("Contradiction graph search was truncated; absence of a conflict is inconclusive.", RuntimeWarning, stacklevel=2)
            conflicts = [(ev.entry, 0.0) for ev in report.evidence if ev.kind == "conflict" and 0.0 < threshold]
            return report.consistency, conflicts

        entries, _ = self.retrieve(
            new_statement_emb, top_k=8, person_filter=person,
            tier_filter=[TIER_SEMANTIC, TIER_EPISODIC],
        )
        if not entries:
            return 1.0, []
        device = self.episodic.keys.device
        aligned = self._fit_dim(new_statement_emb.detach().to(device).reshape(-1)).unsqueeze(0)
        query = self.value_encoder(aligned)[0]
        records = [FactRecord(entry=e, fact=fact_from_metadata(e["metadata"])) for e in entries]
        nodes = torch.stack([e["value"].to(device) for e in entries])
        adjacency = build_adjacency(records, device=device)
        ages = torch.tensor([
            max(0, self.session_tick - int(e["metadata"].get("written_at", self.session_tick)))
            for e in entries
        ], device=device, dtype=nodes.dtype)
        scores = self.score_graph(query, nodes, adjacency, ages, hops=max_hops)
        conflicts = [(e, float(s.item())) for e, s in zip(entries, scores) if float(s.item()) < threshold]
        return float(scores.mean().item()), conflicts

    @torch.no_grad()
    def consolidate(self, lr: float = 0.01) -> dict[str, int]:
        self._assert_writable()
        if self.heldout_guard is not None:
            audit = self.audit_holdouts()
            if audit["failed_slots"]:
                raise HoldoutViolation(f"cannot consolidate unaudited memory: {audit['failures'][:3]}")
        promoted = 0
        if bool(self.episodic.used.any()):
            used_epi = self.episodic.used.nonzero(as_tuple=False).squeeze(-1)
            ages = (
                float(self.session_tick) - self.episodic.last_access[used_epi].float()
            ).clamp_min(0.0)
            imps = self.importance_scorer(
                self.episodic.pred_errors[used_epi],
                self.episodic.access_counts[used_epi],
                ages,
                self.episodic.values[used_epi],
            )
            for local_pos, imp_val in zip(used_epi.tolist(), imps.tolist()):
                li = int(local_pos)
                md = self.episodic.metadata[li]
                if md.get("promoted_to_semantic") or md.get("no_promote"):
                    continue
                if (
                    float(imp_val) >= self.promotion_threshold
                    and int(self.episodic.access_counts[li].item()) >= 2
                ):
                    cluster_id = str(
                        md.get("concept")
                        or md.get("person")
                        or md.get("type", f"epi_{li}")
                    )
                    val_vec = self.episodic.values[li].clone()
                    key_vec = self.key_encoder(val_vec.unsqueeze(0))[0]
                    sem_md = dict(md)
                    sem_md["promoted_from_episodic"] = li
                    self.semantic.write_or_merge(
                        key=key_vec,
                        value=val_vec,
                        metadata=sem_md,
                        current_tick=self.session_tick,
                        prediction_error=float(self.episodic.pred_errors[li].item()),
                        importance_scorer=self.importance_scorer,
                        cluster_key=cluster_id,
                        ema_rate=min(max(float(lr) * 10.0, 0.05), 0.5),
                        _guard_checked=True,
                    )
                    md["promoted_to_semantic"] = True
                    promoted += 1

        merged_dups = self.semantic.prune_near_duplicate_exemplars(self.key_encoder)

        refreshed_proc = 0
        if bool(self.procedural.used.any()) and lr > 0.0:
            blend = min(max(float(lr), 0.0), 0.25)
            used_proc = self.procedural.used.nonzero(as_tuple=False).squeeze(-1)
            reencoded_k = self.key_encoder(self.procedural.values[used_proc])
            self.procedural.keys[used_proc] = (
                (1.0 - blend) * self.procedural.keys[used_proc] + blend * reencoded_k
            )
            refreshed_proc = int(used_proc.numel())

        return {
            "promoted_to_semantic": promoted,
            "merged_duplicate_exemplars": merged_dups,
            "refreshed_procedural": refreshed_proc,
        }

    @torch.no_grad()
    def person_summary(self, person: str) -> dict[str, Any]:
        entries, _ = self.retrieve(
            torch.zeros(self.d_model, device=self.episodic.keys.device),
            top_k=self.capacity,
            person_filter=person,
        )
        actions: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        skills: list[dict[str, Any]] = []
        tiers_seen: set[str] = set()
        for entry in entries:
            md = entry["metadata"]
            tiers_seen.add(str(entry["tier"]))
            t = str(md.get("type", "fact"))
            if t in ("action", "did", "done_to"):
                actions.append(md)
            elif entry["tier"] == TIER_PROCEDURAL or t.startswith("skill"):
                skills.append(md)
            else:
                facts.append(md)
        all_md = facts + actions + skills
        return {
            "person": person,
            "n_memories": len(entries),
            "facts": facts,
            "actions": actions,
            "skills": skills,
            "tiers_seen": sorted(tiers_seen),
            "modalities_seen": sorted(
                {str(md.get("modality")) for md in all_md if md.get("modality") is not None}
            ),
        }

    @torch.no_grad()
    def tier_stats(self) -> dict[str, Any]:
        sem_exemplars_active = int(
            (self.semantic.exemplar_used & self.semantic.used.unsqueeze(-1)).sum().item()
        )
        return {
            "episodic_used": int(self.episodic.used.sum().item()),
            "episodic_capacity": self.epi_cap,
            "semantic_slots_used": int(self.semantic.used.sum().item()),
            "semantic_exemplars_active": sem_exemplars_active,
            "semantic_capacity": self.sem_cap,
            "n_exemplars_per_slot": self.n_exemplars,
            "procedural_used": int(self.procedural.used.sum().item()),
            "procedural_capacity": self.proc_cap,
            "session_tick": self.session_tick,
        }

    def save(self, path: str) -> None:
        if self.heldout_guard is not None:
            audit = self.audit_holdouts()
            if audit["failed_slots"]:
                raise HoldoutViolation(f"cannot save unaudited memory: {audit['failures'][:3]}")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                # Preserve legacy schema 3 for unguarded memories. Schema 4
                # forces older code to reject, rather than silently drop a guard.
                "schema_version": self.SCHEMA_VERSION if self.heldout_guard else 3,
                "holdout_guard": None if self.heldout_guard is None else self.heldout_guard.to_state(),
                "d_model": self.d_model,
                "capacity": self.capacity,
                "epi_cap": self.epi_cap,
                "sem_cap": self.sem_cap,
                "proc_cap": self.proc_cap,
                "n_exemplars": self.n_exemplars,
                "max_trace_steps": self.max_trace_steps,
                "max_pattern_bytes": self.max_pattern_bytes,
                "max_explanation_bytes": self.max_explanation_bytes,
                "session_tick": self.session_tick,
                "module_state": self.state_dict(),
                "episodic_metadata": self.episodic.metadata,
                "semantic_metadata": self.semantic.metadata,
                "semantic_concept_to_slot": self.semantic.concept_to_slot,
                "procedural_metadata": self.procedural.metadata,
            },
            path,
        )

    def load(
        self,
        path: str,
        map_location: str | torch.device = "cpu",
    ) -> MemoryModel:
        self._assert_writable()
        state = torch.load(path, map_location=map_location, weights_only=True)
        schema = int(state.get("schema_version", 1))
        if schema > self.SCHEMA_VERSION:
            raise ValueError("memory checkpoint was created by a newer schema")
        saved_policy = state.get("holdout_guard")
        if schema >= 4 and saved_policy is None:
            raise HoldoutViolation("guarded schema is missing its registry")
        saved_guard = HoldoutGuard.from_state(saved_policy) if saved_policy is not None else None
        if self.heldout_guard is not None and saved_guard is not None:
            self.heldout_guard.assert_compatible(saved_guard)
        guard = self.heldout_guard or saved_guard
        if guard is not None:
            audit = guard.audit_checkpoint(state)
            if audit["failed_slots"]:
                raise HoldoutViolation(f"checkpoint fails holdout audit: {audit['failures'][:3]}")

        if state.get("schema_version", 1) >= 2 and "module_state" in state:
            # Audit before any mutation; strict migration fills only known additions.
            self._loading_audited_state = True
            try:
                self.load_state_dict(state["module_state"], strict=True)
            finally:
                self._loading_audited_state = False
            self.procedural.trace_lens.clamp_(min=0, max=self.procedural.max_trace_steps)
            self.session_tick = int(state.get("session_tick", 0))
            if "episodic_metadata" in state:
                self.episodic.metadata = list(state["episodic_metadata"])
            if "semantic_metadata" in state:
                self.semantic.metadata = list(state["semantic_metadata"])
            if "semantic_concept_to_slot" in state:
                self.semantic.concept_to_slot = dict(state["semantic_concept_to_slot"])
            if "procedural_metadata" in state:
                self.procedural.metadata = list(state["procedural_metadata"])
            if guard is not None:
                self.set_holdout_guard(guard)
            return self

        # Legacy v1 flat checkpoint migration into Hierarchical Tiers
        for mod_name in ("key_encoder", "value_encoder", "query_encoder", "context_proj"):
            if mod_name in state:
                getattr(self, mod_name).load_state_dict(state[mod_name])
        if "contradiction_scorer" in state:
            self.contradiction_scorer.pair_scorer.load_state_dict(state["contradiction_scorer"])
        if "person_linker" in state:
            pl_state = state["person_linker"]
            if pl_state["weight"].shape == self.person_linker.weight.shape:
                self.person_linker.load_state_dict(pl_state)

        old_used = state.get("used", torch.zeros(0, dtype=torch.bool))
        old_keys = state.get("keys", torch.zeros(0, self.d_model))
        old_vals = state.get("values", torch.zeros(0, self.d_model))
        old_counts = state.get("access_counts", torch.zeros(0, dtype=torch.long))
        old_last = state.get("last_access", torch.zeros(0, dtype=torch.long))
        old_meta = state.get("metadata", [])

        device = self.episodic.keys.device
        for i in old_used.nonzero(as_tuple=False).squeeze(-1).tolist():
            idx = int(i)
            md = dict(old_meta[idx]) if idx < len(old_meta) else {}
            t_name = self.infer_tier(md)
            k_i = self._fit_dim(old_keys[idx].to(device))
            v_i = self._fit_dim(old_vals[idx].to(device))
            w_tick = int(md.get("written_at", idx))
            if t_name == TIER_EPISODIC:
                li = self.episodic.write_slot(
                    k_i, v_i, md, w_tick, 0.5, self.importance_scorer
                )
                self.episodic.access_counts[li] = int(old_counts[idx].item())
                self.episodic.last_access[li] = int(old_last[idx].item())
            elif t_name == TIER_SEMANTIC:
                li = self.semantic.write_or_merge(
                    k_i, v_i, md, w_tick, 0.5, self.importance_scorer
                )
                self.semantic.access_counts[li] = int(old_counts[idx].item())
                self.semantic.last_access[li] = int(old_last[idx].item())
            else:
                li = self.procedural.write_skill(
                    k_i,
                    v_i,
                    v_i.unsqueeze(0),
                    1,
                    md,
                    w_tick,
                    0.5,
                    self.importance_scorer,
                )
                self.procedural.access_counts[li] = int(old_counts[idx].item())
                self.procedural.last_access[li] = int(old_last[idx].item())

        self.session_tick = int(state.get("session_tick", 0))
        return self
