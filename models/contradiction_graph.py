"""Temporally aware, bounded multi-hop reasoning over explicit memory facts.

The neural message-passing scorer is trainable.  The graph also provides a
conservative symbolic check for explicitly structured facts: it never treats
unrelated facts, non-overlapping time intervals, or merely different values of
a non-exclusive relation as contradictions.  This is evidence checking, not a
claim that a person is lying or that stored information is true.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identifier(value: Any, name: str) -> bytes:
    if isinstance(value, str):
        value = value.encode("utf-8")
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{name} must be non-empty bytes or a UTF-8 string")
    # ASCII-only folding; no tokenizer or linguistic normalization.
    return value.lower()


def _time(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric time or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class Fact:
    """An explicit fact with optional half-open validity interval [from, to).

    ``exclusive`` declares at most one positive object for this relation at a
    time. ``transitive`` explicitly permits chaining that predicate. Neither
    property is inferred from text. All fact times must use the same scale.
    """

    subject: bytes
    predicate: bytes
    object: bytes
    polarity: bool = True
    valid_from: float | None = None
    valid_to: float | None = None
    exclusive: bool = False
    transitive: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _identifier(self.subject, "subject"))
        object.__setattr__(self, "predicate", _identifier(self.predicate, "predicate"))
        object.__setattr__(self, "object", _identifier(self.object, "object"))
        for name in ("polarity", "exclusive", "transitive"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        start = _time(self.valid_from, "valid_from")
        end = _time(self.valid_to, "valid_to")
        if start is not None and end is not None and start >= end:
            raise ValueError("valid_from must be earlier than valid_to")
        object.__setattr__(self, "valid_from", start)
        object.__setattr__(self, "valid_to", end)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Fact:
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            polarity=data.get("polarity", True),
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            exclusive=data.get("exclusive", False),
            transitive=data.get("transitive", False),
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "polarity": self.polarity,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "exclusive": self.exclusive,
            "transitive": self.transitive,
        }


def fact_from_metadata(metadata: Mapping[str, Any]) -> Fact | None:
    source = metadata.get("structured_fact", metadata)
    if not isinstance(source, Mapping) or not all(k in source for k in ("subject", "predicate", "object")):
        return None
    return Fact.from_mapping(source)


@dataclass(frozen=True)
class FactRecord:
    entry: dict[str, Any]
    fact: Fact | None


@dataclass(frozen=True)
class GraphEvidence:
    """A supporting or conflicting fact and the actual chain used to reach it."""

    entry: dict[str, Any]
    fact: Fact
    kind: str  # "support" or "conflict"
    path_slots: tuple[int, ...]
    valid_from: float | None
    valid_to: float | None
    learned_consistency: float


@dataclass(frozen=True)
class ContradictionReport:
    status: str  # "support", "conflict", "mixed", or "unknown"
    consistency: float
    evidence: tuple[GraphEvidence, ...]
    truncated: bool = False


class TemporalGraphReasoner(nn.Module):
    """Trainable temporal graph message passing followed by a pair scorer.

    Adjacency is node-to-node; messages flow for ``hops`` rounds. The zero
    initialization on residual gates makes migration from the original pair
    scorer prediction-preserving while allowing the gates to learn. Symbolic
    temporal/path checks are intentionally separate from this learned score.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.pair_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, 1), nn.Sigmoid()
        )
        self.time_mlp = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.message_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.time_gate = nn.Parameter(torch.zeros(()))
        self.message_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        query: torch.Tensor,
        nodes: torch.Tensor,
        adjacency: torch.Tensor,
        age: torch.Tensor,
        *,
        hops: int = 2,
    ) -> torch.Tensor:
        """Return one differentiable consistency score per node, shape (N,)."""
        if nodes.ndim != 2 or nodes.shape[1] != self.d_model:
            raise ValueError("nodes must have shape (N, d_model)")
        n = nodes.shape[0]
        if query.numel() != self.d_model or adjacency.shape != (n, n) or age.shape != (n,):
            raise ValueError("query, adjacency, and age shapes do not match graph nodes")
        if not 0 <= hops <= 16:
            raise ValueError("hops must be between 0 and 16")
        if n == 0:
            return nodes.new_empty((0,))
        if not bool(torch.isfinite(adjacency).all() and (adjacency >= 0).all() and torch.isfinite(age).all() and (age >= 0).all()):
            raise ValueError("graph adjacency and ages must be finite and non-negative")
        adj = adjacency.to(device=nodes.device, dtype=nodes.dtype)
        adj = adj + torch.eye(n, device=nodes.device, dtype=nodes.dtype)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        age_features = torch.log1p(age.to(device=nodes.device, dtype=nodes.dtype).clamp_min(0)).unsqueeze(-1)
        h = nodes + torch.tanh(self.time_gate) * self.time_mlp(age_features)
        for _ in range(hops):
            messages = adj @ h
            h = h + torch.tanh(self.message_gate) * self.message_mlp(torch.cat([h, messages], dim=-1))
        q = query.reshape(1, self.d_model).to(device=nodes.device, dtype=nodes.dtype).expand(n, -1)
        learned = self.pair_scorer(torch.cat([h, q], dim=-1)).squeeze(-1)
        cosine = F.cosine_similarity(h, q, dim=-1)
        return 0.5 * learned + 0.25 * (cosine + 1.0)


def _bounds(fact: Fact) -> tuple[float, float]:
    return (
        fact.valid_from if fact.valid_from is not None else -math.inf,
        fact.valid_to if fact.valid_to is not None else math.inf,
    )


def build_adjacency(records: list[FactRecord], *, device: torch.device) -> torch.Tensor:
    """Link co-valid facts sharing an entity or an explicitly recorded person."""
    n = len(records)
    result = torch.zeros(n, n, device=device)
    for i, left in enumerate(records):
        for j in range(i + 1, n):
            right = records[j]
            lf, rf = left.fact, right.fact
            if lf is not None and rf is not None:
                l0, l1 = _bounds(lf)
                r0, r1 = _bounds(rf)
                if max(l0, r0) >= min(l1, r1):
                    continue
                linked = bool({lf.subject, lf.object} & {rf.subject, rf.object})
            else:
                lm = left.entry.get("metadata", {})
                rm = right.entry.get("metadata", {})
                linked = bool(lm.get("person") is not None and lm.get("person") == rm.get("person"))
            if linked:
                result[i, j] = 1.0
                result[j, i] = 1.0
    return result


def find_evidence(
    records: list[FactRecord],
    query: Fact,
    learned_scores: torch.Tensor,
    *,
    as_of: float | None = None,
    max_hops: int = 3,
    max_states: int = 4096,
) -> tuple[list[GraphEvidence], bool]:
    """Find explicit evidence along alias and declared-transitive paths.

    Paths and their validity intervals are tracked separately, so an expired
    alias cannot make a current fact appear reachable. Work is bounded and the
    returned truncation flag makes an incomplete search visible to callers.
    """
    if not 1 <= max_hops <= 16 or max_states < 1:
        raise ValueError("max_hops must be 1..16 and max_states must be positive")
    if learned_scores.shape != (len(records),):
        raise ValueError("one learned score is required for each record")
    at = _time(as_of, "as_of")
    q0, q1 = _bounds(query)
    if at is not None and not (q0 <= at < q1):
        return [], False

    # current entity, prior record indices, interval, seen entities, traversed
    # a transitive predicate edge
    queue = deque([(query.subject, (), q0, q1, frozenset([query.subject]), False)])
    evidence: list[GraphEvidence] = []
    seen_evidence: set[tuple[str, tuple[int, ...]]] = set()
    processed = 0
    truncated = False
    while queue:
        if processed >= max_states:
            truncated = True
            break
        entity, path, lo, hi, seen, via_transitive = queue.popleft()
        processed += 1
        for i, rec in enumerate(records):
            fact = rec.fact
            if fact is None or i in path:
                continue
            f0, f1 = _bounds(fact)
            new_lo, new_hi = max(lo, f0), min(hi, f1)
            if new_lo >= new_hi or (at is not None and not (new_lo <= at < new_hi)):
                continue
            final_path = path + (i,)
            if fact.subject == entity and fact.predicate == query.predicate:
                # A negative fact cannot be composed through a positive transitive
                # chain; nor can undeclared relations be chained.
                composable = not via_transitive or (fact.polarity and (fact.transitive or query.transitive))
                if composable:
                    same_object = fact.object == query.object
                    conflict = (same_object and fact.polarity != query.polarity) or (
                        not same_object and fact.polarity and query.polarity
                        and (fact.exclusive or query.exclusive) and not via_transitive
                    )
                    support = same_object and fact.polarity == query.polarity
                    kind = "conflict" if conflict else ("support" if support else "")
                    slots = tuple(int(records[k].entry["slot"]) for k in final_path)
                    key = (kind, slots)
                    if kind and key not in seen_evidence:
                        seen_evidence.add(key)
                        evidence.append(GraphEvidence(
                            entry=rec.entry, fact=fact, kind=kind, path_slots=slots,
                            valid_from=None if new_lo == -math.inf else new_lo,
                            valid_to=None if new_hi == math.inf else new_hi,
                            learned_consistency=float(learned_scores[i].detach().item()),
                        ))
            if len(final_path) >= max_hops or not fact.polarity:
                continue
            next_entity: bytes | None = None
            is_transitive = via_transitive
            if fact.predicate == b"same_as":
                if fact.subject == entity:
                    next_entity = fact.object
                elif fact.object == entity:
                    next_entity = fact.subject
            elif fact.subject == entity and fact.predicate == query.predicate and (fact.transitive or query.transitive):
                next_entity = fact.object
                is_transitive = True
            if next_entity is not None and next_entity not in seen:
                if processed + len(queue) >= max_states:
                    truncated = True
                else:
                    queue.append((next_entity, final_path, new_lo, new_hi, seen | {next_entity}, is_transitive))
    return evidence, truncated
