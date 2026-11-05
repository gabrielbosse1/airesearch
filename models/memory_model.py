"""ByteCore-DualMemory: one unified 8192-dim skill/event store.

Phase 25 replaces the two physical FIFO stores with ONE policy-ranked bank.
``DualMemory.episodic`` remains a compatibility alias of ``procedural``;
events and skills share capacity, decay policy, and the adapter's timeline.
The geometry is unchanged:

* **keys**   ``(capacity, key_dim=1024)``   -- the SAME width as the core
  model's hidden state, so retrieval is high-fidelity matching, not a
  down-projected guess;
* **values** ``(capacity, value_dim=8192)`` -- the wide skill matrix. Text,
  image, audio and video are all stored as 8192-dim vectors; the core model
  is agnostic to which encoder produced them.

``ProceduralSkillBank`` holds both skills and session events. Short-term
slots decay; Forever slots have exactly zero effective decay and cannot be
selected for overwrite. ``DualMemory`` exposes one :class:`SkillBankView`.

Contract (unchanged mission invariant):
* writes are ``@torch.no_grad()`` storage events -- memory is a store, not a
  gradient update;
* reads are differentiable with respect to the *query* and to any
  batch-local view built by the (trainable) skill projectors;
* persistent keys/values are detached: a write never retains an autograd graph;
* only the small per-slot importance/decay policy parameters are trainable.

A :class:`SkillBankView` is either *global* (``keys (N, D)``, shared by the
whole batch -- the persistent banks) or *batch-local* (``keys (B, N, D)`` --
freshly projected multimodal matrices that still carry gradients).
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

KEY_DIM = 1024
VALUE_DIM = 8192
MODALITIES: tuple[str, ...] = ("text", "image", "audio", "video", "code")
_MODALITY_CODE = {name: index for index, name in enumerate(MODALITIES)}

MEMORY_SCHEMA = 2
FOREVER_IMPORTANCE_FLOOR = 2.0
DEFAULT_DECAY = 1.0


@dataclass(frozen=True)
class SkillBankView:
    """Read-only tensor view of memory slots handed to the GatedSkillAdapter.

    Global:    keys ``(N, key_dim)``,   values ``(N, value_dim)``, mask ``(N,)``
    Batch-local: keys ``(B, N, key_dim)``, values ``(B, N, value_dim)``,
                 mask ``(B, N)``
    ``mask`` is True for active slots. Batch-local tensors may require grad
    (they are the output of the trainable SkillProjectors).
    """

    keys: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.keys.ndim not in (2, 3):
            raise ValueError("keys must be (N, D) or (B, N, D)")
        if self.values.ndim != self.keys.ndim:
            raise ValueError("keys and values must have the same rank")
        if self.mask.dtype != torch.bool:
            raise ValueError("mask must be a bool tensor")
        if self.keys.shape[:-1] != self.values.shape[:-1]:
            raise ValueError("keys and values must agree on slot dimensions")
        if self.mask.shape != self.keys.shape[:-1]:
            raise ValueError("mask must be (N,) or (B, N)")
        if self.keys.ndim == 3 and self.mask.shape[0] != self.keys.shape[0]:
            raise ValueError("batched mask must match the batch dimension")

    @property
    def batched(self) -> bool:
        return self.keys.ndim == 3

    @property
    def key_dim(self) -> int:
        return int(self.keys.shape[-1])

    @property
    def value_dim(self) -> int:
        return int(self.values.shape[-1])

    @property
    def empty(self) -> bool:
        return self.keys.shape[-2] == 0 or not bool(self.mask.any())

    def to(self, device: torch.device | str) -> "SkillBankView":
        return SkillBankView(self.keys.to(device), self.values.to(device), self.mask.to(device))


def stack_views(*views: SkillBankView | None, batch_size: int | None = None) -> SkillBankView | None:
    """Concatenate views along the slot axis, broadcasting global -> batched.

    Used to merge the persistent banks with batch-local multimodal matrices
    into the single view the adapter reads. Returns ``None`` when nothing
    active remains.
    """
    active = [view for view in views if view is not None and view.keys.shape[-2] > 0]
    if not active:
        return None
    batched = any(view.batched for view in active)
    if batched:
        if batch_size is None:
            batch_size = next(view.keys.shape[0] for view in active if view.batched)
        expanded = []
        for view in active:
            if view.batched:
                if view.keys.shape[0] != batch_size:
                    raise ValueError("batched views disagree on batch size")
                expanded.append(view)
            else:
                slots = view.keys.shape[0]
                expanded.append(SkillBankView(
                    view.keys.unsqueeze(0).expand(batch_size, slots, view.key_dim),
                    view.values.unsqueeze(0).expand(batch_size, slots, view.value_dim),
                    view.mask.unsqueeze(0).expand(batch_size, slots),
                ))
        active = expanded
    keys = torch.cat([view.keys for view in active], dim=-2)
    values = torch.cat([view.values for view in active], dim=-2)
    mask = torch.cat([view.mask for view in active], dim=-1)
    return SkillBankView(keys, values, mask)


@dataclass(frozen=True)
class EvictionEvent:
    """Detached snapshot of a full-bank eviction decision (training only).

    Indices refer to per-slot policy parameters, which intentionally survive
    replacement. Ages are measured at the decision, not recomputed later.
    The trainer consumes events each micro-batch before delayed writes can
    change a slot's tier. Events are transient and not checkpointed.
    """

    victim: int
    retained: torch.Tensor
    ages: torch.Tensor
    victim_age: float
    victim_forever: bool


class ProceduralSkillBank(nn.Module):
    """Unified skill/event store with a learnable dynamic eviction policy.

    ``score_i = importance_i - decay_rate_i * age_i``.
    Age is clock ticks since write/explicit reinforcement divided by capacity.
    Short importance is sigmoid(raw) in (0,1); Forever importance is
    2 + sigmoid(raw) in (2,3), with decay exactly zero. Consequently EVERY
    Forever score exceeds EVERY short score, even at infinite age. Eviction
    additionally excludes Forever indices: permanence is a hard guard, not a
    learned preference that a large age could overwhelm.

    Raw policy parameters are per-slot, initialised once, and NEVER reset on
    replacement. Eviction snapshots supply gradients to importance/decay;
    key/value writes remain detached. ``reinforce`` is explicit: merely
    exposing a distractor to retrieval must not make the entire bank fresh.
    """

    def __init__(self, capacity: int = 1024, key_dim: int = KEY_DIM,
                 value_dim: int = VALUE_DIM, *, max_eviction_events: int = 256) -> None:
        super().__init__()
        if int(capacity) < 1:
            raise ValueError("capacity must be a positive integer")
        if int(key_dim) < 1 or int(value_dim) < 1:
            raise ValueError("key_dim and value_dim must be positive integers")
        if int(max_eviction_events) < 1:
            raise ValueError("max_eviction_events must be positive")
        self.capacity, self.key_dim, self.value_dim = int(capacity), int(key_dim), int(value_dim)
        self.register_buffer("keys", torch.zeros(self.capacity, self.key_dim))
        self.register_buffer("values", torch.zeros(self.capacity, self.value_dim))
        self.register_buffer("used", torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer("forever", torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer("written_at", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("last_read_at", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("read_count", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("modality", torch.zeros(self.capacity, dtype=torch.uint8))
        self.importance_raw = nn.Parameter(torch.zeros(self.capacity))
        self.decay_raw = nn.Parameter(torch.full(
            (self.capacity,), math.log(math.expm1(DEFAULT_DECAY))))
        self.metadata: list[dict[str, Any]] = [{} for _ in range(self.capacity)]
        self.eviction_events: deque[EvictionEvent] = deque(maxlen=int(max_eviction_events))
        self.forever_overwrite_attempts = 0
        self._tick = 0

    def __len__(self) -> int:
        return int(self.used.sum().item())

    @property
    def decay_rate(self) -> torch.Tensor:
        """Non-negative, finite for finite logits; identically zero if Forever."""
        return torch.where(self.forever, torch.zeros_like(self.decay_raw),
                           F.softplus(self.decay_raw))

    @property
    def importance(self) -> torch.Tensor:
        return torch.where(self.forever,
                           FOREVER_IMPORTANCE_FLOOR + torch.sigmoid(self.importance_raw),
                           torch.sigmoid(self.importance_raw))

    def ages(self) -> torch.Tensor:
        return (float(self._tick) - self.written_at.float()) / float(self.capacity)

    def scores(self) -> torch.Tensor:
        return self.importance - self.decay_rate * self.ages()

    def _check_vector(self, tensor: torch.Tensor, width: int, name: str) -> None:
        if tensor.ndim != 1 or tensor.shape[0] != width:
            raise ValueError(f"{name} must have shape ({width},), got {tuple(tensor.shape)}")

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor, *, modality: str = "text",
              metadata: Mapping[str, Any] | None = None, forever: bool = False) -> int:
        """Store one detached skill/event; evict the lowest-scoring short slot.

        A Forever slot can be upgraded/rewritten only by explicitly targeting
        it through ``write_forever``. Ordinary writes NEVER overwrite it.
        """
        if modality not in _MODALITY_CODE:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        if not isinstance(forever, bool):
            raise ValueError("forever must be a bool")
        self._check_vector(key, self.key_dim, "key")
        self._check_vector(value, self.value_dim, "value")
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("memory keys and values must be finite")
        free = (~self.used).nonzero(as_tuple=False).squeeze(-1)
        if free.numel() > 0:
            slot = int(free[0].item())
        else:
            evictable = self.used & ~self.forever
            if not bool(evictable.any()):
                self.forever_overwrite_attempts += 1
                ages = self.ages().detach()
                victim = int(torch.argmin(self.importance.detach()).item())
                self.eviction_events.append(EvictionEvent(
                    victim, torch.empty(0, dtype=torch.long, device=self.keys.device),
                    ages, float(ages[victim]), True))
                raise MemoryError("bank is full of Forever slots; refusing overwrite")
            ages = self.ages().detach()
            scores = self.scores().detach().masked_fill(~evictable, torch.inf)
            slot = int(torch.argmin(scores).item())
            retained = evictable.clone()
            retained[slot] = False
            self.eviction_events.append(EvictionEvent(
                slot, retained.nonzero(as_tuple=False).squeeze(-1).clone(),
                ages.clone(), float(ages[slot]), bool(self.forever[slot])))
        self._tick += 1
        self.keys[slot] = key.detach().to(self.keys.dtype)
        self.values[slot] = value.detach().to(self.values.dtype)
        self.used[slot] = True
        self.forever[slot] = bool(forever)
        self.written_at[slot] = self._tick
        self.last_read_at[slot] = 0
        self.read_count[slot] = 0
        self.modality[slot] = _MODALITY_CODE[modality]
        record = dict(metadata or {})
        record.update(modality=modality, written_at=self._tick, forever=bool(forever))
        self.metadata[slot] = record
        # Do NOT assign importance_raw/decay_raw here: replacement must not
        # erase the policy learned from previous occupant eviction decisions.
        return slot

    @torch.no_grad()
    def write_forever(self, slot: int, key: torch.Tensor, value: torch.Tensor, *,
                      modality: str = "text", metadata: Mapping[str, Any] | None = None) -> int:
        """Explicit host curation of a permanent slot (the only overwrite path)."""
        if not 0 <= int(slot) < self.capacity:
            raise ValueError("slot outside bank capacity")
        if self.used[slot] and not self.forever[slot]:
            raise ValueError("write_forever cannot overwrite a short-term slot")
        self._check_vector(key, self.key_dim, "key")
        self._check_vector(value, self.value_dim, "value")
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("memory keys and values must be finite")
        if modality not in _MODALITY_CODE:
            raise ValueError(f"unknown modality {modality!r}")
        self._tick += 1
        self.keys[slot] = key.detach()
        self.values[slot] = value.detach()
        self.used[slot], self.forever[slot] = True, True
        self.written_at[slot] = self._tick
        self.modality[slot] = _MODALITY_CODE[modality]
        self.metadata[slot] = {**dict(metadata or {}), "modality": modality,
                               "written_at": self._tick, "forever": True}
        return int(slot)

    @torch.no_grad()
    def mark_forever(self, slot: int) -> None:
        if not 0 <= int(slot) < self.capacity or not bool(self.used[slot]):
            raise ValueError("cannot pin an unused/out-of-range slot")
        self.forever[slot] = True
        self.metadata[slot]["forever"] = True

    @torch.no_grad()
    def reinforce(self, indices: torch.Tensor | list[int]) -> None:
        """Refresh age only for slots actually used, not all distractors."""
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.keys.device)
        if idx.ndim != 1 or (idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.capacity)):
            raise ValueError("invalid reinforcement indices")
        if idx.numel() and not bool(self.used[idx].all()):
            raise ValueError("cannot reinforce unused slots")
        self.written_at[idx] = self._tick
        self.last_read_at[idx] = self._tick
        self.read_count[idx] += 1

    def policy_loss(self, *, margin: float = 0.25) -> torch.Tensor:
        """Bounded, stable auxiliary loss for the learnable eviction policy.

        stale eviction: victim scores below fresher retained slots, and
        softplus(margin - decay*age) rewards efficient stale removal;
        freshness: softplus(decay*age - margin) discourages premature decay;
        Forever: penalise nonzero raw decay AND any score-margin violation.
        The hard tier bounds make the latter term zero in a valid bank; an
        attempted all-Forever overwrite is rejected and counted separately.
        Softplus/BCE-with-logits avoids exp(decay*age) and log(0) gradients.
        """
        if margin <= 0:
            raise ValueError("margin must be positive")
        device, dtype = self.decay_raw.device, self.decay_raw.dtype
        zero = (self.importance_raw.sum() + self.decay_raw.sum()) * 0.0
        terms = [zero]
        imp, decay = self.importance, self.decay_rate
        short = self.used & ~self.forever
        if bool(short.any()):
            terms.append(F.softplus(decay[short] * self.ages()[short] - margin).mean())
        for event in self.eviction_events:
            victim_score = imp[event.victim] - decay[event.victim] * event.victim_age
            if event.retained.numel():
                ages = event.ages[event.retained]
                fresh = ages < 0.5 * event.victim_age
                if bool(fresh.any()):
                    retained_scores = imp[event.retained[fresh]] - decay[event.retained[fresh]] * ages[fresh]
                    terms.append(F.softplus(margin + victim_score - retained_scores).mean())
            if event.victim_age >= 0.5:
                terms.append(F.softplus(margin - decay[event.victim] * event.victim_age))
            if event.victim_forever:  # defensive: ordinary eviction never does this
                terms.append(F.softplus(decay[event.victim])
                             + F.relu(margin + 1.0 - imp[event.victim]))
        if bool(self.forever.any()):
            terms.append(F.softplus(self.decay_raw[self.forever]).mean())
            if bool(short.any()):
                violation = F.relu(margin + imp[short].max() - imp[self.forever].min())
                terms.append(violation + zero)
        return torch.stack(terms).mean()

    def clear_policy_events(self) -> None:
        self.eviction_events.clear()

    def view(self) -> SkillBankView:
        idx = self.used.nonzero(as_tuple=False).squeeze(-1)
        return SkillBankView(self.keys[idx], self.values[idx],
                             torch.ones(idx.numel(), dtype=torch.bool, device=self.keys.device))

    @torch.no_grad()
    def clear(self) -> None:
        for name in ("keys", "values", "used", "forever", "written_at",
                     "last_read_at", "read_count", "modality"):
            getattr(self, name).zero_()
        self.importance_raw.zero_()
        self.decay_raw.fill_(math.log(math.expm1(DEFAULT_DECAY)))
        self.metadata = [{} for _ in range(self.capacity)]
        self.eviction_events.clear()
        self.forever_overwrite_attempts = 0
        self._tick = 0

    def state_payload(self) -> dict[str, Any]:
        return {"schema_version": MEMORY_SCHEMA, "capacity": self.capacity,
                "key_dim": self.key_dim, "value_dim": self.value_dim,
                "buffers": {name: getattr(self, name).clone()
                            for name in ("keys", "values", "used", "forever", "written_at",
                                         "last_read_at", "read_count", "modality")},
                "parameters": {"importance_raw": self.importance_raw.detach().clone(),
                               "decay_raw": self.decay_raw.detach().clone()},
                "metadata": list(self.metadata), "tick": self._tick,
                "forever_overwrite_attempts": self.forever_overwrite_attempts}

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != MEMORY_SCHEMA:
            raise ValueError("not a unified dynamic-memory payload (schema 2)")
        if (payload["capacity"], payload["key_dim"], payload["value_dim"]) != (
                self.capacity, self.key_dim, self.value_dim):
            raise ValueError("bank geometry does not match the payload")
        expected_buffers = {"keys", "values", "used", "forever", "written_at",
                            "last_read_at", "read_count", "modality"}
        if set(payload["buffers"]) != expected_buffers or set(payload["parameters"]) != {
                "importance_raw", "decay_raw"}:
            raise ValueError("incomplete dynamic-memory state")
        with torch.no_grad():
            for name, tensor in payload["buffers"].items():
                getattr(self, name).copy_(tensor)
            self.importance_raw.copy_(payload["parameters"]["importance_raw"])
            self.decay_raw.copy_(payload["parameters"]["decay_raw"])
        self.metadata = list(payload["metadata"])
        self._tick = int(payload["tick"])
        self.forever_overwrite_attempts = int(payload.get("forever_overwrite_attempts", 0))
        self.eviction_events.clear()


class EpisodicEventBank(ProceduralSkillBank):
    """Compatibility constructor; unified banks store events and skills together."""

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor, *, modality: str = "text",
              metadata: Mapping[str, Any] | None = None) -> int:
        return super().write(key, value, modality=modality, metadata=metadata)


class DualMemory(nn.Module):
    """Unified physical bank; episodic/procedural are aliases, not two stores.

    Legacy capacity arguments are summed for configuration compatibility.
    New configs should use ``capacity``. The default remains 2048 total slots.
    """

    def __init__(self, procedural_capacity: int = 1024, episodic_capacity: int = 1024,
                 key_dim: int = KEY_DIM, value_dim: int = VALUE_DIM, *,
                 capacity: int | None = None) -> None:
        super().__init__()
        self.key_dim, self.value_dim = int(key_dim), int(value_dim)
        total = int(capacity) if capacity is not None else int(procedural_capacity) + int(episodic_capacity)
        self.procedural = ProceduralSkillBank(total, key_dim, value_dim)
        self.episodic = self.procedural  #the same nn.Module, registered only once

    @property
    def bank(self) -> ProceduralSkillBank:
        return self.procedural

    def __len__(self) -> int:
        return len(self.procedural)

    @torch.no_grad()
    def write_skill(self, key: torch.Tensor, value: torch.Tensor, *, modality: str = "image",
                    metadata: Mapping[str, Any] | None = None, forever: bool = False) -> int:
        record = {**dict(metadata or {}), "tier": "procedural"}
        return self.procedural.write(key, value, modality=modality, metadata=record, forever=forever)

    @torch.no_grad()
    def write_event(self, key: torch.Tensor, value: torch.Tensor, *,
                    metadata: Mapping[str, Any] | None = None, forever: bool = False) -> int:
        record = {**dict(metadata or {}), "tier": "episodic"}
        return self.procedural.write(key, value, modality="text", metadata=record, forever=forever)

    def view(self) -> SkillBankView | None:
        return self.procedural.view()

    def policy_loss(self, **kwargs: Any) -> torch.Tensor:
        return self.procedural.policy_loss(**kwargs)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": MEMORY_SCHEMA,
                    "bank": self.procedural.state_payload()}, target)
        return target

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != MEMORY_SCHEMA or "bank" not in payload:
            raise ValueError("not a unified dynamic-memory checkpoint (schema 2)")
        self.procedural.load_payload(payload["bank"])

    def manifest(self) -> str:
        rows = [{"bank": "unified", "slot": slot, **meta}
                for slot, meta in enumerate(self.procedural.metadata)
                if bool(self.procedural.used[slot])]
        return json.dumps(rows, ensure_ascii=False, indent=2)
