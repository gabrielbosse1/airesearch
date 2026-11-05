"""ByteCore-DualMemory: the 8192-dim skill/event store (Phase 24).

Two persistent banks share one geometry:

* **keys**   ``(capacity, key_dim=1024)``   -- the SAME width as the core
  model's hidden state, so retrieval is high-fidelity matching, not a
  down-projected guess;
* **values** ``(capacity, value_dim=8192)`` -- the wide skill matrix. Text,
  image, audio and video are all stored as 8192-dim vectors; the core model
  is agnostic to which encoder produced them.

``ProceduralSkillBank`` holds skills (multimodal matrices, learned procedures);
``EpisodicEventBank`` holds session events. ``DualMemory`` bundles both and
exposes a single :class:`SkillBankView` for the Gated Skill Adapter.

Contract (unchanged mission invariant):
* writes are ``@torch.no_grad()`` storage events -- memory is a store, not a
  gradient update;
* reads are differentiable with respect to the *query* and to any
  batch-local view built by the (trainable) skill projectors;
* persistent buffers are detached: a write never retains an autograd graph.

A :class:`SkillBankView` is either *global* (``keys (N, D)``, shared by the
whole batch -- the persistent banks) or *batch-local* (``keys (B, N, D)`` --
freshly projected multimodal matrices that still carry gradients).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

KEY_DIM = 1024
VALUE_DIM = 8192
MODALITIES: tuple[str, ...] = ("text", "image", "audio", "video", "code")
_MODALITY_CODE = {name: index for index, name in enumerate(MODALITIES)}

MEMORY_SCHEMA = 1


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


class ProceduralSkillBank(nn.Module):
    """Persistent skill matrices: 1024-dim keys, 8192-dim values, FIFO eviction.

    All state lives in registered buffers (no parameters): the bank itself is
    never trained. Learning happens in the projectors that *produce* keys and
    values and in the adapter that *reads* them.
    """

    def __init__(self, capacity: int = 1024, key_dim: int = KEY_DIM,
                 value_dim: int = VALUE_DIM) -> None:
        super().__init__()
        if int(capacity) < 1:
            raise ValueError("capacity must be a positive integer")
        if int(key_dim) < 1 or int(value_dim) < 1:
            raise ValueError("key_dim and value_dim must be positive integers")
        self.capacity = int(capacity)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.register_buffer("keys", torch.zeros(self.capacity, self.key_dim))
        self.register_buffer("values", torch.zeros(self.capacity, self.value_dim))
        self.register_buffer("used", torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer("written_at", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("modality", torch.zeros(self.capacity, dtype=torch.uint8))
        self.metadata: list[dict[str, Any]] = [{} for _ in range(self.capacity)]
        self._tick = 0

    def __len__(self) -> int:
        return int(self.used.sum().item())

    def _check_vector(self, tensor: torch.Tensor, width: int, name: str) -> None:
        if tensor.ndim != 1 or tensor.shape[0] != width:
            raise ValueError(f"{name} must have shape ({width},), got {tuple(tensor.shape)}")

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor, *, modality: str = "text",
              metadata: Mapping[str, Any] | None = None) -> int:
        """Store one skill. Gradient-free by contract; tensors are detached."""
        if modality not in _MODALITY_CODE:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        self._check_vector(key, self.key_dim, "key")
        self._check_vector(value, self.value_dim, "value")
        free = (~self.used).nonzero(as_tuple=False).squeeze(-1)
        if free.numel() > 0:
            slot = int(free[0].item())
        else:
            # FIFO: evict the oldest written slot. Deterministic, auditable.
            ages = self.written_at.clone()
            ages[~self.used] = torch.iinfo(torch.long).max
            slot = int(torch.argmin(ages).item())
        self._tick += 1
        self.keys[slot] = key.detach().to(self.keys.dtype)
        self.values[slot] = value.detach().to(self.values.dtype)
        self.used[slot] = True
        self.written_at[slot] = self._tick
        self.modality[slot] = _MODALITY_CODE[modality]
        record = dict(metadata or {})
        record["modality"] = modality
        record["written_at"] = self._tick
        self.metadata[slot] = record
        return slot

    def view(self) -> SkillBankView:
        """Active slots as a global, detached view for the adapter."""
        idx = self.used.nonzero(as_tuple=False).squeeze(-1)
        return SkillBankView(self.keys[idx], self.values[idx],
                             torch.ones(idx.numel(), dtype=torch.bool, device=self.keys.device))

    @torch.no_grad()
    def clear(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.used.zero_()
        self.written_at.zero_()
        self.modality.zero_()
        self.metadata = [{} for _ in range(self.capacity)]
        self._tick = 0

    def state_payload(self) -> dict[str, Any]:
        return {"schema_version": MEMORY_SCHEMA, "capacity": self.capacity,
                "key_dim": self.key_dim, "value_dim": self.value_dim,
                "buffers": {name: getattr(self, name).clone()
                            for name in ("keys", "values", "used", "written_at", "modality")},
                "metadata": list(self.metadata), "tick": self._tick}

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != MEMORY_SCHEMA:
            raise ValueError("not a Phase 24 skill-bank payload")
        if (payload["capacity"], payload["key_dim"], payload["value_dim"]) != (
                self.capacity, self.key_dim, self.value_dim):
            raise ValueError("bank geometry does not match the payload")
        with torch.no_grad():
            for name, tensor in payload["buffers"].items():
                getattr(self, name).copy_(tensor)
        self.metadata = list(payload["metadata"])
        self._tick = int(payload["tick"])


class EpisodicEventBank(ProceduralSkillBank):
    """Same geometry, event semantics: session facts written as text matrices."""

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor, *, modality: str = "text",
              metadata: Mapping[str, Any] | None = None) -> int:
        return super().write(key, value, modality=modality, metadata=metadata)


class DualMemory(nn.Module):
    """Episodic events + procedural skills behind one read interface."""

    def __init__(self, procedural_capacity: int = 1024, episodic_capacity: int = 1024,
                 key_dim: int = KEY_DIM, value_dim: int = VALUE_DIM) -> None:
        super().__init__()
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.procedural = ProceduralSkillBank(procedural_capacity, key_dim, value_dim)
        self.episodic = EpisodicEventBank(episodic_capacity, key_dim, value_dim)

    def __len__(self) -> int:
        return len(self.procedural) + len(self.episodic)

    @torch.no_grad()
    def write_skill(self, key: torch.Tensor, value: torch.Tensor, *, modality: str = "image",
                    metadata: Mapping[str, Any] | None = None) -> int:
        return self.procedural.write(key, value, modality=modality, metadata=metadata)

    @torch.no_grad()
    def write_event(self, key: torch.Tensor, value: torch.Tensor, *,
                    metadata: Mapping[str, Any] | None = None) -> int:
        return self.episodic.write(key, value, modality="text", metadata=metadata)

    def view(self) -> SkillBankView | None:
        return stack_views(self.episodic.view(), self.procedural.view())

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": MEMORY_SCHEMA,
                    "procedural": self.procedural.state_payload(),
                    "episodic": self.episodic.state_payload()}, target)
        return target

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != MEMORY_SCHEMA:
            raise ValueError("not a Phase 24 dual-memory checkpoint")
        self.procedural.load_payload(payload["procedural"])
        self.episodic.load_payload(payload["episodic"])

    def manifest(self) -> str:
        """Human-readable audit of stored slots (no tensor dumps)."""
        rows = []
        for name, bank in (("episodic", self.episodic), ("procedural", self.procedural)):
            for slot, meta in enumerate(bank.metadata):
                if bool(bank.used[slot]):
                    rows.append({"bank": name, "slot": slot, **meta})
        return json.dumps(rows, ensure_ascii=False, indent=2)
