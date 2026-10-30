"""Concept bank: a differentiable key-value memory that supports *write without
retraining*.

New concepts are inserted directly into memory (a detached tensor copy), so
learning a new concept requires no gradient step -- the model "stores" the
concept and can recall/condition generation on it instantly at inference.

The bank is a torch Module (so keys/values are saved in state_dict) but its
entries are updated by a plain memory write, never by an optimizer step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any

from .holdout import HoldoutGuard, HoldoutViolation


class ConceptBank(nn.Module):
    def __init__(self, capacity: int = 256, d: int = 64):
        super().__init__()
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1 or not isinstance(d, int) or isinstance(d, bool) or d < 1:
            raise ValueError("concept capacity and dimension must be positive integers")
        self.capacity = capacity
        self.d = d
        # Memory entries (buffers: persisted, not optimized).
        self.register_buffer("keys", torch.zeros(capacity, d))
        self.register_buffer("vals", torch.zeros(capacity, d))
        self.register_buffer("used", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.name_to_idx = {}  # concept name -> slot index
        self.heldout_guard: HoldoutGuard | None = None
        self.slot_provenance: dict[int, list[dict[str, Any]]] = {}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if self.heldout_guard is not None:
            raise HoldoutViolation("load concepts through audited ConceptBank.load")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _audit_state(self, state: dict[str, Any], guard: HoldoutGuard) -> None:
        used, names = state["used"], state["name_to_idx"]
        provenance = state.get("slot_provenance", {})
        for name, idx in names.items():
            if not isinstance(name, str) or not isinstance(idx, int) or not 0 <= idx < len(used) or not bool(used[idx]):
                raise HoldoutViolation("invalid concept name-to-slot mapping")
            guard._check_metadata({"name": name})
        for idx in used.nonzero(as_tuple=False).flatten().tolist():
            records = provenance.get(idx, provenance.get(str(idx)))
            if not isinstance(records, list) or not records:
                raise HoldoutViolation("concept has no auditable raw-byte provenance")
            for record in records:
                guard.validate_provenance(record)

    def set_holdout_guard(self, guard: HoldoutGuard) -> None:
        if not isinstance(guard, HoldoutGuard):
            raise TypeError("guard must be a HoldoutGuard")
        if self.heldout_guard is not None:
            self.heldout_guard.assert_compatible(guard)
        self._audit_state({"used": self.used, "name_to_idx": self.name_to_idx,
                           "slot_provenance": self.slot_provenance}, guard)
        self.heldout_guard = guard

    def audit_holdouts(self) -> None:
        if self.heldout_guard is None:
            raise HoldoutViolation("no concept holdout registry is attached")
        self._audit_state({"used": self.used, "name_to_idx": self.name_to_idx,
                           "slot_provenance": self.slot_provenance}, self.heldout_guard)

    def preflight_load(self, path: str) -> dict[str, Any]:
        container = torch.load(path, map_location="cpu", weights_only=True)
        if "guarded_concept_schema" in container:
            if container["guarded_concept_schema"] != 1 or not isinstance(container.get("guarded_state"), dict):
                raise HoldoutViolation("unsupported guarded concept checkpoint")
            state = container["guarded_state"]
            if state.get("holdout_guard") is None:
                raise HoldoutViolation("guarded concept checkpoint lacks a holdout registry")
        else:
            state = container
        if (tuple(state["keys"].shape) != (self.capacity, self.d)
                or tuple(state["vals"].shape) != (self.capacity, self.d)
                or tuple(state["used"].shape) != (self.capacity,)):
            raise ValueError("concept checkpoint dimensions do not match this bank")
        saved = state.get("holdout_guard")
        saved_guard = HoldoutGuard.from_state(saved) if saved is not None else None
        if self.heldout_guard is not None and saved_guard is not None:
            self.heldout_guard.assert_compatible(saved_guard)
        guard = self.heldout_guard or saved_guard
        if guard is not None:
            self._audit_state(state, guard)
        return state

    # ------------------------------------------------------------------ #
    # Write: store / update a concept (no gradient, no retraining)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def write(self, name: str, key: torch.Tensor, val: torch.Tensor, *, raw_input: bytes | None = None) -> int:
        """Store concept ``name`` with key/value vectors.

        Returns the slot index. If the name already exists, its slot is
        overwritten (exponential moving average merge with the old value so
        repeated writes reinforce rather than destroy).
        """
        if not isinstance(name, str):
            raise TypeError("concept name must be a string")
        provenance = None
        if self.heldout_guard is not None:
            provenance = self.heldout_guard.authorize(raw_input, metadata={"name": name})
        key = key.detach().reshape(-1).float()
        val = val.detach().reshape(-1).float()
        if key.shape[0] != self.d or val.shape[0] != self.d:
            raise ValueError(f"key/val must have dim {self.d}, got "
                             f"{key.shape[0]}/{val.shape[0]}")
        if not bool(torch.isfinite(key).all() and torch.isfinite(val).all()):
            raise ValueError("concept vectors must be finite")

        if name in self.name_to_idx:
            idx = self.name_to_idx[name]
            records = self.slot_provenance.get(idx, [])
            if provenance is not None and provenance not in records and len(records) >= 4096:
                raise HoldoutViolation("concept provenance limit reached; allocate a new concept")
            # Reinforce existing concept with an EMA merge.
            old_v = self.vals[idx]
            self.vals[idx] = 0.5 * old_v + 0.5 * val
            self.keys[idx] = key  # refresh key
            self.used[idx] = True
            if provenance is not None and provenance not in records:
                self.slot_provenance.setdefault(idx, []).append(provenance)
            return idx

        # Allocate a free slot; overwrite oldest if full (simple FIFO).
        if int(self.count.item()) < self.capacity:
            free = (~self.used).nonzero(as_tuple=False)
            if free.shape[0] > 0:
                idx = int(free[0])
            else:
                idx = int(self.count.item())
        else:
            idx = int(self.count.item() % self.capacity)
        if bool(self.used[idx]):
            self.name_to_idx = {old_name: old_idx for old_name, old_idx in self.name_to_idx.items() if old_idx != idx}
        self.slot_provenance.pop(idx, None)
        self.keys[idx] = key
        self.vals[idx] = val
        self.used[idx] = True
        self.name_to_idx[name] = idx
        self.count += 1
        if provenance is not None:
            self.slot_provenance[idx] = [provenance]
        return idx

    # ------------------------------------------------------------------ #
    # Read: soft-attention recall over used keys
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def read(self, query: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
        """Recall a concept vector for ``query`` via key-attention.

        query: (B, d) or (d,). Returns (B, d) concept vector. If the bank is
        empty, returns zeros.
        """
        single = query.dim() == 1
        q = query.detach().float()
        if single:
            q = q.unsqueeze(0)  # (1, d)

        if not bool(self.used.any()):
            out = torch.zeros(q.shape[0], self.d, device=q.device)
            return out[0] if single else out

        used_keys = self.keys[self.used]      # (M, d)
        used_vals = self.vals[self.used]      # (M, d)

        # Scaled dot-product attention over keys.
        logits = (q @ used_keys.T) / math.sqrt(self.d)  # (B, M)
        if top_k is not None and top_k < used_keys.shape[0]:
            # Keep only top-k logits.
            vals, idx = torch.topk(logits, k=top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, idx, vals)
            logits = mask
        attn = F.softmax(logits, dim=-1)  # (B, M)
        out = attn @ used_vals  # (B, d)
        return out[0] if single else out

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if self.heldout_guard is not None:
            self.audit_holdouts()
        state = {
            "guard_schema_version": 1 if self.heldout_guard is not None else 0,
            "holdout_guard": None if self.heldout_guard is None else self.heldout_guard.to_state(),
            "slot_provenance": self.slot_provenance,
            "keys": self.keys,
            "vals": self.vals,
            "used": self.used,
            "count": self.count,
            "name_to_idx": self.name_to_idx,
            "capacity": self.capacity,
            "d": self.d,
        }
        # Old ConceptBank implementations require top-level keys and cannot
        # safely load this wrapper while silently dropping its holdout policy.
        payload = {"guarded_concept_schema": 1, "guarded_state": state} if self.heldout_guard is not None else state
        torch.save(payload, path)

    def load(self, path: str):
        state = self.preflight_load(path)
        saved = state.get("holdout_guard")
        guard = self.heldout_guard or (HoldoutGuard.from_state(saved) if saved is not None else None)
        self.keys = state["keys"].to(self.keys.device)
        self.vals = state["vals"].to(self.vals.device)
        self.used = state["used"].to(self.used.device)
        self.count = state["count"].to(self.count.device)
        self.name_to_idx = state["name_to_idx"]
        self.slot_provenance = {int(k): v for k, v in state.get("slot_provenance", {}).items()}
        self.heldout_guard = guard
        self.capacity = int(state.get("capacity", self.capacity))
        self.d = int(state.get("d", self.d))
        return self
