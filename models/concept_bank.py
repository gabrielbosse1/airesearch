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


class ConceptBank(nn.Module):
    def __init__(self, capacity: int = 256, d: int = 64):
        super().__init__()
        self.capacity = capacity
        self.d = d
        # Memory entries (buffers: persisted, not optimized).
        self.register_buffer("keys", torch.zeros(capacity, d))
        self.register_buffer("vals", torch.zeros(capacity, d))
        self.register_buffer("used", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.name_to_idx = {}  # concept name -> slot index

    # ------------------------------------------------------------------ #
    # Write: store / update a concept (no gradient, no retraining)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def write(self, name: str, key: torch.Tensor, val: torch.Tensor) -> int:
        """Store concept ``name`` with key/value vectors.

        Returns the slot index. If the name already exists, its slot is
        overwritten (exponential moving average merge with the old value so
        repeated writes reinforce rather than destroy).
        """
        key = key.detach().reshape(-1).float()
        val = val.detach().reshape(-1).float()
        if key.shape[0] != self.d or val.shape[0] != self.d:
            raise ValueError(f"key/val must have dim {self.d}, got "
                             f"{key.shape[0]}/{val.shape[0]}")

        if name in self.name_to_idx:
            idx = self.name_to_idx[name]
            # Reinforce existing concept with an EMA merge.
            old_v = self.vals[idx]
            self.vals[idx] = 0.5 * old_v + 0.5 * val
            self.keys[idx] = key  # refresh key
            self.used[idx] = True
            return idx

        # Allocate a free slot; overwrite oldest if full (simple FIFO).
        if int(self.count.item()) < self.capacity:
            free = (~self.used).nonzero(as_tuple=False)
            if free.shape[0] > 0:
                idx = int(free[0])
            else:
                idx = int(self.count.item())
        else:
            idx = int((self.count.item() - 1) % self.capacity)

        self.keys[idx] = key
        self.vals[idx] = val
        self.used[idx] = True
        self.name_to_idx[name] = idx
        if not bool(self.used[idx]):
            self.count += 1
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
        torch.save({
            "keys": self.keys,
            "vals": self.vals,
            "used": self.used,
            "count": self.count,
            "name_to_idx": self.name_to_idx,
            "capacity": self.capacity,
            "d": self.d,
        }, path)

    def load(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.keys = state["keys"].to(self.keys.device)
        self.vals = state["vals"].to(self.vals.device)
        self.used = state["used"].to(self.used.device)
        self.count = state["count"].to(self.count.device)
        self.name_to_idx = state["name_to_idx"]
        self.capacity = int(state.get("capacity", self.capacity))
        self.d = int(state.get("d", self.d))
        return self
