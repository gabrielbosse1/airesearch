"""MemoryModel: persistent long-term memory that lives *outside* the context window.

Design goals (from the user's request):
  * General memory of things even outside the context window.
  * Remember people: names, what they did, what was done to them.
  * Multitask in memory: aware that a conversation started in *text* and is
    now continuing in *voice* (cross-modal session continuity).
  * Detect when a user is lying / contradicting previously stored facts.
  * Learn new skills from several examples (external examples adjust memory
    via the diffusion system, no full retraining).
  * Serve as the "teacher" context provider for a co-trained generator.

The memory is a differentiable key-value bank. Entries can be written in two
ways:
  1. Fast/direct write  -- store a new fact/person/skill immediately (no
     gradient step). Used for "I just met Alice" or "here is a new example."
  2. Slow consolidation -- gradient-based reinforcement of frequently
     accessed slots. Used to turn repeated examples into stable knowledge.

All vectors live in a shared d_model space so the MemoryModel and the
GeneratorModel can be co-trained.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModel(nn.Module):
    """Long-term episodic + semantic memory with person/entity tracking."""

    def __init__(self, d_model: int = 64, capacity: int = 512, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.n_heads = n_heads

        # ------------------------------------------------------------------
        # Differentiable key-value memory bank
        # ------------------------------------------------------------------
        self.register_buffer("keys", torch.zeros(capacity, d_model))
        self.register_buffer("values", torch.zeros(capacity, d_model))
        self.register_buffer("used", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("access_counts", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("last_access", torch.zeros(capacity, dtype=torch.long))

        # Metadata per slot (plain Python list, indexed by slot)
        self.metadata = []

        # ------------------------------------------------------------------
        # Encoders (shared d_model space with the generator)
        # ------------------------------------------------------------------
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

        # Retrieved context + current input -> fused context vector
        self.context_proj = nn.Linear(d_model * 2, d_model)

        # Contradiction / lie detector: (stored_fact, new_statement) -> consistency [0,1]
        self.contradiction_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Person/entity linker: given a query, predict which person slot it refers to
        self.person_linker = nn.Linear(d_model, capacity)

        self.apply(self._init_weights)

        # Session clock (monotonic, for "what happened before")
        self.session_tick = 0

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # Differentiable encode: content -> (key, value). Used by co-training so
    # gradients can flow through the memory encoders. Buffers are NOT touched
    # here (those are updated by the fast no_grad write for persistence).
    # ------------------------------------------------------------------ #
    def encode(self, content_emb: torch.Tensor):
        """Differentiably encode content into (key, value) pairs.

        content_emb: (B, d_model). Returns keys, values each (B, d_model).
        """
        key = self.key_encoder(content_emb)   # (B, d)
        value = self.value_encoder(content_emb)  # (B, d)
        return key, value

    # ------------------------------------------------------------------ #
    # Fast write: store a new fact/person/skill immediately (no gradients)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def write(
        self,
        content_emb: torch.Tensor,
        metadata: dict,
    ) -> int:
        """Store one memory entry.

        Parameters
        ----------
        content_emb : (d_model,) or (1, d_model) -- the memory content vector.
        metadata : dict -- {person, modality, session, timestamp, type, ...}

        Returns the slot index.
        """
        content_emb = content_emb.detach().float().reshape(-1)
        if content_emb.shape[0] != self.d_model:
            # Project any size into d_model deterministically.
            proj = nn.Linear(content_emb.shape[0], self.d_model, bias=False)
            with torch.no_grad():
                nn.init.orthogonal_(proj.weight)
            content_emb = proj(content_emb.unsqueeze(0)).squeeze(0)

        key = self.key_encoder(content_emb.unsqueeze(0))[0]
        value = self.value_encoder(content_emb.unsqueeze(0))[0]

        # Allocate slot: free slot first, else LRU replacement.
        if int(self.count.item()) < self.capacity:
            free = (~self.used).nonzero(as_tuple=False)
            idx = int(free[0]) if free.shape[0] > 0 else int(self.count.item())
        else:
            idx = int(torch.argmin(self.access_counts[self.used]).item())
            # map back to global index
            used_idx = self.used.nonzero(as_tuple=False).squeeze(1)
            idx = int(used_idx[idx])

        self.keys[idx] = key
        self.values[idx] = value
        self.used[idx] = True
        while len(self.metadata) <= idx:
            self.metadata.append({})
        self.metadata[idx] = dict(metadata)
        self.metadata[idx]["written_at"] = self.session_tick
        self.access_counts[idx] = 0
        self.last_access[idx] = self.session_tick
        self.session_tick += 1
        if not bool(self.used[idx]):
            self.count += 1
        return idx

    # ------------------------------------------------------------------ #
    # Content-addressable retrieval
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def retrieve(
        self,
        query_emb: torch.Tensor,
        top_k: int = 5,
        modality_filter: str | None = None,
        person_filter: str | None = None,
    ):
        """Retrieve the top-k most relevant memory entries for a query.

        Returns (entries, context_vector) where context_vector is the
        attention-weighted sum of retrieved values (d_model,).
        """
        q = self.query_encoder(query_emb.detach().float().reshape(1, -1))  # (1, d)
        used = self.used
        if not bool(used.any()):
            return [], torch.zeros(self.d_model, device=q.device)

        used_keys = self.keys[used]           # (M, d)
        used_vals = self.values[used]         # (M, d)
        used_idx = used.nonzero(as_tuple=False).squeeze(1)

        logits = (q @ used_keys.T) / math.sqrt(self.d_model)  # (1, M)
        logits = logits.squeeze(0)            # (M,)

        # Apply modality / person filters by masking out non-matching slots.
        mask = torch.ones_like(logits, dtype=torch.bool)
        for i, gi in enumerate(used_idx):
            md = self.metadata[int(gi)]
            if modality_filter is not None and md.get("modality") != modality_filter:
                mask[i] = False
            if person_filter is not None and md.get("person") != person_filter:
                mask[i] = False
        logits = logits.masked_fill(~mask, float("-inf"))

        attn = F.softmax(logits, dim=-1)
        k = min(top_k, int(used.sum().item()))
        top_vals, top_pos = torch.topk(attn, k=k)

        entries = []
        for pos, w in zip(top_pos.tolist(), top_vals.tolist()):
            gi = int(used_idx[pos])
            entries.append(
                {
                    "slot": gi,
                    "weight": float(w),
                    "metadata": self.metadata[gi],
                    "value": self.values[gi].clone(),
                }
            )
            # Mark access (for consolidation).
            self.access_counts[gi] += 1
            self.last_access[gi] = self.session_tick

        # Context vector = weighted sum of retrieved values.
        context = (used_vals * attn.unsqueeze(-1)).sum(dim=0)
        return entries, context

    # ------------------------------------------------------------------ #
    # Contradiction / lie detection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def check_contradiction(
        self,
        new_statement_emb: torch.Tensor,
        person: str | None = None,
        threshold: float = 0.5,
    ):
        """Check whether ``new_statement_emb`` contradicts stored facts.

        Returns (contradiction_score, conflicting_entries). score near 1 =
        likely consistent; near 0 = likely contradiction/lie.
        """
        entries, _ = self.retrieve(
            new_statement_emb, top_k=8, person_filter=person
        )
        if not entries:
            return 1.0, []  # nothing stored yet -> no contradiction

        new_vec = new_statement_emb.detach().float().reshape(1, -1)
        scores = []
        conflicts = []
        for e in entries:
            stored = e["value"].reshape(1, -1)
            pair = torch.cat([stored, new_vec], dim=-1)
            consist = self.contradiction_scorer(pair).item()
            scores.append(consist)
            if consist < threshold:
                conflicts.append((e, consist))

        overall = sum(scores) / len(scores) if scores else 1.0
        return overall, conflicts

    # ------------------------------------------------------------------ #
    # Slow consolidation: gradient-based reinforcement of hot slots
    # ------------------------------------------------------------------ #
    def consolidate(self, lr: float = 0.01):
        """Reinforce frequently-accessed memory slots via a small gradient
        step that pulls their keys/values toward their access-weighted
        centroid. Call occasionally (not every step)."""
        if not bool(self.used.any()):
            return
        used_idx = self.used.nonzero(as_tuple=False).squeeze(1)
        counts = self.access_counts[used_idx].float()
        total = counts.sum()
        if total == 0:
            return
        weights = counts / total
        centroid = (self.values[used_idx] * weights.unsqueeze(-1)).sum(dim=0)
        # Gentle pull of hot values toward centroid (stabilizes knowledge).
        with torch.no_grad():
            for gi, w in zip(used_idx.tolist(), weights.tolist()):
                gi = int(gi)
                self.values[gi] = self.values[gi] + lr * w * (
                    centroid - self.values[gi]
                )
        # Re-encode keys to match updated values.
        with torch.no_grad():
            for gi in used_idx.tolist():
                self.keys[gi] = self.key_encoder(self.values[gi].unsqueeze(0))[0]

    # ------------------------------------------------------------------ #
    # Person/entity memory helper
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def person_summary(self, person: str) -> dict:
        """Gather everything memory knows about a person."""
        entries, _ = self.retrieve(
            torch.zeros(self.d_model), top_k=self.capacity, person_filter=person
        )
        actions = []
        facts = []
        for e in entries:
            md = e["metadata"]
            t = md.get("type", "fact")
            if t in ("action", "did", "done_to"):
                actions.append(md)
            else:
                facts.append(md)
        return {
            "person": person,
            "n_memories": len(entries),
            "facts": facts,
            "actions": actions,
            "modalities_seen": sorted({md.get("modality") for md in facts + actions}),
        }

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "keys": self.keys,
                "values": self.values,
                "used": self.used,
                "count": self.count,
                "access_counts": self.access_counts,
                "last_access": self.last_access,
                "metadata": self.metadata,
                "session_tick": self.session_tick,
                "key_encoder": self.key_encoder.state_dict(),
                "value_encoder": self.value_encoder.state_dict(),
                "query_encoder": self.query_encoder.state_dict(),
                "context_proj": self.context_proj.state_dict(),
                "contradiction_scorer": self.contradiction_scorer.state_dict(),
                "person_linker": self.person_linker.state_dict(),
                "d_model": self.d_model,
                "capacity": self.capacity,
            },
            path,
        )

    def load(self, path: str, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        self.keys = state["keys"].to(self.keys.device)
        self.values = state["values"].to(self.values.device)
        self.used = state["used"].to(self.used.device)
        self.count = state["count"].to(self.count.device)
        self.access_counts = state["access_counts"].to(self.access_counts.device)
        self.last_access = state["last_access"].to(self.last_access.device)
        self.metadata = state["metadata"]
        self.session_tick = state.get("session_tick", 0)
        self.key_encoder.load_state_dict(state["key_encoder"])
        self.value_encoder.load_state_dict(state["value_encoder"])
        self.query_encoder.load_state_dict(state["query_encoder"])
        self.context_proj.load_state_dict(state["context_proj"])
        self.contradiction_scorer.load_state_dict(state["contradiction_scorer"])
        self.person_linker.load_state_dict(state["person_linker"])
        self.d_model = int(state.get("d_model", self.d_model))
        self.capacity = int(state.get("capacity", self.capacity))
        return self
