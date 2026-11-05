"""ByteCore v3 foundation: causal byte Transformer + Gated Skill Adapter.

Phase 24. A decoder-only Transformer over raw bytes (no tokenizer) whose
1024-dim core stream reads an 8192-dim skill/memory space through a gated
bridge in EVERY layer:

    hidden (B, T, 1024)
        -> query  W_q: 1024 -> 1024          (high-fidelity key matching)
        -> scores against Memory Keys (N, 1024)
        -> softmax retrieval of Skill Values (N, 8192)   -> (B, T, 8192)
        -> GLU    W_g: 8192 -> 2048, split into two 1024 halves
                  out = Information * sigmoid(Filter)    -> (B, T, 1024)
        -> hidden = hidden + scale * out    (residual injection, all layers)

The GLU is the "V replacement": a plain Linear(8192 -> 1024) would blend every
skill dimension into every output channel. The multiplicative sigmoid gate
instead lets each channel veto or pass a concept, keeping retrieval sharp.
``scale`` is initialised to zero, so a freshly initialised model is exactly
the base Transformer and the bridge opens gradually during training.

Vocabulary: bytes 0-255 plus nine control ids (VOCAB_SIZE = 265).

=================  =========================================================
``BOS`` 256        beginning of sequence
``EOS`` 257        end of sequence (the only learned stop)
``PAD`` 258        padding, masked out of the loss
``TOOL_SEARCH`` 259      opens a search query
``TOOL_SEARCH_END`` 260  closes it; the host executes a sandboxed search
``OBS_START`` 261        opens a host-injected observation
``OBS_END`` 262          closes it
``EDIT`` 263             ordinary learned marker -- NO host semantics
``EDIT_END`` 264         ordinary learned marker -- NO host semantics
=================  =========================================================

Tooling contract (Phase 24): detection is by TOKEN ID, never by regex over
decoded text. ``[TOOL_SEARCH]`` queries are executed by a sandboxed callback
(see ``training/train.py``); ``[EDIT]`` is standard vocabulary trained by the
same cross-entropy as every other byte. There are no host-side text hacks.

The adaptive "think" loop: ``IterativeRefinement`` runs up to
``refinement_steps`` weight-tied passes. A ``Linear(d_model, 1) -> sigmoid``
confidence head reads each intermediate state; at inference the loop breaks
as soon as confidence exceeds ``confidence_threshold`` (default 0.9). During
training all passes run and an auxiliary BCE loss teaches the head to fire on
low-entropy (decided) states.

Phase 24.4 adds the LATENT THOUGHT MATRIX: a ``LatentThoughtBuffer`` of shape
``(B, max_thought_steps, d_model)`` inside the loop. Bidirectional
cross-attention lets the scratchpad read the context and the context
frontier read the scratchpad; the confidence head evaluates the combined
state. The buffer is purely latent -- it NEVER passes through the byte head
(``head`` only ever sees refined context hidden states), and the write-back
touches only the last position so no future byte leaks through the buffer.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

BOS = 256
EOS = 257
PAD = 258
TOOL_SEARCH = 259
TOOL_SEARCH_END = 260
OBS_START = 261
OBS_END = 262
EDIT = 263
EDIT_END = 264
VOCAB_SIZE = 265
CONTROL_NAMES = {BOS: "<bos>", EOS: "<eos>", PAD: "<pad>",
                 TOOL_SEARCH: "[TOOL_SEARCH]", TOOL_SEARCH_END: "[/TOOL_SEARCH]",
                 OBS_START: "[OBS]", OBS_END: "[/OBS]",
                 EDIT: "[EDIT]", EDIT_END: "[/EDIT]"}


@dataclass(frozen=True)
class FoundationConfig:
    """Every knob that changes the model. Nothing else is architecture."""

    vocab_size: int = VOCAB_SIZE
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    max_seq_len: int = 256
    dropout: float = 0.1
    norm_type: str = "rmsnorm"          # "layernorm" | "rmsnorm"
    ffn_type: str = "swiglu"            # "gelu" | "swiglu"
    position_type: str = "rope"         # "learned" | "rope"
    n_kv_heads: int = 0                 # 0 => MHA; >0 => GQA
    refinement_steps: int = 0           # max weight-tied "think" passes (4 prod)
    refinement_rank: int = 256          # low-rank bottleneck inside the loop
    max_thought_steps: int = 4          # latent scratchpad slots (0 disables; 4 prod)
    confidence_threshold: float = 0.9   # early-break threshold at inference
    memory_dim: int = 0                 # 0 disables the adapter; 8192 in prod
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"vocab_size must be {VOCAB_SIZE} "
                             "(256 bytes + 9 control ids)")
        for name in ("d_model", "n_layers", "n_heads", "d_ff", "max_seq_len"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")
        if self.ffn_type not in ("gelu", "swiglu"):
            raise ValueError("ffn_type must be 'gelu' or 'swiglu'")
        if self.position_type not in ("learned", "rope"):
            raise ValueError("position_type must be 'learned' or 'rope'")
        if not isinstance(self.n_kv_heads, int) or self.n_kv_heads < 0:
            raise ValueError("n_kv_heads must be a non-negative integer")
        if self.n_kv_heads and (self.n_kv_heads > self.n_heads or self.n_heads % self.n_kv_heads):
            raise ValueError("n_kv_heads must divide n_heads")
        if not isinstance(self.refinement_steps, int) or self.refinement_steps < 0:
            raise ValueError("refinement_steps must be a non-negative integer")
        if not isinstance(self.refinement_rank, int) or self.refinement_rank < 1:
            raise ValueError("refinement_rank must be a positive integer")
        if not isinstance(self.max_thought_steps, int) or self.max_thought_steps < 0:
            raise ValueError("max_thought_steps must be a non-negative integer (0 disables)")
        if not 0.0 < float(self.confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not isinstance(self.memory_dim, int) or self.memory_dim < 0:
            raise ValueError("memory_dim must be a non-negative integer (0 disables)")
        if float(self.rope_theta) <= 0 or float(self.norm_eps) <= 0:
            raise ValueError("rope_theta and norm_eps must be positive")
        if self.position_type == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FoundationConfig":
        """Build from a YAML section; unknown keys are rejected, not ignored."""
        if mapping is None:
            return cls()
        known = set(cls().__dict__)
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown foundation config keys: {unknown}")
        return cls(**{key: mapping[key] for key in mapping})

    def scaled(self, **overrides: Any) -> "FoundationConfig":
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Wide-model components
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Pre-norm RMS normalisation; the reduction runs in float32 even under
    fp16/bf16 autocast (squaring fp16 activations at width 1024 can overflow)."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        values = hidden.float()
        values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + self.eps)
        return (values * self.weight.float()).to(dtype)


class SwiGLUFFN(nn.Module):
    """``[SiLU(x W_value) * (x W_branch)] W_out`` -- bias-free, 3 projections."""

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.w_value = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_branch = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_out = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        activated = F.silu(self.w_value(hidden)) * self.w_branch(hidden)
        return self.dropout(self.w_out(activated))


def rotary_tables(length: int, head_dim: int, theta: float, device: torch.device,
                  dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inverse = 1.0 / (float(theta) ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    positions = torch.arange(length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse)
    doubled = torch.cat((angles, angles), dim=-1)[None, None, :, :]
    return doubled.cos().to(dtype), doubled.sin().to(dtype)


def rotate_half(hidden: torch.Tensor) -> torch.Tensor:
    left, right = hidden.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rotary(hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return hidden * cos + rotate_half(hidden) * sin


def repeat_kv(hidden: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden
    batch, heads, length, width = hidden.shape
    return (hidden[:, :, None, :, :]
            .expand(batch, heads, repeats, length, width)
            .reshape(batch, heads * repeats, length, width))


def make_norm(config: FoundationConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model, config.norm_eps)
    return nn.LayerNorm(config.d_model, eps=config.norm_eps)


# ---------------------------------------------------------------------------
# The Gated Skill Adapter: 1024 <-> 8192 bridge (Phase 24 core)
# ---------------------------------------------------------------------------
class GatedSkillAdapter(nn.Module):
    """High-fidelity bridge between the core stream and the skill bank.

    Shapes (production: ``d_model=1024``, ``memory_dim=8192``)::

        hidden   (B, T, 1024)
        query    W_q(hidden)                       (B, T, 1024)
        keys     (N, 1024) or (B, N, 1024)
        scores   q @ keys^T / sqrt(1024)           (B, T, N)
        weights  softmax(scores, masked slots)     (B, T, N)
        retrieved weights @ values (N, 8192)       (B, T, 8192)
        glu      W_g: 8192 -> 2048                 (B, T, 2048)
        split    Information | Filter              (B, T, 1024) each
        output   Information * sigmoid(Filter)     (B, T, 1024)

    ``scale`` (a learned scalar, zero-initialised) multiplies the output so
    injection is the identity at initialisation. The gate is the sharp
    bouncer: sigmoid(Filter) ~ 0 vetoes a channel, ~ 1 passes it intact --
    no dense blending of 8192 concepts into mush.
    """

    def __init__(self, d_model: int = 1024, memory_dim: int = 8192) -> None:
        super().__init__()
        if d_model < 1 or memory_dim < 1:
            raise ValueError("d_model and memory_dim must be positive")
        self.d_model = int(d_model)
        self.memory_dim = int(memory_dim)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.glu = nn.Linear(memory_dim, 2 * d_model, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))
        self._query_scale = 1.0 / math.sqrt(d_model)

    def retrieve(self, hidden: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        """Softmax retrieval of 8192-dim skill values. Differentiable."""
        query = self.query(hidden)
        if keys.ndim == 2:
            scores = (query @ keys.transpose(0, 1)) * self._query_scale
        else:
            scores = torch.einsum("btd,bnd->btn", query, keys) * self._query_scale
        if mask is not None:
            mask_b = mask.unsqueeze(1) if mask.ndim == 2 else mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~mask_b, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            # Rows with NO active slot must retrieve exactly zero, not NaN.
            any_active = mask_b.any(dim=-1, keepdim=True)
            weights = torch.where(any_active, weights, torch.zeros_like(weights))
        else:
            weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        if values.ndim == 2:
            return weights @ values
        return torch.einsum("btn,bnm->btm", weights, values)

    def glu_gate(self, retrieved: torch.Tensor) -> torch.Tensor:
        """The GLU: project memory_dim -> 2*d_model, split, gate, multiply."""
        if retrieved.shape[-1] != self.memory_dim:
            raise ValueError(f"expected skill vectors of width {self.memory_dim}, "
                             f"got {retrieved.shape[-1]}")
        information, filter_half = self.glu(retrieved).chunk(2, dim=-1)
        return information * torch.sigmoid(filter_half)

    def forward(self, hidden: torch.Tensor, memory: Any | None = None) -> torch.Tensor:
        """Return the delta to add to the residual stream (zero when empty)."""
        if memory is None or memory.empty:
            return torch.zeros_like(hidden)
        retrieved = self.retrieve(hidden, memory.keys, memory.values, memory.mask)
        return self.scale * self.glu_gate(retrieved)


# ---------------------------------------------------------------------------
# Iterative refinement with the model-decided adaptive loop
# ---------------------------------------------------------------------------
class LatentThoughtBuffer(nn.Module):
    """Isolated latent scratchpad for the adaptive loop (Phase 24.4).

    A dedicated ``(B, max_thought_steps, d_model)`` workspace where the model
    "thinks" without emitting bytes and without polluting the context window:

    * **Buffer processes** -- at loop step ``t``, slot ``t`` is updated from
      slot ``t-1`` and the context:
      ``state_t = state_(t-1) + g_buf * CrossAttn(state_(t-1) + slot_t, h)``
      (Q = one thought slot, K/V = the full context ``(B, T, D)``).
    * **Context thinks** -- the context reads the written slots back through
      a second cross-attention (Q = context, K/V = slots ``0..t``).

    **Causality contract (critical).** The buffer is a GLOBAL summary of the
    whole ``(B, T)`` window, so writing it back into every position would
    leak future bytes into earlier predictions through the scratchpad. Only
    the FRONTIER position (the last one, whose next-byte prediction may
    depend on the entire window) reads the buffer back; positions ``< T-1``
    are untouched. ``tests/test_foundation.py`` asserts this byte-for-byte.

    **Isolation contract.** The buffer never reaches the byte head: the
    transformer passes only the refined context hidden states to
    ``head(norm_final(h))``. The scratchpad influences bytes exclusively
    through the gated frontier cross-attention above.

    **Step-0 stability (asymmetric ReZero).** ``context_gate`` (write side)
    initialised to 0 and slot embeddings to zero: a fresh model's logits and
    confidence are EXACTLY the pre-24.4 architecture's. ``buffer_gate`` (read
    side) initialised to 1, NOT 0: with both gates at zero the system
    deadlocks -- zero slots make the write-back attention output exactly
    zero, so ``context_gate`` gets zero gradient, while ``buffer_gate``'s
    consumers are all gated off (observed in the smoke probe: norms stayed
    0.0 through training). With the read side open, slots are non-zero from
    the first forward, ``dL/d(context_gate) = <dL/dh, inject> != 0``, and the
    write side opens smoothly from step 1 while the byte pathway starts
    unperturbed.

    Attention cost: scores are ``(B, heads, S, T)`` / ``(B, heads, 1, S)`` --
    linear in T, ~0.4% of one T x T self-attention map at production shapes.
    """

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.capacity = int(min(config.max_thought_steps, config.refinement_steps))
        if self.capacity < 1:
            raise ValueError("LatentThoughtBuffer needs at least one thought slot")
        d_model = config.d_model
        self.slot_embeddings = nn.Parameter(torch.zeros(1, self.capacity, d_model))
        self.read_context = nn.MultiheadAttention(d_model, config.n_heads,
                                                  dropout=config.dropout, batch_first=True)
        self.write_back = nn.MultiheadAttention(d_model, config.n_heads,
                                                dropout=config.dropout, batch_first=True)
        self.norm_thought_q = make_norm(config)
        self.norm_context_kv = make_norm(config)
        self.norm_context_q = make_norm(config)
        self.norm_thought_kv = make_norm(config)
        # Asymmetric ReZero: read side OPEN (buffer thinks from step 0), write
        # side CLOSED (bytes unperturbed at init). Double-zero would deadlock:
        # zero slots -> zero inject -> zero gradient on BOTH gates forever.
        self.buffer_gate = nn.Parameter(torch.ones(()))
        self.context_gate = nn.Parameter(torch.zeros(()))
        # Telemetry (detached): the last forward's buffer, for tests/smoke.
        self.last_buffer: torch.Tensor | None = None
        self.last_norm: float = 0.0
        self.last_steps: int = 0
        self._slots: list[torch.Tensor] = []
        self._state: torch.Tensor | None = None

    def begin(self) -> None:
        """Reset the per-forward recurrence (each forward thinks anew)."""
        self._slots = []
        self._state = None

    def step(self, hidden: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One loop step: update slot ``index``, inject into the frontier.

        Returns ``(hidden, pooled_thought)`` where ``pooled_thought`` (B, D)
        is the mean of the written slots -- the combined-state signal the
        confidence head reads.
        """
        slot = min(index, self.capacity - 1)   # loop longer than capacity: roll last slot
        if self._state is None:
            prev = hidden.new_zeros(hidden.shape[0], hidden.shape[2])
        else:
            prev = self._state
        query = (prev + self.slot_embeddings[:, slot]).unsqueeze(1)      # (B, 1, D)
        context = self.norm_context_kv(hidden)                           # (B, T, D)
        delta, _ = self.read_context(self.norm_thought_q(query), context, context,
                                     need_weights=False)
        state = prev + self.buffer_gate * delta.squeeze(1)               # (B, D)
        self._state = state
        self._slots.append(state)
        # Stack (never in-place): earlier slots stay version-stable for autograd.
        written = torch.stack(self._slots, dim=1)                        # (B, t+1, D)
        frontier = hidden[:, -1:, :]                                     # (B, 1, D)
        kv = self.norm_thought_kv(written)
        inject, _ = self.write_back(self.norm_context_q(frontier), kv, kv,
                                    need_weights=False)
        hidden = torch.cat([hidden[:, :-1], frontier + self.context_gate * inject], dim=1)
        return hidden, written.mean(dim=1)

    def finalize(self) -> torch.Tensor:
        """Expose the full ``(B, capacity, D)`` buffer (unwritten slots keep
        their learned init) as detached telemetry."""
        written = torch.stack(self._slots, dim=1)
        if written.shape[1] < self.capacity:
            rest = self.slot_embeddings[:, written.shape[1]:].expand(written.shape[0], -1, -1)
            written = torch.cat([written, rest], dim=1)
        self.last_buffer = written.detach()
        self.last_norm = float(written.detach().float().norm().item())
        self.last_steps = len(self._slots)
        return self.last_buffer


class IterativeRefinement(nn.Module):
    """Weight-tied low-rank "think" loop with a confidence-based early exit.

    ``h_(t+1) = h_t + alpha_t * Cell(RMSNorm(h_t))``, ``alpha`` zero-init.
    The confidence head ``Linear(d_model, 1) -> sigmoid`` reads each state.

    Phase 24.4: a ``LatentThoughtBuffer`` scratchpad runs inside the loop.
    Each pass (1) updates thought slot ``t`` from slot ``t-1`` + context,
    (2) lets the context frontier read the written slots back (bidirectional
    cross-attention, ReZero-gated), then (3) applies the weight-tied cell.
    The confidence head evaluates the COMBINED state: its logit gains a
    zero-init ``thought_confidence`` term read from the pooled scratchpad, so
    an uncertain buffer keeps the loop running and a converged one lets it
    break. The buffer itself never reaches the byte head.

    * Training: ALL ``steps`` passes run; intermediate states are returned so
      the trainer can build the auxiliary BCE target (low byte-entropy).
    * Inference (``adaptive=True``): break as soon as every position's
      confidence exceeds ``threshold`` (default 0.9), at most ``steps`` passes.
    """

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.steps = int(config.refinement_steps)
        self.threshold = float(config.confidence_threshold)
        rank = min(config.refinement_rank, config.d_model)
        self.norm = make_norm(config)
        self.down = nn.Linear(config.d_model, rank, bias=False)
        self.branch = nn.Linear(config.d_model, rank, bias=False)
        self.up = nn.Linear(rank, config.d_model, bias=False)
        self.step_scales = nn.Parameter(torch.zeros(self.steps))
        self.confidence_head = nn.Linear(config.d_model, 1)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)
        # Phase 24.4 latent scratchpad (max_thought_steps=0 keeps the legacy
        # architecture exactly: no extra parameters, no extra compute).
        if int(config.max_thought_steps) > 0:
            self.thoughts = LatentThoughtBuffer(config)
            self.thought_norm = make_norm(config)
            self.thought_confidence = nn.Linear(config.d_model, 1)
            nn.init.zeros_(self.thought_confidence.weight)
            nn.init.zeros_(self.thought_confidence.bias)
        else:
            self.thoughts = None

    def cell(self, hidden: torch.Tensor) -> torch.Tensor:
        values = self.norm(hidden)
        return self.up(F.silu(self.down(values)) * self.branch(values))

    def confidence(self, hidden: torch.Tensor,
                   thought: torch.Tensor | None = None) -> torch.Tensor:
        """Per-position confidence in (0, 1), shape ``(B, T)``.

        ``thought`` (B, D) is the pooled scratchpad state: the head evaluates
        the COMBINED context + thought state. The thought term is zero-init,
        so confidence is neutral with respect to the buffer at step 0.
        """
        logit = self.confidence_head(self.norm(hidden)).squeeze(-1)
        if thought is not None and self.thoughts is not None:
            logit = logit + self.thought_confidence(self.thought_norm(thought))
        return torch.sigmoid(logit)

    def forward(self, hidden: torch.Tensor, *, use_checkpoint: bool = False,
                adaptive: bool = False, threshold: float | None = None,
                collect_states: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        threshold = self.threshold if threshold is None else float(threshold)
        states: list[torch.Tensor] | None = [] if collect_states else None
        steps_run = 0
        confidence_value: float | None = None
        thought_pools: list[torch.Tensor] = []
        if self.thoughts is not None:
            self.thoughts.begin()
        for index in range(self.steps):
            # Phase 24.4: the scratchpad reads the problem, the context
            # frontier reads the scratchpad (both ReZero-gated, causal).
            if self.thoughts is not None:
                hidden, pool = self.thoughts.step(hidden, index)
                thought_pools.append(pool)
            alpha = self.step_scales[index]
            if use_checkpoint and self.training and torch.is_grad_enabled():
                delta = checkpoint(self.cell, hidden, use_reentrant=False)
            else:
                delta = self.cell(hidden)
            hidden = hidden + alpha * delta
            steps_run += 1
            if states is not None:
                states.append(hidden)
            if adaptive and not self.training:
                conf = self.confidence(hidden, thought_pools[-1] if thought_pools else None)
                confidence_value = float(conf.min().item())
                if confidence_value > threshold:
                    break
        if self.thoughts is not None:
            self.thoughts.finalize()
        return hidden, {"steps_run": steps_run, "confidence": confidence_value,
                        "states": states,
                        "thought_states": thought_pools if thought_pools else None,
                        "thought_steps": self.thoughts.last_steps if self.thoughts is not None else 0,
                        "thought_norm": self.thoughts.last_norm if self.thoughts is not None else 0.0}


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with optional GQA and RoPE, via fused SDPA."""

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = config.d_model // self.n_heads
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim)
        self.value = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout_p = float(config.dropout)
        self.position_type = config.position_type
        self.rope_theta = float(config.rope_theta)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = hidden.shape
        q = self.query(hidden).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.position_type == "rope":
            cos, sin = rotary_tables(length, self.head_dim, self.rope_theta,
                                     hidden.device, q.dtype)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        dropout = self.dropout_p if self.training else 0.0
        if self.repeats > 1:
            try:
                context = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=dropout, enable_gqa=True)
            except TypeError:  # torch < 2.5 has no enable_gqa
                context = F.scaled_dot_product_attention(
                    q, repeat_kv(k, self.repeats), repeat_kv(v, self.repeats),
                    is_causal=True, dropout_p=dropout)
        else:
            context = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.n_heads * self.head_dim)
        return self.output(context)


class TransformerBlock(nn.Module):
    """Pre-norm block: attention, FFN, then the gated skill injection.

    ``hidden = hidden + GatedSkillAdapter(norm(hidden), memory)`` runs in
    EVERY layer when ``config.memory_dim > 0`` (all 8 in production).
    """

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.norm_1 = make_norm(config)
        self.attention = CausalSelfAttention(config)
        self.norm_2 = make_norm(config)
        if config.ffn_type == "swiglu":
            self.feed_forward = SwiGLUFFN(config)
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(config.dropout),
            )
        self.has_skills = config.memory_dim > 0
        if self.has_skills:
            self.skill_norm = make_norm(config)
            self.skill_adapter = GatedSkillAdapter(config.d_model, config.memory_dim)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None,
                memory: Any | None = None) -> torch.Tensor:
        hidden = hidden + self.attention(self.norm_1(hidden), mask)
        hidden = hidden + self.feed_forward(self.norm_2(hidden))
        if self.has_skills:
            hidden = hidden + self.skill_adapter(self.skill_norm(hidden), memory)
        return hidden


class CausalByteTransformer(nn.Module):
    """Next-byte prediction over raw bytes, conditioned on skill matrices."""

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        config = config if isinstance(config, FoundationConfig) else FoundationConfig(**dict(config))
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (nn.Embedding(config.max_seq_len, config.d_model)
                                   if config.position_type == "learned" else None)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.refinement = (IterativeRefinement(config)
                           if config.refinement_steps > 0 else None)
        self.norm_final = make_norm(config)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        self.apply(self._initialize)
        # Re-zero the confidence head after the blanket init: neutral 0.5.
        if self.refinement is not None:
            nn.init.zeros_(self.refinement.confidence_head.weight)
            nn.init.zeros_(self.refinement.confidence_head.bias)
            # The thought-confidence term must also stay neutral at step 0.
            if self.refinement.thoughts is not None:
                nn.init.zeros_(self.refinement.thought_confidence.weight)
                nn.init.zeros_(self.refinement.thought_confidence.bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings only -- used to build TEXT skill matrices."""
        return self.token_embedding(ids)

    def forward(self, ids: torch.Tensor, memory: Any | None = None, *,
                adaptive: bool | None = None, collect_states: bool = False,
                return_info: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Next-byte logits ``(B, T, vocab)``; ``memory`` is a SkillBankView.

        ``adaptive=None`` means: adaptive early-exit at eval, full loop in
        training. ``return_info`` adds refinement steps/confidence/states.
        """
        if ids.ndim != 2:
            raise ValueError("ids must have shape (batch, length)")
        if int(ids.max()) >= self.config.vocab_size or int(ids.min()) < 0:
            raise ValueError("ids must be within the vocabulary")
        length = ids.shape[1]
        if length > self.config.max_seq_len:
            raise ValueError(f"sequence length {length} exceeds max_seq_len {self.config.max_seq_len}")
        hidden = self.embed(ids)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(torch.arange(length, device=ids.device))
        hidden = self.embedding_dropout(hidden)
        use_checkpoint = (self.gradient_checkpointing and self.training
                          and torch.is_grad_enabled())
        for block in self.blocks:
            if use_checkpoint:
                hidden = checkpoint(block, hidden, self.causal_mask, memory, use_reentrant=False)
            else:
                hidden = block(hidden, self.causal_mask, memory)
        info: dict[str, Any] = {"steps_run": 0, "confidence": None, "states": None}
        if self.refinement is not None:
            adaptive_flag = (not self.training) if adaptive is None else bool(adaptive)
            hidden, info = self.refinement(hidden, use_checkpoint=use_checkpoint,
                                           adaptive=adaptive_flag, collect_states=collect_states)
        logits = self.head(self.norm_final(hidden))
        return (logits, info) if return_info else logits

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def loss(self, ids: torch.Tensor, targets: torch.Tensor | None = None,
             memory: Any | None = None, *, confidence_weight: float = 0.1,
             entropy_threshold: float = 0.35,
             return_metrics: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Causal-LM cross-entropy + auxiliary confidence BCE.

        The BCE target for each refinement state is ``1`` when the byte
        distribution at that state has normalised entropy below
        ``entropy_threshold`` (a "decided" state), else ``0``. Targets are
        detached: the confidence head learns to READ certainty, it must not
        change the byte distribution to manufacture it.

        One masking contract for every data group (PT-BR text, code, English,
        multimodal captions/codecs): both ``ce`` and the auxiliary BCE are
        means over positions where ``targets != PAD``, so the loss is
        token-weighted and a 50% PT-BR batch is the exact blend of its parts.
        A batch with no unmasked target returns a graph-carrying zero loss
        (cross-entropy would be ``0/0 = NaN``).
        """
        if targets is None:
            targets = ids[:, 1:]
            ids = ids[:, :-1]
            length = min(ids.shape[1], targets.shape[1])
            ids, targets = ids[:, :length], targets[:, :length]
        want_conf = (self.refinement is not None and confidence_weight > 0.0
                     and self.training)
        logits, info = self.forward(ids, memory, collect_states=want_conf, return_info=True)
        valid = targets != PAD
        if not bool(valid.any()):
            # A fully masked batch has no supervision: cross_entropy would
            # return 0/0 = NaN and poison every weight. Return a zero loss that
            # still carries the graph (backward is a no-op, never NaN).
            total = logits.sum() * 0.0
            if return_metrics:
                return total, {"ce": 0.0, "confidence_bce": 0.0, "total": 0.0,
                               "mean_entropy": None,
                               "steps_run": int(info["steps_run"]),
                               "thought_norm": float(info.get("thought_norm") or 0.0)}
            return total
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             targets.reshape(-1), ignore_index=PAD)
        aux = torch.zeros((), device=ce.device, dtype=ce.dtype)
        mean_entropy = None
        if want_conf:
            # Both terms are means over NON-PAD positions: a batch that is 50%
            # PT-BR is the token-weighted blend of its parts, so the auxiliary
            # BCE cannot be diluted by short (heavily padded) caption rows.
            states = info["states"]
            with torch.no_grad():
                step_logits = torch.stack(
                    [self.head(self.norm_final(state)) for state in states]).float()
                log_probs = torch.log_softmax(step_logits, dim=-1)
                entropy = -(log_probs.exp() * log_probs).sum(-1) / math.log(self.config.vocab_size)
                conf_targets = (entropy < float(entropy_threshold)).float()
                mean_entropy = float(entropy[..., valid].mean().item())
            terms = []
            thought_states = info.get("thought_states") or [None] * len(states)
            for state, target_t, thought in zip(states, conf_targets, thought_states):
                conf = self.refinement.confidence(state, thought).float().clamp(1e-6, 1.0 - 1e-6)
                terms.append(F.binary_cross_entropy(conf[valid], target_t[valid]))
            aux = torch.stack(terms).mean().to(ce.dtype)
        total = ce + float(confidence_weight) * aux
        if return_metrics:
            return total, {"ce": float(ce.item()), "confidence_bce": float(aux.item()),
                           "total": float(total.item()),
                           "mean_entropy": mean_entropy,
                           "steps_run": int(info["steps_run"]),
                           "thought_norm": float(info.get("thought_norm") or 0.0)}
        return total

    @property
    def parameters_count(self) -> int:
        return sum(int(tensor.numel()) for tensor in self.parameters())

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, *, max_new_bytes: int = 192,
                 temperature: float = 0.0, stop: tuple[int, ...] = (EOS,),
                 memory: Any | None = None) -> torch.Tensor:
        """Greedy (or sampled) continuation. Returns only the new ids."""
        self.eval()
        if prefix.ndim != 1:
            raise ValueError("generate expects a 1-D prefix of ids")
        if not 0.0 <= temperature < 5.0:
            raise ValueError("temperature must be in [0, 5)")
        generated: list[int] = []
        ids = prefix.clone()
        for _ in range(max_new_bytes):
            window = ids[-self.config.max_seq_len:]
            logits = self.forward(window.unsqueeze(0), memory)[0, -1]
            if temperature <= 0.0:
                next_id = int(logits.argmax().item())
            else:
                next_id = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).item())
            if next_id in stop:
                break
            generated.append(next_id)
            ids = torch.cat((ids, torch.tensor([next_id], dtype=ids.dtype, device=ids.device)))
        return torch.tensor(generated, dtype=torch.long, device=prefix.device)

    @torch.no_grad()
    def generate_with_tools(self, prefix: torch.Tensor,
                            search_fn: Callable[[bytes], bytes], *,
                            max_new_bytes: int = 512, max_tool_calls: int = 4,
                            temperature: float = 0.0, memory: Any | None = None,
                            observation_limit: int = 2048) -> tuple[torch.Tensor, dict[str, Any]]:
        """Token-level tool loop. Detection is by ID, never by regex.

        When the model emits ``TOOL_SEARCH ... TOOL_SEARCH_END``, the decoded
        query bytes go to ``search_fn`` (sandboxed; mock ddgs for training)
        and the result is appended as ``OBS_START <bytes> OBS_END``. ``EDIT``
        ids receive NO special treatment -- they are ordinary vocabulary.
        """
        self.eval()
        if prefix.ndim != 1:
            raise ValueError("generate_with_tools expects a 1-D prefix of ids")
        ids = prefix.clone()
        generated: list[int] = []
        queries: list[bytes] = []
        query_start: int | None = None
        tool_calls = 0
        for _ in range(max_new_bytes):
            window = ids[-self.config.max_seq_len:]
            logits = self.forward(window.unsqueeze(0), memory)[0, -1]
            if temperature <= 0.0:
                next_id = int(logits.argmax().item())
            else:
                next_id = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).item())
            if next_id == EOS:
                break
            generated.append(next_id)
            ids = torch.cat((ids, torch.tensor([next_id], dtype=ids.dtype, device=ids.device)))
            if next_id == TOOL_SEARCH:
                query_start = ids.shape[0]
            elif next_id == TOOL_SEARCH_END and query_start is not None:
                query = decode_ids(ids[query_start:-1])
                queries.append(query)
                result = bytes(search_fn(query))[:observation_limit]
                obs = [OBS_START] + list(result) + [OBS_END]
                obs_tensor = torch.tensor(obs, dtype=ids.dtype, device=ids.device)
                ids = torch.cat((ids, obs_tensor))
                generated.extend(obs)
                query_start = None
                tool_calls += 1
                if tool_calls >= max_tool_calls:
                    break
        stats = {"tool_calls": tool_calls, "queries": queries}
        return torch.tensor(generated, dtype=torch.long, device=prefix.device), stats


# ---------------------------------------------------------------------------
# Byte <-> id helpers. PURE byte mapping: no scanning, no regex, no markers.
# Special tokens are inserted programmatically by data builders / the tool
# runtime, never parsed out of text.
# ---------------------------------------------------------------------------
def encode_bytes(data: bytes, *, bos: bool = True, eos: bool = False) -> torch.Tensor:
    """Encode raw bytes to ids. The literal text '[TOOL_SEARCH]' stays eight
    ordinary bytes -- only the control ID 259 means a tool call."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    ids = ([BOS] if bos else []) + list(bytes(data)) + ([EOS] if eos else [])
    return torch.tensor(ids, dtype=torch.long)


def decode_ids(ids: torch.Tensor | list[int], *, keep_controls: bool = False) -> bytes:
    """Decode ids to raw bytes. Control ids are dropped by default; with
    ``keep_controls=True`` they render as their marker names (debugging)."""
    values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    out = bytearray()
    for value in values:
        if 0 <= value <= 255:
            out.append(value)
        elif keep_controls and value in CONTROL_NAMES and value not in (BOS, EOS, PAD):
            out += CONTROL_NAMES[value].encode("utf8")
    return bytes(out)


def pad_batch(sequences: list[torch.Tensor], length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad id sequences to ``length``; returns (ids, shifted targets)."""
    if length < 1:
        raise ValueError("length must be positive")
    batch = len(sequences)
    ids = torch.full((batch, length), PAD, dtype=torch.long)
    targets = torch.full((batch, length), PAD, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        truncated = sequence[:length]
        size = truncated.shape[0]
        ids[index, :size] = truncated
        targets[index, : size - 1] = truncated[1:]
    return ids, targets


CHECKPOINT_SCHEMA = 4  # v4: latent thought buffer (Phase 24.4)


def save_foundation(model: CausalByteTransformer, path: str | Path, *,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Save weights plus the config, so a checkpoint is self-describing."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "parameters": model.parameters_count,
    }
    if extra:
        state.update(dict(extra))
    torch.save(state, target)
    return {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "parameters": state["parameters"]}


def load_foundation(path: str | Path, *, device: torch.device | str = "cpu") -> CausalByteTransformer:
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("not a ByteCore v4 foundation checkpoint")
    model = CausalByteTransformer(FoundationConfig(**state["config"]))
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device)
