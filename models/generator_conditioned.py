"""Raw-byte generator with RETRO memory, cognitive editing and latent thought.

Per-layer top-K memory injection supports sequence-level or causally retrieved
chunked neighbors. A session-local scratchpad and bounded host-control loop
perform non-monotonic editing without emitting latent/control tokens as bytes.
A parallel causal thought stream and supervised confidence head support opt-in
adaptive computation. Raw data remain bytes 0-255; IDs 256-263 are internal
function/cognitive controls. No external tokenizer is used.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cognitive import (
    CognitiveAction, CognitiveController, CognitiveEvent, CognitiveObservation,
    CognitiveResult, apply_edit,
)
from .cognitive_arguments import decode_edit_arguments, encode_edit_arguments
from .epistemic import ConfidenceHead, ThoughtStreamLayer
from .adaptive import configured_threshold
from .procedural import raw_bytes
from .conversation_intents import NUM_INTENTS


@dataclass(frozen=True)
class GeneratorState:
    """Optional differentiable generator outputs; defaults remain unchanged."""

    output: torch.Tensor
    hidden: torch.Tensor
    thought: torch.Tensor
    confidence_logits: torch.Tensor
    confidence: torch.Tensor
    memory_attention: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class AdaptiveThought:
    scratchpad: torch.Tensor
    confidence: torch.Tensor
    steps: torch.Tensor


@dataclass(frozen=True)
class CognitivePolicyOutput:
    """Differentiable action and edit-argument logits for supervised training."""

    action_logits: torch.Tensor
    kind_logits: torch.Tensor
    byte_logits: torch.Tensor
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    destination_logits: torch.Tensor
    context: torch.Tensor
    confidence: torch.Tensor


class RetroCrossAttention(nn.Module):
    """Multi-head cross-attention over top-K memory slots with a learnable null sink.

    Supports both:
      * Sequence-level multi-vector memory: (B, K, d_model)
      * Chunked RETRO neighbor memory:      (B, N_chunks, K, d_model)

    A learnable null/sink key paired with an exact zero value vector allows each
    byte position to selectively ignore irrelevant memories, preserves non-zero
    Q/K gradients even when K=1, and prevents NaNs when all slots are masked.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_memory_slots: int = 64,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_memory_slots = max_memory_slots

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.rank_emb = nn.Embedding(max_memory_slots, d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable null/sink key in head space; its value is identically 0.
        self.null_key = nn.Parameter(
            torch.randn(1, n_heads, 1, self.head_dim) * 0.02
        )
        self.attn_dropout = nn.Dropout(dropout)

    def _attend_3d(
        self,
        h: torch.Tensor,
        mem: torch.Tensor,
        mem_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, l, d = h.shape
        k_slots = mem.shape[1]
        device = h.device

        rank_ids = torch.arange(k_slots, device=device).clamp(
            max=self.max_memory_slots - 1
        )
        mem_ranked = self.norm_kv(mem + self.rank_emb(rank_ids).unsqueeze(0))

        q = (
            self.q_proj(self.norm_q(h))
            .view(b, l, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(mem_ranked)
            .view(b, k_slots, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(mem_ranked)
            .view(b, k_slots, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        k_null = self.null_key.expand(b, self.n_heads, 1, self.head_dim)
        v_null = torch.zeros(
            b, self.n_heads, 1, self.head_dim, device=device, dtype=h.dtype
        )
        k_all = torch.cat([k_null, k], dim=2)
        v_all = torch.cat([v_null, v], dim=2)

        scores = (q @ k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mem_mask is not None:
            sink_valid = torch.ones(b, 1, device=device, dtype=torch.bool)
            full_mask = torch.cat([sink_valid, mem_mask.bool()], dim=1)
            scores = scores.masked_fill(~full_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn_dropped = self.attn_dropout(attn)
        ctx = (
            (attn_dropped @ v_all)
            .transpose(1, 2)
            .contiguous()
            .view(b, l, d)
        )
        return self.out_proj(ctx), attn[:, :, :, 1:]

    def forward(
        self,
        h: torch.Tensor,
        mem: torch.Tensor,
        mem_mask: torch.Tensor | None = None,
        *,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mem.dim() == 3:
            return self._attend_3d(h, mem, mem_mask)

        if mem.dim() != 4:
            raise ValueError(
                f"Expected 3D or 4D memory tensor, got shape {tuple(mem.shape)}"
            )

        b, l, d = h.shape
        _, n_chunks, k_slots, _ = mem.shape
        if n_chunks < 1:
            raise ValueError("chunked memory must have at least one chunk")
        chunk_len = int(chunk_size) if chunk_size is not None else max(math.ceil(l / n_chunks), 1)
        if chunk_len < 1 or (n_chunks - 1) * chunk_len >= l or n_chunks * chunk_len < l:
            raise ValueError("chunk_size and number of memory chunks do not cover the sequence")
        padded_len = n_chunks * chunk_len
        h_pad = F.pad(h, (0, 0, 0, padded_len - l)) if padded_len > l else h

        h_chunks = h_pad.reshape(b * n_chunks, chunk_len, d)
        mem_chunks = mem.reshape(b * n_chunks, k_slots, d)
        mask_chunks = (
            mem_mask.reshape(b * n_chunks, k_slots) if mem_mask is not None else None
        )
        out_chunks, attn_chunks = self._attend_3d(h_chunks, mem_chunks, mask_chunks)
        out = out_chunks.reshape(b, padded_len, d)[:, :l, :]
        attn = attn_chunks.reshape(b, n_chunks, self.n_heads, chunk_len, k_slots)
        return out, attn


class RetroMemoryInjectionLayer(nn.Module):
    """Per-layer multi-vector memory injector: virtual prefix tokens + gated CCA."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_memory_slots: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_memory_slots = max_memory_slots

        self.prefix_norm = nn.LayerNorm(d_model)
        self.prefix_rank_emb = nn.Embedding(max_memory_slots, d_model)
        self.prefix_proj = nn.Linear(d_model, d_model)
        self.prefix_gate = nn.Parameter(torch.tensor(0.25))

        self.cross_attn = RetroCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_memory_slots=max_memory_slots,
        )
        self.attn_gate = nn.Parameter(torch.tensor(0.25))

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, max(d_ff // 2, d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, d_model), d_model),
            nn.Dropout(dropout),
        )
        self.ffn_gate = nn.Parameter(torch.tensor(0.15))

    def build_prefix_tokens(self, mem: torch.Tensor) -> torch.Tensor:
        k_slots = mem.shape[1]
        rank_ids = torch.arange(k_slots, device=mem.device).clamp(
            max=self.max_memory_slots - 1
        )
        ranked = self.prefix_norm(mem + self.prefix_rank_emb(rank_ids).unsqueeze(0))
        return torch.tanh(self.prefix_gate) * self.prefix_proj(ranked)

    def apply_cross_attention(
        self,
        h: torch.Tensor,
        mem: torch.Tensor,
        mem_mask: torch.Tensor | None = None,
        *,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ca_out, attn_weights = self.cross_attn(h, mem, mem_mask=mem_mask, chunk_size=chunk_size)
        if mem_mask is None:
            active: torch.Tensor | float = 1.0
        elif mem.ndim == 3:
            active = mem_mask.any(dim=-1)[:, None, None].to(h.dtype)
        else:
            size = chunk_size if chunk_size is not None else math.ceil(h.shape[1] / mem.shape[1])
            active = mem_mask.any(dim=-1).repeat_interleave(size, dim=1)[:, :h.shape[1], None].to(h.dtype)
        # Completely masked rows/chunks must behave exactly like no memory,
        # even after out-projection/FFN biases have been trained.
        h = h + active * torch.tanh(self.attn_gate) * ca_out
        h = h + active * torch.tanh(self.ffn_gate) * self.ffn(self.norm_ffn(h))
        return h, attn_weights


class GeneratorConditioned(nn.Module):
    """Byte-level generator with Per-Layer Multi-Vector Memory Injection (RETRO).

    Raw bytes occupy 0-255; 256-258 are function controls and 259-263
    are internal cognitive controls. Controls are never emitted as canvas bytes.
    """

    VOCAB_BYTES: int = 256
    FUNC_CALL: int = 256
    FUNC_ARG: int = 257
    FUNC_END: int = 258
    LOAD_SPAN: int = 259
    SAVE_SPAN: int = 260
    THINK: int = 261
    EDIT_OP: int = 262
    DONE: int = 263
    # Phase 15 external tool use. These controls are never serialized as bytes
    # and are never executed by the model: the host intercepts TOOL_CALL, runs
    # an external sandbox and injects TOOL_RESULT. See models/tool_use.py.
    TOOL_CALL: int = 264
    TOOL_ARG_START: int = 265
    TOOL_ARG_END: int = 266
    TOOL_RESULT: int = 267
    VOCAB_SIZE: int = 268
    TOOL_TOKEN_START: int = 264
    # A checkpoint that predates the tool controls keeps its old predictions
    # exactly: new rows are seeded conservatively and biased against emission.
    TOOL_UNTRAINED_BIAS: float = -8.0
    VALID_MODES: tuple[str, str, str] = ("hybrid", "cross_attn", "prefix")

    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 256,
        max_len: int = 128,
        dropout: float = 0.1,
        memory_top_k: int = 4,
        memory_inject_every: int = 1,
        retro_chunk_size: int = 16,
        injection_mode: str = "hybrid",
        max_memory_slots: int = 64,
        scratchpad_slots: int = 8,
        max_canvas_bytes: int = 4096,
        max_think_steps: int = 32,
    ) -> None:
        super().__init__()
        if scratchpad_slots < 1 or max_canvas_bytes < 1 or max_think_steps < 1 or max_memory_slots < 1:
            raise ValueError("scratchpad, canvas, thought, and memory limits must be positive")
        if max_len < 1 or n_layers < 1:
            raise ValueError("max_len and n_layers must be positive")
        if injection_mode not in self.VALID_MODES:
            raise ValueError(
                f"injection_mode must be one of {self.VALID_MODES}, "
                f"got {injection_mode!r}"
            )
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.memory_top_k = max(int(memory_top_k), 1)
        self.memory_inject_every = max(int(memory_inject_every), 1)
        self.retro_chunk_size = max(int(retro_chunk_size), 1)
        self.injection_mode = injection_mode
        self.max_memory_slots = max_memory_slots
        self.scratchpad_slots = int(scratchpad_slots)
        self.max_canvas_bytes = int(max_canvas_bytes)
        self.max_think_steps = int(max_think_steps)

        n_ctrl = self.VOCAB_SIZE - self.VOCAB_BYTES
        self.byte_emb = nn.Embedding(self.VOCAB_BYTES, d_model)
        self.control_emb = nn.Embedding(n_ctrl, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.injected_layers: tuple[int, ...] = tuple(
            i for i in range(n_layers) if i % self.memory_inject_every == 0
        )
        self.memory_adapters = nn.ModuleDict(
            {
                str(i): RetroMemoryInjectionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    max_memory_slots=max_memory_slots,
                )
                for i in self.injected_layers
            }
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.VOCAB_SIZE)
        self.dropout = nn.Dropout(dropout)

        # Thought states are parallel to output positions; scratchpad state is
        # cloned per cognitive call and is never persisted or shared by sessions.
        self.thought_init = nn.Linear(d_model, d_model)
        self.thought_layers = nn.ModuleList(ThoughtStreamLayer(d_model) for _ in range(n_layers))
        self.confidence_head = ConfidenceHead(2 * d_model, d_model)
        self.register_buffer("scratchpad", torch.zeros(1, self.scratchpad_slots, d_model), persistent=False)
        self.scratchpad_reader = nn.GRUCell(d_model, d_model)
        self.scratchpad_thinker = nn.GRUCell(d_model, d_model)
        self.think_context = nn.Linear(2 * d_model, d_model)
        self.think_norm = nn.LayerNorm(d_model)
        self.pointer_boundary = nn.Parameter(torch.zeros(d_model))
        self.cognitive_state_emb = nn.Embedding(4, d_model)
        self.pointer_query = nn.Linear(d_model, d_model)
        self.pointer_start = nn.Linear(d_model, 1)
        self.pointer_end = nn.Linear(d_model, 1)
        self.pointer_destination = nn.Linear(d_model, 1)
        self.edit_kind_head = nn.Linear(d_model, 4)
        self.edit_byte_head = nn.Linear(d_model, 256)
        # Optional structured-request conditioning. These projections influence
        # argument logits, not action selection or host-side edit execution.
        # They start at zero, preserving all old request-free predictions.
        self.request_start_proj = nn.Linear(d_model, d_model, bias=False)
        self.request_end_proj = nn.Linear(d_model, d_model, bias=False)
        self.request_destination_proj = nn.Linear(d_model, d_model, bias=False)
        self.request_value_proj = nn.Linear(d_model, d_model, bias=False)
        self.request_kind_proj = nn.Linear(d_model, 4, bias=False)
        self.register_buffer("structured_edit_trained", torch.zeros((), dtype=torch.bool))
        self.register_buffer("structured_edit_trained_kinds", torch.zeros(4, dtype=torch.bool))
        # Auxiliary training head: predicts a target-minus-current byte sketch
        # from post-THINK scratchpad state. It is never given the target at inference.
        self.reasoning_predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        # Phase 10: classify the complete raw-byte message before selecting an
        # action. This head is independent of existing byte/cognitive logits.
        # A small trainable byte n-gram encoder compensates for the frozen edit
        # backbone's weak conversational features. It cannot affect edit logits.
        intent_channels = max(8, d_model // 2)
        self.intent_encoder = nn.Sequential(
            nn.Conv1d(d_model, intent_channels, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv1d(intent_channels, d_model, kernel_size=5, padding=2), nn.SiLU(),
        )
        self.intent_classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, NUM_INTENTS)
        )
        # A legacy checkpoint has no trained intent model. Consumers must check
        # this flag rather than treating a randomly initialized head as usable.
        self.register_buffer("intent_trained", torch.zeros((), dtype=torch.bool))

        self.apply(self._init_weights)
        for projection in (self.request_start_proj, self.request_end_proj, self.request_destination_proj,
                           self.request_value_proj, self.request_kind_proj):
            nn.init.zeros_(projection.weight)
        for adapter in self.memory_adapters.values():
            nn.init.constant_(adapter.prefix_gate, 0.25)
            nn.init.constant_(adapter.attn_gate, 0.25)
            nn.init.constant_(adapter.ffn_gate, 0.15)
            nn.init.normal_(adapter.cross_attn.null_key, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Migrate 259-token v1/v2 checkpoints; preserve old rows exactly.

        Old single-vector adapters are approximated by the first RETRO prefix
        projection; no claim of functionally identical memory conditioning is
        made for that older architecture. Unrelated mismatches remain errors.
        """
        migrated = OrderedDict(state_dict)
        old_vocab = (
            "head.weight" in migrated and migrated["head.weight"].shape[0] == 259
        )
        is_legacy = not any(k.startswith("memory_adapters.") for k in migrated) and (
            "memory_gate.weight" in migrated or "memory_proj.weight" in migrated
        )
        defaults = self.state_dict()
        pre_tool_head_rows = int(migrated["head.weight"].shape[0]) if "head.weight" in migrated else self.VOCAB_SIZE
        pre_tool_control_rows = int(migrated["control_emb.weight"].shape[0]) if "control_emb.weight" in migrated else self.VOCAB_SIZE - self.VOCAB_BYTES
        if old_vocab:
            for key, old_rows in (("control_emb.weight", 3), ("head.weight", 259), ("head.bias", 259)):
                if key not in migrated or migrated[key].shape[0] != old_rows:
                    raise RuntimeError(f"inconsistent legacy vocabulary tensor: {key}")
                old = migrated[key]
                target = defaults[key].detach().clone().to(device=old.device, dtype=old.dtype)
                if old.shape[1:] != target.shape[1:]:
                    raise RuntimeError(f"incompatible legacy vocabulary dimensions: {key}")
                target[:old_rows].copy_(old)
                migrated[key] = target
        if is_legacy:
            projection = migrated.get("memory_proj.weight")
            projection_bias = migrated.get("memory_proj.bias")
            for key in list(migrated):
                if key.startswith(("memory_gate.", "memory_proj.")):
                    del migrated[key]
            for key, value in defaults.items():
                if key.startswith("memory_adapters."):
                    migrated.setdefault(key, value)
            if projection is not None and "memory_adapters.0.prefix_proj.weight" in migrated:
                if projection.shape == migrated["memory_adapters.0.prefix_proj.weight"].shape:
                    migrated["memory_adapters.0.prefix_proj.weight"] = projection
                if projection_bias is not None and projection_bias.shape == migrated["memory_adapters.0.prefix_proj.bias"].shape:
                    migrated["memory_adapters.0.prefix_proj.bias"] = projection_bias
        if old_vocab:
            new_prefixes = (
                "thought_init.", "thought_layers.", "confidence_head.",
                "scratchpad_reader.", "scratchpad_thinker.", "think_context.",
                "think_norm.", "pointer_boundary", "cognitive_state_emb.", "pointer_query.",
                "pointer_start.", "pointer_end.", "pointer_destination.",
                "edit_kind_head.", "edit_byte_head.",
            )
            for key, value in defaults.items():
                if key.startswith(new_prefixes):
                    migrated.setdefault(key, value)
        # Phase 15: grow any pre-tool vocabulary tensor in place. Existing rows
        # are copied unchanged into freshly allocated targets, so the caller's
        # tensors are never mutated and old predictions cannot shift.
        for key in ("control_emb.weight", "head.weight", "head.bias"):
            old = migrated.get(key)
            current = defaults[key]
            if old is None or tuple(old.shape) == tuple(current.shape):
                continue
            if (old.ndim != current.ndim or old.shape[0] > current.shape[0]
                    or tuple(old.shape[1:]) != tuple(current.shape[1:])):
                raise RuntimeError(f"incompatible vocabulary tensor: {key}")
            if old.shape[0] < 1:
                raise RuntimeError(f"incompatible vocabulary tensor: {key}")
            target = current.detach().clone().to(device=old.device, dtype=old.dtype)
            target[: old.shape[0]].copy_(old)
            migrated[key] = target
        # A checkpoint that predates the tool controls must behave exactly as
        # it did before: seed the new rows from the old ones and bias the head
        # against emitting tool IDs until a tool head is actually trained.
        tool_control_start = self.TOOL_TOKEN_START - self.VOCAB_BYTES
        if pre_tool_control_rows <= tool_control_start:
            control = migrated["control_emb.weight"].clone()
            control[tool_control_start:] = control[:tool_control_start].mean(0, keepdim=True)
            migrated["control_emb.weight"] = control
        if pre_tool_head_rows <= self.TOOL_TOKEN_START:
            weight = migrated["head.weight"].clone()
            weight[self.TOOL_TOKEN_START:] = weight[: self.VOCAB_BYTES].mean(0, keepdim=True)
            bias = migrated["head.bias"].clone()
            bias[self.TOOL_TOKEN_START:] = self.TOOL_UNTRAINED_BIAS
            migrated["head.weight"] = weight
            migrated["head.bias"] = bias
        # Phase 1-6 checkpoints have no reasoning head. Initialize only when the
        # whole head is absent; a partially missing/corrupt new head must still
        # fail strict loading. Existing parameters are never overwritten.
        if not any(key.startswith("reasoning_predictor.") for key in migrated):
            for key, value in defaults.items():
                if key.startswith("reasoning_predictor."):
                    migrated[key] = value
        request_prefixes = ("request_start_proj.", "request_end_proj.", "request_destination_proj.",
                            "request_value_proj.", "request_kind_proj.")
        has_request_weights = any(key.startswith(request_prefixes) for key in migrated)
        if not has_request_weights:
            if "structured_edit_trained" in migrated or "structured_edit_trained_kinds" in migrated:
                raise ValueError("checkpoint contains structured edit metadata but has no request weights")
            for key, value in defaults.items():
                if key.startswith(request_prefixes):
                    migrated[key] = torch.zeros_like(value)
            migrated.setdefault("structured_edit_trained", torch.zeros_like(self.structured_edit_trained))
            migrated.setdefault("structured_edit_trained_kinds", torch.zeros_like(self.structured_edit_trained_kinds))
        # If any request weight exists, missing scope metadata or a partially
        # missing adapter remains a strict-loading error rather than being
        # misidentified as an old checkpoint.
        intent_prefixes = ("intent_encoder.", "intent_classifier.")
        has_intent_weights = any(key.startswith(intent_prefixes) for key in migrated)
        if not has_intent_weights and "intent_trained" not in migrated:
            # Fully legacy checkpoint: deterministic initialization, independent
            # of previously loaded heads and without changing global RNG state.
            rng = torch.Generator(device="cpu").manual_seed(10101)
            for key, value in defaults.items():
                if key.startswith(intent_prefixes):
                    if key == "intent_classifier.0.weight":
                        fresh = torch.randn(value.shape, generator=rng, dtype=torch.float32) * 0.02
                        migrated[key] = fresh.to(device=value.device, dtype=value.dtype)
                    elif key.startswith("intent_encoder.") and key.endswith(".weight"):
                        fan_in = value.shape[1] * value.shape[2]
                        fresh = torch.randn(value.shape, generator=rng, dtype=torch.float32) / math.sqrt(fan_in)
                        migrated[key] = fresh.to(device=value.device, dtype=value.dtype)
                    else:
                        migrated[key] = torch.zeros_like(value)
            migrated["intent_trained"] = torch.zeros_like(self.intent_trained)
        # Never silently repair a partially present encoder/head or missing flag.
        return super().load_state_dict(migrated, strict=strict, assign=assign)

    def mark_structured_edit_trained(self, kinds: tuple[str, ...] = ("replace",)) -> None:
        """Record trained operation kinds; this metadata is not an accuracy guarantee."""
        names = ("insert", "delete", "replace", "move")
        if not isinstance(kinds, tuple) or not kinds or any(kind not in names for kind in kinds):
            raise ValueError("kinds must be a nonempty tuple of supported edit kinds")
        for kind in kinds:
            self.structured_edit_trained_kinds[names.index(kind)] = True
        self.structured_edit_trained.copy_(self.structured_edit_trained_kinds.any())

    def _fit_mem_dim(self, t: torch.Tensor) -> torch.Tensor:
        d = t.shape[-1]
        if d == self.d_model:
            return t
        if d < self.d_model:
            return F.pad(t, (0, self.d_model - d))
        return t[..., : self.d_model]

    def _retrieve_chunked_retro(
        self,
        token_emb: torch.Tensor,
        memory_model: Any,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal chunked RETRO retrieval: chunk c queries memory using chunk c-1."""
        b, l, d = token_emb.shape
        device = token_emb.device
        c_size = self.retro_chunk_size
        n_chunks = max(math.ceil(l / c_size), 1)

        mem_chunks = torch.zeros(
            b, n_chunks, top_k, d, device=device, dtype=token_emb.dtype
        )
        mask_chunks = torch.zeros(b, n_chunks, top_k, device=device, dtype=torch.bool)

        if n_chunks <= 1:
            return mem_chunks, mask_chunks

        padded_len = n_chunks * c_size
        emb_pad = (
            F.pad(token_emb, (0, 0, 0, padded_len - l))
            if padded_len > l
            else token_emb[:, :padded_len, :]
        )
        chunk_means = emb_pad.view(b, n_chunks, c_size, d)[:, :-1].mean(dim=2)
        flat_queries = chunk_means.reshape(b * (n_chunks - 1), d)

        if hasattr(memory_model, "retrieve_differentiable") and self.training:
            vecs, _, vmask = memory_model.retrieve_differentiable(
                flat_queries, top_k=top_k
            )
            k_eff = min(vecs.shape[1], top_k)
            mem_chunks[:, 1:, :k_eff, :] = vecs[:, :k_eff, :].view(
                b, n_chunks - 1, k_eff, d
            )
            mask_chunks[:, 1:, :k_eff] = vmask[:, :k_eff].view(
                b, n_chunks - 1, k_eff
            )
        else:
            for bi in range(b):
                for ci in range(n_chunks - 1):
                    entries, m_vecs = memory_model.retrieve(
                        chunk_means[bi, ci], top_k=top_k
                    )
                    if m_vecs.dim() == 1 and entries:
                        m_vecs = m_vecs.unsqueeze(0)
                    if m_vecs.numel() > 0 and m_vecs.shape[0] > 0:
                        k_eff = min(m_vecs.shape[0], top_k)
                        mem_chunks[bi, ci + 1, :k_eff] = m_vecs[:k_eff].to(
                            device=device, dtype=token_emb.dtype
                        )
                        mask_chunks[bi, ci + 1, :k_eff] = True

        return mem_chunks, mask_chunks

    def _prepare_memory_tokens(
        self,
        memory_context: torch.Tensor | list[dict[str, Any]] | tuple[Any, ...] | None,
        memory_mask: torch.Tensor | None,
        memory_model: Any | None,
        token_emb: torch.Tensor,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        b = token_emb.shape[0]
        device = token_emb.device
        dtype = token_emb.dtype
        k_req = top_k if top_k is not None else self.memory_top_k

        if memory_context is None:
            if memory_model is not None:
                return self._retrieve_chunked_retro(token_emb, memory_model, k_req)
            return None, None

        if isinstance(memory_context, tuple) and len(memory_context) == 2:
            entries_part, vec_part = memory_context
            memory_context = vec_part if torch.is_tensor(vec_part) else entries_part

        if isinstance(memory_context, list):
            if len(memory_context) == 0:
                return None, None
            stacked = torch.stack(
                [
                    self._fit_mem_dim(
                        e["value"].to(device=device, dtype=dtype).reshape(-1)
                    )
                    for e in memory_context
                ],
                dim=0,
            )
            mem = stacked.unsqueeze(0).expand(b, -1, -1)
            mask = (
                memory_mask.to(device=device, dtype=torch.bool)
                if memory_mask is not None
                else torch.ones(b, mem.shape[1], device=device, dtype=torch.bool)
            )
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(b, -1)
            return mem, mask

        if not torch.is_tensor(memory_context):
            raise TypeError(f"Unsupported memory_context type: {type(memory_context)}")

        mc = memory_context.to(device=device, dtype=dtype)
        if mc.numel() == 0:
            return None, None
        mc = self._fit_mem_dim(mc)

        if mc.dim() == 1:
            mem = mc.view(1, 1, self.d_model).expand(b, 1, self.d_model)
        elif mc.dim() == 2:
            if b == 1:
                mem = mc.unsqueeze(0)
            elif mc.shape[0] == b:
                mem = mc.unsqueeze(1)
            else:
                mem = mc.unsqueeze(0).expand(b, -1, -1)
        elif mc.dim() == 3:
            if mc.shape[0] == 1 and b > 1:
                mem = mc.expand(b, -1, -1)
            elif mc.shape[0] != b:
                raise ValueError(
                    f"memory_context batch dim {mc.shape[0]} != input batch dim {b}"
                )
            else:
                mem = mc
        elif mc.dim() == 4:
            if mc.shape[0] == 1 and b > 1:
                mem = mc.expand(b, -1, -1, -1)
            elif mc.shape[0] != b:
                raise ValueError(
                    f"4D memory_context batch dim {mc.shape[0]} != input batch dim {b}"
                )
            else:
                mem = mc
        else:
            raise ValueError(f"Unsupported memory_context shape: {tuple(mc.shape)}")

        if memory_mask is not None:
            mask = memory_mask.to(device=device, dtype=torch.bool)
            if mask.dim() == 1 and mem.dim() == 3:
                mask = mask.unsqueeze(0).expand(b, -1)
        elif mem.dim() == 3:
            mask = torch.ones(b, mem.shape[1], device=device, dtype=torch.bool)
        else:
            mask = torch.ones(
                b, mem.shape[1], mem.shape[2], device=device, dtype=torch.bool
            )
        return mem, mask

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_context: torch.Tensor | list[dict[str, Any]] | tuple[Any, ...] | None = None,
        return_logits: bool = True,
        *,
        memory_mask: torch.Tensor | None = None,
        memory_model: Any | None = None,
        return_memory_attn: bool = False,
        return_state: bool = False,
        memory_chunk_size: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | GeneratorState:
        """Predict bytes; optionally expose latent thought and confidence state.

        With live RETRO retrieval, each chunk's neighbors are queried only from
        the preceding complete chunk. An explicitly supplied 4-D memory tensor
        may set ``memory_chunk_size``; otherwise its chunks divide the sequence
        evenly. Normal outputs and the old ``return_memory_attn`` contract are
        unchanged unless ``return_state=True`` is requested.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty (batch, sequence) tensor")
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError("input_ids must contain integer byte/control IDs")
        if bool(((input_ids < 0) | (input_ids >= self.VOCAB_SIZE)).any()):
            raise ValueError(f"input_ids must be in [0, {self.VOCAB_SIZE - 1}]")
        input_ids = input_ids.long()
        b, l = input_ids.shape
        device = input_ids.device
        n_ctrl = self.VOCAB_SIZE - self.VOCAB_BYTES

        is_control = input_ids >= self.VOCAB_BYTES
        byte_ids = input_ids.clamp(min=0, max=self.VOCAB_BYTES - 1)
        ctrl_ids = (input_ids - self.VOCAB_BYTES).clamp(min=0, max=n_ctrl - 1)
        # Preserve the established control embedding behavior for old weights.
        token_emb = self.byte_emb(byte_ids) + self.control_emb(ctrl_ids) * is_control.unsqueeze(-1).to(self.byte_emb.weight.dtype)
        pos_idx = torch.arange(l, device=device).clamp(max=self.max_len - 1)
        h = self.dropout(token_emb + self.pos_emb(pos_idx).unsqueeze(0))
        thought = torch.tanh(self.thought_init(token_emb))

        mem_tokens, mem_mask_norm = self._prepare_memory_tokens(
            memory_context=memory_context, memory_mask=memory_mask,
            memory_model=memory_model, token_emb=token_emb,
        )
        chunk_size: int | None = None
        if mem_tokens is not None:
            if mem_mask_norm is not None:
                expected = mem_tokens.shape[:-1]
                if mem_mask_norm.shape[0] == 1 and b > 1:
                    mem_mask_norm = mem_mask_norm.expand(b, *mem_mask_norm.shape[1:])
                if tuple(mem_mask_norm.shape) != tuple(expected):
                    raise ValueError(f"memory_mask must have shape {tuple(expected)}")
                if not bool(mem_mask_norm.any()):
                    mem_tokens = None
                    mem_mask_norm = None
            if mem_tokens is not None and mem_tokens.ndim == 4:
                n_chunks = mem_tokens.shape[1]
                if n_chunks < 1:
                    raise ValueError("chunked memory must contain at least one chunk")
                chunk_size = memory_chunk_size
                if chunk_size is None:
                    chunk_size = self.retro_chunk_size if memory_context is None and memory_model is not None else math.ceil(l / n_chunks)
                if chunk_size < 1 or (n_chunks - 1) * chunk_size >= l or n_chunks * chunk_size < l:
                    raise ValueError("memory_chunk_size and memory chunks must cover the sequence")

        causal_mask = torch.triu(torch.ones(l, l, device=device, dtype=torch.bool), diagonal=1)
        attn_maps: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            key_str = str(layer_idx)
            has_inj = mem_tokens is not None and key_str in self.memory_adapters
            if has_inj and self.injection_mode in ("hybrid", "prefix"):
                assert mem_tokens is not None
                adapter: RetroMemoryInjectionLayer = self.memory_adapters[key_str]  # type: ignore[assignment]
                if mem_tokens.ndim == 4:
                    _, n_chunks, k_per_chunk, _ = mem_tokens.shape
                    flat_mem = mem_tokens.reshape(b, n_chunks * k_per_chunk, self.d_model)
                    flat_mask = mem_mask_norm.reshape(b, -1) if mem_mask_norm is not None else None
                else:
                    flat_mem, flat_mask = mem_tokens, mem_mask_norm
                prefix = adapter.build_prefix_tokens(flat_mem)
                # Route fully masked batch rows through the ordinary block.
                # This avoids all-masked prefix queries/NaNs without introducing
                # a synthetic prefix that changes other rows' outputs.
                valid_rows = (torch.ones(b, device=device, dtype=torch.bool)
                              if flat_mask is None else flat_mask.any(dim=1))
                active_idx = valid_rows.nonzero(as_tuple=False).flatten()
                inactive_idx = (~valid_rows).nonzero(as_tuple=False).flatten()
                k_total = prefix.shape[1]
                pmask = torch.zeros(k_total + l, k_total + l, device=device, dtype=torch.bool)
                pmask[:k_total, k_total:] = True
                pmask[k_total:, k_total:] = causal_mask
                if mem_tokens.ndim == 4:
                    assert chunk_size is not None
                    pmask[k_total:, :k_total] = True
                    for ci in range(n_chunks):
                        lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, l)
                        pmask[k_total + lo:k_total + hi, ci * k_per_chunk:(ci + 1) * k_per_chunk] = False
                active_h = h.index_select(0, active_idx)
                ext_h = torch.cat([prefix.index_select(0, active_idx), active_h], dim=1)
                active_mask = flat_mask.index_select(0, active_idx) if flat_mask is not None else None
                key_pad = None
                if active_mask is not None and not bool(active_mask.all()):
                    key_pad = ~torch.cat([
                        active_mask, torch.ones(active_idx.numel(), l, device=device, dtype=torch.bool)
                    ], dim=1)
                ext_out = block(ext_h, src_mask=pmask, src_key_padding_mask=key_pad, is_causal=False)
                original_h = h
                h = h.index_copy(0, active_idx, ext_out[:, k_total:, :])
                if inactive_idx.numel():
                    unconditioned = block(original_h.index_select(0, inactive_idx), src_mask=causal_mask, is_causal=True)
                    h = h.index_copy(0, inactive_idx, unconditioned)
            else:
                h = block(h, src_mask=causal_mask, is_causal=True)
            if has_inj and self.injection_mode in ("hybrid", "cross_attn"):
                assert mem_tokens is not None
                adapter = self.memory_adapters[key_str]  # type: ignore[assignment]
                h, attn_w = adapter.apply_cross_attention(h, mem_tokens, mem_mask=mem_mask_norm, chunk_size=chunk_size)
                if return_memory_attn or return_state:
                    attn_maps.append(attn_w)
            h, thought = self.thought_layers[layer_idx](h, thought)

        h = self.norm(h)
        out = self.head(h) if return_logits else h
        if return_state:
            confidence_logits = self.confidence_head(torch.cat([h, thought], dim=-1))
            return GeneratorState(
                output=out, hidden=h, thought=thought,
                confidence_logits=confidence_logits,
                confidence=self.confidence_head.probability(confidence_logits),
                memory_attention=tuple(attn_maps),
            )
        if return_memory_attn:
            return out, attn_maps
        return out

    def _validate_conversation_input(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> None:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty (batch, sequence) tensor")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("conversation input_ids must be integer raw bytes")
        if input_ids.shape[1] > self.max_len or bool(((input_ids < 0) | (input_ids > 255)).any()):
            raise ValueError("conversation inputs must be raw bytes within max_len")
        if not torch.is_tensor(lengths) or lengths.dtype not in (torch.int32, torch.int64):
            raise TypeError("lengths must be an integer tensor")
        if lengths.shape != (input_ids.shape[0],) or lengths.device != input_ids.device:
            raise ValueError("lengths must have one value per input on the same device")
        if bool(((lengths < 1) | (lengths > input_ids.shape[1])).any()):
            raise ValueError("each length must identify a nonempty input")

    def conversation_backbone_features(
        self, input_ids: torch.Tensor, lengths: torch.Tensor, *,
        memory_context: torch.Tensor | list[dict[str, Any]] | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool causal byte states with optional, separately supplied memory vectors.

        Padding is identified by lengths, never byte value zero. The caller is
        responsible for access control and for scoping retrieved memory.
        """
        self._validate_conversation_input(input_ids, lengths)
        state = self(input_ids, memory_context=memory_context, memory_mask=memory_mask,
                     return_logits=False, return_state=True)
        assert isinstance(state, GeneratorState)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        mean = (state.hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
        last = state.hidden[torch.arange(input_ids.shape[0], device=input_ids.device), lengths.long() - 1]
        return (last + mean) * 0.5

    def conversation_features(self, input_ids: torch.Tensor, lengths: torch.Tensor, *,
                              focus_start: torch.Tensor | None = None,
                              backbone_features: torch.Tensor | None = None,
                              memory_context: torch.Tensor | list[dict[str, Any]] | None = None,
                              memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Combine causal context and trainable n-gram features of the user turn.

        ``focus_start`` is the caller-supplied start of the current turn; it is
        not inferred from user-supplied delimiter bytes. Cached backbone features
        are an optional training optimization and must correspond to the input.
        """
        self._validate_conversation_input(input_ids, lengths)
        if focus_start is None:
            focus_start = torch.zeros_like(lengths)
        if (not torch.is_tensor(focus_start) or focus_start.dtype not in (torch.int32, torch.int64)
                or focus_start.shape != lengths.shape or focus_start.device != lengths.device):
            raise TypeError("focus_start must be an integer tensor matching lengths")
        if bool(((focus_start < 0) | (focus_start >= lengths)).any()):
            raise ValueError("focus_start must identify at least one current byte")
        if backbone_features is None:
            backbone_features = self.conversation_backbone_features(
                input_ids, lengths, memory_context=memory_context, memory_mask=memory_mask)
        elif memory_context is not None or memory_mask is not None:
            raise ValueError("cached backbone features cannot be combined with a live memory context")
        elif backbone_features.shape != (input_ids.shape[0], self.d_model) or backbone_features.device != input_ids.device:
            raise ValueError("cached backbone features have an incompatible shape or device")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        focus = (positions >= focus_start.unsqueeze(1)) & (positions < lengths.unsqueeze(1))
        mask = focus.unsqueeze(1)
        embedded = self.byte_emb(input_ids.long()).transpose(1, 2) * mask
        hidden = self.intent_encoder[1](self.intent_encoder[0](embedded)) * mask
        hidden = self.intent_encoder[3](self.intent_encoder[2](hidden)) * mask
        mean = hidden.sum(dim=-1) / focus.sum(dim=-1, keepdim=True)
        maximum = hidden.masked_fill(~mask, float("-inf")).amax(dim=-1)
        return backbone_features + 0.5 * (mean + maximum)

    def classify_intent(self, input_ids: torch.Tensor, lengths: torch.Tensor, *,
                        focus_start: torch.Tensor | None = None,
                        memory_context: torch.Tensor | list[dict[str, Any]] | None = None,
                        memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return learned intent logits; callers may provide scoped memory vectors."""
        return self.intent_classifier(self.conversation_features(
            input_ids, lengths, focus_start=focus_start,
            memory_context=memory_context, memory_mask=memory_mask))

    def read_scratchpad(self, scratchpad: torch.Tensor, byte_ids: torch.Tensor) -> torch.Tensor:
        """Differentiably READ byte spans into a session-local latent buffer."""
        if scratchpad.ndim != 3 or scratchpad.shape[1:] != (self.scratchpad_slots, self.d_model):
            raise ValueError("scratchpad must have shape (B, scratchpad_slots, d_model)")
        if byte_ids.ndim != 2 or byte_ids.shape[0] != scratchpad.shape[0]:
            raise ValueError("byte_ids must have shape (B, span_length)")
        if byte_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("byte_ids must be integer raw bytes")
        if bool(((byte_ids < 0) | (byte_ids > 255)).any()):
            raise ValueError("scratchpad READ accepts only raw bytes")
        b, length = byte_ids.shape
        if length:
            emb = self.byte_emb(byte_ids.long())
            pos = torch.arange(length, device=byte_ids.device).clamp(max=self.max_len - 1)
            emb = emb + self.pos_emb(pos).unsqueeze(0)
            pooled = F.adaptive_avg_pool1d(emb.transpose(1, 2), self.scratchpad_slots).transpose(1, 2)
        else:
            pooled = torch.zeros_like(scratchpad)
        return self.scratchpad_reader(
            pooled.reshape(-1, self.d_model), scratchpad.reshape(-1, self.d_model)
        ).reshape(b, self.scratchpad_slots, self.d_model)

    def replay_latent_trace(self, scratchpad: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        """Replay stored latent steps into a session scratchpad in order."""
        if scratchpad.ndim != 3 or scratchpad.shape[1:] != (self.scratchpad_slots, self.d_model):
            raise ValueError("scratchpad has an incompatible shape")
        if trace.ndim != 2 or trace.shape[1] != self.d_model or trace.shape[0] > self.max_think_steps:
            raise ValueError("trace must have shape (steps, d_model) within max_think_steps")
        if not bool(torch.isfinite(trace).all()):
            raise ValueError("trace must be finite")
        b, slots, d = scratchpad.shape
        out = scratchpad
        for step in trace.to(device=out.device, dtype=out.dtype):
            drive = step.view(1, 1, d).expand(b, slots, d)
            out = self.scratchpad_reader(drive.reshape(-1, d), out.reshape(-1, d)).reshape(b, slots, d)
        return out

    def thought_confidence_logits(self, scratchpad: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return logits for collecting task-specific thought-calibration data."""
        if scratchpad.ndim != 3 or scratchpad.shape[1:] != (self.scratchpad_slots, self.d_model):
            raise ValueError("scratchpad has an incompatible shape")
        if context.shape != (scratchpad.shape[0], self.d_model):
            raise ValueError("context must have shape (B, d_model)")
        return self.confidence_head(torch.cat([context, scratchpad.mean(dim=1)], dim=-1))

    def think_scratchpad(
        self,
        scratchpad: torch.Tensor,
        context: torch.Tensor,
        *,
        steps: int = 1,
        confidence_threshold: float | None = None,
        min_steps: int = 1,
    ) -> AdaptiveThought:
        """THINK without emitting bytes; optionally halt each sample separately.

        Confidence-based stopping is opt-in and unsafe until validated and
        calibrated. Without a threshold exactly ``steps`` iterations are used.
        """
        if scratchpad.ndim != 3 or scratchpad.shape[1:] != (self.scratchpad_slots, self.d_model):
            raise ValueError("scratchpad has an incompatible shape")
        b, slots, d = scratchpad.shape
        if b < 1 or context.shape != (b, d):
            raise ValueError("context must have shape (positive B, d_model)")
        if (not isinstance(steps, int) or isinstance(steps, bool)
                or not isinstance(min_steps, int) or isinstance(min_steps, bool)
                or not 1 <= steps <= self.max_think_steps or not 1 <= min_steps <= steps):
            raise ValueError("thought steps/min_steps must be integers within configured bounds")
        if confidence_threshold is not None:
            if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
                raise ValueError("confidence_threshold must be in [0, 1]")
            self.confidence_head.warn_if_uncalibrated("thought")
        active = torch.ones(b, dtype=torch.bool, device=scratchpad.device)
        used = torch.zeros(b, dtype=torch.long, device=scratchpad.device)
        confidence = torch.zeros(b, dtype=scratchpad.dtype, device=scratchpad.device)
        state = scratchpad
        for iteration in range(steps):
            indices = active.nonzero(as_tuple=False).flatten()
            active_state = state.index_select(0, indices)
            active_context = context.index_select(0, indices)
            expanded = active_context.unsqueeze(1).expand(indices.numel(), slots, d)
            drive = F.silu(self.think_context(torch.cat([self.think_norm(active_state), expanded], dim=-1)))
            proposal = self.scratchpad_thinker(
                drive.reshape(-1, d), active_state.reshape(-1, d)
            ).reshape(indices.numel(), slots, d)
            state = state.index_copy(0, indices, proposal)
            logits = self.thought_confidence_logits(proposal, active_context)
            current = self.confidence_head.probability(logits)
            confidence = confidence.index_copy(0, indices, current)
            used = used.index_add(0, indices, torch.ones_like(indices))
            if confidence_threshold is not None and iteration + 1 >= min_steps:
                stop = torch.isfinite(current) & (current >= confidence_threshold)
                active = active.clone()
                active[indices[stop]] = False
                if not bool(active.any()):
                    break
        return AdaptiveThought(state, confidence, used)

    def predict_edit_intent(self, scratchpad: torch.Tensor) -> torch.Tensor:
        """Predict an edit-intent sketch from each post-THINK scratchpad.

        This auxiliary output is not fed back into the policy. The supervised
        target is constructed only by the training code, never at inference.
        """
        if scratchpad.ndim != 3 or scratchpad.shape[1:] != (self.scratchpad_slots, self.d_model):
            raise ValueError("scratchpad must have shape (B, scratchpad_slots, d_model)")
        return self.reasoning_predictor(scratchpad.mean(dim=1))

    def _pointer_positions(self, count: int, *, device: torch.device) -> torch.Tensor:
        """Learned in-window positions plus deterministic long-range features.

        Older checkpoints clamp every position beyond max_len to one embedding.
        Retain their in-window behavior exactly, but distinguish longer pointer
        coordinates with small sinusoidal features (no new parameters). Long-
        range prediction still requires training and is not guaranteed.
        """
        positions = torch.arange(count, device=device)
        learned = self.pos_emb(positions.clamp(max=self.max_len - 1))
        beyond = positions >= self.max_len
        if not bool(beyond.any()):
            return learned
        dimensions = torch.arange(self.d_model, device=device)
        scales = torch.exp(-math.log(10000.0) * (2 * (dimensions // 2)).float() / self.d_model)
        angles = positions.float().unsqueeze(1) * scales.unsqueeze(0)
        sinusoid = torch.where((dimensions % 2 == 0).unsqueeze(0), angles.sin(), angles.cos())
        return learned + beyond.unsqueeze(1).to(learned.dtype) * (0.02 * sinusoid).to(learned.dtype)

    def _request_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Fixed Fourier keys for explicit request coordinates, including >255."""
        half = self.d_model // 2
        if half == 0:
            raise ValueError("structured edit requests require d_model >= 2")
        frequencies = torch.remainder((torch.arange(half, device=positions.device, dtype=torch.float32) + 1)
                                      * 1.618033988749895, math.pi)
        angles = positions.to(dtype=torch.float32).unsqueeze(-1) * frequencies
        result = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if result.shape[-1] < self.d_model:
            result = F.pad(result, (0, self.d_model - result.shape[-1]))
        return F.normalize(result, dim=-1).to(dtype=self.byte_emb.weight.dtype)

    def cognitive_policy(
        self,
        canvas_view: bytes | bytearray,
        scratchpad: torch.Tensor,
        *,
        pointer_bytes: bytes | bytearray | None = None,
        memory_context: torch.Tensor | list[dict[str, Any]] | tuple[Any, ...] | None = None,
        memory_model: Any | None = None,
        loaded: bool = False,
        dirty: bool = False,
        last_token: int | None = None,
        structured_request: bytes | bytearray | None = None,
    ) -> CognitivePolicyOutput:
        """Differentiable control/edit logits for one byte canvas.

        This can be teacher-forced from supervised action traces. The canvas
        remains literal bytes; model controls are kept in a separate vocabulary.
        """
        canvas = raw_bytes(canvas_view, "canvas_view")
        points = canvas if pointer_bytes is None else raw_bytes(pointer_bytes, "pointer_bytes")
        if len(canvas) > self.max_canvas_bytes or len(points) > self.max_canvas_bytes:
            raise ValueError("canvas exceeds max_canvas_bytes")
        if scratchpad.shape != (1, self.scratchpad_slots, self.d_model):
            raise ValueError("cognitive_policy currently requires a single scratchpad")
        if dirty and not loaded:
            raise ValueError("a dirty span must be loaded")
        request_action: CognitiveAction | None = None
        if structured_request is not None:
            request_action = decode_edit_arguments(structured_request, max_position=self.max_canvas_bytes,
                                                   max_payload=self.max_canvas_bytes)
            if request_action.kind in ("insert", "replace") and len(request_action.data) != 1:
                raise ValueError("structured policy requests currently require a single payload byte")
        if last_token is not None and not self.LOAD_SPAN <= last_token <= self.DONE:
            raise ValueError("last_token must be a cognitive control token")
        device = self.byte_emb.weight.device
        if scratchpad.device != device:
            raise ValueError("scratchpad and generator must be on the same device")
        window = list(canvas[-self.max_len:]) or [self.LOAD_SPAN]
        ids = torch.tensor([window], device=device, dtype=torch.long)
        token_emb = self.byte_emb(ids.clamp(max=255))
        external: torch.Tensor | None = None
        ext_mask: torch.Tensor | None = None
        if memory_context is not None:
            external, ext_mask = self._prepare_memory_tokens(memory_context, None, None, token_emb)
            if external is not None and external.ndim != 3:
                raise ValueError("cognitive_policy expects sequence-level, not chunked, external memory")
        elif memory_model is not None:
            query = token_emb.mean(dim=1) + scratchpad.mean(dim=1)
            if self.training and torch.is_grad_enabled() and hasattr(memory_model, "retrieve_differentiable"):
                external, _, ext_mask = memory_model.retrieve_differentiable(query, top_k=self.memory_top_k)
                external = external.to(device=device, dtype=scratchpad.dtype)
                ext_mask = ext_mask.to(device=device)
            else:
                _, found = memory_model.retrieve(query[0], top_k=self.memory_top_k)
                if found.numel():
                    external = found.to(device=device, dtype=scratchpad.dtype).reshape(1, -1, self.d_model)
                    ext_mask = torch.ones(1, external.shape[1], device=device, dtype=torch.bool)
        if external is None:
            mem = scratchpad
            mem_mask = torch.ones(1, scratchpad.shape[1], device=device, dtype=torch.bool)
        else:
            mem = torch.cat([external, scratchpad], dim=1)
            if ext_mask is None:
                ext_mask = torch.ones(1, external.shape[1], device=device, dtype=torch.bool)
            mem_mask = torch.cat([ext_mask, torch.ones(1, scratchpad.shape[1], device=device, dtype=torch.bool)], dim=1)
        state = self.forward(ids, mem, memory_mask=mem_mask, return_state=True)
        assert isinstance(state, GeneratorState)
        state_id = int(loaded) + 2 * int(dirty)
        context = state.hidden[0, -1] + scratchpad.mean(dim=1)[0] + self.cognitive_state_emb.weight[state_id]
        if last_token is not None:
            context = context + self.control_emb.weight[last_token - self.VOCAB_BYTES]
        point_ids = torch.tensor(list(points), device=device, dtype=torch.long)
        point_emb = self.byte_emb(point_ids) if point_ids.numel() else self.byte_emb.weight.new_empty((0, self.d_model))
        point_emb = torch.cat([point_emb, self.pointer_boundary.unsqueeze(0)], dim=0)
        features = point_emb + self._pointer_positions(len(points) + 1, device=device)
        shared = (features * self.pointer_query(context)).sum(dim=-1) / math.sqrt(self.d_model)
        start_logits = self.pointer_start(features).squeeze(-1) + shared
        end_logits = self.pointer_end(features).squeeze(-1) + shared
        destination_logits = self.pointer_destination(features).squeeze(-1) + shared
        kind_logits = self.edit_kind_head(context)
        byte_logits = self.edit_byte_head(context)
        if request_action is not None and loaded:
            # The caller supplies the desired structured edit, not a target
            # document. A trainable compatibility score steers each head; the
            # normal policy still chooses every action and argument. No request
            # field is written directly into the output or canvas.
            keys = self._request_positions(torch.arange(len(points) + 1, device=device))
            def query(position: int) -> torch.Tensor:
                return self._request_positions(torch.tensor([position], device=device))[0]
            assert request_action.start is not None and request_action.kind is not None
            start_logits = start_logits + 16.0 * (keys @ self.request_start_proj(query(request_action.start)))
            if request_action.end is not None:
                end_logits = end_logits + 16.0 * (keys @ self.request_end_proj(query(request_action.end)))
            if request_action.destination is not None:
                destination_logits = destination_logits + 16.0 * (keys @ self.request_destination_proj(query(request_action.destination)))
            kinds = ("insert", "delete", "replace", "move")
            kind_code = kinds.index(request_action.kind)
            byte_keys = F.normalize(self.byte_emb.weight, dim=-1)
            kind_logits = kind_logits + 8.0 * self.request_kind_proj(byte_keys[kind_code])
            if request_action.data:
                byte_logits = byte_logits + 8.0 * (byte_keys @ self.request_value_proj(byte_keys[request_action.data[0]]))
        return CognitivePolicyOutput(
            action_logits=self.head(context)[self.LOAD_SPAN:self.DONE + 1],
            kind_logits=kind_logits, byte_logits=byte_logits,
            start_logits=start_logits, end_logits=end_logits, destination_logits=destination_logits,
            context=context, confidence=state.confidence[0, -1],
        )

    @torch.no_grad()
    def cognitive_generate(
        self,
        canvas: bytes | bytearray,
        *,
        controller: CognitiveController | None = None,
        memory_context: torch.Tensor | list[dict[str, Any]] | tuple[Any, ...] | None = None,
        memory_model: Any | None = None,
        initial_trace: torch.Tensor | None = None,
        max_steps: int = 64,
        confidence_threshold: float | None = None,
        temperature: float = 0.0,
        instruction_bytes: bytes | bytearray | None = None,
        structured_request: bytes | bytearray | None = None,
        allow_untrained_structured: bool = False,
        autonomous_think_steps: int = 1,
        min_think_steps: int = 1,
        adaptive_config: Mapping[str, Any] | None = None,
    ) -> CognitiveResult:
        """READ -> THINK -> non-monotonic EDIT -> SAVE, until DONE or a cap.

        With no controller, the learned control-token logits and pointer/byte
        heads choose actions. An optional host controller supplies the exact same
        typed actions and is useful for supervision or deterministic workflows.
        A dirty span is never silently committed or discarded. Optional raw
        instruction bytes prime the scratchpad but do not guarantee instruction
        following. An optional structured_request is a canonical EDIT argument
        frame, explicitly supplied by the caller and used only to condition
        learned argument logits. It is not a controller and is not automatically
        executed. Old checkpoints have no trained request adapter.
        """
        if adaptive_config is not None:
            if confidence_threshold is not None:
                raise ValueError("pass either adaptive_config or an explicit threshold, not both")
            confidence_threshold = configured_threshold(adaptive_config, self.confidence_head, "thought")
            if confidence_threshold is not None:
                autonomous_think_steps = adaptive_config.get("max_thought_steps", autonomous_think_steps)
                min_think_steps = adaptive_config.get("min_thought_steps", min_think_steps)
        data = raw_bytes(canvas, "canvas")
        instruction = None if instruction_bytes is None else raw_bytes(instruction_bytes, "instruction_bytes")
        request = None if structured_request is None else raw_bytes(structured_request, "structured_request")
        if request is not None:
            if controller is not None:
                raise ValueError("structured_request conditions the autonomous policy, not a host controller")
            decoded_request = decode_edit_arguments(request, max_position=self.max_canvas_bytes,
                                                     max_payload=self.max_canvas_bytes)
            if decoded_request.kind in ("insert", "replace") and len(decoded_request.data) != 1:
                raise ValueError("structured policy requests currently require a single payload byte")
            if (decoded_request.kind == "replace" and decoded_request.end != decoded_request.start + 1
                    and not allow_untrained_structured):
                raise RuntimeError("structured request training only covers one-byte replacement spans")
            kinds = ("insert", "delete", "replace", "move")
            assert decoded_request.kind is not None
            if not bool(self.structured_edit_trained_kinds[kinds.index(decoded_request.kind)]) and not allow_untrained_structured:
                raise RuntimeError(f"structured {decoded_request.kind} request adapter has not been trained")
        if instruction is not None and len(instruction) > self.max_canvas_bytes:
            raise ValueError("instruction exceeds max_canvas_bytes")
        if (not isinstance(autonomous_think_steps, int) or isinstance(autonomous_think_steps, bool)
                or not 1 <= autonomous_think_steps <= self.max_think_steps
                or not isinstance(min_think_steps, int) or isinstance(min_think_steps, bool)
                or not 1 <= min_think_steps <= autonomous_think_steps):
            raise ValueError("autonomous/min thought steps must lie within the configured limit")
        if len(data) > self.max_canvas_bytes:
            raise ValueError("canvas exceeds max_canvas_bytes")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 4096:
            raise ValueError("max_steps must be an integer between 1 and 4096")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if confidence_threshold is not None and (not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1):
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.eval()
        scratch = self.scratchpad.clone().to(device=self.byte_emb.weight.device, dtype=self.byte_emb.weight.dtype)
        if initial_trace is not None:
            scratch = self.replay_latent_trace(scratch, initial_trace)
        if instruction:
            ids = torch.tensor([list(instruction)], device=scratch.device, dtype=torch.long)
            scratch = self.read_scratchpad(scratch, ids)
        workspace: bytes | None = None
        span: tuple[int, int] | None = None
        dirty = False
        events: list[CognitiveEvent] = []
        last_confidence: float | None = None
        done = False

        def choose(logits: torch.Tensor) -> int:
            if temperature == 0:
                return int(logits.argmax().item())
            return int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).item())

        for iteration in range(max_steps):
            view = data if workspace is None or span is None else data[:span[0]] + workspace + data[span[1]:]
            pointer = data if workspace is None else workspace
            previous = events[-1].action.token if events else None
            request_kwargs = {"structured_request": request} if request is not None else {}
            policy = self.cognitive_policy(
                view, scratch, pointer_bytes=pointer, memory_context=memory_context, memory_model=memory_model,
                loaded=workspace is not None, dirty=dirty, last_token=previous, **request_kwargs,
            )
            if controller is not None:
                observation = CognitiveObservation(
                    step=iteration, canvas=data, workspace=workspace, span=span,
                    dirty=dirty, scratchpad=scratch.detach().clone(), confidence=last_confidence,
                )
                action = controller(observation)
                if not isinstance(action, CognitiveAction):
                    raise TypeError("controller must return CognitiveAction")
            else:
                logits = policy.action_logits.clone()
                if workspace is None:
                    logits[self.SAVE_SPAN - self.LOAD_SPAN] = -torch.inf
                    logits[self.EDIT_OP - self.LOAD_SPAN] = -torch.inf
                if dirty:
                    logits[self.LOAD_SPAN - self.LOAD_SPAN] = -torch.inf
                    logits[self.DONE - self.LOAD_SPAN] = -torch.inf
                token = self.LOAD_SPAN + choose(logits)
                if token == self.LOAD_SPAN:
                    # LOAD pointers address the complete committed canvas.
                    if workspace is not None:
                        policy = self.cognitive_policy(
                            view, scratch, pointer_bytes=data, memory_context=memory_context, memory_model=memory_model,
                            loaded=True, dirty=False, last_token=previous, **request_kwargs,
                        )
                    start = choose(policy.start_logits)
                    end = start + choose(policy.end_logits[start:])
                    action = CognitiveAction(token, start=start, end=end)
                elif token == self.EDIT_OP:
                    kinds = ("insert", "delete", "replace", "move")
                    kind = kinds[choose(policy.kind_logits)]
                    start = choose(policy.start_logits)
                    end = start + choose(policy.end_logits[start:])
                    dest = choose(policy.destination_logits)
                    if start < dest < end:
                        dest = end
                    payload = bytes([choose(policy.byte_logits)]) if kind in ("insert", "replace") else b""
                    action = CognitiveAction(token, start=start, end=end, kind=kind, data=payload, destination=dest)  # type: ignore[arg-type]
                else:
                    action = CognitiveAction(token, think_steps=autonomous_think_steps if token == self.THINK else 1)
            thought_steps = 0
            argument_bytes: bytes | None = None
            if action.token == self.LOAD_SPAN:
                if dirty:
                    raise ValueError("SAVE the edited span before loading another")
                if (not isinstance(action.start, int) or isinstance(action.start, bool)
                        or not isinstance(action.end, int) or isinstance(action.end, bool)
                        or not 0 <= action.start <= action.end <= len(data)):
                    raise ValueError("LOAD_SPAN indices must be a valid half-open canvas range")
                span = (action.start, action.end)
                workspace = data[span[0]:span[1]]
                ids = torch.tensor([list(workspace)], device=scratch.device, dtype=torch.long)
                scratch = self.read_scratchpad(scratch, ids)
            elif action.token == self.THINK:
                if not isinstance(action.think_steps, int) or isinstance(action.think_steps, bool):
                    raise ValueError("think_steps must be an integer")
                thought = self.think_scratchpad(
                    scratch, policy.context.unsqueeze(0), steps=action.think_steps,
                    confidence_threshold=confidence_threshold, min_steps=min_think_steps,
                )
                scratch = thought.scratchpad
                thought_steps = int(thought.steps[0].item())
                last_confidence = float(thought.confidence[0].item())
            elif action.token == self.EDIT_OP:
                if workspace is None or span is None:
                    raise ValueError("LOAD_SPAN is required before EDIT_OP")
                max_workspace = self.max_canvas_bytes - (len(data) - (span[1] - span[0]))
                applied_action = action
                if controller is None:
                    # The policy heads choose the fields; serialize and parse
                    # their literal byte arguments before applying the edit.
                    # Keep the original typed action in events so legacy RL
                    # scoring still sees every sampled argument.
                    argument_bytes = encode_edit_arguments(action)
                    applied_action = decode_edit_arguments(
                        argument_bytes, max_position=self.max_canvas_bytes,
                        max_payload=self.max_canvas_bytes,
                    )
                updated = apply_edit(workspace, applied_action, max_length=max_workspace)
                dirty = dirty or updated != workspace
                workspace = updated
            elif action.token == self.SAVE_SPAN:
                if workspace is None or span is None:
                    raise ValueError("LOAD_SPAN is required before SAVE_SPAN")
                data = data[:span[0]] + workspace + data[span[1]:]
                span = (span[0], span[0] + len(workspace))
                dirty = False
            elif action.token == self.DONE:
                if dirty:
                    raise ValueError("SAVE the edited span before DONE")
                done = True
            else:
                raise ValueError("unsupported cognitive control token")
            events.append(CognitiveEvent(
                action=action, canvas_length=len(data),
                workspace_length=None if workspace is None else len(workspace),
                thought_steps=thought_steps, confidence=last_confidence,
                argument_bytes=argument_bytes,
            ))
            if done:
                break
        return CognitiveResult(
            canvas=data, done=done, halt_reason="done" if done else "max_steps",
            events=tuple(events), scratchpad=scratch.detach().clone(),
            pending_workspace=workspace if dirty else None,
            pending_span=span if dirty else None,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        memory_context: torch.Tensor | list[dict[str, Any]] | tuple[Any, ...] | None = None,
        max_new: int = 40,
        temperature: float = 0.8,
        func_handler: Callable[[str], Any] | None = None,
        *,
        memory_model: Any | None = None,
        top_k_memories: int | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        """Generate one sequence, with cognitive controls reserved for the loop."""
        self.eval()
        x = prompt_ids.clone().unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids.clone()
        if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] == 0:
            raise ValueError("generate accepts one non-empty prompt")
        if not isinstance(max_new, int) or max_new < 0:
            raise ValueError("max_new must be a non-negative integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        temp = float(temperature)

        active_mem = memory_context
        if active_mem is None and memory_model is not None:
            q_emb = self.byte_emb(x[:, -self.max_len :].clamp(min=0, max=255)).mean(
                dim=1
            )[0]
            _, active_mem = memory_model.retrieve(
                q_emb, top_k=top_k_memories or self.memory_top_k
            )

        func_calls: list[str] = []
        generated_count = 0
        while generated_count < max_new:
            ctx_window = x[:, -self.max_len :]
            logits = self.forward(
                ctx_window,
                active_mem,
                memory_model=memory_model if active_mem is None else None,
            )
            assert isinstance(logits, torch.Tensor)
            next_logits = logits[0, -1].clone() / temp
            next_logits[self.LOAD_SPAN:] = -torch.inf
            probs = F.softmax(next_logits, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())
            x = torch.cat([x, torch.tensor([[nxt]], device=x.device)], dim=1)
            generated_count += 1

            if nxt == self.FUNC_CALL and func_handler is not None:
                arg_tokens: list[int] = []
                terminated = False
                for _ in range(min(max(self.max_len // 2, 16), max_new - generated_count)):
                    ctx_window = x[:, -self.max_len :]
                    logits = self.forward(ctx_window, active_mem)
                    assert isinstance(logits, torch.Tensor)
                    nl = logits[0, -1].clone() / temp
                    nl[self.LOAD_SPAN:] = -torch.inf
                    pr = F.softmax(nl, dim=-1)
                    nt = int(torch.multinomial(pr, num_samples=1).item())
                    x = torch.cat([x, torch.tensor([[nt]], device=x.device)], dim=1)
                    generated_count += 1
                    if nt == self.FUNC_END:
                        terminated = True
                        break
                    if nt < self.VOCAB_BYTES:
                        arg_tokens.append(nt)
                if not terminated:
                    # Never dispatch a truncated or unterminated model call.
                    break
                func_name = bytes(arg_tokens).decode("utf-8", errors="replace")
                func_calls.append(func_name)
                result = func_handler(func_name)
                result_bytes = list(f"RESULT:{result}".encode("utf-8"))
                if result_bytes:
                    rb_tensor = torch.tensor(
                        [result_bytes], dtype=torch.long, device=x.device
                    )
                    x = torch.cat([x, rb_tensor], dim=1)

        return x[0], func_calls
