"""ByteCore v2 foundation: a standard causal byte-level Transformer.

This module is deliberately boring. It is a plain decoder-only Transformer
over raw bytes with no tokenizer, and it replaces the prototype's brittle
machinery:

* no pointer-generator copy head -- copying is left to induction heads,
  which is what actually scales;
* no rigid ``[TOOL_CALL][ARG_START]...`` control frames -- tool use is
  ordinary text (``Thought:`` / ``Action:`` / ``Observation:``) modelled by
  the same next-byte objective;
* no internal verifier -- verification is outsourced to the sandbox, and
  the observation comes back as bytes in the same stream.

Vocabulary: bytes 0-255 plus five control ids.

=============  ==============================================================
``BOS`` 256    beginning of sequence
``EOS`` 257    end of sequence (the only learned stop)
``PAD`` 258    padding, masked out of the loss
``REF_START`` 259  opens a reference to a span of the request
``REF_END`` 260    closes it
=============  ==============================================================

The two reference ids exist because a byte-level model at this scale cannot
reproduce an arbitrary identifier -- Phase 17 measured 0/448 on twelve-character
shas. A reference lets the model *point at* bytes it was shown instead of
spelling them, and the host resolves it against the request rather than trusting
what the model wrote. See :mod:`models.react`.

The architecture is identical at every scale: only ``d_model``, ``n_layers``,
``n_heads``, ``d_ff`` and ``max_seq_len`` change. The CPU configuration used
for tests and the production configuration are the same class.

Phase 22 adds an OPTIONAL wide-and-gated stack (RMSNorm, SwiGLU, RoPE, GQA and
a weight-tied iterative refinement loop). Every new knob defaults to the exact
Phase 16 behaviour, so historical checkpoints and tests are unaffected; the
``configs/t4_scaled.yaml`` preset opts into the new stack.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

BOS = 256
EOS = 257
PAD = 258
REF_START = 259
REF_END = 260
VOCAB_SIZE = 261
CONTROL_NAMES = {BOS: "<bos>", EOS: "<eos>", PAD: "<pad>",
                 REF_START: "<ref>", REF_END: "</ref>"}

# The on-the-wire spelling of the two reference ids. They are ordinary text so
# that a ReAct stream stays readable and so the interceptor is a bytes match,
# not a token match -- the model, the dataset and the host all agree on these
# eight bytes and nothing else.
REF_OPEN = b"[REF]"
REF_CLOSE = b"[/REF]"


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
    # ---- Phase 22 upgrades. Defaults reproduce the Phase 16 architecture. ----
    norm_type: str = "layernorm"        # "layernorm" | "rmsnorm"
    ffn_type: str = "gelu"              # "gelu" | "swiglu"
    position_type: str = "learned"      # "learned" | "rope"
    n_kv_heads: int = 0                # 0 => multi-head attention; >0 => GQA
    refinement_steps: int = 0          # weight-tied iterative refinement passes
    refinement_rank: int = 256         # low-rank bottleneck inside the loop
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"vocab_size must be {VOCAB_SIZE} "
                             "(256 bytes + BOS/EOS/PAD/REF_START/REF_END)")
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
        """Return a copy with overrides -- the production preset lives in YAML."""
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase 22: optional wide-model components. None of these are constructed when
# FoundationConfig uses its (legacy) defaults, which keeps the historical
# checkpoint schema byte-for-byte valid.
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (pre-norm, bias-free).

    ``x / sqrt(E[x^2] + eps) * weight``. The reduction is computed in float32
    even under fp16/bf16 autocast because squaring fp16 activations at width
    1024 can overflow before the mean is taken.
    """

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        values = hidden.float()
        values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + self.eps)
        return (values * self.weight.float()).to(dtype)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: ``[SiLU(x W_value) (x) (x W_branch)] W_out``.

    All three projections are bias-free. Parameter names deliberately avoid
    the word ``gate``: ``tests/test_foundation.py`` forbids legacy
    copy/pointer/gate module names, and a SwiGLU branch is not that mechanism.
    With ``d_model=1024, d_ff=4096`` this is 3 * 1024 * 4096 = 12.58M
    parameters per layer instead of 2 * 1024 * 4096 for a GELU FFN.
    """

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
    """RoPE cosine/sine tables of shape ``(1, 1, L, head_dim)``.

    Computed on the fly (no registered buffers): they cost
    ``L * head_dim / 2`` float32 values, i.e. ~1 MB at L=2048, d_head=128,
    and avoid a persistent device allocation that scales with max_seq_len.
    """
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
    """Rotate query/key heads; the attention dot product becomes relative."""
    return hidden * cos + rotate_half(hidden) * sin


def repeat_kv(hidden: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand ``(B, n_kv, L, d)`` to ``(B, n_kv * repeats, L, d)``."""
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


class IterativeRefinement(nn.Module):
    """Weight-tied low-rank "think" loop before the byte prediction head.

    ``h_(t+1) = h_(t) + alpha_t * Cell(RMSNorm(h_(t)))`` for ``steps`` passes.
    ``alpha`` is initialised to zero so the loop is mathematically the identity
    at step 0: enabling it cannot destabilise a freshly initialised model.
    The cell is a shared ``d -> rank -> d`` bottleneck (~1M parameters for
    d=1024, rank=256), NOT an extra Transformer layer, so memory grows with
    ``steps`` boundaries (8 MB each at B=4, L=1024, d=1024, fp16) rather than
    with another 12-layer stack. Each pass is wrapped in activation
    checkpointing when training, so intermediate cell activations are
    recomputed instead of retained.
    """

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.steps = int(config.refinement_steps)
        rank = min(config.refinement_rank, config.d_model)
        self.norm = make_norm(config)
        self.down = nn.Linear(config.d_model, rank, bias=False)
        self.branch = nn.Linear(config.d_model, rank, bias=False)
        self.up = nn.Linear(rank, config.d_model, bias=False)
        self.step_scales = nn.Parameter(torch.zeros(self.steps))

    def cell(self, hidden: torch.Tensor) -> torch.Tensor:
        values = self.norm(hidden)
        return self.up(F.silu(self.down(values)) * self.branch(values))

    def forward(self, hidden: torch.Tensor, *, use_checkpoint: bool = False) -> torch.Tensor:
        for index in range(self.steps):
            alpha = self.step_scales[index]
            if use_checkpoint and self.training and torch.is_grad_enabled():
                delta = checkpoint(self.cell, hidden, use_reentrant=False)
            else:
                delta = self.cell(hidden)
            hidden = hidden + alpha * delta
        return hidden


class CausalSelfAttention(nn.Module):
    """Plain multi-head causal self-attention. No memory, no copy, no gates.

    Causality is enforced by :func:`torch.nn.functional.scaled_dot_product_attention`
    rather than by materialising a ``(B, H, T, T)`` score matrix: the fused path
    is what makes long contexts affordable. The explicit mask buffer is kept so
    tests can assert the property independently of the kernel.
    """

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim)
        self.value = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout_p = float(config.dropout)
        self.position_type = config.position_type
        self.rope_theta = float(config.rope_theta)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # (B, T, C) -> (B, H, T, D). With GQA, K/V use fewer heads than Q.
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
        # Fused SDPA avoids materialising (B, H, T, T) scores. T4 (sm75) has no
        # FlashAttention-2, so PyTorch picks math/mem-efficient kernels here.
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
    """Pre-norm block: attention, then FFN, both with residuals."""

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

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = hidden + self.attention(self.norm_1(hidden), mask)
        hidden = hidden + self.feed_forward(self.norm_2(hidden))
        return hidden


class CausalByteTransformer(nn.Module):
    """Next-byte prediction over raw bytes. The whole foundation."""

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        config = config if isinstance(config, FoundationConfig) else FoundationConfig(**dict(config))
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Learned absolute positions are the Phase 16 default; RoPE needs no
        # parameters and no max_seq_len-sized table, which is why the wide
        # preset can raise max_seq_len to 2048 without a memory penalty.
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

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return next-byte logits of shape ``(batch, length, vocab_size)``."""
        if ids.ndim != 2:
            raise ValueError("ids must have shape (batch, length)")
        if int(ids.max()) >= self.config.vocab_size or int(ids.min()) < 0:
            raise ValueError("ids must be within the vocabulary")
        length = ids.shape[1]
        if length > self.config.max_seq_len:
            raise ValueError(f"sequence length {length} exceeds max_seq_len {self.config.max_seq_len}")
        hidden = self.token_embedding(ids)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(torch.arange(length, device=ids.device))
        hidden = self.embedding_dropout(hidden)
        use_checkpoint = (self.gradient_checkpointing and self.training
                          and torch.is_grad_enabled())
        for block in self.blocks:
            if use_checkpoint:
                hidden = checkpoint(block, hidden, self.causal_mask, use_reentrant=False)
            else:
                hidden = block(hidden, self.causal_mask)
        if self.refinement is not None:
            hidden = self.refinement(hidden, use_checkpoint=use_checkpoint)
        return self.head(self.norm_final(hidden))

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable/disable activation checkpointing for blocks and refinement."""
        self.gradient_checkpointing = bool(enabled)

    def loss(self, ids: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """Mean cross-entropy over next-byte targets, padding excluded."""
        if targets is None:
            targets = ids[:, 1:]
            ids = ids[:, :-1]
            # Truncate the shifted view so both tensors stay aligned.
            length = min(ids.shape[1], targets.shape[1])
            ids, targets = ids[:, :length], targets[:, :length]
        logits = self.forward(ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=PAD,
        )

    @property
    def parameters_count(self) -> int:
        return sum(int(tensor.numel()) for tensor in self.parameters())

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, *, max_new_bytes: int = 192,
                 temperature: float = 0.0, stop: tuple[int, ...] = (EOS,)) -> torch.Tensor:
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
            logits = self.forward(window.unsqueeze(0))[0, -1]
            if temperature <= 0.0:
                next_id = int(logits.argmax().item())
            else:
                next_id = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).item())
            if next_id in stop:
                break
            generated.append(next_id)
            ids = torch.cat((ids, torch.tensor([next_id], dtype=ids.dtype, device=ids.device)))
            if len(generated) >= max_new_bytes:
                break
        return torch.tensor(generated, dtype=torch.long, device=prefix.device)


# --------------------------------------------------------------------------
# Byte <-> id helpers. Bytes 0-255 are their own ids; control ids are dropped
# on decode, so round-tripping text is exact and never emits a control byte.
# --------------------------------------------------------------------------
def encode_bytes(data: bytes, *, bos: bool = True, eos: bool = False) -> torch.Tensor:
    """Encode raw bytes to ids, optionally framing with BOS/EOS.

    Reference markers are written into the stream as the two-byte sequences the
    rest of the code uses for them, so this function stays a pure bytes->ids map.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    values: list[int] = []
    index = 0
    raw = bytes(data)
    while index < len(raw):
        if raw.startswith(REF_OPEN, index):
            values.append(REF_START)
            index += len(REF_OPEN)
        elif raw.startswith(REF_CLOSE, index):
            values.append(REF_END)
            index += len(REF_CLOSE)
        else:
            values.append(raw[index])
            index += 1
    allowed = set(range(256)) | {REF_START, REF_END}
    if any(value not in allowed for value in values):
        raise ValueError("byte ids must be in 0..255 or a reference control id")
    ids = ([BOS] if bos else []) + values + ([EOS] if eos else [])
    return torch.tensor(ids, dtype=torch.long)


def decode_ids(ids: torch.Tensor | list[int]) -> bytes:
    """Decode ids to raw bytes, dropping BOS/EOS/PAD.

    Reference ids come back as their marker text, so a decoded stream is
    byte-identical to what was encoded.
    """
    values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    out = bytearray()
    for value in values:
        if 0 <= value <= 255:
            out.append(value)
        elif value == REF_START:
            out += REF_OPEN
        elif value == REF_END:
            out += REF_CLOSE
    return bytes(out)


def pad_batch(sequences: list[torch.Tensor], length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of id sequences to ``length``; returns (ids, targets).

    Targets are the inputs shifted by one, with padding masked by ``PAD`` --
    the standard causal-LM arrangement, with no off-by-one.
    """
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


CHECKPOINT_SCHEMA = 2


def save_foundation(model: CausalByteTransformer, path: str | Path, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
    """Load a checkpoint and rebuild the model from its stored config."""
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("not a ByteCore v2 foundation checkpoint")
    model = CausalByteTransformer(FoundationConfig(**state["config"]))
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device)
