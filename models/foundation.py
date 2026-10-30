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

Vocabulary: bytes 0-255 plus three control ids.

===========  ==========================================
``BOS`` 256  beginning of sequence
``EOS`` 257  end of sequence (the only learned stop)
``PAD`` 258  padding, masked out of the loss
===========  ==========================================

The architecture is identical at every scale: only ``d_model``, ``n_layers``,
``n_heads``, ``d_ff`` and ``max_seq_len`` change. The CPU configuration used
for tests and the production configuration are the same class.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

BOS = 256
EOS = 257
PAD = 258
VOCAB_SIZE = 259
CONTROL_NAMES = {BOS: "<bos>", EOS: "<eos>", PAD: "<pad>"}


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

    def __post_init__(self) -> None:
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"vocab_size must be {VOCAB_SIZE} (256 bytes + BOS/EOS/PAD)")
        for name in ("d_model", "n_layers", "n_heads", "d_ff", "max_seq_len"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

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
        self.head_dim = config.d_model // config.n_heads
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout_p = float(config.dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # (B, T, C) -> (B, H, T, D)
        batch, length, _ = hidden.shape
        shape = (batch, length, self.n_heads, self.head_dim)
        q = self.query(hidden).view(shape).transpose(1, 2)
        k = self.key(hidden).view(shape).transpose(1, 2)
        v = self.value(hidden).view(shape).transpose(1, 2)
        context = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.n_heads * self.head_dim)
        return self.output(context)


class TransformerBlock(nn.Module):
    """Pre-norm block: attention, then FFN, both with residuals."""

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        # Learned absolute positions: this is a standard GPT-style baseline, so
        # the usable context is capped at max_seq_len. Documented, not hidden.
        mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)
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
        positions = torch.arange(length, device=ids.device)
        hidden = self.token_embedding(ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, self.causal_mask)
        return self.head(self.norm_final(hidden))

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
    """Encode raw bytes to ids, optionally framing with BOS/EOS."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    values = list(data)
    if any(not 0 <= value <= 255 for value in values):
        raise ValueError("byte ids must be in 0..255")
    ids = ([BOS] if bos else []) + values + ([EOS] if eos else [])
    return torch.tensor(ids, dtype=torch.long)


def decode_ids(ids: torch.Tensor | list[int]) -> bytes:
    """Decode ids to raw bytes, dropping BOS/EOS/PAD."""
    values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    return bytes(value for value in values if 0 <= value <= 255)


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
