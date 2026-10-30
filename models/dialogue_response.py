"""Small autoregressive raw-byte response decoder for bounded social dialogue.

The decoder is separate from the Phase 9 edit policy, so training it cannot
change the edit model. Its local EOS class 256 is an internal stop signal; it
is never serialized into the user's byte stream. This is not an open-domain
language model and must not be treated as a source of factual knowledge.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


class ByteDialogueDecoder(nn.Module):
    EOS = 256
    IGNORE = -100

    def __init__(self, context_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = int(context_dim if hidden_dim is None else hidden_dim)
        if context_dim < 1 or hidden_dim < 1:
            raise ValueError("decoder dimensions must be positive")
        self.context_dim = int(context_dim)
        self.hidden_dim = hidden_dim
        # Set by the trusted checkpoint loader; not a learned parameter.
        self.expected_generator_heads: int | None = None
        self.byte_embedding = nn.Embedding(256, hidden_dim)
        self.start = nn.Parameter(torch.zeros(hidden_dim))
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.byte_head = nn.Linear(hidden_dim, 257)
        self.register_buffer("response_trained", torch.zeros((), dtype=torch.bool))

    def forward(self, context: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits; targets contain raw bytes, EOS, and -100 padding."""
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape (batch, context_dim)")
        if targets.ndim != 2 or targets.shape[0] != context.shape[0] or targets.shape[1] < 1:
            raise ValueError("targets must have shape (batch, positive_length)")
        if targets.dtype not in (torch.int32, torch.int64) or targets.device != context.device:
            raise TypeError("targets must be integer IDs on the context device")
        if bool(((targets < 0) & (targets != self.IGNORE)).any()) or bool((targets > self.EOS).any()):
            raise ValueError("targets may contain only bytes, EOS, and loss padding")
        previous = targets[:, :-1].clamp(min=0, max=255).long()
        embedded = self.byte_embedding(previous)
        start = self.start.view(1, 1, -1).expand(context.shape[0], 1, -1)
        teacher_input = torch.cat((start, embedded), dim=1)
        initial = torch.tanh(self.context_projection(context)).unsqueeze(0)
        states, _ = self.recurrent(teacher_input, initial)
        return self.byte_head(states)

    @torch.no_grad()
    def generate(self, context: torch.Tensor, *, max_bytes: int = 96) -> tuple[bytes, bool]:
        """Greedy raw-byte response and whether EOS was reached within the limit."""
        if context.shape != (1, self.context_dim):
            raise ValueError("generation requires one context vector")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        state = torch.tanh(self.context_projection(context)).unsqueeze(0)
        current = self.start.view(1, 1, -1)
        result = bytearray()
        for _ in range(max_bytes + 1):
            states, state = self.recurrent(current, state)
            token = int(self.byte_head(states[:, -1]).argmax(dim=-1).item())
            if token == self.EOS:
                return bytes(result), True
            if len(result) == max_bytes:
                return bytes(result), False
            result.append(token)
            current = self.byte_embedding(torch.tensor([[token]], device=context.device))
        raise AssertionError("unreachable")


def load_dialogue_decoder(path: str | Path, *, generator_path: str | Path,
                          device: str | torch.device = "cpu") -> ByteDialogueDecoder:
    """Load a trained decoder only with the exact generator it was paired with."""
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("unsupported response checkpoint schema")
    expected = state.get("generator_sha256")
    actual = hashlib.sha256(Path(generator_path).read_bytes()).hexdigest()
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("response decoder and generator checkpoints do not match")
    context_dim, hidden_dim = state.get("context_dim"), state.get("hidden_dim")
    heads = state.get("generator_n_heads")
    if not isinstance(context_dim, int) or not isinstance(hidden_dim, int) or not 1 <= context_dim <= 4096 or not 1 <= hidden_dim <= 4096:
        raise ValueError("invalid response checkpoint dimensions")
    if not isinstance(heads, int) or isinstance(heads, bool) or not 1 <= heads <= context_dim or context_dim % heads:
        raise ValueError("invalid generator attention-head metadata")
    decoder = ByteDialogueDecoder(context_dim, hidden_dim).to(device)
    decoder.expected_generator_heads = heads
    decoder.load_state_dict(state["state_dict"], strict=True)
    if not bool(decoder.response_trained):
        raise RuntimeError("response checkpoint is not marked trained")
    return decoder.eval()
