"""Intent- and episodic-context-conditioned autoregressive byte responses.

A bidirectional raw-byte encoder and pointer/generator decoder can copy bytes
from retrieved user text without a tokenizer. The local EOS class 256 and the
0/1 segment indicators are internal model metadata, never output data. This
small model is not a reliable open-domain language model.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .conversation import ConversationTurn
from .conversation_intents import NUM_INTENTS
from .procedural import raw_bytes


def pack_response_context(message: bytes, history: tuple[ConversationTurn, ...] | list[ConversationTurn], *,
                          max_bytes: int = 384) -> tuple[bytes, bytes]:
    """Pack the complete current message and the newest complete turns that fit.

    Segment IDs distinguish current input from untrusted history independently
    of any delimiters a user may include. Returned IDs are 0/1 internal labels.
    """
    message = raw_bytes(message, "message")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 4:
        raise ValueError("max_bytes must be at least four")
    current = b"Q:" + message
    if not message or len(current) > max_bytes:
        raise ValueError("message is empty or exceeds the response byte budget")
    selected: list[bytes] = []
    used = len(current)
    for turn in reversed(history):
        if not isinstance(turn, ConversationTurn):
            raise TypeError("history must contain ConversationTurn objects")
        entry = b"\nU:" + raw_bytes(turn.user, "history user") + b"\nA:" + raw_bytes(turn.assistant, "history assistant")
        if used + len(entry) > max_bytes:
            break
        selected.insert(0, entry); used += len(entry)
    prior = b"".join(selected)
    return current + prior, bytes([0]) * len(current) + bytes([1]) * len(prior)


class ContextualByteDecoder(nn.Module):
    EOS = 256
    IGNORE = -100

    def __init__(self, context_dim: int, hidden_dim: int = 64, *, max_input_bytes: int = 384) -> None:
        super().__init__()
        if not isinstance(context_dim, int) or context_dim < 1 or not isinstance(hidden_dim, int) or hidden_dim < 2 or hidden_dim % 2:
            raise ValueError("positive context_dim and an even hidden_dim >= 2 are required")
        if not isinstance(max_input_bytes, int) or not 4 <= max_input_bytes <= 4096:
            raise ValueError("invalid decoder input byte limit")
        self.context_dim, self.hidden_dim, self.max_input_bytes = context_dim, hidden_dim, max_input_bytes
        self.expected_generator_heads: int | None = None
        self.byte_embedding = nn.Embedding(256, hidden_dim)
        self.segment_embedding = nn.Embedding(2, hidden_dim)
        self.intent_embedding = nn.Embedding(NUM_INTENTS, hidden_dim)
        self.source_encoder = nn.GRU(hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.state_projection = nn.Linear(context_dim + 2 * hidden_dim, hidden_dim)
        self.start = nn.Parameter(torch.zeros(hidden_dim))
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.byte_head = nn.Linear(hidden_dim, 257)
        self.copy_gate = nn.Linear(3 * hidden_dim, 1)
        self.register_buffer("response_trained", torch.zeros((), dtype=torch.bool))

    def _encode(self, context: torch.Tensor, intent: torch.Tensor, source: torch.Tensor,
                segments: torch.Tensor, lengths: torch.Tensor):
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape (batch, context_dim)")
        batch = context.shape[0]
        if source.ndim != 2 or source.shape[0] != batch or not 1 <= source.shape[1] <= self.max_input_bytes:
            raise ValueError("source must contain bounded byte sequences")
        for name, tensor in (("source", source), ("segments", segments), ("intent", intent), ("lengths", lengths)):
            if tensor.dtype not in (torch.int32, torch.int64) or tensor.device != context.device:
                raise TypeError(f"{name} must be an integer tensor on the context device")
        if segments.shape != source.shape or intent.shape != (batch,) or lengths.shape != (batch,):
            raise ValueError("source, segment, intent and length shapes do not match")
        if bool(((source < 0) | (source > 255)).any()) or bool(((segments < 0) | (segments > 1)).any()):
            raise ValueError("source must contain raw bytes and segment IDs must be 0 or 1")
        if bool(((intent < 0) | (intent >= NUM_INTENTS)).any()) or bool(((lengths < 1) | (lengths > source.shape[1])).any()):
            raise ValueError("intent or source length is outside bounds")
        embedded = self.byte_embedding(source.long()) + self.segment_embedding(segments.long())
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, final = self.source_encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=source.shape[1])
        summary = torch.cat((final[-2], final[-1]), dim=-1)
        intent_vector = self.intent_embedding(intent.long())
        initial = torch.tanh(self.state_projection(torch.cat((context, intent_vector, summary), dim=-1))).unsqueeze(0)
        mask = torch.arange(source.shape[1], device=source.device).unsqueeze(0) < lengths.unsqueeze(1)
        return encoded, self.key_projection(encoded), initial, mask

    def _log_distribution(self, states: torch.Tensor, previous: torch.Tensor, encoded: torch.Tensor,
                          keys: torch.Tensor, mask: torch.Tensor, source: torch.Tensor, *,
                          return_attention_scores: bool = False):
        scores = torch.matmul(self.query_projection(states), keys.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, encoded)
        logits = self.byte_head(F.silu(self.output_projection(torch.cat((states, attended), dim=-1))))
        generated = torch.softmax(logits, dim=-1)
        copy_probs = torch.zeros_like(generated)
        indexes = source.long().unsqueeze(1).expand(-1, states.shape[1], -1)
        copy_probs.scatter_add_(2, indexes, attention)
        gate = torch.sigmoid(self.copy_gate(torch.cat((states, attended, previous), dim=-1)))
        mixed = (1.0 - gate) * generated + gate * copy_probs
        log_probs = torch.log(mixed.clamp_min(1e-12))
        return (log_probs, scores) if return_attention_scores else log_probs

    def forward(self, context: torch.Tensor, intent: torch.Tensor, source: torch.Tensor,
                segments: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor, *,
                return_attention_scores: bool = False):
        """Teacher-forced byte/EOS log probabilities; optional training-only attention logits."""
        if targets.ndim != 2 or targets.shape[0] != context.shape[0] or targets.shape[1] < 1:
            raise ValueError("targets must have shape (batch, positive_length)")
        if targets.device != context.device or targets.dtype not in (torch.int32, torch.int64):
            raise TypeError("targets must be integers on the context device")
        if bool(((targets < 0) & (targets != self.IGNORE)).any()) or bool((targets > self.EOS).any()):
            raise ValueError("targets may contain only bytes, EOS and ignore padding")
        encoded, keys, initial, mask = self._encode(context, intent, source, segments, lengths)
        previous_ids = targets[:, :-1].clamp(min=0, max=255).long()
        previous = torch.cat((self.start.view(1, 1, -1).expand(context.shape[0], 1, -1),
                              self.byte_embedding(previous_ids)), dim=1)
        states, _ = self.recurrent(previous, initial)
        return self._log_distribution(states, previous, encoded, keys, mask, source,
                                      return_attention_scores=return_attention_scores)

    @torch.no_grad()
    def generate(self, context: torch.Tensor, intent: torch.Tensor, source: torch.Tensor,
                 segments: torch.Tensor, lengths: torch.Tensor, *, max_bytes: int = 96,
                 temperature: float = 0.0, seed: int | None = None, top_k: int = 8) -> tuple[bytes, bool]:
        if context.shape != (1, self.context_dim):
            raise ValueError("generation requires one context vector")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or not 1 <= max_bytes <= 512:
            raise ValueError("max_bytes must be in 1..512")
        if not math.isfinite(temperature) or temperature < 0 or not 1 <= top_k <= 257:
            raise ValueError("invalid sampling configuration")
        encoded, keys, state, mask = self._encode(context, intent, source, segments, lengths)
        previous = self.start.view(1, 1, -1)
        result = bytearray()
        rng = None
        if seed is not None:
            rng = torch.Generator(device=context.device).manual_seed(int(seed))
        for _ in range(max_bytes + 1):
            states, state = self.recurrent(previous, state)
            log_probs = self._log_distribution(states, previous, encoded, keys, mask, source)[0, 0]
            if temperature == 0:
                token = int(log_probs.argmax().item())
            else:
                top_values, top_indexes = torch.topk(log_probs, top_k)
                probabilities = torch.softmax(top_values / temperature, dim=0)
                token = int(top_indexes[torch.multinomial(probabilities, 1, generator=rng)].item())
            if token == self.EOS:
                return bytes(result), True
            if len(result) == max_bytes:
                return bytes(result), False
            result.append(token)
            previous = self.byte_embedding(torch.tensor([[token]], device=context.device))
        raise AssertionError("unreachable")


def load_contextual_decoder(path: str | Path, *, generator_path: str | Path,
                            device: str | torch.device = "cpu") -> ContextualByteDecoder:
    """Load a trained decoder bound to generator bytes and attention-head count."""
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != 2:
        raise ValueError("unsupported contextual response checkpoint schema")
    actual = hashlib.sha256(Path(generator_path).read_bytes()).hexdigest()
    if state.get("generator_sha256") != actual:
        raise ValueError("contextual decoder and generator checkpoints do not match")
    context_dim, hidden_dim, max_input, heads = (state.get(key) for key in
                                                ("context_dim", "hidden_dim", "max_input_bytes", "generator_n_heads"))
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (context_dim, hidden_dim, max_input, heads)):
        raise ValueError("invalid contextual decoder dimensions")
    if not 1 <= context_dim <= 4096 or not 2 <= hidden_dim <= 2048 or hidden_dim % 2 or not 4 <= max_input <= 4096:
        raise ValueError("invalid contextual decoder dimensions")
    if not 1 <= heads <= context_dim or context_dim % heads:
        raise ValueError("invalid generator attention-head metadata")
    decoder = ContextualByteDecoder(context_dim, hidden_dim, max_input_bytes=max_input).to(device)
    decoder.expected_generator_heads = heads
    decoder.load_state_dict(state["state_dict"], strict=True)
    if not bool(decoder.response_trained):
        raise RuntimeError("contextual response checkpoint is untrained")
    return decoder.eval()
