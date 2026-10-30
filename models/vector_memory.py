"""Bounded, owner-scoped vector facts stored in the existing episodic bank.

This is an explicitly structured, lossless bit-vector codec, NOT a learned
semantic summary or a claim of unrestricted memory capacity. At write time a
narrow grammar extracts a value from the user's observed statement. Each dense
64-d vector holds up to eight raw bytes; at most five slots hold forty bytes.
At read time only vectors, not the statement or extracted value, are supplied
to a trained neural byte reader. User IDs are namespaces, not authentication.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .conversation import ConversationTurn, _FLAGS, _KEY, _RECALL, _REMEMBER, _key, _visible_text
from .episodic_conversation import EpisodicConversationMemory, _EPISODE_TYPE
from .memory_model import TIER_EPISODIC
from .procedural import byte_pattern_embedding, raw_bytes


_VECTOR_TYPE = "conversation_vector_fact_v12"
_MY_STATEMENT = re.compile(r"^my\s+" + _KEY + r"\s+is\s+(.+?)[.!?]?$", _FLAGS)
_RECALL_EXTRA = re.compile(r"^(?:what's|what\s+are|what\s+were)\s+my\s+" + _KEY + r"[.!?]?$", _FLAGS)
_WORK_STATEMENT = re.compile(r"^([a-z][a-z0-9 -]{0,31}?)\s+works\s+(?:at|for)\s+(.+?)[.!?]?$", _FLAGS)
_WORK_QUESTION = re.compile(r"^where\s+does\s+([a-z][a-z0-9 -]{0,31}?)\s+work[.!?]?$", _FLAGS)
_WORK_FORGET = re.compile(r"^(?:(?:please|could you|can you)\s+)?(?:forget|delete|remove|erase)\s+"
                          r"([a-z][a-z0-9 -]{0,31}?)['’]s\s+(?:workplace|employer)[.!?]?$", _FLAGS)


def statement_fact(message: bytes) -> tuple[bytes, bytes] | None:
    """Parse only affirmative supported fact forms; no conversational inference."""
    message = raw_bytes(message, "message")
    try:
        text = message.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    match = _MY_STATEMENT.fullmatch(text) or _REMEMBER.fullmatch(text)
    if match is not None:
        value = match.group(2).strip().encode("utf-8")
        return (b"my:" + _key(match.group(1)), value) if _visible_text(value) else None
    match = _WORK_STATEMENT.fullmatch(text)
    if match is not None:
        value = match.group(2).strip().encode("utf-8")
        return (b"person:" + _key(match.group(1)) + b":workplace", value) if _visible_text(value) else None
    return None


def question_topic(message: bytes) -> bytes | None:
    message = raw_bytes(message, "message")
    try:
        text = message.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    match = _RECALL.fullmatch(text) or _RECALL_EXTRA.fullmatch(text)
    if match is not None:
        return b"my:" + _key(match.group(1))
    match = _WORK_QUESTION.fullmatch(text)
    if match is not None:
        return b"person:" + _key(match.group(1)) + b":workplace"
    return None


def workplace_forget_topic(message: bytes) -> bytes | None:
    """Accept only an anchored affirmative workplace-forget command."""
    message = raw_bytes(message, "message")
    try:
        text = message.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    match = _WORK_FORGET.fullmatch(text)
    return b"person:" + _key(match.group(1)) + b":workplace" if match is not None else None


def topic_hash(topic: bytes) -> str:
    return hashlib.sha256(b"vector.fact.topic.v1\0" + raw_bytes(topic, "topic")).hexdigest()


class VectorFactCodec(nn.Module):
    """Encode 1–40 raw bytes into 1–5 dense 64-dimensional vectors.

    Signed bit planes are rotated by a fixed, checkpointed orthogonal matrix.
    Padding is zero, distinct from all raw bytes including NUL. The codec is
    lossless but does not compress, encrypt or understand the value. Writes
    are detached; the learned neural reader is trained from retrieved values.
    """
    DIM = 64
    BYTES_PER_SLOT = 8
    MAX_SLOTS = 5
    MAX_BYTES = BYTES_PER_SLOT * MAX_SLOTS

    def __init__(self, *, seed: int = 120120) -> None:
        super().__init__()
        rng = torch.Generator(device="cpu").manual_seed(seed)
        matrix, _ = torch.linalg.qr(torch.randn(self.DIM, self.DIM, generator=rng))
        self.register_buffer("rotation", matrix.contiguous())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.rotation.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    @torch.no_grad()
    def encode(self, value: bytes | bytearray) -> torch.Tensor:
        value = raw_bytes(value, "fact value")
        if not 1 <= len(value) <= self.MAX_BYTES:
            raise ValueError("vector facts must contain 1..40 raw bytes")
        count = math.ceil(len(value) / self.BYTES_PER_SLOT)
        values = torch.tensor(list(value), dtype=torch.long, device=self.rotation.device)
        shifts = torch.arange(7, -1, -1, dtype=torch.long, device=self.rotation.device)
        bits = (((values[:, None] >> shifts) & 1).to(self.rotation.dtype) * 2) - 1
        padded = torch.zeros((count * self.BYTES_PER_SLOT, 8), dtype=self.rotation.dtype, device=self.rotation.device)
        padded[:len(value)] = bits
        return padded.reshape(count, self.DIM) @ self.rotation


class VectorMemoryReader(nn.Module):
    """Learn to decode answer bytes from actual retrieved dense memory vectors.

    A trainable inverse projection and byte-presence head produce independent
    raw-byte/EOS distributions. The caller supplies no stored episode bytes or
    expected value; query features may modulate the readout. This is a learned
    vector readout, not an autoregressive language model or semantic reasoner.
    """
    EOS = 256
    IGNORE = -100

    def __init__(self, context_dim: int = 64) -> None:
        super().__init__()
        if not isinstance(context_dim, int) or context_dim < 1:
            raise ValueError("context_dim must be positive")
        self.context_dim = context_dim
        self.projection = nn.Linear(VectorFactCodec.DIM, VectorFactCodec.DIM, bias=False)
        self.context_gate = nn.Linear(context_dim, VectorFactCodec.DIM)
        self.presence = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        shifts = torch.arange(7, -1, -1, dtype=torch.long)
        patterns = ((((torch.arange(256)[:, None] >> shifts) & 1).float() * 2) - 1)
        self.register_buffer("byte_patterns", patterns, persistent=False)
        self.register_buffer("reader_trained", torch.zeros((), dtype=torch.bool))

    def forward(self, context: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape (batch, context_dim)")
        if (vectors.ndim != 3 or vectors.shape[0] != context.shape[0] or vectors.shape[2] != VectorFactCodec.DIM
                or not 1 <= vectors.shape[1] <= VectorFactCodec.MAX_SLOTS or vectors.device != context.device):
            raise ValueError("vectors must have shape (batch, 1..5, 64) on the context device")
        if not vectors.is_floating_point() or not bool(torch.isfinite(vectors).all()):
            raise ValueError("memory vectors must be finite floating point")
        if mask is None:
            mask = torch.ones(vectors.shape[:2], dtype=torch.bool, device=context.device)
        if mask.dtype != torch.bool or mask.shape != vectors.shape[:2] or mask.device != context.device:
            raise ValueError("memory mask must be boolean and match batch and slots")
        decoded = self.projection(vectors)
        modulation = 1 + 0.1 * torch.tanh(self.context_gate(context)).unsqueeze(1)
        decoded = decoded * modulation * mask.unsqueeze(-1)
        decoded = decoded.reshape(context.shape[0], vectors.shape[1] * VectorFactCodec.BYTES_PER_SLOT, 8)
        bit_logits = decoded * F.softplus(self.logit_scale)
        signs = self.byte_patterns.to(device=context.device, dtype=context.dtype)
        log_bits = F.logsigmoid(bit_logits.unsqueeze(-2) * signs).sum(dim=-1)
        feature = decoded.square().mean(dim=-1, keepdim=True)
        present = self.presence(feature).squeeze(-1)
        byte_log = log_bits + F.logsigmoid(present).unsqueeze(-1)
        eos_log = F.logsigmoid(-present).unsqueeze(-1)
        return torch.cat((byte_log, eos_log), dim=-1)

    @torch.no_grad()
    def generate(self, context: torch.Tensor, vectors: torch.Tensor) -> tuple[bytes, bool]:
        if context.shape != (1, self.context_dim) or vectors.ndim != 3 or vectors.shape[0] != 1:
            raise ValueError("generation requires one context and its memory vectors")
        probabilities = self(context, vectors)[0]
        output = bytearray()
        for scores in probabilities:
            byte = int(scores.argmax().item())
            if byte == self.EOS:
                return bytes(output), bool(output)
            output.append(byte)
        # A completely full final slot has no padding/EOS byte.
        return bytes(output), bool(output)


def load_vector_reader(path: str | Path, *, generator_path: str | Path) -> tuple[VectorFactCodec, VectorMemoryReader]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    actual = hashlib.sha256(Path(generator_path).read_bytes()).hexdigest()
    if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("generator_sha256") != actual:
        raise ValueError("vector reader checkpoint is incompatible with this generator")
    codec = VectorFactCodec()
    codec.load_state_dict(state["codec_state_dict"], strict=True)
    if codec.fingerprint != state.get("codec_fingerprint"):
        raise ValueError("vector codec fingerprint mismatch")
    reader = VectorMemoryReader(state["context_dim"])
    reader.load_state_dict(state["reader_state_dict"], strict=True)
    if not bool(reader.reader_trained):
        raise ValueError("vector reader checkpoint is not trained")
    return codec.eval(), reader.eval()


@dataclass(frozen=True)
class RetrievedVectorFact:
    vectors: torch.Tensor  # (1..5, 64); no raw value or transcript bytes
    slots: tuple[int, ...]


class VectorConversationMemory(EpisodicConversationMemory):
    """Store old episodes and durable, independently retrieved vector facts."""

    def __init__(self, memory, codec: VectorFactCodec, path=None) -> None:
        if memory.d_model != VectorFactCodec.DIM or not isinstance(codec, VectorFactCodec):
            raise ValueError("vector memory requires a width-64 bank and codec")
        self.codec = codec.eval()
        super().__init__(memory, path)

    def _write_fact(self, user_id: bytes, source: bytes, topic: bytes, value: bytes) -> int:
        scope = self.scope(user_id)
        vectors = self.codec.encode(value).to(self.memory.episodic.keys.device)
        hashed = topic_hash(topic)
        metadata_base = {"type": _VECTOR_TYPE, "person": scope, "topic_hash": hashed,
                         "codec_hash": self.codec.fingerprint, "modality": "text", "no_promote": True}
        self.memory.validate_holdout_write(source, value, metadata=metadata_base)
        self.memory.delete_episodic_entries(person=scope, event_type=_VECTOR_TYPE, metadata_filter={"topic_hash": hashed})
        content = byte_pattern_embedding(topic, self.memory.d_model, device=self.memory.episodic.keys.device)
        for index, vector in enumerate(vectors):
            if bool(self.memory.episodic.used.all()):
                removed = self.memory.delete_episodic_entries(person=scope, event_type=_EPISODE_TYPE, limit=1)
                if not removed:
                    raise ValueError("episodic memory is full; vector facts from other users are not evicted")
            metadata = dict(metadata_base, chunk=index, chunks=len(vectors))
            self.memory.write(content, metadata, tier=TIER_EPISODIC, raw_input=source, raw_output=value,
                              value_override=vector)
        return len(vectors)

    def record_episode(self, user_id: bytes, turn: ConversationTurn) -> None:
        if not isinstance(turn, ConversationTurn):
            raise TypeError("turn must be a ConversationTurn")
        fact = statement_fact(turn.user)
        # Unsupported long values remain ordinary recent episodes, not
        # silently truncated vector facts.
        supported = fact is not None and 1 <= len(fact[1]) <= self.codec.MAX_BYTES
        def write() -> None:
            if supported:
                assert fact is not None
                self._write_fact(user_id, turn.user, fact[0], fact[1])
            self._write_episode(user_id, turn)
        self._mutate_banks(write)

    @torch.no_grad()
    def retrieve_fact(self, user_id: bytes, question: bytes) -> RetrievedVectorFact | None:
        topic = question_topic(question)
        if topic is None:
            return None
        scope = self.scope(user_id)
        device = self.memory.episodic.keys.device
        query = byte_pattern_embedding(topic, self.memory.d_model, device=device)
        with self._lock, self.memory.read_only():
            entries, _ = self.memory.retrieve(query, top_k=self.codec.MAX_SLOTS, person_filter=scope,
                                              tier_filter=TIER_EPISODIC,
                                              metadata_filter={"type": _VECTOR_TYPE, "topic_hash": topic_hash(topic),
                                                               "codec_hash": self.codec.fingerprint})
        if not entries:
            return None
        ordered = sorted(entries, key=lambda item: item["metadata"].get("chunk", -1))
        expected = ordered[0]["metadata"].get("chunks")
        if (not isinstance(expected, int) or not 1 <= expected <= self.codec.MAX_SLOTS or len(ordered) != expected
                or [entry["metadata"].get("chunk") for entry in ordered] != list(range(expected))
                or any(entry["metadata"].get("chunks") != expected for entry in ordered)):
            raise RuntimeError("incomplete or malformed vector fact; refusing to decode")
        return RetrievedVectorFact(torch.stack([entry["value"].detach().clone() for entry in ordered]),
                                   tuple(int(entry["slot"]) for entry in ordered))

    def forget_topic_and_purge_history(self, user_id: bytes, topic: bytes) -> tuple[int, int]:
        """Delete a scoped vector topic and this owner's ordinary transcript."""
        scope, hashed = self.scope(user_id), topic_hash(topic)
        def remove() -> tuple[int, int]:
            vectors = self.memory.delete_episodic_entries(person=scope, event_type=_VECTOR_TYPE,
                                                           metadata_filter={"topic_hash": hashed})
            episodes = self.memory.delete_episodic_entries(person=scope, event_type=_EPISODE_TYPE)
            return vectors, episodes
        return self._mutate_banks(remove)

    def forget_and_purge_history(self, user_id: bytes, key: bytes) -> tuple[int, int]:
        frame, scope = self._frame(user_id, key), self.scope(user_id)
        hashed = topic_hash(b"my:" + key)
        def remove() -> tuple[int, int]:
            facts = self.memory.delete_procedures(frame)
            vectors = self.memory.delete_episodic_entries(person=scope, event_type=_VECTOR_TYPE,
                                                           metadata_filter={"topic_hash": hashed})
            episodes = self.memory.delete_episodic_entries(person=scope, event_type=_EPISODE_TYPE)
            return facts + vectors, episodes
        return self._mutate_banks(remove)
