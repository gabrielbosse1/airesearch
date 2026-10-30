"""Clean memory integration for the ByteCore v2 foundation.

Memory enters the model as **prefix tokens** and nothing else:

    [MEM] my:favorite_color = blue
    [MEM] person:alice:workplace = Acme Labs
    Request: What is my favorite color?
    Thought: ...

Why a prefix and not a bespoke side-channel:

* it is ordinary bytes, so the transformer attends over retrieved facts with
  the same causal self-attention it uses everywhere -- the injection is
  differentiable with respect to the embeddings and the whole stack;
* it is independent of ``d_model``: nothing about the mechanism has to be
  re-tuned when the model is scaled;
* it degrades gracefully: with no retrieval the sequence is simply shorter.

Retrieval itself stays gradient-free. Persistent memory writes in this codebase
are deliberately not differentiable (a write is a storage event, not a
gradient update), so the retrieved entries arrive as bytes, and what flows
through the network afterwards is fully differentiable.

The retriever is injected, so the foundation depends on no particular store.
:class:`VectorFactRetriever` adapts Phase 12's vector facts -- the same store
and the same trained readout that score 10/10 -- to that interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch

MEMORY_MARK = b"[MEM]"
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class MemoryEntry:
    """One retrieved fact: where it came from and the bytes it carries."""

    topic: bytes
    value: bytes
    source: str = "vector"
    score: float = 1.0


class PrefixMemory:
    """Retrieve top-K entries and render them as prefix bytes."""

    def __init__(self, retriever: Callable[[bytes], Sequence[MemoryEntry]], *,
                 top_k: int = DEFAULT_TOP_K, marker: bytes = MEMORY_MARK,
                 max_entry_bytes: int = 96) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_entry_bytes < 1:
            raise ValueError("max_entry_bytes must be positive")
        self.retriever = retriever
        self.top_k = top_k
        self.marker = marker
        self.max_entry_bytes = max_entry_bytes

    def retrieve(self, query: bytes) -> list[MemoryEntry]:
        """Top-K entries for a query; an absent or failing store yields none."""
        if not isinstance(query, (bytes, bytearray)):
            raise TypeError("query must be bytes")
        try:
            entries = list(self.retriever(bytes(query)) or [])
        except Exception:
            return []  # retrieval must never break generation
        entries = [entry for entry in entries if entry.value][: self.top_k]
        return entries

    def render(self, entries: Iterable[MemoryEntry]) -> bytes:
        """Render entries as ``[MEM] topic = value`` lines."""
        lines: list[bytes] = []
        for entry in entries:
            value = entry.value[: self.max_entry_bytes]
            lines.append(self.marker + b" " + entry.topic + b" = " + value)
        return b"\n".join(lines)

    def prefix(self, query: bytes) -> bytes:
        """Retrieve and render in one call -- what the loop actually uses."""
        return self.render(self.retrieve(query))


class ScriptedRetriever:
    """Deterministic retriever for tests: a topic -> value mapping."""

    def __init__(self, facts: dict[bytes, bytes] | None = None) -> None:
        self.facts = dict(facts or {})

    def __call__(self, query: bytes) -> list[MemoryEntry]:
        topic = _topic_of(query)
        if topic is not None and topic in self.facts:
            return [MemoryEntry(topic=topic, value=self.facts[topic], source="scripted")]
        return []


def _topic_of(query: bytes) -> bytes | None:
    """Pull a topic out of a synthetic question; used by the scripted store."""
    import re

    text = query.decode("utf-8", "replace").strip().lower().rstrip("?")
    patterns = (r"^what is my (.+)$", r"^where does (.+) work$", r"^when is the (.+) project due$",
                r"^what port does the (.+) server use$", r"^who owns the (.+) device$",
                r"^who leads the (.+) team$")
    mapping = ("my:{0}", "person:{0}:workplace", "project:{0}:deadline",
               "server:{0}:port", "device:{0}:owner", "team:{0}:lead")
    for pattern, template in zip(patterns, mapping):
        match = re.match(pattern, text)
        if match:
            return template.format(match.group(1).replace(" ", "_")).encode("utf-8")
    return None


class VectorFactRetriever:
    """Phase 12 vector facts -> prefix entries, read by the trained readout.

    The exact-match fact for the query comes first; the remaining slots are
    filled from topics observed during the session (most recent first), so the
    prefix can carry up to ``top_k`` entries when several facts are known.
    """

    def __init__(self, generator, memory, reader, *, user_id: bytes = b"default",
                 top_k: int = DEFAULT_TOP_K) -> None:
        self.generator = generator
        self.memory = memory
        self.reader = reader
        self.user_id = user_id
        self.top_k = top_k
        self.known: list[tuple[bytes, bytes]] = []  # (topic, question) in write order

    def observe(self, topic: bytes, question: bytes) -> None:
        """Record that a topic exists, so it can fill a prefix slot."""
        pair = (topic, question)
        if pair not in self.known:
            self.known.append(pair)
            self.known = self.known[-8:]

    def _read(self, question: bytes) -> tuple[bytes, bytes] | None:
        from models.conversation import pack_conversation

        model = self.generator
        payload = pack_conversation(question, (), model.max_len)
        ids = torch.tensor([list(payload)], dtype=torch.long)
        lengths = torch.tensor([len(payload)], dtype=torch.long)
        focus = torch.tensor([2], dtype=torch.long)
        base = model.conversation_features(ids, lengths, focus_start=focus)
        try:
            retrieved = self.memory.retrieve_fact(self.user_id, question)
        except (ValueError, RuntimeError):
            return None
        if retrieved is None:
            return None
        vectors = retrieved.vectors.unsqueeze(0).to(dtype=base.dtype)
        context = model.conversation_features(ids, lengths, focus_start=focus, memory_context=vectors)
        value, complete = self.reader.generate(context, vectors)
        if not complete or not value:
            return None
        return topic_of_question(question), value

    def __call__(self, query: bytes) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        seen: set[bytes] = set()
        candidates: list[bytes] = [bytes(query)]
        for topic, question in reversed(self.known):
            if question not in candidates:
                candidates.append(question)
        for question in candidates[: self.top_k]:
            read = self._read(question)
            if read is None:
                continue
            topic, value = read
            if not topic or topic in seen:
                continue
            seen.add(topic)
            entries.append(MemoryEntry(topic=topic, value=value, source="vector_reader",
                                       score=1.0 if question == query else 0.5))
        return entries


def topic_of_question(question: bytes) -> bytes:
    """Topic key for a supported question form, or empty bytes."""
    return _topic_of(question) or b""


def decode_vectors(vectors: torch.Tensor, rotation: torch.Tensor) -> bytes:
    """Exact inverse of :meth:`models.vector_memory.VectorFactCodec.encode`.

    The codec is a signed bit packing followed by an orthogonal rotation, so the
    inverse is the transpose -- a linear, differentiable operation. It exists to
    *verify* the store is lossless and that the learned readout agrees with it;
    the runtime path uses the trained reader, which is what the 10/10 benchmark
    measures.
    """
    if vectors.ndim != 2 or vectors.shape[1] != rotation.shape[0]:
        raise ValueError("vectors must have shape (slots, 64)")
    flat = (vectors @ rotation.T).reshape(-1, 8)
    bits = (flat > 0).to(torch.uint8)
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.long)
    values = (bits.to(torch.long) * weights).sum(dim=-1)
    # Padding rows are exactly zero before the rotation and only numerically
    # zero after it, so a magnitude threshold -- not an equality test -- decides
    # where the stored value ends.
    present = torch.nonzero(flat.abs().max(dim=-1).values >= 0.25).flatten()
    if present.numel() == 0:
        return b""
    end = int(present[-1].item()) + 1
    return bytes(values[:end].tolist())


@dataclass
class MemoryPrefixReport:
    """What a prefix lookup did -- reported, never silently assumed."""

    query: bytes
    entries: list[MemoryEntry] = field(default_factory=list)
    prefix: bytes = b""

    @property
    def count(self) -> int:
        return len(self.entries)

    def as_dict(self) -> dict:
        return {"query": self.query.decode("utf-8", "replace"),
                "entries": [{"topic": entry.topic.decode("utf-8", "replace"),
                             "value": entry.value.decode("utf-8", "replace"),
                             "source": entry.source} for entry in self.entries],
                "prefix": self.prefix.decode("utf-8", "replace")}
