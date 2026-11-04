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

The retriever is injected, so the foundation depends on no particular store:
any callable ``bytes -> Sequence[MemoryEntry]`` works, and
:class:`ScriptedRetriever` is the deterministic one used by tests. Phase 22/23
ships no learned retriever; the hierarchical store in
:mod:`models.memory_model` is read through its own tensor API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


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


def topic_of_question(question: bytes) -> bytes:
    """Topic key for a supported question form, or empty bytes."""
    return _topic_of(question) or b""


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
