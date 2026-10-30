"""Bounded, auditable synthetic-training replay; never stores user transcripts.

Sampling is reproducible with the saved local RNG state. A complete batch is
preflighted before insertion, and file replacement is atomic on ordinary OS
errors. Exact holdout guards do not prove semantic novelty or audit weights.
"""
from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from models.conversation_intents import NUM_INTENTS
from models.holdout import HoldoutGuard, HoldoutViolation
from models.procedural import raw_bytes


@dataclass(frozen=True)
class ReplayExample:
    message: bytes
    label: int
    phase: str
    task: str = "intent"
    output: bytes | None = None
    context: tuple[bytes, ...] = ()
    style: int | None = None


class ReplayBuffer:
    """FIFO-capacity replay with independent RNG and mandatory save preflight."""

    def __init__(self, capacity: int = 1000, *, seed: int = 0, guard: HoldoutGuard | None = None) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or not 1 <= capacity <= 1_000_000:
            raise ValueError("capacity must be a positive bounded integer")
        if guard is not None and not isinstance(guard, HoldoutGuard):
            raise TypeError("guard must be a HoldoutGuard")
        self.capacity = capacity
        self.guard = guard
        self._items: list[ReplayExample] = []
        self._rng = random.Random(seed)

    @property
    def buffer(self) -> tuple[ReplayExample, ...]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def _check(self, item: ReplayExample) -> ReplayExample:
        if not isinstance(item, ReplayExample):
            raise TypeError("replay entries must be ReplayExample instances")
        message = raw_bytes(item.message, "replay message")
        if not message or len(message) > 4096:
            raise ValueError("replay messages must contain 1..4096 raw bytes")
        if not isinstance(item.label, int) or isinstance(item.label, bool) or not 0 <= item.label < NUM_INTENTS:
            raise ValueError("invalid replay intent label")
        if not isinstance(item.phase, str) or not item.phase or len(item.phase) > 64 or not item.phase.isascii():
            raise ValueError("invalid replay provenance phase")
        if item.task not in {"intent", "vector", "diverse_response", "contextual_response"}:
            raise ValueError("invalid replay task")
        output = None if item.output is None else raw_bytes(item.output, "replay output")
        if output is not None and (not output or len(output) > 4096):
            raise ValueError("replay outputs must contain 1..4096 bytes")
        if item.task != "intent" and output is None:
            raise ValueError("non-intent replay requires a supervised output")
        if not isinstance(item.context, tuple) or len(item.context) > 16:
            raise ValueError("replay context must be a bounded tuple of raw bytes")
        context = tuple(raw_bytes(raw, "replay context") for raw in item.context)
        if any(len(raw) > 4096 for raw in context):
            raise ValueError("replay context field exceeds byte limit")
        if item.style is not None and (not isinstance(item.style, int) or isinstance(item.style, bool) or not 0 <= item.style < 1024):
            raise ValueError("invalid replay style")
        if self.guard is not None:
            self.guard.validate(message, output, metadata={"source_phase": item.phase, "split": "train", "task": item.task},
                                extra=context)
        return ReplayExample(message, int(item.label), item.phase, item.task, output, context, item.style)

    def add(self, examples: Iterable[ReplayExample]) -> None:
        checked = [self._check(example) for example in examples]
        # Whole batch is checked before the live buffer is modified.
        self._items = (self._items + checked)[-self.capacity:]

    def sample(self, batch_size: int, *, task: str | None = None) -> list[ReplayExample]:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 0:
            raise ValueError("batch_size must be a non-negative integer")
        eligible = self._items if task is None else [row for row in self._items if row.task == task]
        return self._rng.sample(eligible, min(batch_size, len(eligible)))

    def save(self, path: str | Path) -> None:
        if self.guard is None:
            raise HoldoutViolation("persistent replay requires a holdout guard")
        # Revalidate before persistence; do not trust previous caller state.
        items = [self._check(item) for item in self._items]
        state = {"schema_version": 1, "capacity": self.capacity, "guard_fingerprint": self.guard.fingerprint,
                 "examples": [{"message": row.message, "label": row.label, "phase": row.phase, "task": row.task,
                               "output": row.output, "context": row.context, "style": row.style} for row in items],
                 "rng_state": self._rng.getstate()}
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".replay-", suffix=".pt", dir=str(destination.parent))
        os.close(fd)
        try:
            torch.save(state, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path, *, guard: HoldoutGuard) -> ReplayBuffer:
        if not isinstance(guard, HoldoutGuard):
            raise TypeError("loading replay requires an active holdout guard")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("unsupported replay checkpoint")
        if state.get("guard_fingerprint") != guard.fingerprint:
            raise HoldoutViolation("replay checkpoint uses a different holdout guard")
        result = cls(state["capacity"], guard=guard)
        examples = state.get("examples")
        if not isinstance(examples, list) or len(examples) > result.capacity:
            raise ValueError("invalid replay checkpoint size")
        result.add(ReplayExample(row["message"], row["label"], row["phase"], row.get("task", "intent"),
                                 row.get("output"), tuple(row.get("context", ())), row.get("style")) for row in examples)
        result._rng.setstate(state["rng_state"])
        return result


def mixed_examples(new_examples: list[ReplayExample], replay: ReplayBuffer, *, batch_size: int,
                   rng: random.Random, old_fraction: float = 0.30) -> list[ReplayExample]:
    """Produce exactly 70/30 new/old (up to integer rounding), never 70/130."""
    if not new_examples or batch_size < 1 or not 0 <= old_fraction <= 1:
        raise ValueError("invalid replay batch request")
    task = new_examples[0].task
    if any(row.task != task for row in new_examples):
        raise ValueError("one replay batch must contain a single task")
    old_count = round(batch_size * old_fraction)
    if sum(row.task == task for row in replay.buffer) < old_count:
        raise ValueError("not enough old examples of this task for the requested replay ratio")
    new_count = batch_size - old_count
    new = [new_examples[rng.randrange(len(new_examples))] for _ in range(new_count)]
    mixed = new + replay.sample(old_count, task=task)
    rng.shuffle(mixed)
    return mixed
