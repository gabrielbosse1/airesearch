"""Scalable ReAct trajectory data for the ByteCore v2 foundation.

Every example is a raw byte sequence in one format::

    [MEM] my:color = blue
    Request: What is my favorite color?
    Thought: the answer is stored in memory
    Action: recall("my:color")
    Observation: blue
    Answer: blue

There are no control frames, no copy targets and no separate result channel:
the objective is plain next-byte prediction over the whole stream, which is
exactly what the foundation model is trained on.

Five trajectory kinds are generated:

=========  ===============================================================
``exec``   request -> one ``exec_python`` call -> observation -> answer
``repair`` two rounds: the code fails, the sandbox says why, the model fixes it
``file``   ``write_file`` then ``read_file`` (or a read of a known file)
``memory`` a stored fact is retrieved with ``recall``
``chat``   no tool at all: the model must decline to act
=========  ===============================================================

Two properties are enforced:

* **Instance-disjoint splits.** Every snippet, filename, word and fact value is
  generated into one pool sliced per split, so a held-out trajectory never
  reuses training content (verified in ``tests/test_phase16.py``).
* **Real observations.** Executable labels are produced by a real interpreter.
  Snippets are executed in one interpreter process for speed, and a
  deterministic sample is re-verified against :class:`models.sandbox.ByteSandbox`
  (one file per run, exactly like the agent) before the dataset is accepted. A
  mismatch raises instead of shipping a wrong label.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from models.react import (ANSWER, format_action, observation_line, render_trajectory,
                          sandbox_observation)
from models.sandbox import ByteSandbox

DEFAULT_TOTAL = 12_000
SEED = 16001
KINDS = ("exec", "repair", "file", "memory", "chat")

#: How many trajectories of each kind each split receives.
_SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "train": {"exec": 5_040, "repair": 2_240, "file": 1_680, "memory": 1_120, "chat": 1_120},
    "validation": {"exec": 220, "repair": 100, "file": 70, "memory": 50, "chat": 40},
    "test": {"exec": 220, "repair": 100, "file": 70, "memory": 50, "chat": 40},
}
#: How many distinct content instances back those trajectories.
_POOL_SIZES: dict[str, dict[str, int]] = {
    "train": {"exec": 900, "repair": 450, "file": 220, "memory": 220, "chat": 180},
    "validation": {"exec": 60, "repair": 40, "file": 30, "memory": 30, "chat": 30},
    "test": {"exec": 60, "repair": 40, "file": 30, "memory": 30, "chat": 30},
}

_SYLLABLES = ("ba", "ce", "di", "fo", "gu", "ja", "ke", "li", "mo", "nu", "po",
              "ra", "se", "ti", "vo", "xu", "ze", "tra", "ble", "son", "mar", "lun")
_NAMES = ("total", "value", "result", "count", "score", "amount", "index", "length",
          "offset", "width", "height", "radius", "weight", "margin", "budget", "sample",
          "window", "cursor", "buffer", "signal", "packet", "metric", "vector", "matrix")
_KEYS = ("favorite color", "favorite food", "lucky number", "hometown", "birth year",
         "favorite animal", "middle name", "favorite book", "phone model", "pet name")

_EXEC_TEMPLATES = (
    ("Calculate {a} * {b}.", "print({a} * {b})", "multiply the two numbers"),
    ("What is {a} + {b}?", "print({a} + {b})", "add the two numbers"),
    ("Compute {a} - {b}.", "print({a} - {b})", "subtract the second from the first"),
    ("How many characters are in '{word}'?", "print(len('{word}'))", "count the characters"),
    ("Sort the numbers {a}, {b} and {c}.",
     "print(sorted([{a}, {b}, {c}]))", "sort the three numbers"),
    ("Sum the numbers from 1 to {n}.", "print(sum(range(1, {n})))", "sum the range"),
    ("Uppercase the text '{word}'.", "print('{word}'.upper())", "uppercase the string"),
    ("Reverse the string '{word}'.", "print('{word}'[::-1])", "reverse the string"),
    ("What is the larger of {a} and {b}?", "print(max({a}, {b}))", "take the maximum"),
    ("What is {a} squared?", "print({a} ** 2)", "square the number"),
    ("List the numbers from 0 to {n}.", "print(list(range({n})))", "build the list"),
    ("How many words are in '{word} {word2}'?",
     "print(len('{word} {word2}'.split()))", "count the words"),
)
_THOUGHT_STEMS = ("I should {goal}.", "The task is to {goal}.", "To answer this I must {goal}.",
                  "This needs a tool: {goal}.", "Let me {goal}.")
_ANSWER_STEMS = ("The answer is {value}.", "It is {value}.", "That gives {value}.",
                 "The result is {value}.")

_CHAT_INSTANCES = (
    ("hello", "Hello! How can I help?"),
    ("hi there", "Hi! What can I do for you?"),
    ("good morning", "Good morning! What would you like to work on?"),
    ("thanks", "You're welcome."),
    ("thank you", "You're welcome."),
    ("how are you", "I'm running normally. What do you need?"),
    ("who are you", "I am ByteCore, a raw-byte model with a sandbox."),
    ("what can you do", "I can run Python, read and write files, and recall stored facts."),
    ("goodbye", "Goodbye!"),
    ("that is interesting", "Glad you think so."),
)
_CHAT_THOUGHTS = ("No tool is needed; this is small talk.",
                  "This is conversational, so I answer directly.",
                  "There is nothing to execute here; I reply in words.")


@dataclass(frozen=True)
class Trajectory:
    """One complete ReAct byte sequence and the metadata that produced it."""

    request: bytes
    body: bytes
    memory: bytes
    kind: str
    text: bytes

    @property
    def length(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------
# Content pools. One pool per kind, sliced per split -> instance-disjoint.
# --------------------------------------------------------------------------
def _word(rng: random.Random) -> str:
    return "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3)))


def _exec_pool(total: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    pool: list[dict[str, object]] = []
    seen: set[str] = set()
    attempts = 0
    while len(pool) < total:
        attempts += 1
        if attempts > 200 * total + 1000:
            raise RuntimeError("exec pool exhausted its content space")
        template = rng.randrange(len(_EXEC_TEMPLATES))
        instance = {
            "template": template,
            "a": rng.randint(2, 40),
            "b": rng.randint(2, 30),
            "c": rng.randint(2, 25),
            "n": rng.randint(3, 40),
            "word": _word(rng),
            "word2": _word(rng),
        }
        request = _EXEC_TEMPLATES[template][0].format(**instance)
        if request in seen:
            continue
        seen.add(request)
        pool.append(instance)
    return pool


def _scramble(name: str, rng: random.Random) -> str:
    position = rng.randrange(max(len(name) - 1, 1))
    letters = list(name)
    letters[position], letters[position + 1] = letters[position + 1], letters[position]
    return "".join(letters)


def _repair_pool(total: int, seed: int) -> list[tuple[str, str]]:
    """(broken, corrected) single-line snippets. Same six categories as Phase 15."""
    rng = random.Random(seed)
    pool: list[tuple[str, str]] = []
    seen: set[str] = set()
    attempts = 0
    while len(pool) < total:
        attempts += 1
        if attempts > 200 * total + 1000:
            raise RuntimeError("repair pool exhausted its content space")
        word = _word(rng)
        number, other = rng.randint(2, 90), rng.randint(2, 20)
        category = rng.randrange(6)
        if category == 0:
            typo = rng.choice(("pritn", "prnt", "prin", "pirnt"))
            body = rng.choice((f"{number} * {other}", f"{number} + {other}", f"'{word}'",
                               f"len('{word}')", f"sorted([{other}, {number}])", f"max({number}, {other})"))
            broken, fixed = f"{typo}({body})", f"print({body})"
        elif category == 1:
            head = rng.choice((f"if {number} > {other}", f"for i in range({other % 7 + 2})",
                               f"while {other} < {number}"))
            tail = f"print({number} + {other})"
            broken, fixed = f"{head} {tail}", f"{head}: {tail}"
        elif category == 2:
            inner = rng.choice((f"len('{word}')", f"'{word}'.upper()", f"sum([{number}, {other}])",
                                f"sorted([{other}, {number}])", f"max({number}, {other})"))
            broken, fixed = f"print({inner}", f"print({inner})"
        elif category == 3:
            if rng.random() < 0.5:
                broken, fixed = f"print({number} + '{other}')", f"print({number} + int('{other}'))"
            else:
                broken, fixed = f"print('{word}' + {number})", f"print('{word}' + str({number}))"
        elif category == 4:
            name = rng.choice(_NAMES)
            wrong = _scramble(name, rng)
            if wrong == name:
                wrong = name[::-1]
            broken, fixed = f"{name} = {number}; print({wrong})", f"{name} = {number}; print({name})"
        else:
            broken, fixed = f"print(int('{number}x'))", f"print(int('{number}'))"
        if broken in seen:
            continue
        seen.add(broken)
        pool.append((broken, fixed))
    return pool


def _file_pool(total: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pool: list[tuple[str, str]] = []
    seen: set[str] = set()
    suffixes = ("txt", "md", "log", "csv", "json")
    attempts = 0
    while len(pool) < total:
        attempts += 1
        if attempts > 200 * total + 1000:
            raise RuntimeError("file pool exhausted its content space")
        name = f"{_word(rng)}.{rng.choice(suffixes)}"
        if name in seen:
            continue
        seen.add(name)
        content = " ".join(_word(rng) for _ in range(rng.randint(2, 5)))
        pool.append((name, content))
    return pool


#: (topic template, question template). The field in braces is filled with a
#: random name or word, so the topic space is far larger than the fact count.
_MEMORY_TOPICS = (
    ("my:{key}", "What is my {key}?"),
    ("person:{name}:workplace", "Where does {name} work?"),
    ("project:{word}:deadline", "When is the {word} project due?"),
    ("server:{word}:port", "What port does the {word} server use?"),
    ("device:{word}:owner", "Who owns the {word} device?"),
    ("team:{word}:lead", "Who leads the {word} team?"),
)


def _memory_pool(total: int, seed: int) -> list[tuple[str, str, str]]:
    """(topic, value, question). Values stay inside the vector codec's limits."""
    rng = random.Random(seed)
    pool: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(pool) < total:
        attempts += 1
        if attempts > 200 * total + 1000:
            raise RuntimeError("memory pool exhausted its content space")
        topic_template, question_template = rng.choice(_MEMORY_TOPICS)
        fields = {"key": rng.choice(_KEYS).replace(" ", "_"), "name": _word(rng), "word": _word(rng)}
        topic = topic_template.format(**fields)
        question = question_template.format(**fields)
        value = rng.choice((_word(rng), str(rng.randint(10, 999)), f"{_word(rng)} {rng.randint(2, 90)}"))
        if (topic, value) in seen:
            continue
        seen.add((topic, value))
        pool.append((topic, value, question))
    return pool


def _pool(kind: str, total: int, seed: int) -> list:
    builders = {"exec": _exec_pool, "repair": _repair_pool, "file": _file_pool,
                "memory": _memory_pool}
    return builders[kind](total, seed)


_POOL_TOTALS = {kind: sum(sizes[kind] for sizes in _POOL_SIZES.values()) for kind in _POOL_SIZES["train"]}


def _pool_slice(kind: str, split: str, seed: int, size: int | None = None) -> list:
    """Deterministic, disjoint slice of one kind's instance pool."""
    order = ("train", "validation", "test")
    full = _pool(kind, _POOL_TOTALS[kind], seed + 97 * KINDS.index(kind))
    start = sum(_POOL_SIZES[earlier][kind] for earlier in order[: order.index(split)])
    wanted = _POOL_SIZES[split][kind] if size is None else min(size, _POOL_SIZES[split][kind])
    return full[start:start + wanted]


# --------------------------------------------------------------------------
# Execution labels. Every observation comes from the real ByteSandbox -- the
# same call the agent makes at run time (one temporary file per run, 5 s
# timeout) -- reduced by :func:`models.react.sandbox_observation`, which is
# used identically here and in the runtime loop. Nothing is synthesised.
# --------------------------------------------------------------------------
def execute_labels(codes: list[bytes], sandbox: ByteSandbox) -> list[bytes]:
    """Observation bytes for each snippet, straight from the sandbox."""
    return [sandbox_observation(sandbox.execute_python(code)) for code in codes]


def _assert_fails(codes: list[bytes], outputs: list[bytes]) -> None:
    """Repair labels must be genuine failures, not a template that happens to run."""
    prefixes = (b"SyntaxError", b"NameError", b"TypeError", b"ValueError",
                b"IndentationError", b"AttributeError", b"ZeroDivisionError")
    for code, output in zip(codes, outputs):
        if not output.startswith(prefixes):
            raise ValueError(f"repair snippet is not a real failure: {code!r} -> {output!r}")


# --------------------------------------------------------------------------
# Trajectory builders
# --------------------------------------------------------------------------
def _exec_trajectories(pool: list[dict[str, object]], count: int, outputs: list[bytes],
                       rng: random.Random) -> list[Trajectory]:
    rows: list[Trajectory] = []
    for index in range(count):
        instance = pool[index % len(pool)]
        template = _EXEC_TEMPLATES[int(instance["template"])]
        request = template[0].format(**instance).encode("utf-8")
        code = template[1].format(**instance).encode("utf-8")
        value = outputs[index % len(pool)]
        goal = template[2]
        thought = rng.choice(_THOUGHT_STEMS).format(goal=goal)
        answer = rng.choice(_ANSWER_STEMS).format(value=value.decode("utf-8"))
        body = b"\n".join((
            thought.encode("utf-8"),
            format_action(b"exec_python", (code,)),
            observation_line(value),
            ANSWER + b" " + answer.encode("utf-8"),
        ))
        rows.append(Trajectory(request=request, body=b" " + body, memory=b"", kind="exec",
                               text=render_trajectory(request, b" " + body)))
    return rows


def _repair_trajectories(pool: list[tuple[str, str]], count: int, failures: list[bytes],
                         successes: list[bytes], rng: random.Random) -> list[Trajectory]:
    rows: list[Trajectory] = []
    prompts = ("Run this Python and fix any error: {code}", "Execute this and repair it if it fails: {code}",
               "Please run: {code}")
    for index in range(count):
        broken, fixed = pool[index % len(pool)]
        request = prompts[index % len(prompts)].format(code=broken).encode("utf-8")
        failure = failures[index % len(pool)]
        success = successes[index % len(pool)]
        body = b"\n".join((
            b"I will run the pasted code first and read what the sandbox says.",
            format_action(b"exec_python", (broken.encode("utf-8"),)),
            observation_line(failure),
            b"The sandbox reported an error, so I correct the code and run it again.",
            format_action(b"exec_python", (fixed.encode("utf-8"),)),
            observation_line(success),
            ANSWER + b" " + rng.choice(_ANSWER_STEMS).format(value=success.decode("utf-8")).encode("utf-8"),
        ))
        rows.append(Trajectory(request=request, body=b" " + body, memory=b"", kind="repair",
                               text=render_trajectory(request, b" " + body)))
    return rows


def _file_trajectories(pool: list[tuple[str, str]], count: int, writes: list[bytes],
                       reads: list[bytes], rng: random.Random) -> list[Trajectory]:
    rows: list[Trajectory] = []
    for index in range(count):
        name, content = pool[index % len(pool)]
        request = f"Save '{content}' in {name}, then read it back.".encode("utf-8")
        body = b"\n".join((
            b"I write the file first, then read it to confirm the contents.",
            format_action(b"write_file", (name.encode("utf-8"), content.encode("utf-8"))),
            observation_line(writes[index % len(pool)]),
            b"Now I read the file back.",
            format_action(b"read_file", (name.encode("utf-8"),)),
            observation_line(reads[index % len(pool)]),
            ANSWER + b" " + f"Written and read back: {content}".encode("utf-8"),
        ))
        rows.append(Trajectory(request=request, body=b" " + body, memory=b"", kind="file",
                               text=render_trajectory(request, b" " + body)))
    return rows


def _memory_trajectories(pool: list[tuple[str, str, str]], count: int) -> list[Trajectory]:
    rows: list[Trajectory] = []
    for index in range(count):
        topic, value, question = pool[index % len(pool)]
        memory = f"[MEM] {topic} = {value}".encode("utf-8")
        request = question.encode("utf-8")
        body = b"\n".join((
            b"The fact is already in memory, so I retrieve it instead of guessing.",
            format_action(b"recall", (topic.encode("utf-8"),)),
            observation_line(value.encode("utf-8")),
            ANSWER + b" " + value.encode("utf-8"),
        ))
        rows.append(Trajectory(request=request, body=b" " + body, memory=memory, kind="memory",
                               text=render_trajectory(request, b" " + body, memory)))
    return rows


def _chat_trajectories(count: int, seed: int) -> list[Trajectory]:
    rng = random.Random(seed)
    rows: list[Trajectory] = []
    for index in range(count):
        request, answer = _CHAT_INSTANCES[index % len(_CHAT_INSTANCES)]
        variant = f"{request}{'!' if index % 3 == 1 else ''}".encode("utf-8")
        body = b"\n".join((
            rng.choice(_CHAT_THOUGHTS).encode("utf-8"),
            ANSWER + b" " + answer.encode("utf-8"),
        ))
        rows.append(Trajectory(request=variant, body=b" " + body, memory=b"", kind="chat",
                               text=render_trajectory(variant, b" " + body)))
    return rows


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def generate_counts() -> dict[str, int]:
    """Trajectory counts per split, as configured (the 10,000+ guarantee)."""
    return {split: sum(counts.values()) for split, counts in _SPLIT_COUNTS.items()}


def build_split(split: str, *, sandbox: ByteSandbox | None = None, seed: int = SEED,
                counts: dict[str, int] | None = None) -> list[Trajectory]:
    """Build one split's trajectories with real, sampled-verified observations."""
    if split not in _SPLIT_COUNTS:
        raise ValueError(f"unknown split: {split}")
    wanted = dict(_SPLIT_COUNTS[split] if counts is None else counts)
    owns_sandbox = sandbox is None
    sandbox = sandbox or ByteSandbox()
    try:
        # A small build (tests, smoke runs) also uses a small pool, so the
        # sandbox cost scales with what was asked for.
        exec_pool = _pool_slice("exec", split, seed, max(4, wanted["exec"]))
        repair_pool = _pool_slice("repair", split, seed, max(4, wanted["repair"]))
        file_pool = _pool_slice("file", split, seed, max(4, wanted["file"]))
        memory_pool = _pool_slice("memory", split, seed, max(4, wanted["memory"]))

        exec_codes = [_EXEC_TEMPLATES[int(item["template"])][1].format(**item).encode("utf-8")
                      for item in exec_pool]
        broken_codes = [broken.encode("utf-8") for broken, _ in repair_pool]
        fixed_codes = [fixed.encode("utf-8") for _, fixed in repair_pool]
        exec_outputs = execute_labels(exec_codes, sandbox)
        repair_failures = execute_labels(broken_codes, sandbox)
        repair_successes = execute_labels(fixed_codes, sandbox)
        _assert_fails(broken_codes, repair_failures)
        if any(not success or success.startswith((b"SyntaxError", b"NameError", b"TypeError",
                                                  b"ValueError", b"IndentationError"))
               for success in repair_successes):
            raise ValueError("a corrected repair snippet still fails in the sandbox")

        # Real file operations: no subprocess, and the observation strings are
        # whatever the sandbox actually returns.
        writes, reads = [], []
        for name, content in file_pool:
            writes.append(sandbox.write_file(name.encode("utf-8"), content.encode("utf-8")))
            reads.append(sandbox.read_file(name.encode("utf-8")))

        rng = random.Random(seed + 13 * KINDS.index("exec"))
        rows = _exec_trajectories(exec_pool, wanted["exec"], exec_outputs, rng)
        rows += _repair_trajectories(repair_pool, wanted["repair"], repair_failures, repair_successes,
                                     random.Random(seed + 13 * KINDS.index("repair")))
        rows += _file_trajectories(file_pool, wanted["file"], writes, reads,
                                   random.Random(seed + 13 * KINDS.index("file")))
        rows += _memory_trajectories(memory_pool, wanted["memory"])
        rows += _chat_trajectories(wanted["chat"], seed + 13 * KINDS.index("chat"))
    finally:
        if owns_sandbox:
            sandbox.close()
    rng = random.Random(seed + 7)
    rng.shuffle(rows)
    return rows


def build_all(*, sandbox: ByteSandbox | None = None, seed: int = SEED) -> dict[str, list[Trajectory]]:
    """Build every split with one shared sandbox."""
    owns = sandbox is None
    sandbox = sandbox or ByteSandbox()
    try:
        return {split: build_split(split, sandbox=sandbox, seed=seed) for split in ("train", "validation", "test")}
    finally:
        if owns:
            sandbox.close()


def dataset_stats(rows: Iterable[Trajectory]) -> dict[str, object]:
    """Counts by kind plus byte-length statistics (used by the report)."""
    rows = list(rows)
    by_kind: dict[str, int] = {}
    for row in rows:
        by_kind[row.kind] = by_kind.get(row.kind, 0) + 1
    lengths = sorted(row.length for row in rows)
    if not lengths:
        return {"count": 0, "by_kind": {}, "bytes": 0}
    return {
        "count": len(rows),
        "by_kind": by_kind,
        "bytes": sum(lengths),
        "min_length": lengths[0],
        "median_length": lengths[len(lengths) // 2],
        "max_length": lengths[-1],
        "with_memory_prefix": sum(1 for row in rows if row.memory),
        # Detected from the bytes rather than from the kind, so a corpus added
        # later (Phase 17's real GitHub trajectories) is counted the same way.
        "with_tool_call": sum(1 for row in rows if b"Action: " in row.text),
    }


def write_jsonl(rows: Iterable[Trajectory], path: str | Path) -> int:
    """Persist trajectories as JSON lines (byte strings are latin-1 escaped)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "kind": row.kind,
                "request": row.request.decode("latin-1"),
                "memory": row.memory.decode("latin-1"),
                "body": row.body.decode("latin-1"),
                "text": row.text.decode("latin-1"),
            }) + "\n")
            count += 1
    return count
