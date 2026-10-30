"""Reproducible, explicitly instructed single-byte replacement examples.

The requested value and zero-based index are contained in each natural-language
instruction, not derived from an unknown target at inference. Splits use
separate RNG namespaces; callers should still preflight byte-content overlap.
The user-specified Hello example is deliberately a training example.
"""
from __future__ import annotations

import random
from typing import Any

from models.cognitive import CognitiveAction, action_to_record
from models.cognitive_arguments import encode_edit_arguments
from models.generator_conditioned import GeneratorConditioned

_ALPHABET = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
_SPLIT_SEED = {"train": 0, "validation": 100_000, "test": 200_000}
DIAGNOSTIC_SOURCE = b"Hello Worl"
DIAGNOSTIC_POSITION = 9
DIAGNOSTIC_VALUE = 100


def instruction_for(position: int, old: int, value: int) -> bytes:
    def character(byte: int) -> str:
        return repr(chr(byte)) if 32 <= byte <= 126 and byte not in (39, 92) else "'\\x%02x'" % byte
    return (f"Replace the byte at zero-based position {position} from {old} ({character(old)}) "
            f"to {value} ({character(value)}).").encode("ascii")


def replacement_record(source: bytes, position: int, value: int, *, identifier: str, split: str) -> dict[str, Any]:
    if not isinstance(source, bytes) or not isinstance(position, int) or isinstance(position, bool) or not 0 <= position < len(source):
        raise ValueError("replacement requires raw source bytes and an in-range position")
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255 or value == source[position]:
        raise ValueError("replacement must be a different raw byte")
    target = source[:position] + bytes([value]) + source[position + 1:]
    g = GeneratorConditioned
    actions = [CognitiveAction(g.LOAD_SPAN, start=0, end=len(source)),
               CognitiveAction(g.THINK),
               CognitiveAction(g.EDIT_OP, start=position, end=position + 1, kind="replace", data=bytes([value])),
               CognitiveAction(g.SAVE_SPAN), CognitiveAction(g.DONE)]
    return {"id": identifier, "split": split, "kind": "positional_replace", "input": source,
            "output": target, "instruction_bytes": instruction_for(position, source[position], value),
            "position": position, "requested_byte": value,
            "structured_request": encode_edit_arguments(actions[2]),
            "trace_actions": [action_to_record(action) for action in actions]}


def diagnostic_record() -> dict[str, Any]:
    return replacement_record(DIAGNOSTIC_SOURCE, DIAGNOSTIC_POSITION, DIAGNOSTIC_VALUE,
                              identifier="user-diagnostic-training-example", split="train")


def positional_examples(split: str, count: int, *, seed: int = 9009, long_count: int = 0) -> list[dict[str, Any]]:
    if split not in _SPLIT_SEED or count < 0 or long_count < 0:
        raise ValueError("split, count, or long_count is invalid")
    rng = random.Random(seed + _SPLIT_SEED[split])
    records = []
    for i in range(count + long_count):
        is_long = i >= count
        length = rng.randrange(260, 301) if is_long else rng.choice((8, 9, 10, 10, 10, 12, 16, 20))
        source = bytes(rng.choice(_ALPHABET) for _ in range(length))
        position = rng.randrange(256, length) if is_long else rng.randrange(length)
        values = [value for value in _ALPHABET if value != source[position]]
        value = rng.choice(values)
        records.append(replacement_record(source, position, value,
                                         identifier=f"positional-{split}-{i}", split=split))
    return records
