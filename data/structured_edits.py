"""Disjoint structured requests covering raw byte values and long positions."""
from __future__ import annotations

import random
from typing import Any

from data.positional_edits import replacement_record

_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}


def structured_examples(split: str, count: int, *, seed: int = 9509, long_count: int = 0) -> list[dict[str, Any]]:
    if split not in _OFFSETS or count < 0 or long_count < 0:
        raise ValueError("invalid structured dataset configuration")
    rng = random.Random(seed + _OFFSETS[split])
    result = []
    for i in range(count + long_count):
        is_long = i >= count
        length = rng.randrange(260, 305) if is_long else rng.choice((8, 9, 10, 10, 12, 16, 20, 32))
        binary = i % 2 == 1
        source = bytearray(rng.randrange(256) if binary else rng.choice(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
                           for _ in range(length))
        pos = rng.randrange(256, length) if is_long else rng.randrange(length)
        # For count >= 256, every requested byte appears; source/request pairs
        # and masks remain disjoint across RNG namespaces.
        value = (i * 73 + seed) % 256
        if source[pos] == value:
            source[pos] = (value + 1) % 256
        record = replacement_record(bytes(source), pos, value, identifier=f"structured-{split}-{i}", split=split)
        record["binary_source"] = binary
        result.append(record)
    return result
