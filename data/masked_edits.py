"""Deterministic raw-byte examples for mask-conditioned replacement training."""
from __future__ import annotations

import random
from typing import Any

_SPLIT_SEED = {"train": 0, "validation": 100_000, "test": 200_000, "coverage": 300_000}


def masked_examples(split: str, per_length: int, *, seed: int = 9090,
                    lengths: tuple[int, ...] = (10, 16, 32, 64, 128),
                    probability: float = 0.30) -> list[dict[str, Any]]:
    if split not in _SPLIT_SEED or per_length < 1 or not lengths or not 0 < probability <= 1:
        raise ValueError("invalid masked edit dataset configuration")
    rng = random.Random(seed + _SPLIT_SEED[split])
    records = []
    for length in lengths:
        if not isinstance(length, int) or not 1 <= length <= 128:
            raise ValueError("lengths must be in [1, 128]")
        for i in range(per_length):
            source = bytes(rng.randrange(256) for _ in range(length))
            mask = bytearray(int(rng.random() < probability) for _ in range(length))
            if not any(mask):
                mask[rng.randrange(length)] = 1
            requested = bytearray(length)
            target = bytearray(source)
            for pos, selected in enumerate(mask):
                if selected:
                    value = (source[pos] + rng.randrange(1, 256)) % 256
                    requested[pos] = value
                    target[pos] = value
            records.append({"id": f"mask-{split}-{length}-{i}", "split": split, "input": source,
                            "output": bytes(target), "edit_mask": bytes(mask), "requested_bytes": bytes(requested)})
    return records


def byte_coverage_examples(*, seed: int = 9090) -> list[dict[str, Any]]:
    """One unseen source/request combination for each possible requested byte."""
    rng = random.Random(seed + _SPLIT_SEED["coverage"])
    records = []
    for value in range(256):
        source = bytearray(rng.randrange(256) for _ in range(10))
        position = value % len(source)
        if source[position] == value:
            source[position] = (value + 1) % 256
        mask = bytearray(len(source)); mask[position] = 1
        requested = bytearray(len(source)); requested[position] = value
        target = source.copy(); target[position] = value
        records.append({"id": f"mask-coverage-{value}", "split": "test", "input": bytes(source),
                        "output": bytes(target), "edit_mask": bytes(mask), "requested_bytes": bytes(requested)})
    return records
