"""Synthetic observed fact -> question -> response pairs for vector training.

Values are generated from raw UTF-8 bytes, including multibyte examples; no
external tokenizer is used. Split-specific pseudorandom seeds provide disjoint
examples, not proof of semantic or distributional novelty.
"""
from __future__ import annotations

import random
import string

_KEYS = ("favorite color", "name", "favorite food", "pet", "hometown", "favorite drink", "project", "hobby",
         "preferred language", "instrument", "nickname", "favorite city")
_PEOPLE = ("Maya", "Oscar", "Nora", "Eli", "Theo", "Luna", "Arun", "Mila", "Hugo", "Inez")
_QUESTIONS = ("What is my {key}?", "Do you remember my {key}?", "Tell me my {key}.",
              "What was my {key}?", "Can you recall my {key}?")


def vector_training_examples(split: str, *, count: int | None = None) -> list[dict[str, bytes]]:
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    if count is None:
        count = 2200 if split == "train" else 240
    if not isinstance(count, int) or not 1 <= count <= 50_000:
        raise ValueError("invalid synthetic example count")
    rng = random.Random(120021 if split == "train" else 120022)
    result, used = [], set()
    chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    unicode_parts = ("é", "ñ", "ç", "Ω", "雪", "ô", "ü", "ø", "á", "í")
    for i in range(count):
        while True:
            n = rng.randint(1, 18)
            value = "".join(rng.choice(chars) for _ in range(n))
            if i % 5 == 0:
                value = "a " + value
            if i % 7 == 0:
                value = value + rng.choice(unicode_parts)
            raw = value.encode("utf-8")
            if 1 <= len(raw) <= 40 and raw not in used:
                used.add(raw); break
        if i % 11 == 0:
            person = _PEOPLE[(i // 11) % len(_PEOPLE)]
            statement = f"{person} works at {value}.".encode("utf-8")
            question = f"Where does {person} work?".encode("utf-8")
        else:
            key = _KEYS[i % len(_KEYS)]
            statement = f"My {key} is {value}.".encode("utf-8")
            question = rng.choice(_QUESTIONS).format(key=key).encode("utf-8")
        result.append({"statement": statement, "question": question, "target": raw})
    return result
