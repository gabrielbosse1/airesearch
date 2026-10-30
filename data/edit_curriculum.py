"""Reproducible, explicitly instructed raw-byte editing curriculum.

These small generated examples exercise task structure; they are not a real
world benchmark. Separate ``train``/``validation``/``test`` namespaces have
non-overlapping identifiers. Nothing in an instruction is an entire target
document, and no code in a canvas is executed.
"""
from __future__ import annotations

import random
from typing import Any

from data.ingest_real_traces import make_control_trace
from models.cognitive import CognitiveAction, action_to_record, apply_edit
from models.generator_conditioned import GeneratorConditioned
from training.co_train import CURRICULUM_STAGES

_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_SPLIT_OFFSET = {"train": 0, "validation": 1000, "test": 2000}


def _suffix(number: int) -> bytes:
    return bytes((_ALPHABET[number // len(_ALPHABET)], _ALPHABET[number % len(_ALPHABET)]))


def curriculum_examples(*, split: str, count_per_stage: int, seed: int = 881) -> list[dict[str, Any]]:
    """Create labelled examples spanning all five stages, as raw-byte records.

    Stage 1 changes exactly one byte. Stage 2 replaces a whole word. Stage 3
    replaces one line. Stage 4 alternates multi-line insert/delete. Stage 5
    moves a two-line code block with one MOVE action (rather than a Levenshtein
    substitute/delete/insert approximation).
    """
    if split not in _SPLIT_OFFSET or not isinstance(count_per_stage, int) or not 1 <= count_per_stage <= 1000:
        raise ValueError("split must be train/validation/test and count_per_stage must be in [1, 1000]")
    rng = random.Random(seed + _SPLIT_OFFSET[split])
    examples: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(CURRICULUM_STAGES):
        for i in range(count_per_stage):
            marker = _suffix(_SPLIT_OFFSET[split] + i)
            label = b"# " + marker + b"\n"
            actions: list[CognitiveAction]
            if stage_index == 0:
                # R: is a publicly specified grammar, not a target-derived hint.
                source = b"R:" + bytes([rng.choice(b"abcdefgh")]) + marker
                target = source[:2] + b"?" + source[3:]
                instruction = b"Replace the first byte after R: with ?."
            elif stage_index == 1:
                source = label + b"color old end\n"
                target = label + b"color new end\n"
                instruction = b"Replace the word old with new."
            elif stage_index == 2:
                source = label + b"top()\nx=0\nbottom()\n"
                target = label + b"top()\ny=1\nbottom()\n"
                instruction = b"Replace the line x=0 with y=1."
            elif stage_index == 3:
                if i % 2 == 0:
                    source = label + b"top()\nbottom()\n"
                    target = label + b"top()\na()\nb()\nbottom()\n"
                    instruction = b"Insert lines a() and b() between top() and bottom()."
                else:
                    source = label + b"top()\na()\nb()\nbottom()\n"
                    target = label + b"top()\nbottom()\n"
                    instruction = b"Delete the two lines a() and b()."
            else:
                block = b"a()\nb()\n"
                source = label + block + b"c()\n"
                move = CognitiveAction(GeneratorConditioned.EDIT_OP, start=len(label),
                                       end=len(label) + len(block), destination=len(source), kind="move")
                target = apply_edit(source, move, max_length=4096)
                instruction = b"Move the consecutive lines a() and b() after c()."
            if stage_index == 4:
                actions = [CognitiveAction(GeneratorConditioned.LOAD_SPAN, start=0, end=len(source)),
                           CognitiveAction(GeneratorConditioned.THINK), move,
                           CognitiveAction(GeneratorConditioned.SAVE_SPAN), CognitiveAction(GeneratorConditioned.DONE)]
            else:
                _, actions = make_control_trace(source, target)
            examples.append({
                "id": f"curriculum-{split}-{stage_index}-{i}", "split": split,
                "kind": "synthetic", "curriculum_stage": stage,
                "input": source, "output": target, "instruction_bytes": instruction,
                "trace_actions": [action_to_record(action) for action in actions],
            })
    return examples
