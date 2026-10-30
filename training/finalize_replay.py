"""Create the bounded synthetic multi-task replay snapshot after Phase 12.

Capacity limits require sampling rather than saving every training row. This
module imports only public synthetic fixtures, never application conversations.
It preserves a task label so later trainers cannot mix incompatible losses.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from data.conversation_response_v2 import copy_generalization_examples, response_examples
from data.diverse_dialogue import response_bytes, training_prompts
from data.phase12_eval_sets import DIVERSITY_VALIDATION_PROMPTS
from data.vector_memory_pairs import vector_training_examples
from models.conversation_intents import ConversationIntent
from models.holdout import HoldoutGuard, HoldoutViolation
from training.replay import ReplayBuffer, ReplayExample
from training.train_replay_intents import phase12_guard


def final_guard(args):
    base = phase12_guard(args)
    extra = HoldoutGuard.from_records([{"input": prompt} for prompt in DIVERSITY_VALIDATION_PROMPTS])
    return HoldoutGuard(base.contents | extra.contents, base.held_out_hashes | extra.held_out_hashes)


def _preflight(guard, rows):
    accepted, blocked = [], 0
    scratch = ReplayBuffer(1, guard=guard)
    for row in rows:
        try:
            scratch._check(row)
        except HoldoutViolation:
            blocked += 1; continue
        accepted.append(row)
    return accepted, blocked


def build(args):
    rng = random.Random(args.seed)
    base_guard, guard = phase12_guard(args), final_guard(args)
    previous = ReplayBuffer.load(Path(args.output_dir) / "replay_phase12.pt", guard=base_guard)
    old = [row for row in previous.buffer if row.task == "intent" and row.phase == "phase10"]
    new = [row for row in previous.buffer if row.task == "intent" and row.phase == "phase12"]
    vector = [ReplayExample(row["statement"], int(ConversationIntent.RECALL), "phase12", "vector",
                            row["target"], (row["question"],)) for row in vector_training_examples("train")]
    diverse = [ReplayExample(prompt, int(ConversationIntent.CHAT), "phase12", "diverse_response", response,
                             (), style) for prompt in training_prompts() for style, response in enumerate(response_bytes())]
    contextual = [ReplayExample(row.message, int(row.intent), "phase12", "contextual_response", row.response,
                                tuple(raw for turn in row.history for raw in (turn.user, turn.assistant)))
                  for row in response_examples("train") + copy_generalization_examples(count=400)]
    categories = {"old_intent": (old, 300), "new_intent": (new, 300), "vector": (vector, 160),
                  "diverse_response": (diverse, 160), "contextual_response": (contextual, 80)}
    selected, details = [], {}
    for name, (rows, count) in categories.items():
        accepted, blocked = _preflight(guard, rows)
        if len(accepted) < count:
            raise RuntimeError(f"not enough preflighted {name} examples: {len(accepted)} < {count}")
        selected += rng.sample(accepted, count)
        details[name] = {"available": len(accepted), "blocked": blocked, "stored": count}
    rng.shuffle(selected)
    replay = ReplayBuffer(1000, seed=args.seed + 1, guard=guard)
    replay.add(selected)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "replay_multitask_phase12.pt"
    replay.save(checkpoint)
    # Verify the saved snapshot under precisely the same holdout registry.
    reloaded = ReplayBuffer.load(checkpoint, guard=guard)
    report = {"capacity": replay.capacity, "stored": len(reloaded), "guard_fingerprint": guard.fingerprint,
              "categories": details, "task_counts": dict(Counter(row.task for row in reloaded.buffer)),
              "source": "Synthetic fixture training rows only; no user conversation is automatically inserted.",
              "scope": "Bounded sampled snapshot; the 70/30 optimization was applied to intent training, not every newly introduced decoder task."}
    (out / "replay_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="runs/phase12")
    parser.add_argument("--seed", type=int, default=121240)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
