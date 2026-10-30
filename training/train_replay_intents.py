"""Phase 12 intent rehearsal with auditable 70/30 new/old replay.

The edit backbone remains frozen. Validation-only selection balances historical
micro-accuracy, new balanced-intent accuracy and the vague/specific boundary.
Previously inspected test fixtures are not used for selection. This cannot
promise protection against forgetting outside these small synthetic sets.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from data.conversation_examples import ConversationExample, conversation_examples
from data.conversation_response_v2 import response_examples
from data.phase11_eval_sets import overall_intent_examples
from data.phase12_eval_sets import DIVERSITY_TEST_PROMPT, intent_test_examples, memory_test_cases
from data.vector_memory_pairs import vector_training_examples
from data.vague_intent_pairs import hard_negative_examples
from models.conversation_intents import ConversationIntent, INTENT_NAMES, NUM_INTENTS
from models.holdout import HoldoutGuard, HoldoutViolation
from training.replay import ReplayBuffer, ReplayExample
from training.train_confidence import load_generator
from training.train_conversation import batch_features, collect_features, encode_cached, intent_metrics, sha256
from training.train_vague_intents import _LEGACY_VAGUE_EDIT, _normal, prior_records


def phase12_guard(args):
    previous = prior_records(args)
    protected = [{"input": msg} for msg, _ in intent_test_examples()]
    protected += [{"input": case.statement, "output": case.value} for case in memory_test_cases()]
    protected += [{"input": row["statement"], "output": row["target"]} for row in vector_training_examples("validation")]
    protected += [{"input": DIVERSITY_TEST_PROMPT}]
    fresh = HoldoutGuard.from_records(protected)
    return HoldoutGuard(previous.contents | fresh.contents, previous.held_out_hashes | fresh.held_out_hashes)


def _old_examples(guard):
    accepted, blocked = [], 0
    for row in conversation_examples("train"):
        label = (ConversationIntent.CLARIFY if row.intent == ConversationIntent.EDIT
                 and _normal(row.message) in _LEGACY_VAGUE_EDIT else row.intent)
        try:
            guard.validate(row.message, metadata={"split": "train", "source_phase": "phase10"})
        except HoldoutViolation:
            blocked += 1; continue
        accepted.append(ReplayExample(row.message, int(label), "phase10"))
    return accepted, blocked


def _new_examples(guard):
    rows = [ReplayExample(msg, int(label), "phase11") for msg, label in hard_negative_examples("train")]
    rows += [ReplayExample(row.message, int(ConversationIntent.CHAT), "phase11")
             for row in response_examples("train") if row.intent == ConversationIntent.CHAT and row.message.lower().startswith(b"my ")]
    unique = {}
    for row in rows:
        if row.message in unique and unique[row.message].label != row.label:
            raise ValueError(f"conflicting new labels for {row.message!r}")
        unique[row.message] = row
    accepted, blocked = [], 0
    for row in unique.values():
        try:
            guard.validate(row.message, metadata={"split": "train", "source_phase": row.phase})
        except HoldoutViolation:
            blocked += 1; continue
        accepted.append(row)
    return accepted, blocked


def _as_conversation(rows):
    return [ConversationExample(row.message, ConversationIntent(row.label)) for row in rows]


@torch.no_grad()
def _metrics(model, cache, batch_size, threshold):
    return intent_metrics(model.intent_classifier, encode_cached(model, cache, batch_size), cache[4], threshold)


def train(args):
    if min(args.epochs, args.batch_size, args.feature_batch_size, args.threads, args.steps_per_epoch) < 1:
        raise ValueError("training counts must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.old_fraction <= 1:
        raise ValueError("invalid optimizer or replay fraction")
    if not math.isfinite(args.distill_weight) or args.distill_weight < 0:
        raise ValueError("invalid distillation weight")
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = load_generator(args.generator, "cpu", args.generator_heads).eval()
    teacher = load_generator(args.teacher, "cpu", args.generator_heads).eval()
    if not bool(model.intent_trained) or not bool(teacher.intent_trained):
        raise RuntimeError("both source and teacher intent heads must be trained")
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    guard = phase12_guard(args)
    old, old_blocked = _old_examples(guard)
    new, new_blocked = _new_examples(guard)
    if len(old) < args.replay_capacity or len(new) < args.batch_size:
        raise RuntimeError("not enough audited new or old examples")
    chosen = rng.sample(old, args.replay_capacity)
    replay = ReplayBuffer(args.replay_capacity, seed=args.seed + 1, guard=guard)
    replay.add(chosen)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    replay.save(out / "replay_phase10.pt")
    print(f"preflight: old={len(old)} ({old_blocked} blocked), new={len(new)} ({new_blocked} blocked); "
          f"stored old={len(replay)}, protected={len(guard.contents)}", flush=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = list(model.intent_encoder.parameters()) + list(model.intent_classifier.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    new_cache = collect_features(model, _as_conversation(new), include_history=False, batch_size=args.feature_batch_size)
    old_cache = collect_features(model, _as_conversation(chosen), include_history=False, batch_size=args.feature_batch_size)
    teacher_cache = collect_features(teacher, _as_conversation(chosen), include_history=False, batch_size=args.feature_batch_size)
    with torch.no_grad():
        teacher_features = encode_cached(teacher, teacher_cache, args.feature_batch_size)
        teacher_logits = teacher.intent_classifier(teacher_features).detach()
    val = {
        "legacy": collect_features(model, conversation_examples("validation"), include_history=False, batch_size=args.feature_batch_size),
        "overall": collect_features(model, [ConversationExample(m, y) for m, y in overall_intent_examples("validation")],
                                    include_history=False, batch_size=args.feature_batch_size),
        "boundary": collect_features(model, [ConversationExample(m, y) for m, y in hard_negative_examples("validation")],
                                     include_history=False, batch_size=args.feature_batch_size),
    }
    old_lookup = {(row.message, row.label, row.phase): i for i, row in enumerate(chosen)}
    if len(old_lookup) != len(chosen):
        raise RuntimeError("replay identities must be unique")
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    class_weights = torch.tensor([1.0, 1.0, 1.0, args.remember_weight, 1.0, args.forget_weight, 1.0])
    old_count = round(args.batch_size * args.old_fraction)
    new_count = args.batch_size - old_count
    best, rank, selected_epoch, history = None, None, 0, []
    for epoch in range(args.epochs):
        total = 0.0
        for _ in range(args.steps_per_epoch):
            new_idx = torch.tensor([rng.randrange(len(new)) for _ in range(new_count)], dtype=torch.long)
            old_rows = replay.sample(old_count)
            old_idx = torch.tensor([old_lookup[(r.message, r.label, r.phase)] for r in old_rows], dtype=torch.long)
            features_new = batch_features(model, new_cache, new_idx)
            features_old = batch_features(model, old_cache, old_idx)
            logits_new = model.intent_classifier(features_new)
            logits_old = model.intent_classifier(features_old)
            logits = torch.cat((logits_new, logits_old))
            targets = torch.cat((new_cache[4][new_idx], old_cache[4][old_idx]))
            loss = F.cross_entropy(logits, targets, weight=class_weights)
            if args.distill_weight:
                temperature = 2.0
                knowledge = F.kl_div(F.log_softmax(logits_old / temperature, dim=-1),
                                     F.softmax(teacher_logits[old_idx] / temperature, dim=-1),
                                     reduction="batchmean") * temperature ** 2
                loss = loss + args.distill_weight * knowledge
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite mixed loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 2.0, error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())
        legacy = _metrics(model, val["legacy"], args.feature_batch_size, args.threshold)
        overall = _metrics(model, val["overall"], args.feature_batch_size, args.threshold)
        boundary = _metrics(model, val["boundary"], args.feature_batch_size, args.threshold)
        # Selection uses validation only. Feasibility avoids selecting
        # a legacy-only model that has forgotten new boundaries.
        feasible = overall["accuracy"] >= 0.85 and boundary["accuracy"] >= 0.80
        score = 0.45 * legacy["accuracy"] + 0.35 * overall["accuracy"] + 0.20 * boundary["accuracy"]
        current_rank = (int(feasible), score, legacy["accuracy"], -overall["nll"])
        if rank is None or current_rank > rank:
            rank, selected_epoch = current_rank, epoch + 1
            best = {"encoder": {k: v.detach().clone() for k, v in model.intent_encoder.state_dict().items()},
                    "classifier": {k: v.detach().clone() for k, v in model.intent_classifier.state_dict().items()}}
        row = {"epoch": epoch + 1, "loss": total / args.steps_per_epoch, "old_validation_accuracy": legacy["accuracy"],
               "overall_validation_accuracy": overall["accuracy"], "boundary_validation_accuracy": boundary["accuracy"],
               "feasible": feasible, "selection_score": score}
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch+1}/{args.epochs}: old-val={legacy['accuracy']:.3f} "
                  f"new-val={overall['accuracy']:.3f} boundary={boundary['accuracy']:.3f} loss={row['loss']:.3f}", flush=True)
    assert best is not None
    model.intent_encoder.load_state_dict(best["encoder"]); model.intent_classifier.load_state_dict(best["classifier"])
    for name, value in model.state_dict().items():
        if not name.startswith(("intent_encoder.", "intent_classifier.")) and not torch.equal(value, initial[name]):
            raise RuntimeError(f"frozen generator tensor changed: {name}")
    path = out / "generator.pt"; torch.save(model.state_dict(), path)
    # Snapshot the accepted new training examples after this phase. FIFO
    # eviction is explicit; users should not interpret this as infinite replay.
    replay.add([ReplayExample(row.message, row.label, "phase12") for row in new])
    replay.save(out / "replay_phase12.pt")
    report = {"configuration": vars(args), "generator_sha256": sha256(path), "source_sha256": sha256(args.generator),
              "teacher_sha256": sha256(args.teacher), "guard_fingerprint": guard.fingerprint,
              "preflight": {"old_accepted": len(old), "old_blocked": old_blocked, "new_accepted": len(new),
                            "new_blocked": new_blocked, "protected_hashes": len(guard.contents),
                            "old_replay_class_counts": {INTENT_NAMES[k]: v for k, v in Counter(r.label for r in chosen).items()}},
              "mix_per_batch": {"new": new_count, "old": old_count, "fraction_old": old_count / args.batch_size},
              "selection_rule": "validation only: feasible new-overall >=.85 and boundary >=.80, then .45 legacy micro + .35 new micro + .20 boundary",
              "selection_history": "The feasibility threshold was raised from .82 to the user-requested .85 after an initial validation-only run; final test results were not inspected for this change.",
              "selected_epoch": selected_epoch, "epochs": history,
              "validation": {name: _metrics(model, cache, args.feature_batch_size, args.threshold) for name, cache in val.items()},
              "frozen_non_intent_tensors_unchanged": True,
              "limitations": "Rehearsal and teacher distillation are not guarantees against forgetting; old development and fresh tests are evaluated after selection."}
    (out / "intent_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator", default="runs/phase11/generator.pt")
    p.add_argument("--teacher", default="runs/phase10/generator.pt")
    p.add_argument("--generator-heads", type=int, default=4)
    p.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--output-dir", default="runs/phase12")
    p.add_argument("--epochs", type=int, default=55)
    p.add_argument("--steps-per-epoch", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--feature-batch-size", type=int, default=32)
    p.add_argument("--replay-capacity", type=int, default=1000)
    p.add_argument("--old-fraction", type=float, default=0.30)
    p.add_argument("--remember-weight", type=float, default=2.0)
    p.add_argument("--forget-weight", type=float, default=2.0)
    p.add_argument("--distill-weight", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=0.00025)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=121210)
    p.add_argument("--threads", type=int, default=1)
    train(p.parse_args())


if __name__ == "__main__":
    main()
