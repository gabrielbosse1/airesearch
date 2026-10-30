"""Fine-tune Phase 10 intent components on paired missing-argument examples.

The edit backbone is frozen. Validation-only selection balances the new
boundary/overall sets with the earlier validation set. All final test examples
are registered before preflight and evaluated only after model selection.
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
from data.conversation_response_v2 import developmental_retention_cases, response_examples, retention_cases
from data.phase11_eval_sets import overall_intent_examples
from data.vague_intent_pairs import hard_negative_examples
from evaluation.phase10_benchmark import _CHALLENGE
from models.conversation import pack_conversation
from models.conversation_intents import ConversationIntent, INTENT_NAMES, NUM_INTENTS
from models.holdout import HoldoutGuard, HoldoutViolation
from training.train_confidence import load_generator
from training.train_conversation import (_HISTORY, batch_features, collect_features, encode_cached,
                                         intent_metrics, make_guard, sha256)


_LEGACY_VAGUE_EDIT = {
    b"please edit the document", b"fix this typo", b"modify this text",
    b"insert a character into this text", b"change this sentence",
}


def _normal(data: bytes) -> bytes:
    return b" ".join(data.lower().rstrip(b".!? ").split())


def prior_records(args: argparse.Namespace):
    old_val = conversation_examples("validation")
    old_test = conversation_examples("test")
    base = make_guard(args, old_val, old_test)
    additions = [{"input": msg} for group in _CHALLENGE.values() for msg in group]
    additions += [{"input": msg} for split in ("validation", "test")
                  for msg, _ in hard_negative_examples(split) + overall_intent_examples(split)]
    additions += [{"input": case.statement} for case in retention_cases() + developmental_retention_cases()]
    # Protect new response-validation histories as well; repeated response styles
    # are not held-out target strings, so no claim of novel response targets.
    additions += [{"input": example.history[0].user} for example in response_examples("validation") if example.history]
    new = HoldoutGuard.from_records(additions)
    return HoldoutGuard(base.contents | new.contents, base.held_out_hashes | new.held_out_hashes)


def train_rows(guard: HoldoutGuard, max_len: int):
    raw = []
    for row in conversation_examples("train"):
        label = ConversationIntent.CLARIFY if row.intent == ConversationIntent.EDIT and _normal(row.message) in _LEGACY_VAGUE_EDIT else row.intent
        raw.append(ConversationExample(row.message, label))
    raw += [ConversationExample(msg, label) for msg, label in hard_negative_examples("train")]
    # Mere first-person statements are conversational content, not authorization
    # to run an edit, delete a record, or claim an explicit memory command.
    for example in response_examples("train"):
        if example.intent == ConversationIntent.CHAT and example.message.lower().startswith(b"my "):
            raw.append(ConversationExample(example.message, ConversationIntent.CHAT))
    unique: dict[bytes, ConversationExample] = {}
    for row in raw:
        old = unique.get(row.message)
        if old is not None and old.intent != row.intent:
            raise ValueError(f"conflicting training intent labels: {row.message!r}")
        unique[row.message] = row
    accepted, blocked = [], 0
    for row in unique.values():
        try:
            guard.validate(row.message)
            for h in _HISTORY:
                guard.validate(pack_conversation(row.message, [h], max_len), extra=(h.user, h.assistant))
        except HoldoutViolation:
            blocked += 1
            continue
        accepted.append(row)
    counts = Counter(row.intent for row in accepted)
    if any(counts[intent] < 8 for intent in ConversationIntent):
        raise RuntimeError(f"insufficient examples after preflight: {counts}")
    return accepted, blocked


def _examples(pairs):
    return [ConversationExample(msg, label) for msg, label in pairs]


@torch.no_grad()
def score(model, cache, batch_size: int, threshold: float):
    features = encode_cached(model, cache, batch_size)
    return intent_metrics(model.intent_classifier, features, cache[4], threshold)


def train(args: argparse.Namespace):
    if min(args.epochs, args.batch_size, args.feature_batch_size, args.threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("invalid training hyperparameters")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); random.seed(args.seed)
    model = load_generator(args.generator, "cpu", args.generator_heads).eval()
    if not bool(model.intent_trained):
        raise RuntimeError("source must have a trained Phase 10 intent model")
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    guard = prior_records(args)
    rows, blocked = train_rows(guard, model.max_len)
    boundary_val = _examples(hard_negative_examples("validation"))
    overall_val = _examples(overall_intent_examples("validation"))
    previous_val = conversation_examples("validation")
    print(f"preflight: {len(rows)} accepted, {blocked} blocked, {len(guard.contents)} protected raw hashes", flush=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.intent_encoder.parameters()) + list(model.intent_classifier.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    train_cache = collect_features(model, rows, include_history=True, batch_size=args.feature_batch_size)
    caches = {
        "boundary": collect_features(model, boundary_val, include_history=False, batch_size=args.feature_batch_size),
        "overall": collect_features(model, overall_val, include_history=False, batch_size=args.feature_batch_size),
        "previous": collect_features(model, previous_val, include_history=False, batch_size=args.feature_batch_size),
    }
    labels = train_cache[4]
    counts = torch.bincount(labels, minlength=NUM_INTENTS).float()
    class_weights = counts.sum() / (NUM_INTENTS * counts)
    class_weights[int(ConversationIntent.CLARIFY)] *= args.clarify_weight
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    best, best_rank, best_epoch, history = None, None, 0, []
    for epoch in range(args.epochs):
        total = 0.0
        for indexes in torch.randperm(len(labels)).split(args.batch_size):
            features = batch_features(model, train_cache, indexes)
            loss = F.cross_entropy(model.intent_classifier(features), labels[indexes], weight=class_weights)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite classifier loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2.0, error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach()) * len(indexes)
        boundary = score(model, caches["boundary"], args.feature_batch_size, args.threshold)
        overall = score(model, caches["overall"], args.feature_batch_size, args.threshold)
        previous = score(model, caches["previous"], args.feature_batch_size, args.threshold)
        # Predeclared, validation-only selection. Boundary accuracy covers both
        # vague and specific examples and avoids claiming safe performance from
        # predicting every example as CLARIFY.
        combined = 0.4 * boundary["accuracy"] + 0.35 * overall["macro_accuracy"] + 0.25 * previous["macro_accuracy"]
        rank = (combined, boundary["accuracy"], -overall["nll"])
        if best_rank is None or rank > best_rank:
            best, best_rank, best_epoch = {
                "encoder": {k: v.detach().clone() for k, v in model.intent_encoder.state_dict().items()},
                "classifier": {k: v.detach().clone() for k, v in model.intent_classifier.state_dict().items()},
            }, rank, epoch + 1
        history.append({"epoch": epoch + 1, "training_loss": total / len(labels), "selection_score": combined,
                        "boundary_accuracy": boundary["accuracy"], "boundary_vague": boundary["per_class"]["clarify"],
                        "overall_macro": overall["macro_accuracy"], "previous_macro": previous["macro_accuracy"]})
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch+1}/{args.epochs}: loss={history[-1]['training_loss']:.4f} "
                  f"boundary={boundary['correct']}/{boundary['count']} overall_macro={overall['macro_accuracy']:.3f} "
                  f"previous_macro={previous['macro_accuracy']:.3f}", flush=True)
    assert best is not None
    model.intent_encoder.load_state_dict(best["encoder"], strict=True)
    model.intent_classifier.load_state_dict(best["classifier"], strict=True)
    for name, value in model.state_dict().items():
        if not name.startswith(("intent_encoder.", "intent_classifier.")) and not torch.equal(value, original[name]):
            raise RuntimeError(f"unrelated generator tensor changed: {name}")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "generator.pt"; torch.save(model.state_dict(), checkpoint)
    # No training or epoch selection occurs after these test metrics are read.
    boundary_test = _examples(hard_negative_examples("test"))
    overall_test = _examples(overall_intent_examples("test"))
    def test_metrics(rows):
        cache = collect_features(model, rows, include_history=False, batch_size=args.feature_batch_size)
        return score(model, cache, args.feature_batch_size, args.threshold)
    report = {"configuration": vars(args), "source_sha256": sha256(args.generator), "checkpoint_sha256": sha256(checkpoint),
              "preflight": {"accepted": len(rows), "blocked": blocked, "protected_content_hashes": len(guard.contents),
                            "fingerprint": guard.fingerprint,
                            "class_counts": {INTENT_NAMES[int(k)]: v for k, v in Counter(r.intent for r in rows).items()}},
              "selection_rule": "0.4 boundary-validation accuracy + 0.35 new-overall-validation macro + 0.25 prior-validation macro; test not used",
              "selected_epoch": best_epoch, "epochs": history,
              "validation": {key: score(model, cache, args.feature_batch_size, args.threshold) for key, cache in caches.items()},
              "test": {"boundary": test_metrics(boundary_test), "overall": test_metrics(overall_test),
                       "phase10_development_test": test_metrics(conversation_examples("test"))},
              "frozen_non_intent_tensors_unchanged": True}
    (output / "intent_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="runs/phase10/generator.pt")
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="runs/phase11")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0006)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--clarify-weight", type=float, default=1.25)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=111011)
    parser.add_argument("--threads", type=int, default=1)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
