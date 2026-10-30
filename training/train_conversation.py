"""Train the Phase 10 byte-intent head and bounded social response decoder.

Only the new intent byte encoder/head and separate response decoder are optimized.
The Phase 9 backbone/edit weights are frozen and checked bit-for-bit. Selection is
on validation macro intent accuracy and validation social response loss; test
inputs are evaluated only after selection. All data are synthetic UTF-8 bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.conversation_examples import ConversationExample, conversation_examples
from data.edit_curriculum import curriculum_examples
from data.holdout_registry import synthetic_holdouts
from data.ingest_real_traces import load_pairs
from data.masked_edits import byte_coverage_examples, masked_examples
from data.positional_edits import positional_examples
from data.structured_edits import structured_examples
from models.conversation import ConversationTurn, pack_conversation
from models.conversation_intents import INTENT_NAMES, NUM_INTENTS, ConversationIntent
from models.dialogue_response import ByteDialogueDecoder
from models.holdout import HoldoutGuard, HoldoutViolation
from training.train_confidence import load_generator


_HISTORY = (
    ConversationTurn(b"Earlier we discussed travel plans.", b"I am listening."),
    ConversationTurn(b"I have a project to discuss.", b"Tell me more about it."),
    ConversationTurn(b"The weather is pleasant today.", b"That sounds nice."),
)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_guard(args: argparse.Namespace, validation: list[ConversationExample], test: list[ConversationExample]) -> HoldoutGuard:
    records = load_pairs(args.pairs, split="validation") + load_pairs(args.pairs, split="test")
    records += synthetic_holdouts(args.processed_dir)
    records += curriculum_examples(split="validation", count_per_stage=2, seed=881)
    records += positional_examples("validation", 24, seed=9009, long_count=2)
    records += positional_examples("test", 64, seed=9009, long_count=4)
    records += structured_examples("validation", 64, seed=9509, long_count=8)
    records += structured_examples("test", 256, seed=9509, long_count=8)
    records += masked_examples("validation", 8, seed=9090) + masked_examples("test", 16, seed=9090)
    records += byte_coverage_examples(seed=9090)
    # Protect held-out inputs, not recurring social-response templates. The
    # latter are deliberately shared and are reported as seen target strings.
    records += [{"input": row.message} for row in validation + test]
    return HoldoutGuard.from_records(records)


def guarded_training_rows(rows: list[ConversationExample], guard: HoldoutGuard, *, max_len: int) -> tuple[list[ConversationExample], int]:
    accepted, blocked = [], 0
    for row in rows:
        try:
            guard.validate(row.message, row.response)
            # This ensures every possible context augmentation is preflighted.
            for turn in _HISTORY:
                guard.validate(pack_conversation(row.message, [turn], max_len), row.response,
                               extra=(turn.user, turn.assistant))
        except HoldoutViolation:
            blocked += 1
            continue
        accepted.append(row)
    counts = Counter(row.intent for row in accepted)
    if any(counts[intent] < 8 for intent in ConversationIntent):
        raise RuntimeError(f"holdout preflight left too few examples per class: {counts}")
    return accepted, blocked


@torch.no_grad()
def collect_features(model, rows: list[ConversationExample], *, include_history: bool, batch_size: int):
    """Cache only frozen backbone features; the byte encoder is trained live."""
    samples: list[tuple[bytes, ConversationExample]] = []
    for index, row in enumerate(rows):
        samples.append((pack_conversation(row.message, [], model.max_len), row))
        if include_history:
            samples.append((pack_conversation(row.message, [_HISTORY[index % len(_HISTORY)]], model.max_len), row))
    width = max(len(item[0]) for item in samples)
    ids = torch.zeros((len(samples), width), dtype=torch.long)
    lengths = torch.tensor([len(item[0]) for item in samples], dtype=torch.long)
    focus = torch.tensor([len(payload) - len(row.message) for payload, row in samples], dtype=torch.long)
    labels = torch.tensor([int(row.intent) for _, row in samples], dtype=torch.long)
    ordered = [row for _, row in samples]
    for i, (payload, _) in enumerate(samples):
        ids[i, :len(payload)] = torch.tensor(list(payload), dtype=torch.long)
    base = []
    device = model.byte_emb.weight.device
    for start in range(0, len(samples), batch_size):
        batch_ids = ids[start:start + batch_size].to(device)
        batch_lengths = lengths[start:start + batch_size].to(device)
        batch_ids = batch_ids[:, :int(batch_lengths.max())]
        base.append(model.conversation_backbone_features(batch_ids, batch_lengths).detach().cpu())
    return torch.cat(base), ids, lengths, focus, labels, ordered


def batch_features(model, cached, indices: torch.Tensor) -> torch.Tensor:
    base, ids, lengths, focus, _, _ = cached
    batch_lengths = lengths[indices]
    return model.conversation_features(ids[indices, :int(batch_lengths.max())], batch_lengths,
                                       focus_start=focus[indices], backbone_features=base[indices])


@torch.no_grad()
def encode_cached(model, cached, batch_size: int) -> torch.Tensor:
    return torch.cat([batch_features(model, cached, idx)
                      for idx in torch.arange(len(cached[4])).split(batch_size)])


@torch.no_grad()
def intent_metrics(head, features: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, Any]:
    logits = head(features)
    probabilities = torch.softmax(logits, dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    matrix = torch.zeros(NUM_INTENTS, NUM_INTENTS, dtype=torch.long)
    for actual, predicted in zip(labels.tolist(), predictions.tolist()):
        matrix[actual, predicted] += 1
    per_class = {}
    for index, name in enumerate(INTENT_NAMES):
        count = int(matrix[index].sum().item())
        correct = int(matrix[index, index].item())
        per_class[name] = {"count": count, "correct": correct, "accuracy": correct / count if count else None}
    macro = sum(item["accuracy"] for item in per_class.values() if item["accuracy"] is not None) / NUM_INTENTS
    covered = confidence >= threshold
    return {"count": int(labels.numel()), "correct": int((predictions == labels).sum().item()),
            "accuracy": float((predictions == labels).float().mean()), "macro_accuracy": float(macro),
            "nll": float(F.cross_entropy(logits, labels)), "per_class": per_class,
            "confusion_matrix": matrix.tolist(), "threshold": threshold,
            "threshold_coverage": float(covered.float().mean()),
            "threshold_accuracy": float((predictions[covered] == labels[covered]).float().mean()) if bool(covered.any()) else None}


def response_targets(rows: list[ConversationExample], device: torch.device) -> tuple[torch.Tensor, list[int]]:
    selected = [i for i, row in enumerate(rows) if row.intent == ConversationIntent.CHAT and row.response is not None]
    if not selected:
        raise ValueError("no social response targets")
    targets = [list(rows[i].response or b"") + [ByteDialogueDecoder.EOS] for i in selected]
    width = max(map(len, targets))
    result = torch.full((len(targets), width), ByteDialogueDecoder.IGNORE, device=device, dtype=torch.long)
    for i, target in enumerate(targets):
        result[i, :len(target)] = torch.tensor(target, device=device, dtype=torch.long)
    return result, selected


@torch.no_grad()
def response_metrics(decoder: ByteDialogueDecoder, features: torch.Tensor,
                     rows: list[ConversationExample]) -> dict[str, Any]:
    targets, selected = response_targets(rows, features.device)
    selected_features = features[selected]
    logits = decoder(selected_features, targets)
    loss = float(F.cross_entropy(logits.reshape(-1, 257), targets.reshape(-1), ignore_index=ByteDialogueDecoder.IGNORE))
    results, counts = [], Counter()
    for feature, index in zip(selected_features, selected):
        output, complete = decoder.generate(feature.unsqueeze(0))
        row = rows[index]
        try:
            text_ok = bool(output.decode("utf-8").strip()) and all(c.isprintable() or c in "\n\t" for c in output.decode("utf-8"))
        except UnicodeDecodeError:
            text_ok = False
        exact = complete and output == row.response
        counts["exact_seen_template"] += int(exact)
        counts["complete"] += int(complete)
        counts["valid_text"] += int(complete and text_ok)
        results.append({"input": row.message.decode("utf-8"), "category": row.response_category,
                        "output": output.decode("utf-8", errors="replace"), "complete": complete,
                        "exact_seen_template": exact})
    return {"count": len(selected), "loss": loss, **dict(counts), "examples": results}


def train(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.intent_epochs, args.response_epochs, args.batch_size, args.feature_batch_size, args.threads) < 1:
        raise ValueError("epochs, batches and threads must be positive")
    if not math.isfinite(args.intent_lr) or args.intent_lr <= 0 or not math.isfinite(args.response_lr) or args.response_lr <= 0:
        raise ValueError("learning rates must be finite and positive")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); random.seed(args.seed)
    model = load_generator(args.generator, "cpu", args.generator_heads).eval()
    if model.max_len < 64:
        raise ValueError("conversation training expects at least 64 context bytes")
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_rows, validation, test = (conversation_examples(name) for name in ("train", "validation", "test"))
    sets = [set(row.message for row in split) for split in (train_rows, validation, test)]
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("conversation input splits overlap")
    guard = make_guard(args, validation, test)
    train_rows, blocked = guarded_training_rows(train_rows, guard, max_len=model.max_len)
    print(f"preflight: {len(train_rows)} train inputs accepted, {blocked} blocked; {len(guard.contents)} held-out content hashes", flush=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in (*model.intent_encoder.parameters(), *model.intent_classifier.parameters()):
        parameter.requires_grad_(True)
    print("extracting frozen causal byte features...", flush=True)
    train_cache = collect_features(model, train_rows, include_history=True, batch_size=args.feature_batch_size)
    val_cache = collect_features(model, validation, include_history=True, batch_size=args.feature_batch_size)
    train_y, train_order = train_cache[4], train_cache[5]
    val_y, val_order = val_cache[4], val_cache[5]
    counts = torch.bincount(train_y, minlength=NUM_INTENTS).float()
    weights = counts.sum() / (NUM_INTENTS * counts)
    intent_parameters = list(model.intent_encoder.parameters()) + list(model.intent_classifier.parameters())
    optimizer = torch.optim.Adam(intent_parameters, lr=args.intent_lr)
    best_rank, best_state, best_epoch, history = None, None, 0, []
    for epoch in range(args.intent_epochs):
        permutation = torch.randperm(len(train_y))
        total = 0.0
        for indices in permutation.split(args.batch_size):
            features = batch_features(model, train_cache, indices)
            loss = F.cross_entropy(model.intent_classifier(features), train_y[indices], weight=weights)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite intent loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(intent_parameters, 2.0, error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach()) * len(indices)
        val_x = encode_cached(model, val_cache, args.feature_batch_size)
        metrics = intent_metrics(model.intent_classifier, val_x, val_y, args.threshold)
        rank = (metrics["macro_accuracy"], -metrics["nll"])
        if best_rank is None or rank > best_rank:
            best_rank, best_epoch = rank, epoch + 1
            best_state = {
                "encoder": {key: value.detach().clone() for key, value in model.intent_encoder.state_dict().items()},
                "classifier": {key: value.detach().clone() for key, value in model.intent_classifier.state_dict().items()},
            }
        history.append({"epoch": epoch + 1, "loss": total / len(train_y), "validation_macro_accuracy": metrics["macro_accuracy"],
                        "validation_accuracy": metrics["accuracy"], "validation_nll": metrics["nll"]})
        if (epoch + 1) % max(1, args.intent_epochs // 10) == 0 or epoch == 0:
            print(f"intent {epoch+1}/{args.intent_epochs}: loss={history[-1]['loss']:.4f} val macro={metrics['macro_accuracy']:.3f} val={metrics['accuracy']:.3f}", flush=True)
    assert best_state is not None
    model.intent_encoder.load_state_dict(best_state["encoder"], strict=True)
    model.intent_classifier.load_state_dict(best_state["classifier"], strict=True); model.intent_trained.fill_(True)
    # Ensure this experiment did not modify Phase 9 policy/backbone/buffers.
    for key, value in model.state_dict().items():
        if key != "intent_trained" and not key.startswith(("intent_classifier.", "intent_encoder.")) and not torch.equal(value, original[key]):
            raise RuntimeError(f"non-intent generator tensor changed: {key}")
    train_x = encode_cached(model, train_cache, args.feature_batch_size)
    val_x = encode_cached(model, val_cache, args.feature_batch_size)
    decoder = ByteDialogueDecoder(model.d_model)
    targets, train_selected = response_targets(train_order, torch.device("cpu"))
    social_x = train_x[train_selected]
    val_targets, val_selected = response_targets(val_order, torch.device("cpu"))
    social_val_x = val_x[val_selected]
    social_categories = Counter(train_order[i].response_category for i in train_selected)
    sample_weights = torch.tensor([1.0 / social_categories[train_order[i].response_category] for i in train_selected])
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.response_lr)
    decoder_best, response_epoch, response_loss, response_history = None, 0, float("inf"), []
    for epoch in range(args.response_epochs):
        # Balanced social categories; indices only refer to the training split.
        indices = torch.multinomial(sample_weights, len(sample_weights), replacement=True)
        total = 0.0
        decoder.train()
        for batch in indices.split(args.batch_size):
            logits = decoder(social_x[batch], targets[batch])
            loss = F.cross_entropy(logits.reshape(-1, 257), targets[batch].reshape(-1), ignore_index=ByteDialogueDecoder.IGNORE)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite response loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach()) * len(batch)
        decoder.eval()
        with torch.no_grad():
            logits = decoder(social_val_x, val_targets)
            vloss = float(F.cross_entropy(logits.reshape(-1, 257), val_targets.reshape(-1), ignore_index=ByteDialogueDecoder.IGNORE))
        if vloss < response_loss:
            response_loss, response_epoch = vloss, epoch + 1
            decoder_best = {key: value.detach().clone() for key, value in decoder.state_dict().items()}
        response_history.append({"epoch": epoch + 1, "loss": total / len(indices), "validation_loss": vloss})
        if (epoch + 1) % max(1, args.response_epochs // 10) == 0 or epoch == 0:
            print(f"response {epoch+1}/{args.response_epochs}: loss={response_history[-1]['loss']:.4f} val loss={vloss:.4f}", flush=True)
    assert decoder_best is not None
    decoder.load_state_dict(decoder_best, strict=True); decoder.response_trained.fill_(True); decoder.eval()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    generator_path, decoder_path = output / "generator.pt", output / "response.pt"
    torch.save(model.state_dict(), generator_path)
    torch.save({"schema_version": 1, "context_dim": decoder.context_dim, "hidden_dim": decoder.hidden_dim,
                "generator_sha256": sha256(generator_path), "generator_n_heads": model.n_heads,
                "state_dict": decoder.state_dict()}, decoder_path)
    # Test examples are read only after both independent validation selections.
    test_cache = collect_features(model, test, include_history=False, batch_size=args.feature_batch_size)
    contextual_cache = collect_features(model, test, include_history=True, batch_size=args.feature_batch_size)
    test_x, test_y, test_order = encode_cached(model, test_cache, args.feature_batch_size), test_cache[4], test_cache[5]
    contextual_x, contextual_y = encode_cached(model, contextual_cache, args.feature_batch_size), contextual_cache[4]
    report = {
        "configuration": vars(args), "source_checkpoint_sha256": sha256(args.generator),
        "generator_checkpoint_sha256": sha256(generator_path), "response_checkpoint_sha256": sha256(decoder_path),
        "data": {"train_original": len(sets[0]), "train_after_preflight": len(train_rows), "train_blocked": blocked,
                 "validation_inputs": len(validation), "test_inputs": len(test),
                 "train_class_counts": {INTENT_NAMES[int(k)]: v for k, v in Counter(row.intent for row in train_rows).items()},
                 "holdout_content_hashes": len(guard.contents), "holdout_fingerprint": guard.fingerprint,
                 "split_policy": "disjoint synthetic message templates and raw inputs; guarded against known earlier holdouts",
                 "response_limitation": "three response targets are seen canned templates, not held-out output text"},
        "parameters": {"phase9_generator": sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("intent_classifier.", "intent_encoder."))),
                       "intent_encoder": sum(p.numel() for p in model.intent_encoder.parameters()),
                       "intent_head": sum(p.numel() for p in model.intent_classifier.parameters()),
                       "generator_total": sum(p.numel() for p in model.parameters()),
                       "response_decoder": sum(p.numel() for p in decoder.parameters()),
                       "non_intent_generator_tensors_unchanged": True},
        "intent": {"selection": "highest validation macro accuracy, then lowest validation NLL; test not used",
                   "selected_epoch": best_epoch, "epochs": history,
                   "validation": intent_metrics(model.intent_classifier, val_x, val_y, args.threshold),
                   "test_no_history": intent_metrics(model.intent_classifier, test_x, test_y, args.threshold),
                   "test_mixed_history": intent_metrics(model.intent_classifier, contextual_x, contextual_y, args.threshold)},
        "response": {"selection": "lowest held-out-input validation response loss; response targets are seen templates",
                     "selected_epoch": response_epoch, "epochs": response_history,
                     "validation": response_metrics(decoder, val_x, val_order),
                     "test_no_history": response_metrics(decoder, test_x, test_order)},
    }
    (output / "training_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="runs/phase9/structured_generator.pt")
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="runs/phase10")
    parser.add_argument("--intent-epochs", type=int, default=80)
    parser.add_argument("--response-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--intent-lr", type=float, default=0.005)
    parser.add_argument("--response-lr", type=float, default=0.003)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=101010)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
