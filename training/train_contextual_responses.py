"""Train a raw-byte, intent/context-conditioned pointer response decoder.

The generator and memory encoders are frozen. Training uses varied synthetic
response styles, direct raw-byte history and simulated episode-value vectors;
the runtime supplies actual retrieved vectors. Validation loss selects a
checkpoint. Test values are held out by full first-turn bytes and scoring only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from data.conversation_response_v2 import ResponseExample, copy_generalization_examples, response_examples, retention_cases
from models.contextual_response import ContextualByteDecoder, pack_response_context
from models.conversation import ConversationTurn
from models.conversation_intents import ConversationIntent, INTENT_NAMES, NUM_INTENTS
from models.holdout import HoldoutViolation
from models.memory_model import MemoryModel
from models.procedural import byte_pattern_embedding
from training.train_confidence import load_generator
from training.train_conversation import sha256
from training.train_vague_intents import prior_records


def preflight(rows: list[ResponseExample], guard):
    accepted, blocked = [], 0
    for row in rows:
        try:
            guard.validate(row.message, row.response,
                           extra=tuple(raw for turn in row.history for raw in (turn.user, turn.assistant)))
        except HoldoutViolation:
            blocked += 1; continue
        accepted.append(row)
    counts = Counter(row.intent for row in accepted)
    if any(counts[intent] < 4 for intent in ConversationIntent):
        raise RuntimeError(f"insufficient response examples after preflight: {counts}")
    return accepted, blocked


@torch.no_grad()
def tensorize(model, memory: MemoryModel, rows: list[ResponseExample], *, feature_batch: int,
              max_input_bytes: int):
    source_payloads, segment_payloads, prompts, vectors = [], [], [], []
    for row in rows:
        source, segments = pack_response_context(row.message, row.history, max_bytes=max_input_bytes)
        source_payloads.append(source); segment_payloads.append(segments)
        prompts.append(b"U:" + row.message)
        if row.history:
            turn = row.history[-1]
            sketch = byte_pattern_embedding(turn.user + b"\n" + turn.assistant, model.d_model, device=torch.device("cpu"))
            vectors.append(memory.value_encoder(sketch.unsqueeze(0))[0].detach())
        else:
            vectors.append(None)
    n = len(rows)
    src_width, target_width = max(map(len, source_payloads)), max(len(row.response) + 1 for row in rows)
    source = torch.zeros((n, src_width), dtype=torch.long)
    segments = torch.zeros_like(source)
    lengths = torch.tensor([len(x) for x in source_payloads], dtype=torch.long)
    targets = torch.full((n, target_width), ContextualByteDecoder.IGNORE, dtype=torch.long)
    alignments = torch.full_like(targets, ContextualByteDecoder.IGNORE)
    target_lengths = torch.tensor([len(row.response) + 1 for row in rows], dtype=torch.long)
    intent = torch.tensor([int(row.intent) for row in rows], dtype=torch.long)
    for i, (payload, seg, row) in enumerate(zip(source_payloads, segment_payloads, rows)):
        source[i, :len(payload)] = torch.tensor(list(payload), dtype=torch.long)
        segments[i, :len(seg)] = torch.tensor(list(seg), dtype=torch.long)
        target = list(row.response) + [ContextualByteDecoder.EOS]
        targets[i, :len(target)] = torch.tensor(target, dtype=torch.long)
        if row.intent == ConversationIntent.RECALL and row.history:
            match = re.fullmatch(rb"My [a-z0-9 -]+ is (.+?)[.!?]?", row.history[-1].user, re.IGNORECASE | re.ASCII)
            if match is not None:
                value = match.group(1)
                # Align only a matching value in the source *history* and its
                # response. This label is derived solely from the training row.
                source_start = payload.find(value, len(b"Q:") + len(row.message))
                target_start = row.response.find(value)
                if source_start >= 0 and target_start >= 0:
                    alignments[i, target_start:target_start + len(value)] = torch.arange(source_start, source_start + len(value))
    feature_parts = []
    for start in range(0, n, feature_batch):
        end = min(start + feature_batch, n)
        batch = prompts[start:end]
        width = max(map(len, batch))
        ids = torch.zeros((len(batch), width), dtype=torch.long)
        prompt_lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)
        focus = torch.full((len(batch),), 2, dtype=torch.long)
        memory_vectors = torch.zeros((len(batch), 1, model.d_model))
        memory_mask = torch.zeros((len(batch), 1), dtype=torch.bool)
        for i, payload in enumerate(batch):
            ids[i, :len(payload)] = torch.tensor(list(payload), dtype=torch.long)
            if vectors[start + i] is not None:
                memory_vectors[i, 0] = vectors[start + i]
                memory_mask[i, 0] = True
        features = model.conversation_features(ids, prompt_lengths, focus_start=focus,
                                               memory_context=memory_vectors, memory_mask=memory_mask)
        feature_parts.append(features.detach())
    return {"context": torch.cat(feature_parts), "intent": intent, "source": source, "segments": segments,
            "lengths": lengths, "targets": targets, "alignments": alignments,
            "target_lengths": target_lengths, "rows": rows}


def loss_for(decoder: ContextualByteDecoder, data, indexes: torch.Tensor, *, alignment_weight: float = 0.0) -> torch.Tensor:
    source_width = int(data["lengths"][indexes].max())
    target_width = int(data["target_lengths"][indexes].max())
    targets = data["targets"][indexes, :target_width]
    output = decoder(data["context"][indexes], data["intent"][indexes], data["source"][indexes, :source_width],
                     data["segments"][indexes, :source_width], data["lengths"][indexes], targets,
                     return_attention_scores=alignment_weight > 0)
    if alignment_weight > 0:
        log_probs, attention_scores = output
    else:
        log_probs = output
    loss = F.nll_loss(log_probs.reshape(-1, 257), targets.reshape(-1), ignore_index=ContextualByteDecoder.IGNORE)
    if alignment_weight > 0:
        alignment = data["alignments"][indexes, :target_width]
        if bool((alignment != ContextualByteDecoder.IGNORE).any()):
            loss = loss + alignment_weight * F.cross_entropy(attention_scores.reshape(-1, source_width), alignment.reshape(-1),
                                                             ignore_index=ContextualByteDecoder.IGNORE)
    return loss


@torch.no_grad()
def evaluate_loss(decoder, data, batch_size):
    losses, counts = 0.0, 0
    for indexes in torch.arange(len(data["rows"])).split(batch_size):
        count = int(data["target_lengths"][indexes].sum())
        losses += float(loss_for(decoder, data, indexes)) * count
        counts += count
    return losses / counts


def contains_exact_value(output: bytes, value: bytes) -> bool:
    return bool(re.search(rb"(?<!\w)" + re.escape(value) + rb"(?!\w)", output, re.IGNORECASE | re.ASCII))


@torch.no_grad()
def evaluate_validation_copy(decoder, data):
    """One greedy decode per distinct validation statement/question pair."""
    selected = []
    seen = set()
    for i, row in enumerate(data["rows"]):
        if row.intent != ConversationIntent.RECALL or not row.history:
            continue
        key = (row.message, row.history[-1].user)
        if key not in seen:
            seen.add(key); selected.append(i)
    correct = 0
    for i in selected:
        row = data["rows"][i]
        match = re.fullmatch(rb"My [a-z0-9 -]+ is (.+?)[.!?]?", row.history[-1].user, re.IGNORECASE | re.ASCII)
        if match is None:
            continue
        length = int(data["lengths"][i])
        response, complete = decoder.generate(data["context"][i:i+1], data["intent"][i:i+1],
                                              data["source"][i:i+1, :length], data["segments"][i:i+1, :length],
                                              data["lengths"][i:i+1])
        correct += int(complete and contains_exact_value(response, match.group(1)))
    return {"correct": correct, "count": len(selected)}


@torch.no_grad()
def decoder_only_retention(decoder, model, memory, *, feature_batch, max_input):
    cases = retention_cases()
    examples = [ResponseExample(case.question, (ConversationTurn(case.statement, b"Thanks for sharing."),),
                                ConversationIntent.RECALL, b"") for case in cases]
    data = tensorize(model, memory, examples, feature_batch=feature_batch, max_input_bytes=max_input)
    result = []
    for i, case in enumerate(cases):
        length = int(data["lengths"][i])
        response, complete = decoder.generate(data["context"][i:i+1], data["intent"][i:i+1],
                                              data["source"][i:i+1, :length], data["segments"][i:i+1, :length],
                                              data["lengths"][i:i+1])
        correct = complete and contains_exact_value(response, case.value)
        result.append({"statement": case.statement.decode(), "question": case.question.decode(),
                       "expected_for_scoring": case.value.decode(), "output": response.decode("utf-8", "replace"),
                       "complete": complete, "correct": correct})
    return {"count": len(result), "correct": sum(row["correct"] for row in result), "examples": result,
            "scope": "decoder only with gold recall intent and simulated memory vectors; not an end-to-end score"}


def train(args: argparse.Namespace):
    if min(args.epochs, args.batch_size, args.feature_batch_size, args.threads, args.copy_eval_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("invalid training hyperparameters")
    if (not math.isfinite(args.alignment_weight) or args.alignment_weight < 0
            or not math.isfinite(args.recall_weight) or args.recall_weight <= 0):
        raise ValueError("invalid alignment or recall weighting")
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); random.seed(args.seed)
    model = load_generator(args.generator, "cpu", args.generator_heads).eval()
    if not bool(model.intent_trained):
        raise RuntimeError("intent checkpoint must be trained")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if getattr(args, "phase12_guard", False):
        from data.phase12_eval_sets import DIVERSITY_VALIDATION_PROMPTS
        from models.holdout import HoldoutGuard
        from training.train_replay_intents import phase12_guard
        base = phase12_guard(args)
        extra = HoldoutGuard.from_records([{"input": prompt} for prompt in DIVERSITY_VALIDATION_PROMPTS])
        guard = HoldoutGuard(base.contents | extra.contents, base.held_out_hashes | extra.held_out_hashes)
    else:
        guard = prior_records(args)
    rows, blocked = preflight(response_examples("train") + copy_generalization_examples(count=args.synthetic_facts), guard)
    val = response_examples("validation")
    print(f"preflight: {len(rows)} responses accepted, {blocked} blocked; {len(guard.contents)} protected hashes", flush=True)
    # Same architecture as runtime, with fixed seeded weights. Memory records
    # themselves are not written during response training; raw history was
    # audited above. Runtime uses actual retrieve() values, not these examples.
    memory = MemoryModel(d_model=model.d_model, capacity=3).eval()
    train_data = tensorize(model, memory, rows, feature_batch=args.feature_batch_size, max_input_bytes=args.max_input_bytes)
    val_data = tensorize(model, memory, val, feature_batch=args.feature_batch_size, max_input_bytes=args.max_input_bytes)
    decoder = ContextualByteDecoder(model.d_model, args.hidden_dim, max_input_bytes=args.max_input_bytes)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    counts = torch.bincount(train_data["intent"], minlength=NUM_INTENTS).float()
    sample_weights = 1.0 / counts[train_data["intent"]]
    sample_weights[train_data["intent"] == int(ConversationIntent.RECALL)] *= args.recall_weight
    best, best_epoch, best_loss, best_rank, history = None, 0, float("inf"), None, []
    for epoch in range(args.epochs):
        sampled = torch.multinomial(sample_weights, len(rows), replacement=True)
        total = 0.0
        decoder.train()
        for indices in sampled.split(args.batch_size):
            loss = loss_for(decoder, train_data, indices, alignment_weight=args.alignment_weight)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite response loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach()) * len(indices)
        decoder.eval()
        validation = evaluate_loss(decoder, val_data, args.batch_size)
        copy = None
        if epoch == 0 or (epoch + 1) % args.copy_eval_every == 0 or epoch + 1 == args.epochs:
            copy = evaluate_validation_copy(decoder, val_data)
            rank = (copy["correct"], -validation)
            if best_rank is None or rank > best_rank:
                best_loss, best_epoch, best_rank = validation, epoch + 1, rank
                best = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_loss": total / len(rows), "validation_loss": validation,
                        "validation_exact_copy": copy})
        if copy is not None:
            print(f"response {epoch+1}/{args.epochs}: loss={history[-1]['train_loss']:.4f} val={validation:.4f} "
                  f"validation-copy={copy['correct']}/{copy['count']}", flush=True)
    assert best is not None
    decoder.load_state_dict(best, strict=True); decoder.response_trained.fill_(True); decoder.eval()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "response.pt"
    torch.save({"schema_version": 2, "generator_sha256": sha256(args.generator), "generator_n_heads": model.n_heads,
                "context_dim": decoder.context_dim, "hidden_dim": decoder.hidden_dim, "max_input_bytes": decoder.max_input_bytes,
                "state_dict": decoder.state_dict()}, path)
    test = decoder_only_retention(decoder, model, memory, feature_batch=args.feature_batch_size, max_input=args.max_input_bytes)
    report = {"configuration": vars(args), "generator_sha256": sha256(args.generator), "checkpoint_sha256": sha256(path),
              "preflight": {"accepted": len(rows), "blocked": blocked, "protected_hashes": len(guard.contents),
                            "class_counts": {INTENT_NAMES[int(k)]: v for k, v in Counter(row.intent for row in rows).items()}},
              "parameters": {"decoder": sum(p.numel() for p in decoder.parameters()),
                             "trainable_generator_parameters": 0},
              "selection_rule": "highest greedy exact-copy score on validation, then lowest validation token NLL; test not used",
              "selected_epoch": best_epoch, "epochs": history, "validation_loss": best_loss,
              "validation_exact_copy": evaluate_validation_copy(decoder, val_data),
              "decoder_only_retention": test,
              "limitations": "Synthetic response styles repeat across splits; no open-domain or safety guarantee."}
    (out / "response_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="runs/phase11/generator.pt")
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--phase12-guard", action="store_true", help="also protect Phase 12 final/validation fixtures")
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="runs/phase11")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-input-bytes", type=int, default=384)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--recall-weight", type=float, default=3.0)
    parser.add_argument("--alignment-weight", type=float, default=0.5)
    parser.add_argument("--synthetic-facts", type=int, default=400)
    parser.add_argument("--copy-eval-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=112012)
    parser.add_argument("--threads", type=int, default=1)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
