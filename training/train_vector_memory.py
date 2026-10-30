"""Train a neural byte reader on actual vector-only episodic retrievals.

Held-out statements and values are protected before the first guarded write.
The frozen generator receives only the current question and memory vectors.
Validation (not Phase 12 final tests) selects the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from data.vector_memory_pairs import vector_training_examples
from models.holdout import HoldoutViolation
from models.memory_model import MemoryModel
from models.vector_memory import VectorConversationMemory, VectorFactCodec, VectorMemoryReader
from training.co_train import forced_memory_logits, forced_memory_training_step, prepare_forced_memory_examples
from training.train_confidence import load_generator
from training.train_conversation import sha256
from training.train_replay_intents import phase12_guard


def _memory(codec, guard=None):
    model = MemoryModel(d_model=64, tier_capacities=(128, 4, 4))
    if guard is not None:
        model.set_holdout_guard(guard)
    return VectorConversationMemory(model, codec)


@torch.no_grad()
def _validate(generator, memory, reader, rows, batch_size):
    exact, correct_bytes, total_bytes, loss_sum, batches = 0, 0, 0, 0.0, 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        logits, targets, _ = forced_memory_logits(generator, memory, reader, batch)
        loss = F.nll_loss(logits.reshape(-1, 257), targets.reshape(-1), ignore_index=VectorMemoryReader.IGNORE)
        loss_sum += float(loss); batches += 1
        predictions = logits.argmax(dim=-1)
        for idx, row in enumerate(batch):
            got = []
            for item in predictions[idx, :len(row["vectors"]) * 8].tolist():
                if item == VectorMemoryReader.EOS:
                    break
                got.append(item)
            candidate = bytes(got)
            target = row["target"]
            exact += candidate == target
            correct_bytes += sum(x == y for x, y in zip(candidate, target))
            total_bytes += len(target)
    return {"exact": exact, "count": len(rows), "exact_accuracy": exact / len(rows),
            "byte_accuracy": correct_bytes / total_bytes, "mean_batch_nll": loss_sum / batches}


def train(args):
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("invalid training configuration")
    generator = load_generator(args.generator, "cpu", args.generator_heads).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    codec = VectorFactCodec(seed=args.codec_seed).eval()
    guard = phase12_guard(args)
    train_candidates = vector_training_examples("train", count=args.train_examples)
    accepted, blocked = [], 0
    for row in train_candidates:
        try:
            guard.validate(row["statement"], row["target"], metadata={"split": "train"}, extra=(row["question"],))
        except HoldoutViolation:
            blocked += 1; continue
        accepted.append(row)
    if len(accepted) < args.batch_size:
        raise RuntimeError("not enough vector examples passed holdout preflight")
    train_memory = _memory(codec, guard)
    val_memory = _memory(codec)  # isolated ephemeral evaluation bank; never persisted into replay
    print(f"preflight: accepted={len(accepted)}, blocked={blocked}, protected={len(guard.contents)}; writing training episodes", flush=True)
    prepared = prepare_forced_memory_examples(train_memory, accepted)
    audit = train_memory.memory.audit_holdouts()
    if audit["failed_slots"]:
        raise HoldoutViolation(f"guarded training memory failed audit: {audit['failures'][:3]}")
    validation = prepare_forced_memory_examples(val_memory, vector_training_examples("validation"))
    print(f"retrieved vector-only training={len(prepared)}, validation={len(validation)}", flush=True)
    reader = VectorMemoryReader(generator.d_model)
    optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
    best, rank, selected_epoch, history = None, None, 0, []
    for epoch in range(args.epochs):
        indices = list(range(len(prepared))); rng.shuffle(indices)
        loss_sum = 0.0; updates = 0
        reader.train()
        for start in range(0, len(indices), args.batch_size):
            batch = [prepared[i] for i in indices[start:start + args.batch_size]]
            loss, info = forced_memory_training_step(generator, train_memory, reader, batch, optimizer)
            if info["episode_bytes_in_generator_input"] != 0:
                raise RuntimeError("unexpected episode bytes in model input")
            loss_sum += float(loss.detach()); updates += 1
        reader.eval(); metrics = _validate(generator, val_memory, reader, validation, args.batch_size)
        current_rank = (metrics["exact_accuracy"], -metrics["mean_batch_nll"])
        if rank is None or current_rank > rank:
            rank, selected_epoch = current_rank, epoch + 1
            best = {name: value.detach().clone() for name, value in reader.state_dict().items()}
        row = {"epoch": epoch + 1, "train_nll": loss_sum / updates, "validation": metrics}
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch+1}/{args.epochs}: val={metrics['exact']}/{metrics['count']} "
                  f"byte={metrics['byte_accuracy']:.3f} nll={metrics['mean_batch_nll']:.4f}", flush=True)
    assert best is not None
    reader.load_state_dict(best); reader.reader_trained.fill_(True); reader.eval()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "vector_reader.pt"
    torch.save({"schema_version": 1, "generator_sha256": sha256(args.generator), "context_dim": generator.d_model,
                "codec_state_dict": codec.state_dict(), "codec_fingerprint": codec.fingerprint,
                "reader_state_dict": reader.state_dict()}, checkpoint)
    report = {"configuration": vars(args), "generator_sha256": sha256(args.generator), "reader_sha256": sha256(checkpoint),
              "codec_fingerprint": codec.fingerprint, "guard_fingerprint": guard.fingerprint,
              "preflight": {"accepted": len(accepted), "blocked": blocked, "protected_hashes": len(guard.contents)},
              "selected_epoch": selected_epoch, "selection_rule": "validation exact vector-only bytes, then validation NLL",
              "validation": _validate(generator, val_memory, reader, validation, args.batch_size),
              "reader_parameters": sum(p.numel() for p in reader.parameters()), "epochs": history,
              "limitations": "Structured value extraction and lossless fixed-capacity encoding; not learned semantic compression. Actual retrieved values are cached for training after writes."}
    (output / "vector_reader_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator", default="runs/phase12/generator.pt")
    p.add_argument("--generator-heads", type=int, default=4)
    p.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--output-dir", default="runs/phase12")
    p.add_argument("--train-examples", type=int, default=2200)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--seed", type=int, default=121220)
    p.add_argument("--codec-seed", type=int, default=120120)
    p.add_argument("--threads", type=int, default=1)
    train(p.parse_args())


if __name__ == "__main__":
    main()
