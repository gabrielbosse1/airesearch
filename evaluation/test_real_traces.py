"""Train-only procedure persistence and genuinely autonomous held-out edits.

These are small repository-derived byte excerpts, not a broad external test.
The held-out inputs/targets are never persisted. No target, target-derived
budget, host controller or oracle action is passed to autonomous generation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.ingest_real_traces import ingest_real_traces, load_pairs, make_control_trace
from data.holdout_registry import synthetic_holdouts
from models.cognitive import action_from_record, action_to_record
from models.holdout import HoldoutGuard, HoldoutViolation
from models.memory_model import MemoryModel
from training.cognitive_train import validate_action_trace
from training.train_confidence import load_generator


@torch.no_grad()
def evaluate_policy(generator: Any, records: list[dict[str, Any]], memory: MemoryModel | None = None, *,
                    budget: int = 128, prime_explanation: bool = False) -> dict[str, Any]:
    if not 1 <= budget <= 4096:
        raise ValueError("budget must be between 1 and 4096")
    generator.eval()
    details = []
    with memory.read_only() if memory is not None else nullcontext():
        for record in records:
            # Target is used for scoring only, never for the generation budget.
            result = generator.cognitive_generate(
                record["input"], memory_model=memory, max_steps=budget,
                instruction_bytes=record.get("explanation", b"") if prime_explanation else None,
            )
            distance, _ = make_control_trace(record["input"], record["output"])
            conf = [e.confidence for e in result.events if e.action.token == generator.THINK and e.confidence is not None]
            details.append({"id": record["id"], "kind": record["kind"], "optimal_distance": distance,
                            "exact": result.done and result.canvas == record["output"], "done": result.done,
                            "halt_reason": result.halt_reason, "steps": len(result.events), "budget": budget,
                            "think_steps": sum(e.thought_steps for e in result.events),
                            "thought_confidence": sum(conf) / len(conf) if conf else None,
                            "output_hex": result.canvas.hex(), "target_hex": record["output"].hex(),
                            "actions": [action_to_record(e.action) for e in result.events]})
    by_kind: dict[str, Any] = {}
    for kind in sorted({r["kind"] for r in details}):
        group = [r for r in details if r["kind"] == kind]
        by_kind[kind] = {"count": len(group), "exact": sum(int(r["exact"]) for r in group),
                         "accuracy": sum(int(r["exact"]) for r in group) / len(group)}
    return {"count": len(details), "exact": sum(int(r["exact"]) for r in details),
            "exact_fraction": sum(int(r["exact"]) for r in details) / max(len(details), 1),
            "done_fraction": sum(int(r["done"]) for r in details) / max(len(details), 1),
            "by_kind": by_kind, "examples": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--generator", default="runs/dual_memory/generator.pt")
    parser.add_argument("--baseline-generator", default="runs/dual_memory/generator.pt")
    parser.add_argument("--output-dir", default="runs/phase6/real_traces")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()
    if args.train_epochs < 0:
        raise ValueError("train-epochs cannot be negative")
    torch.manual_seed(337)
    train = load_pairs(args.pairs, split="train")
    validation = load_pairs(args.pairs, split="validation")
    test = load_pairs(args.pairs, split="test")
    if not train or not test:
        raise ValueError("fixture must include both train and test records")
    synthetic = synthetic_holdouts(args.processed_dir, require=False)
    guard = HoldoutGuard.from_records(validation + test + synthetic)
    generator = load_generator(args.generator, args.device, 4)
    baseline_generator = load_generator(args.baseline_generator, args.device, 4)
    memory = MemoryModel(d_model=generator.d_model, capacity=max(128, (len(train) + len(validation) + len(test)) * 5)).to(args.device)
    memory.set_holdout_guard(guard)
    train_prepared, logs = ingest_real_traces(memory, generator, train, epochs=args.train_epochs,
                                               batch_size=args.batch_size, lr=args.lr)
    before = memory.audit_holdouts()
    if before["failed_slots"]:
        raise AssertionError("training memory failed the holdout audit")

    # Exercise the actual write API, not merely the hash helper. Every attempt
    # must fail before allocation, count changes or persistent-byte writes.
    slots_before = int(memory.procedural.used.sum().item())
    rejected = 0
    for record in validation + test:
        try:
            memory.write_procedure(record["input"], record["output"], torch.zeros(1, memory.d_model),
                                   metadata={"split": record["split"]})
        except HoldoutViolation:
            rejected += 1
        else:
            raise AssertionError("held-out write unexpectedly succeeded")
    if rejected != len(validation) + len(test) or int(memory.procedural.used.sum().item()) != slots_before:
        raise AssertionError("failed holdout attempts changed persistent memory")

    directory = Path(args.output_dir); directory.mkdir(parents=True, exist_ok=True)
    memory_path = directory / "memory.pt"
    # Save before held-out model evaluation; test-time retrieval is read-only.
    memory.save(str(memory_path)); torch.save(generator.state_dict(), directory / "generator.pt")
    restored = MemoryModel(d_model=generator.d_model, capacity=memory.capacity).to(args.device)
    restored.load(str(memory_path), map_location=args.device)
    replay = []
    with restored.read_only():
        for example in train_prepared:
            source, target = example["input"], example["output"]
            execution = restored.execute_procedure(source)
            trace_actions = [action_from_record(r) for r in execution.metadata["trace_actions"]]
            symbolic = validate_action_trace(generator, source, trace_actions)
            ok = (execution.output == target and symbolic == target
                  and execution.explanation_bytes == example["context"]["explanation_bytes"]
                  and torch.equal(execution.latent_trace.cpu(), example["latent_trace"].cpu()))
            replay.append({"id": example["context"]["id"], "matched": bool(ok),
                           "distance": example["context"]["distance"], "trace_steps": len(trace_actions),
                           "latent_steps": len(execution.latent_trace), "slot": execution.slot})
    if not all(item["matched"] for item in replay):
        raise AssertionError("one or more stored training procedures failed persistence/replay")
    snapshot = {k: v.detach().clone() for k, v in restored.state_dict().items()}
    baseline = evaluate_policy(baseline_generator, test, budget=args.budget)
    train_eval = evaluate_policy(generator, train, budget=args.budget)
    test_eval = evaluate_policy(generator, test, budget=args.budget)
    conditioned = evaluate_policy(generator, test, memory=restored, budget=args.budget)
    primed = evaluate_policy(generator, test, memory=restored, budget=args.budget, prime_explanation=True)
    unchanged = all(torch.equal(v, restored.state_dict()[k]) for k, v in snapshot.items())
    if not unchanged:
        raise AssertionError("held-out evaluation mutated persistent memory")
    after = restored.audit_holdouts()
    if after["failed_slots"]:
        raise AssertionError("restored memory failed the holdout audit")
    report = {"protocol": {"source": "repository-derived before/after excerpts, not an external benchmark",
               "split": "source-file split; only train procedures are persisted; exact and long-substring guards",
               "distance": "optimal unit-cost byte Levenshtein; MOVE is not a primitive",
               "generation": "autonomous cognitive_generate, fixed budget, no target/controller/oracle action",
               "instruction": "real records have only generic provenance text, not an actionable edit instruction; priming it is experimental",
               "generator_checkpoint": args.generator, "baseline_generator_checkpoint": args.baseline_generator,
               "train_epochs": args.train_epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
               "saved_memory_contains_heldout_targets": False, "memory_read_only_during_test": unchanged,
               "warning": "Train stored-output replay is a lookup, not learned editing; exact hashes cannot detect semantic or unregistered leakage."},
              "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
              "holdout_checks": {"attempts_rejected": rejected, "synthetic_heldout_records": len(synthetic),
                                 "audit_before": before, "audit_after": after},
              "replay": {"count": len(replay), "matched": sum(r["matched"] for r in replay), "examples": replay},
              "co_training": logs, "baseline_test": baseline, "trained_train": train_eval,
              "trained_test": test_eval, "trained_test_with_train_memory": conditioned,
              "trained_test_with_generic_explanation": primed}
    (directory / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "holdout_rejected": rejected,
                      "replay": {"matched": report["replay"]["matched"], "count": len(replay)},
                      "baseline_test": baseline["exact_fraction"], "trained_train": train_eval["exact_fraction"],
                      "trained_test": test_eval["exact_fraction"],
                      "trained_test_with_train_memory": conditioned["exact_fraction"],
                      "test_with_generic_explanation": primed["exact_fraction"]}, indent=2))


if __name__ == "__main__":
    main()
