"""Fine-tune the existing cognitive policy on explicit positional replacements.

The user-provided Hello example is deliberately oversampled and is NOT a test
case. Validation/test examples are generated in disjoint RNG namespaces and
are preflighted against each training byte sequence and trace before updates.
Only cognitive/control parameters train; existing checkpoints remain untouched.

Run: OMP_NUM_THREADS=1 python -m training.train_positional_edits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from data.edit_curriculum import curriculum_examples
from data.holdout_registry import synthetic_holdouts
from data.ingest_real_traces import load_pairs
from data.positional_edits import diagnostic_record, positional_examples
from models.cognitive import action_from_record, action_to_record
from models.generator_conditioned import GeneratorConditioned
from models.holdout import HoldoutGuard
from training.cognitive_train import cognitive_trace_supervision_loss
from training.rl_editing import evaluate_autonomous, preflight_training_examples
from training.train_cognitive import freeze_backbone
from training.train_confidence import load_generator


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def evaluate_positional(model: GeneratorConditioned, records: list[dict[str, Any]], *, budget: int,
                        structured: bool = False, allow_untrained: bool = False) -> dict[str, Any]:
    """Greedy, memory/controller-free full execution; targets are scoring only."""
    details = []
    for record in records:
        request = record["structured_request"] if structured else None
        result = model.cognitive_generate(record["input"], instruction_bytes=record["instruction_bytes"],
                                           structured_request=request, allow_untrained_structured=allow_untrained,
                                           max_steps=budget, temperature=0.)
        edits = [event for event in result.events if event.action.token == model.EDIT_OP]
        correct_arguments = any(
            e.action.kind == "replace" and e.action.start == record["position"] and
            e.action.end == record["position"] + 1 and e.action.data == bytes([record["requested_byte"]])
            and e.argument_bytes is not None for e in edits
        )
        # Source/target are accessed only after the autonomous call.
        exact = bool(result.done and result.canvas == record["output"])
        details.append({"id": record["id"], "length": len(record["input"]), "position": record["position"],
                        "requested_byte": record["requested_byte"], "exact": exact, "done": result.done,
                        "correct_edit_arguments": bool(correct_arguments), "output_hex": result.canvas.hex(),
                        "source_hex": record["input"].hex(), "target_hex": record["output"].hex(),
                        "actions": [dict(action_to_record(e.action),
                                         argument_bytes=None if e.argument_bytes is None else list(e.argument_bytes))
                                    for e in result.events]})
    if not details:
        return {"count": 0, "exact": 0, "correct_edit_arguments": 0, "done": 0, "examples": []}
    return {"count": len(details), "exact": sum(int(d["exact"]) for d in details),
            "correct_edit_arguments": sum(int(d["correct_edit_arguments"]) for d in details),
            "done": sum(int(d["done"]) for d in details), "examples": details}


def train(args: argparse.Namespace) -> dict[str, Any]:
    if (args.epochs < 1 or args.train_examples < 1 or args.validation_examples < 1 or args.test_examples < 1
            or args.diagnostic_repeats < 1 or args.long_train < 0 or args.long_validation < 0 or args.long_test < 0
            or args.anchors_per_stage < 0 or not math.isfinite(args.lr) or args.lr <= 0
            or not 1 <= args.budget <= 4096 or args.threads < 1):
        raise ValueError("invalid training configuration")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = load_generator(args.generator, "cpu", args.generator_heads).eval()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    diagnostic = diagnostic_record()
    positional_train = positional_examples("train", args.train_examples, seed=args.seed, long_count=args.long_train)
    validation = positional_examples("validation", args.validation_examples, seed=args.seed, long_count=args.long_validation)
    test = positional_examples("test", args.test_examples, seed=args.seed, long_count=args.long_test)
    anchors = (curriculum_examples(split="train", count_per_stage=args.anchors_per_stage, seed=881)
               if args.anchors_per_stage else [])
    train_records = [diagnostic] + positional_train + anchors
    real_validation = load_pairs(args.pairs, split="validation")
    real_test = load_pairs(args.pairs, split="test")
    # Include all known earlier held-outs as well as the newly generated splits.
    previous_synthetic = curriculum_examples(split="validation", count_per_stage=2, seed=881)
    processed = synthetic_holdouts(args.processed_dir)
    guard = HoldoutGuard.from_records(real_validation + real_test + previous_synthetic + processed + validation + test)
    preflight_training_examples(model, train_records, guard)
    print(f"preflight passed: {len(train_records)} unique training traces against {len(guard.contents)} protected byte hashes", flush=True)
    baseline_validation = evaluate_positional(model, validation, budget=args.budget)
    baseline_diagnostic = evaluate_positional(model, [diagnostic], budget=args.budget)
    # Mark prior confidence calibration invalid before selective optimization.
    model.train()
    model.eval()
    params = freeze_backbone(model)
    # The auxiliary reasoning head is not part of this supervised objective.
    for parameter in model.reasoning_predictor.parameters():
        parameter.requires_grad_(False)
    params = [parameter for parameter in params if parameter.requires_grad]
    trained_names = {name for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam(params, lr=args.lr)
    best_rank = None
    best_state = None
    best_epoch = 0
    logs = []
    for epoch in range(args.epochs):
        examples = positional_train + anchors + [diagnostic] * args.diagnostic_repeats
        random.Random(args.seed + 10_000 + epoch).shuffle(examples)
        total = 0.
        for record in examples:
            actions = [action_from_record(item) for item in record["trace_actions"]]
            optimizer.zero_grad(set_to_none=True)
            loss = cognitive_trace_supervision_loss(model, record["input"], actions,
                                                      instruction_bytes=record["instruction_bytes"])
            assert isinstance(loss, torch.Tensor)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite positional training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
        val = evaluate_positional(model, validation, budget=args.budget)
        diag = evaluate_positional(model, [diagnostic], budget=args.budget)
        # Selection intentionally prioritizes the requested *training* example;
        # validation breaks ties. The separate test split is untouched here.
        rank = (diag["exact"], val["exact"], val["correct_edit_arguments"])
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log = {"epoch": epoch + 1, "updates": len(examples), "mean_supervised_trace_loss": total / len(examples),
               "training_diagnostic_exact": diag["exact"], "validation_exact": val["exact"],
               "validation_correct_arguments": val["correct_edit_arguments"], "validation_count": val["count"]}
        logs.append(log)
        print(f"epoch {epoch+1}/{args.epochs}: loss={log['mean_supervised_trace_loss']:.4f} "
              f"training-Hello={diag['exact']}/1 validation={val['exact']}/{val['count']} "
              f"arguments={val['correct_edit_arguments']}/{val['count']}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    model.eval()
    for name, value in model.state_dict().items():
        if name in original and name not in trained_names and not name.startswith("confidence_head."):
            if not torch.equal(value, original[name]):
                raise RuntimeError(f"frozen tensor changed during positional training: {name}")
    for key in ("head.weight", "head.bias"):
        if not torch.equal(model.state_dict()[key][:model.LOAD_SPAN], original[key][:model.LOAD_SPAN]):
            raise RuntimeError(f"legacy byte/function rows changed: {key}")
    if not torch.equal(model.control_emb.weight[:model.LOAD_SPAN - model.VOCAB_BYTES],
                       original["control_emb.weight"][:model.LOAD_SPAN - model.VOCAB_BYTES]):
        raise RuntimeError("legacy control embeddings changed")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "generator.pt")
    # Final held-out scoring happens only after selection and saving are complete.
    test_results = evaluate_positional(model, test, budget=args.budget)
    real_test_results = evaluate_autonomous(model, real_test, budget=128)
    report = {
        "configuration": vars(args), "source_checkpoint_sha256": _sha256(args.generator),
        "checkpoint_sha256": _sha256(output / "generator.pt"), "selected_epoch": best_epoch,
        "selection_rule": "prioritize in-training diagnostic exact, then independent positional-validation exact and argument matches; no test-based selection",
        "training_example_counts": {"user_diagnostic": 1, "synthetic_short": args.train_examples,
                                    "synthetic_long": args.long_train, "curriculum_anchors": len(anchors),
                                    "diagnostic_repeats_per_epoch": args.diagnostic_repeats},
        "guard": {"fingerprint": guard.fingerprint, "protected_content_hashes": len(guard.contents),
                  "processed_holdout_records": len(processed), "training_traces_preflighted": len(train_records)},
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "participating_tensor_elements": sum(p.numel() for p in params),
                       "legacy_vocabulary_rows_gradient_masked": True, "new_model_parameters": 0},
        "epochs": logs, "baseline_validation": baseline_validation, "baseline_diagnostic": baseline_diagnostic,
        "selected_validation": evaluate_positional(model, validation, budget=args.budget),
        "selected_training_diagnostic": evaluate_positional(model, [diagnostic], budget=args.budget),
        "heldout_positional_test": test_results, "real_test_no_actionable_instruction": real_test_results,
        "limitations": ["The Hello example is explicitly included and oversampled in training.",
                        "Natural-language inputs share a synthetic template; byte-content disjointness is not semantic independence.",
                        "Long-position representations are distinct but accuracy beyond the pretrained context is unproven.",
                        "The requested byte and position are present in the instruction; full targets are not inference inputs."],
    }
    (output / "cognitive_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"selected epoch {best_epoch}; held-out positional exact={test_results['exact']}/{test_results['count']}; "
          f"real test exact={real_test_results['exact']}/{real_test_results['count']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="runs/phase8/generator.pt")
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="runs/phase9")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-examples", type=int, default=128)
    parser.add_argument("--validation-examples", type=int, default=24)
    parser.add_argument("--test-examples", type=int, default=64)
    parser.add_argument("--long-train", type=int, default=4)
    parser.add_argument("--long-validation", type=int, default=2)
    parser.add_argument("--long-test", type=int, default=4)
    parser.add_argument("--diagnostic-repeats", type=int, default=32)
    parser.add_argument("--anchors-per-stage", type=int, default=2)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=9009)
    parser.add_argument("--threads", type=int, default=1)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
