"""Train optional learned cognitive argument conditioning on explicit requests.

The structured request already names the desired position, operation and byte.
It is an input, not a target document, memory lookup, host controller or forced
edit. The policy must still emit actions and argument bytes. This experiment
measures structured copying, not free-form language understanding.
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
from data.structured_edits import structured_examples
from models.cognitive import action_from_record
from models.holdout import HoldoutGuard
from training.cognitive_train import cognitive_supervision_loss
from training.rl_editing import preflight_training_examples
from training.train_confidence import load_generator
from training.train_positional_edits import evaluate_positional


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def teacher_edit_state(model, record: dict[str, Any]) -> torch.Tensor:
    """Scratch state after gold LOAD and THINK; not used as inference input."""
    device = model.byte_emb.weight.device
    scratch = model.scratchpad.clone()
    instruction = torch.tensor([list(record["instruction_bytes"])], device=device, dtype=torch.long)
    scratch = model.read_scratchpad(scratch, instruction)
    source = record["input"]
    scratch = model.read_scratchpad(scratch, torch.tensor([list(source)], device=device, dtype=torch.long))
    policy = model.cognitive_policy(source, scratch, pointer_bytes=source, loaded=True, last_token=model.LOAD_SPAN)
    return model.think_scratchpad(scratch, policy.context.unsqueeze(0), steps=1).scratchpad.detach()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if (args.epochs < 1 or args.train_examples < 1 or args.validation_examples < 1 or args.test_examples < 1
            or min(args.long_train, args.long_validation, args.long_test) < 0 or not 1 <= args.budget <= 4096
            or not math.isfinite(args.lr) or args.lr <= 0 or args.threads < 1):
        raise ValueError("invalid structured request training configuration")
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    model = load_generator(args.generator, "cpu", args.generator_heads).eval()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_rows = structured_examples("train", args.train_examples, seed=args.seed, long_count=args.long_train)
    validation = structured_examples("validation", args.validation_examples, seed=args.seed, long_count=args.long_validation)
    test = structured_examples("test", args.test_examples, seed=args.seed, long_count=args.long_test)
    previous_val = positional_examples("validation", 24, seed=9009, long_count=2)
    previous_test = positional_examples("test", 64, seed=9009, long_count=4)
    previous_train = positional_examples("train", 128, seed=9009, long_count=4) + [diagnostic_record()]
    previous_anchors = curriculum_examples(split="train", count_per_stage=2, seed=881)
    other_heldout = (load_pairs(args.pairs, split="validation") + load_pairs(args.pairs, split="test")
                     + curriculum_examples(split="validation", count_per_stage=2, seed=881)
                     + synthetic_holdouts(args.processed_dir))
    guard = HoldoutGuard.from_records(other_heldout + previous_val + previous_test + validation + test)
    # The inherited checkpoint's known fine-tuning examples also receive a
    # preflight against the new test split before this experiment begins.
    preflight_training_examples(model, previous_train + previous_anchors + train_rows, guard)
    for row in train_rows:
        guard.validate(row["input"], row["output"], extra=(row["structured_request"],))
    print(f"preflight passed: {len(train_rows)} structured training and {len(previous_train) + len(previous_anchors)} inherited traces", flush=True)
    baseline = evaluate_positional(model, validation, budget=args.budget, structured=True, allow_untrained=True)
    for p in model.parameters():
        p.requires_grad_(False)
    # The move-destination adapter remains zero/untrained in this replacement-only experiment.
    projections = (model.request_start_proj, model.request_end_proj, model.request_value_proj, model.request_kind_proj)
    params = [p for projection in projections for p in projection.parameters()]
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    cached = {row["id"]: teacher_edit_state(model, row) for row in train_rows}
    best_rank = None; best_epoch = 0; best_state = None; logs = []
    for epoch in range(args.epochs):
        indices = list(range(len(train_rows)))
        random.Random(args.seed + epoch + 80_000).shuffle(indices)
        total = 0.
        for index in indices:
            row = train_rows[index]
            edit = action_from_record(row["trace_actions"][2])
            optimizer.zero_grad(set_to_none=True)
            loss, _ = cognitive_supervision_loss(model, row["input"], cached[row["id"]], edit,
                                                  workspace=row["input"], last_token=model.THINK,
                                                  structured_request=row["structured_request"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite structured request loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())
        model.mark_structured_edit_trained()
        val = evaluate_positional(model, validation, budget=args.budget, structured=True)
        rank = (val["exact"], val["correct_edit_arguments"])
        if best_rank is None or rank > best_rank:
            best_rank = rank; best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        logs.append({"epoch": epoch + 1, "mean_edit_loss": total / len(indices), "validation_exact": val["exact"],
                     "validation_correct_arguments": val["correct_edit_arguments"], "validation_count": val["count"]})
        print(f"epoch {epoch+1}/{args.epochs}: loss={logs[-1]['mean_edit_loss']:.4f} "
              f"validation exact={val['exact']}/{val['count']} args={val['correct_edit_arguments']}/{val['count']}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state, strict=True); model.eval()
    trained = {"request_start_proj.weight", "request_end_proj.weight", "request_value_proj.weight", "request_kind_proj.weight",
               "structured_edit_trained", "structured_edit_trained_kinds"}
    for key, value in model.state_dict().items():
        if key not in trained and not torch.equal(value, original[key]):
            raise RuntimeError(f"non-adapter model tensor changed: {key}")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "structured_generator.pt")
    # Disjoint test is accessed only after model selection.
    testing = evaluate_positional(model, test, budget=args.budget, structured=True)
    diagnostic = evaluate_positional(model, [diagnostic_record()], budget=args.budget, structured=True)
    natural_diag = evaluate_positional(model, [diagnostic_record()], budget=args.budget)
    report = {"configuration": vars(args), "source_checkpoint_sha256": _sha(args.generator),
              "checkpoint_sha256": _sha(output / "structured_generator.pt"), "selected_epoch": best_epoch,
              "selection_rule": "maximum structured-validation exact, then correct-argument count; independent test not used",
              "training_examples": len(train_rows), "previous_training_traces_preflighted": len(previous_train) + len(previous_anchors),
              "guard": {"fingerprint": guard.fingerprint, "protected_content_hashes": len(guard.contents)},
              "parameters": {"total": sum(p.numel() for p in model.parameters()),
                             "new_structured_adapter_parameters": sum(p.numel() for name, p in model.named_parameters() if name.startswith("request_")),
                             "trainable_replacement_parameters": sum(p.numel() for p in params),
                             "non_adapter_tensors_unchanged": True},
              "baseline_structured_validation": baseline, "epochs": logs,
              "selected_structured_validation": evaluate_positional(model, validation, budget=args.budget, structured=True),
              "heldout_structured_test": testing, "diagnostic_with_structured_request": diagnostic,
              "diagnostic_natural_only": natural_diag,
              "limitations": ["Structured input already specifies the exact desired edit; this is learned copying, not natural-language parsing.",
                              "Only replacement requests were trained; insert/delete/move request adapters are not validated.",
                              "Long-position Fourier keys are distinct but long-context accuracy must be measured.",
                              "The natural-only diagnostic was included in the inherited checkpoint's training data."]}
    (output / "structured_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"selected epoch {best_epoch}; heldout structured={testing['exact']}/{testing['count']}; "
          f"training Hello natural={natural_diag['exact']}/1", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="runs/phase9/generator.pt")
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="runs/phase9")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--validation-examples", type=int, default=64)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--long-train", type=int, default=32)
    parser.add_argument("--long-validation", type=int, default=8)
    parser.add_argument("--long-test", type=int, default=8)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=9509)
    parser.add_argument("--threads", type=int, default=1)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
