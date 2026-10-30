"""Train and independently evaluate the native, requested-byte mask editor.

Training uses reproducible raw-byte sources, independently chosen 30% masks and
explicitly requested replacement values. Only a zero-initialized mask feature
and requested-value projection are updated. Selection uses validation only;
all-byte coverage and the user diagnostic are scored after selection.

Run: OMP_NUM_THREADS=1 python -m training.train_masked_diffusion
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from data.edit_curriculum import curriculum_examples
from data.holdout_registry import synthetic_holdouts
from data.ingest_real_traces import load_pairs
from data.masked_edits import byte_coverage_examples, masked_examples
from data.positional_edits import diagnostic_record
from models.bytecore_diffusion import ByteCoreDiffusion
from models.holdout import HoldoutGuard


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(config_path: str, runs_dir: str) -> ByteCoreDiffusion:
    settings = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["model"]
    return ByteCoreDiffusion(
        vocab_size=settings["vocab_size"], d_model=settings["d_model"], n_layers=settings["n_layers"],
        n_heads=settings["n_heads"], d_ff=settings["d_ff"], max_len=settings["max_len"],
        diffusion_timesteps=settings["diffusion_timesteps"], concept_capacity=settings["concept_capacity"],
        concept_dim=settings["concept_dim"], dropout=settings["dropout"], device="cpu",
    ).load(runs_dir).eval()


def diagnostic_mask_record() -> dict[str, Any]:
    record = diagnostic_record()
    mask = bytearray(len(record["input"])); mask[record["position"]] = 1
    requested = bytearray(len(mask)); requested[record["position"]] = record["requested_byte"]
    return {"id": "user-diagnostic-unseen-combination", "split": "test", "input": record["input"],
            "output": record["output"], "edit_mask": bytes(mask), "requested_bytes": bytes(requested)}


def tensor_batch(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records or len({len(item["input"]) for item in records}) != 1:
        raise ValueError("tensor batch requires equal-length examples")
    ids = torch.tensor([list(item["input"]) for item in records], dtype=torch.long)
    mask = torch.tensor([list(item["edit_mask"]) for item in records], dtype=torch.bool)
    request = torch.tensor([list(item["requested_bytes"]) for item in records], dtype=torch.long)
    target = torch.tensor([list(item["output"]) for item in records], dtype=torch.long)
    return ids, mask, request, target


@torch.no_grad()
def evaluate_masked(model: ByteCoreDiffusion, records: list[dict[str, Any]], *, seed: int,
                    batch_size: int = 16, allow_untrained: bool = False) -> dict[str, Any]:
    """Evaluate actual native DDPM output, with no target passed to sampling."""
    if not records or batch_size < 1:
        raise ValueError("evaluation requires records and a positive batch size")
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    model.eval()
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_length[len(record["input"])].append(record)
    results = []
    try:
        for length in sorted(by_length):
            group = by_length[length]
            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                ids, mask, request, target = tensor_batch(chunk)
                predicted = model.masked_edit(ids, mask, request, allow_untrained_mask=allow_untrained)
                for i, record in enumerate(chunk):
                    actual = bytes(predicted[i].tolist())
                    expected = bytes(target[i].tolist())  # scoring only
                    original = record["input"]
                    selected = [p for p, flag in enumerate(record["edit_mask"]) if flag]
                    results.append({"id": record["id"], "length": length, "positions": selected,
                                    "requested_bytes": [record["requested_bytes"][p] for p in selected],
                                    "output_hex": actual.hex(), "exact": actual == expected,
                                    "masked_byte_correct": sum(actual[p] == expected[p] for p in selected),
                                    "masked_byte_count": len(selected),
                                    "unmasked_preserved": all(actual[p] == original[p] for p in range(length) if p not in selected)})
    finally:
        torch.random.set_rng_state(state)
    groups = {}
    for length in sorted(by_length):
        current = [item for item in results if item["length"] == length]
        groups[str(length)] = {"count": len(current), "exact": sum(int(item["exact"]) for item in current),
                               "masked_byte_correct": sum(item["masked_byte_correct"] for item in current),
                               "masked_byte_count": sum(item["masked_byte_count"] for item in current)}
    return {"count": len(results), "exact": sum(int(item["exact"]) for item in results),
            "masked_byte_correct": sum(item["masked_byte_correct"] for item in results),
            "masked_byte_count": sum(item["masked_byte_count"] for item in results),
            "all_unmasked_preserved": all(item["unmasked_preserved"] for item in results),
            "by_length": groups, "examples": results}


def train(args: argparse.Namespace) -> dict[str, Any]:
    if (args.steps < 1 or args.train_per_length < 1 or args.validation_per_length < 1
            or args.test_per_length < 1 or args.batch_size < 1 or args.eval_every < 1
            or not math.isfinite(args.lr) or args.lr <= 0 or args.threads < 1):
        raise ValueError("invalid masked diffusion training configuration")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = load_model(args.config, args.runs_dir)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    training = masked_examples("train", args.train_per_length, seed=args.seed)
    validation = masked_examples("validation", args.validation_per_length, seed=args.seed)
    test = masked_examples("test", args.test_per_length, seed=args.seed)
    coverage = byte_coverage_examples(seed=args.seed)
    diagnostic = diagnostic_mask_record()
    real_holdouts = load_pairs(args.pairs, split="validation") + load_pairs(args.pairs, split="test")
    earlier_val = curriculum_examples(split="validation", count_per_stage=2, seed=881)
    processed = synthetic_holdouts(args.processed_dir)
    guard = HoldoutGuard.from_records(real_holdouts + earlier_val + processed + validation + test + coverage + [diagnostic])
    # Audit the entire fixed training set before the first optimizer update.
    for record in training:
        guard.validate(record["input"], record["output"], extra=(record["requested_bytes"],))
    print(f"preflight passed: {len(training)} training examples against {len(guard.contents)} protected byte hashes", flush=True)
    baseline = evaluate_masked(model, validation, seed=args.seed + 77, batch_size=args.batch_size, allow_untrained=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.core.edit_mask_embedding.requires_grad_(True)
    model.core.edit_value_proj.weight.requires_grad_(True)
    parameters = [model.core.edit_mask_embedding, model.core.edit_value_proj.weight]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in training:
        groups[len(record["input"])].append(record)
    rng = random.Random(args.seed + 1)
    losses = []
    logs = []
    best_rank = None
    best_state = None
    best_step = 0
    for step in range(1, args.steps + 1):
        length = rng.choice(sorted(groups))
        records = [rng.choice(groups[length]) for _ in range(args.batch_size)]
        ids, mask, request, target = tensor_batch(records)
        source_emb = model.bytes_to_emb(ids)
        target_emb = model.bytes_to_emb(target)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(target_emb, source_emb=source_emb, edit_mask=mask,
                                  requested_byte_ids=request, confidence_weight=0.)
        assert isinstance(loss, torch.Tensor)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite masked training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        model.mark_mask_edit_trained()
        losses.append(float(loss.detach()))
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate_masked(model, validation, seed=args.seed + 77, batch_size=args.batch_size)
            rank = (val["masked_byte_correct"], val["exact"])
            if best_rank is None or rank > best_rank:
                best_rank = rank; best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            log = {"step": step, "mean_recent_loss": sum(losses[-args.eval_every:]) / len(losses[-args.eval_every:]),
                   "validation_exact": val["exact"], "validation_count": val["count"],
                   "validation_masked_correct": val["masked_byte_correct"],
                   "validation_masked_count": val["masked_byte_count"]}
            logs.append(log)
            print(f"step {step}/{args.steps}: loss={log['mean_recent_loss']:.6f} "
                  f"validation bytes={val['masked_byte_correct']}/{val['masked_byte_count']} "
                  f"rows={val['exact']}/{val['count']}", flush=True)
    assert best_state is not None
    # Restore only the trainable adapter tensors. Loading the whole state
    # directly would bypass, and correctly be rejected by, a guarded concept
    # bank; no memory buffers or provenance are touched here.
    with torch.no_grad():
        model.core.edit_mask_embedding.copy_(best_state["core.edit_mask_embedding"])
        model.core.edit_value_proj.weight.copy_(best_state["core.edit_value_proj.weight"])
        model.mask_edit_trained.copy_(best_state["mask_edit_trained"])
    model.eval()
    changed = {"core.edit_mask_embedding", "core.edit_value_proj.weight", "mask_edit_trained"}
    for name, value in model.state_dict().items():
        if name not in changed and not torch.equal(value, original[name]):
            raise RuntimeError(f"frozen diffusion tensor changed: {name}")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    # Test targets are never passed into the sampler, and test does not select weights.
    tests = evaluate_masked(model, test, seed=args.seed + 88, batch_size=args.batch_size)
    coverage_results = evaluate_masked(model, coverage, seed=args.seed + 99, batch_size=args.batch_size)
    diagnostic_results = evaluate_masked(model, [diagnostic], seed=42, batch_size=1)
    report = {
        "configuration": vars(args), "source_model_sha256": _sha256(Path(args.runs_dir) / "model.pt"),
        "model_sha256": _sha256(output / "model.pt"), "selected_step": best_step,
        "selection_rule": "maximum validation masked-byte accuracy, then exact rows; no test-based selection",
        "training_examples": len(training), "mask_probability": .30, "training_updates": args.steps,
        "guard": {"fingerprint": guard.fingerprint, "protected_content_hashes": len(guard.contents),
                  "processed_holdout_records": len(processed), "preflighted_examples": len(training)},
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "new_trainable_parameters": sum(p.numel() for p in parameters),
                       "frozen_backbone_unchanged": True},
        "training_log": logs, "baseline_validation_untrained_mask_head": baseline,
        "selected_validation": evaluate_masked(model, validation, seed=args.seed + 77, batch_size=args.batch_size),
        "heldout_multimask_test": tests, "heldout_all_byte_values_test": coverage_results,
        "heldout_user_diagnostic": diagnostic_results,
        "limitations": ["Requested replacement values are supplied explicitly; a mask alone does not specify them.",
                        "Only the mask feature and requested-value projection were trained, on synthetic byte substitutions.",
                        "Held-out combinations are synthetic; this does not establish free-form instruction following or structural edits.",
                        "The original diffusion backbone has no padding attention mask; sampling width changes context.",
                        "Legacy confidence calibration is not valid for masked sampling; adaptive masked halting is disabled."],
    }
    report_path = output.parent / "diffusion_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"selected step {best_step}; multimask={tests['exact']}/{tests['count']} "
          f"coverage={coverage_results['exact']}/{coverage_results['count']}; "
          f"Hello={diagnostic_results['exact']}/1", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="runs/phase9/diffusion")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-per-length", type=int, default=128)
    parser.add_argument("--validation-per-length", type=int, default=8)
    parser.add_argument("--test-per-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=9090)
    parser.add_argument("--threads", type=int, default=1)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
