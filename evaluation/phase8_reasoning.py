"""Training-only diagnostic for edit-intent MSE versus a zero-prediction baseline.

This is not a held-out capability metric. It does not load validation or test
pairs, call autonomous generation, mutate the models, or tune checkpoints.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from data.edit_curriculum import curriculum_examples
from data.ingest_real_traces import load_pairs
from models.cognitive import action_from_record
from training.cognitive_train import cognitive_trace_supervision_loss, edit_intent_target
from training.rl_editing import _real_training_examples
from training.train_confidence import load_generator


@torch.no_grad()
def evaluate_train_intents(checkpoint: str, records: list[dict[str, Any]], *, device: str = "cpu",
                           heads: int = 4) -> dict[str, Any]:
    generator = load_generator(checkpoint, device, heads).eval()
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        if record.get("split") != "train":
            raise ValueError("this diagnostic only accepts training examples")
        _, mse, thought_count = cognitive_trace_supervision_loss(
            generator, record["input"], [action_from_record(a) for a in record["trace_actions"]],
            instruction_bytes=record.get("instruction_bytes"), target_bytes=record["output"], return_reasoning=True,
        )
        # Generated traces and real optimal traces each THINK once before edits,
        # so current == input at that point. Assert rather than mislabel a trace.
        think_indices = [i for i, a in enumerate(record["trace_actions"]) if a["token"] == generator.THINK]
        if thought_count != 1 or any(a["token"] == generator.EDIT_OP for a in record["trace_actions"][:think_indices[0]]):
            raise ValueError("zero-baseline diagnostic requires one pre-edit THINK")
        zero_mse = edit_intent_target(generator, record["input"], record["output"]).square().mean()
        grouped[record.get("curriculum_stage", "real_train")].append((float(mse), float(zero_mse)))
    result = {}
    for stage, values in grouped.items():
        result[stage] = {"count": len(values), "predictor_mse": sum(v[0] for v in values) / len(values),
                         "zero_prediction_mse": sum(v[1] for v in values) / len(values)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="runs/phase8/metrics.json")
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--selected", default="runs/phase8/generator.pt")
    parser.add_argument("--final", default="runs/phase8/last_epoch.pt")
    parser.add_argument("--output", default="runs/phase8/reasoning_train_diagnostic.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    protocol = json.loads(Path(args.metrics).read_text(encoding="utf-8"))["protocol"]
    records = curriculum_examples(split="train", count_per_stage=protocol["examples_per_stage"], seed=protocol["seed"])
    records += _real_training_examples(load_pairs(args.pairs, split="train"))
    result = {
        "protocol": "teacher-forced post-THINK MSE on training examples only; a zero vector is the baseline; no test inputs",
        "selected_checkpoint": args.selected, "final_checkpoint": args.final,
        "selected": evaluate_train_intents(args.selected, records, device=args.device),
        "final_epoch": evaluate_train_intents(args.final, records, device=args.device),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
