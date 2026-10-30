"""Separate caller-declared route decisions from real model edit outcomes."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate import corrupt_sequence
from models.bytecore_diffusion import ByteCoreDiffusion
from models.edit_router import EditRequest, EditRouter, route_edit
from training.train_cognitive import task
from training.train_confidence import load_generator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--generator", default="runs/dual_memory/generator.pt")
    p.add_argument("--output", default="runs/phase6/routing.json")
    p.add_argument("--payloads", type=int, default=64)
    p.add_argument("--denoise-examples", type=int, default=100)
    args = p.parse_args()
    if args.payloads < 1 or args.denoise_examples < 1:
        raise ValueError("sample counts must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text()); m = cfg["model"]
    d = ByteCoreDiffusion(vocab_size=m["vocab_size"], d_model=m["d_model"], n_layers=m["n_layers"],
                          n_heads=m["n_heads"], d_ff=m["d_ff"], max_len=m["max_len"],
                          diffusion_timesteps=m["diffusion_timesteps"], concept_capacity=m["concept_capacity"],
                          concept_dim=m["concept_dim"], dropout=m["dropout"], device="cpu").load(args.runs_dir).eval()
    g = load_generator(args.generator, "cpu", 4).eval()
    router = EditRouter(d, g, diffusion_pad_to=cfg["data"]["max_seq_len"])
    checks = [("denoise", None, "diffusion"), ("substitute", None, "diffusion"),
              ("replace", True, "diffusion"), ("replace", False, "cognitive"),
              ("insert", None, "cognitive"), ("delete", None, "cognitive"), ("move", None, "cognitive")]
    dispatch = [{"operation": op, "length_preserving": preserve, "expected": expected,
                 "actual": route_edit(EditRequest(b"x", op, preserve))} for op, preserve, expected in checks]
    payloads = [bytes(x) for x in itertools.product(b"abcdefgh", repeat=3)]; random.Random(23).shuffle(payloads)
    payloads = payloads[112:112+args.payloads]
    if len(payloads) != args.payloads:
        raise ValueError("too many payloads requested")
    mapping = {ord("I"): ("insert", None), ord("D"): ("delete", None),
               ord("R"): ("replace", True), ord("M"): ("move", None)}
    outcomes = {chr(k): {"count": 0, "exact": 0, "engine": None} for k in mapping}
    for i, payload in enumerate(payloads):
        for command in b"IDRM":
            source = bytes([command]) + b":" + payload
            expected = task(g, source)[1]
            op, preserve = mapping[command]
            torch.manual_seed(7500 + 4 * i + command)
            result = router.execute(EditRequest(source, op, preserve), max_steps=8)
            out = outcomes[chr(command)]; out["count"] += 1
            out["exact"] += int(result.completed and result.output == expected)
            out["engine"] = result.engine
    for value in outcomes.values():
        value["exact_fraction"] = value["exact"] / value["count"]

    data_dir = Path(cfg["data"]["processed_dir"])
    rows = {split: [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text().splitlines() if line.strip()]
            for split in ("train", "test")}
    train = {bytes(r["bytes"]) for r in rows["train"]}; seen = set(train); examples = []
    for row in rows["test"]:
        raw = bytes(row["bytes"])
        if raw and raw not in seen:
            examples.append(raw); seen.add(raw)
        if len(examples) == args.denoise_examples:
            break
    if not examples:
        raise ValueError("no distinct test examples")
    original_correct = corrupted_correct = total = 0
    for i, target in enumerate(examples):
        corrupted = bytes(corrupt_sequence(target, 0.10, "substitute", random.Random(9000+i)))
        torch.manual_seed(9100+i)
        output = router.execute(EditRequest(corrupted, "substitute")).output
        original_correct += sum(a == b for a, b in zip(output, target))
        corrupted_correct += sum(a == b for a, b in zip(corrupted, target))
        total += len(target)
    report = {"protocol": {"dispatch": "caller-declared operation and replacement length; not learned routing or edit success",
                           "synthetic": "held-out three-byte command-prefix tasks; no target/controller supplied",
                           "denoising": "10% synthetic substitutions on exact-byte-novel test inputs; output compared to original",
                           "limitation": "diffusion cannot obey target-directed substitution instructions and raises instead of silently ignoring them; real excerpts lack actionable instructions"},
              "dispatch": {"correct": sum(r["actual"] == r["expected"] for r in dispatch), "count": len(dispatch), "cases": dispatch},
              "synthetic_exact": outcomes,
              "substitution_denoising": {"examples": len(examples), "bytes": total,
                                          "corrupted_input_accuracy": corrupted_correct/total,
                                          "output_accuracy_vs_original": original_correct/total},
              "parameters_added": 0}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"dispatch": report["dispatch"], "synthetic_exact": outcomes,
                      "substitution_denoising": report["substitution_denoising"]}, indent=2))


if __name__ == "__main__":
    main()
