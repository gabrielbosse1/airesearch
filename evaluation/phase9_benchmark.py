"""CPU microbenchmark of the Phase 9 diagnostic models, excluding model load."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from evaluation.diagnostic_single_byte_edit import INSTRUCTION, SOURCE
from training.train_confidence import load_generator
from training.train_masked_diffusion import load_model


def summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {"runs": len(values), "median_ms": statistics.median(values) * 1000,
            "p95_ms": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))] * 1000,
            "mean_ms": statistics.mean(values) * 1000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/phase9/latency.json")
    args = parser.parse_args()
    if args.runs < 2:
        raise ValueError("at least two timed runs are required")
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    diffusion = load_model("configs/default.yaml", "runs/phase9/diffusion")
    cognitive = load_generator("runs/phase9/generator.pt", "cpu", 4).eval()
    ids = torch.tensor([list(SOURCE)], dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool); mask[0, 9] = True
    request = torch.zeros_like(ids); request[0, 9] = 100
    def diffusion_call():
        return diffusion.masked_edit(ids, mask, request)
    def cognitive_call():
        return cognitive.cognitive_generate(SOURCE, instruction_bytes=INSTRUCTION, max_steps=12)
    # Warm both pathways, then time single requests sequentially on CPU.
    for _ in range(3):
        diffusion_call(); cognitive_call()
    times = {"diffusion_masked_50_steps": [], "cognitive_natural_5_actions": []}
    for _ in range(args.runs):
        start = time.perf_counter(); diffusion_call(); times["diffusion_masked_50_steps"].append(time.perf_counter() - start)
        start = time.perf_counter(); cognitive_call(); times["cognitive_natural_5_actions"].append(time.perf_counter() - start)
    report = {"description": "single ten-byte input; CPU, one PyTorch thread; warm, model-load excluded",
              "seed": args.seed, "torch": torch.__version__, "measurements": {k: summary(v) for k, v in times.items()},
              "note": "Microbenchmarks vary by machine and do not measure throughput, accuracy, or I/O."}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
