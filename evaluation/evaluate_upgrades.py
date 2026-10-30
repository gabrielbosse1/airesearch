"""Reproducible, narrow functional benchmarks for the four upgrade phases."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from models.bytecore_diffusion import ByteCoreDiffusion
from models.contradiction_graph import Fact
from models.memory_model import MemoryModel
from models.holdout import HoldoutGuard
from data.holdout_registry import synthetic_holdouts
from data.ingest_real_traces import load_pairs
from training.train_cognitive import evaluate as evaluate_cognitive
from training.train_confidence import collect_diffusion, collect_generator, get_loader, load_generator, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--generator", default="runs/dual_memory/generator.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--generator-heads", type=int, default=4)
    ap.add_argument("--eval-examples", type=int, default=300)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc, dc = cfg["model"], cfg["data"]
    torch.manual_seed(42)
    generator = load_generator(args.generator, args.device, args.generator_heads)
    diffusion = ByteCoreDiffusion(
        vocab_size=mc["vocab_size"], d_model=mc["d_model"], n_layers=mc["n_layers"],
        n_heads=mc["n_heads"], d_ff=mc["d_ff"], max_len=mc["max_len"],
        diffusion_timesteps=mc["diffusion_timesteps"], concept_capacity=mc["concept_capacity"],
        concept_dim=mc["concept_dim"], dropout=mc["dropout"], device=args.device,
    ).load(args.runs_dir, map_location=args.device).eval()
    memory = MemoryModel(d_model=generator.d_model, capacity=32).to(args.device)
    protected = synthetic_holdouts(dc["processed_dir"], max_length=dc["max_seq_len"])
    if Path("data/real_edit_pairs.jsonl").exists():
        protected += load_pairs("data/real_edit_pairs.jsonl", split="validation") + load_pairs("data/real_edit_pairs.jsonl", split="test")
    memory.set_holdout_guard(HoldoutGuard.from_records(protected))

    # This is the same fixed disjoint held-out split used by the included toy
    # trainer. It does not measure general-purpose editing or new task formats.
    payloads = [bytes(x) for x in itertools.product(b"abcdefgh", repeat=3)]
    random.Random(23).shuffle(payloads)
    canvases = [bytes([cmd]) + b":" + p for p in payloads[48:112] for cmd in b"IDRM"]
    t0 = time.perf_counter()
    cognitive = evaluate_cognitive(generator, canvases)
    cognitive["ms_per_task"] = (time.perf_counter() - t0) * 1000 / len(canvases)

    trace = torch.arange(2 * generator.d_model, dtype=torch.float32, device=args.device).reshape(2, generator.d_model)
    slot = memory.write_procedure(b"\x00\xffinput", b"\xff\x00output", trace, b"byte replay")
    replay = memory.execute_procedure(b"\x00\xffinput")
    assert replay.slot == slot and replay.output == b"\xff\x00output" and torch.equal(replay.latent_trace, trace)
    a = memory.write_fact(Fact(b"alice", b"same_as", b"alias-one"))
    b = memory.write_fact(Fact(b"alias-one", b"same_as", b"alias-two", valid_from=0, valid_to=20))
    c = memory.write_fact(Fact(b"alias-two", b"language", b"python", exclusive=True, valid_from=5, valid_to=15))
    claim = Fact(b"alice", b"language", b"rust")
    active = memory.explain_contradiction(claim, as_of=10)
    expired = memory.explain_contradiction(claim, as_of=20)
    assert active.status == "conflict" and active.evidence[0].path_slots == (a, b, c)
    assert expired.status == "unknown"

    loader = get_loader(os.path.join(dc["processed_dir"], "test.jsonl"), dc["max_seq_len"], args.eval_examples, 32)
    d_features, d_labels = collect_diffusion(diffusion, loader, seed=103)
    g_features, g_labels = collect_generator(generator, loader, seed=103)
    with torch.no_grad():
        d_logits = diffusion.core.confidence_head(d_features.to(args.device)).cpu()
        g_logits = generator.confidence_head(g_features.to(args.device)).cpu()
    d_temp = float(diffusion.core.confidence_head.temperature.item())
    g_temp = float(generator.confidence_head.temperature.item())

    torch.manual_seed(42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, stats = diffusion.generate(8, 16, confidence_threshold=0.9, min_steps=5, return_stats=True)
    adaptive = {"steps": stats.steps.cpu().tolist(), "mean_steps": float(stats.steps.float().mean()),
                "full_schedule_steps": diffusion.diffusion_timesteps,
                "warnings": [str(w.message) for w in caught]}
    report = {
        "cognitive_toy_held_out": cognitive,
        "procedure": {"round_trip": True, "latent_steps": len(replay.latent_trace), "slot": slot},
        "graph": {"active": active.status, "expired": expired.status, "path_slots": list(active.evidence[0].path_slots)},
        "confidence_denoising_test": metrics(d_logits, d_labels, d_temp),
        "confidence_next_byte_test": metrics(g_logits, g_labels, g_temp),
        "adaptive_sampling_unvalidated_example": adaptive,
        "parameters": {"generator_checkpoint_architecture": sum(p.numel() for p in generator.parameters()),
                       "diffusion": sum(p.numel() for p in diffusion.parameters())},
    }
    out = Path(args.runs_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "upgrade_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    d, g = report["confidence_denoising_test"], report["confidence_next_byte_test"]
    lines = [
        "# ByteCore-DualMemory Upgrade Verification", "",
        "## Cognitive loop (narrow synthetic task)",
        f"- Held-out exact results: {cognitive['exact_task_fraction']:.4f} on {cognitive['count']} tasks; "
        f"DONE fraction: {cognitive['done_fraction']:.4f}.",
        f"- Mean {args.device} wall time: {cognitive['ms_per_task']:.2f} ms/task.",
        "- Commands I/D/R/M operate only on a three-byte a–h payload; this is not general editing accuracy.", "",
        "## Procedural and graph memory",
        "- Exact byte-pattern replay and ordered latent trace round-trip: PASS.",
        f"- Multi-hop path {active.evidence[0].path_slots}: {active.status} at time 10; {expired.status} at time 20.", "",
        "## Held-out confidence measurements",
        f"- Diffusion q-sampled denoising: {d['count']} real-byte predictions, correctness {d['correct_fraction']:.4f}, Brier {d['brier']:.4f}, ECE-10 {d['ece_10_bins']:.4f}.",
        f"- Generator next byte without memory: {g['count']} real-byte predictions, correctness {g['correct_fraction']:.4f}, Brier {g['brier']:.4f}, ECE-10 {g['ece_10_bins']:.4f}.",
        "- Confidence was trained on separate training data and temperature-fitted on validation; "
        "the test set and metrics above do not validate scratchpad or ancestral-sampling halting.", "",
        "## Adaptive computation mechanism (not a quality benchmark)",
        f"- Eight ancestral samples at threshold 0.9 used {adaptive['steps']} steps (full schedule: {adaptive['full_schedule_steps']}).",
        "- These sampling states differ from the calibration data. A scope-mismatch warning is expected; "
        "no correctness or speed/quality claim is implied.",
    ]
    text = "\n".join(lines) + "\n"
    (out / "upgrade_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
