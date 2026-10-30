"""Relate stored-slot importance to end-to-end marginal NLL utility."""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnostics_common import contexts, load_memory, score_generator, test_batches
from evaluation.plotting import canvas, scatter_panel, save
from training.train_confidence import load_generator


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable"); ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def selection_counts(g, m, batches):
    counts = collections.Counter(); all_slots = {}
    for batch in batches:
        ids = batch["input_ids"][:, :-1]; b, length = ids.shape; size = g.retro_chunk_size
        chunks = math.ceil(length / size)
        emb = g.byte_emb(ids)
        padded = F.pad(emb, (0, 0, 0, chunks*size-length)).view(b, chunks, size, g.d_model)
        flat = padded[:, :-1].mean(dim=2).reshape(-1, g.d_model)
        records, logits, _, importance = m._gather_candidates(m.query_encoder(flat))
        for idx, (tier, local, global_slot, _) in enumerate(records):
            all_slots[global_slot] = {"tier": tier, "local_slot": local, "importance": float(importance[idx])}
        if not records:
            continue
        best = logits.topk(min(g.memory_top_k, len(records)), dim=1).indices
        for i in range(b):
            valid_chunks = max(0, math.ceil(max(batch["real_lens"][i] - 1, 0) / size) - 1)
            for ci in range(valid_chunks):
                for idx in best[i*(chunks-1) + ci].tolist():
                    counts[records[idx][2]] += 1
    return all_slots, counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--generator", default="runs/dual_memory/generator.pt")
    p.add_argument("--memory", default="runs/dual_memory/memory.pt")
    p.add_argument("--output", default="runs/importance_validation.json")
    p.add_argument("--plot", default="runs/importance_validation.png")
    p.add_argument("--examples", type=int, default=128)
    p.add_argument("--slots", type=int, default=18)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.slots < 1:
        raise ValueError("slots must be positive")
    g = load_generator(args.generator, args.device, 4); m = load_memory(args.memory, args.device)
    batches = test_batches(args.config, args.examples, 16, args.device)
    slots, counts = selection_counts(g, m, batches)
    if not slots:
        raise ValueError("memory has no active slots")
    # Include at least one from each existing tier, then the most-retrieved slots.
    ordered = sorted(slots, key=lambda s: (-counts[s], -slots[s]["importance"], s))
    chosen = []
    for tier in ("episodic", "semantic", "procedural"):
        match = next((s for s in ordered if slots[s]["tier"] == tier), None)
        if match is not None and len(chosen) < args.slots:
            chosen.append(match)
    for slot in ordered:
        if len(chosen) >= args.slots:
            break
        if slot not in chosen:
            chosen.append(slot)
    baseline = score_generator(g, batches, contexts(g, m, batches))
    rows = []
    for slot in chosen:
        data = slots[slot]; bank = getattr(m, data["tier"]); local = data["local_slot"]
        was_used = bool(bank.used[local])
        bank.used[local] = False
        try:
            ablated = score_generator(g, batches, contexts(g, m, batches))
        finally:
            bank.used[local] = was_used
        rows.append({"global_slot": slot, **data, "retrieval_count": counts[slot],
                     "utility_delta_nll": ablated["nll"] - baseline["nll"],
                     "utility_delta_accuracy": baseline["accuracy"] - ablated["accuracy"]})
    xs = np.asarray([r["importance"] for r in rows]); ys = np.asarray([r["utility_delta_nll"] for r in rows])
    pearson = correlation(xs, ys); spearman = correlation(rank_average(xs), rank_average(ys))
    report = {"protocol": {"utility": "paired NLL after removing one slot and recomputing previous-chunk top-K, minus full-memory NLL; positive means helpful",
                           "scope": "selected active slots in one saved memory and one held-out multimodal set",
                           "selection": "at least one per tier, then most frequently retrieved; correlation is descriptive, not causal extrapolation",
                           "caution": "Unused slots and replacement effects create zeros; significance is not established."},
              "baseline_nll": baseline["nll"], "selected_slots": len(rows), "pearson": pearson, "spearman": spearman,
              "slots": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    image, draw = canvas("Memory importance versus marginal utility", "Slot removal + top-K replacement on the same held-out byte predictions; small samples are descriptive")
    max_y = max(max((abs(y) for y in ys), default=0), 1e-6) * 1.2
    xmax = max(1.0, float(max((r["retrieval_count"] for r in rows), default=0)) * 1.05)
    scatter_panel(draw, (30, 115, 785, 905), f"Importance vs utility (Pearson {pearson:.3f})" if pearson is not None else "Importance vs utility (correlation undefined)",
                  zip(xs, ys), x_label="Importance score", y_label="NLL increase", x_range=(0, 1), y_range=(-max_y, max_y))
    scatter_panel(draw, (815, 115, 1570, 905), "Retrieval frequency vs utility",
                  [(r["retrieval_count"], r["utility_delta_nll"]) for r in rows], x_label="Top-K selections", y_label="NLL increase",
                  x_range=(0, xmax), y_range=(-max_y, max_y), color="#d36a3b")
    save(image, args.plot)
    print(json.dumps({"slots": len(rows), "pearson": pearson, "spearman": spearman,
                      "retrieved": sum(r["retrieval_count"] > 0 for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
