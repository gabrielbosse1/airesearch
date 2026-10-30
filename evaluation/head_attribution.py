"""Attribute task NLL to RETRO cross-attention heads via output ablation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnostics_common import contexts, load_memory, score_generator, test_batches
from evaluation.plotting import canvas, font, save
from training.train_confidence import load_generator


def draw_heatmap(rows, tasks, path):
    img, draw = canvas("Memory attention head attribution", "Delta held-out next-byte NLL when one RETRO cross-attention head is zeroed (positive = helped)")
    left, top, right, bottom = 75, 120, 1525, 915
    draw.rounded_rectangle((left, top, right, bottom), radius=12, fill="white", outline="#dce4ee", width=2)
    x0, y0, cw, ch = 320, 210, 330, min(68, 600 // max(len(rows), 1))
    values = [r["delta_nll"][t] for r in rows for t in tasks]
    scale = max(max((abs(x) for x in values), default=0), 1e-8)
    for j, task in enumerate(tasks):
        draw.text((x0 + j*cw + 90, y0 - 50), task.title(), font=font(22), fill="#344c63")
    for i, row in enumerate(rows):
        y = y0 + i*ch
        draw.text((110, y + 12), f"Layer {row['layer']} / head {row['head']}", font=font(19), fill="#344c63")
        for j, task in enumerate(tasks):
            value = row["delta_nll"][task]; strength = min(abs(value) / scale, 1)
            if value >= 0:
                base = (27, 130, 150)
            else:
                base = (210, 96, 64)
            color = tuple(round(245*(1-strength) + c*strength) for c in base)
            x = x0 + j*cw
            draw.rounded_rectangle((x, y, x+cw-12, y+ch-8), radius=8, fill=color)
            draw.text((x + 90, y + 12), f"{value:+.5f}", font=font(20),
                      fill="#ffffff" if strength > .55 else "#1d3448")
    draw.text((110, 850), f"Color scale +/- {scale:.5f} nats; zeroing a cross-head leaves prefix tokens and self-attention intact.",
              font=font(18), fill="#54677c")
    save(img, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--generator", default="runs/dual_memory/generator.pt")
    p.add_argument("--memory", default="runs/dual_memory/memory.pt")
    p.add_argument("--output", default="runs/head_attribution.json")
    p.add_argument("--plot", default="runs/head_attribution.png")
    p.add_argument("--examples", type=int, default=128)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    g = load_generator(args.generator, args.device, 4); m = load_memory(args.memory, args.device)
    batches = test_batches(args.config, args.examples, 16, args.device)
    mem = contexts(g, m, batches)
    baseline = score_generator(g, batches, mem, collect_attention=True)
    tasks = sorted(baseline["by_modality"])
    layers = sorted(g.memory_adapters.keys(), key=int)
    rows = []
    for layer in layers:
        adapter = g.memory_adapters[layer]
        for head in range(adapter.cross_attn.n_heads):
            dim = adapter.cross_attn.head_dim
            def mask_head(module, inputs, head=head, dim=dim):
                x = inputs[0]
                gate = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
                gate[head*dim:(head+1)*dim] = 0
                return (x * gate,)
            handle = adapter.cross_attn.out_proj.register_forward_pre_hook(mask_head)
            try:
                ablated = score_generator(g, batches, mem)
            finally:
                handle.remove()
            idx = layers.index(layer)
            rows.append({"layer": int(layer), "head": head,
                         "delta_nll": {task: ablated["by_modality"][task]["nll"] - baseline["by_modality"][task]["nll"] for task in tasks},
                         "delta_accuracy": {task: ablated["by_modality"][task]["accuracy"] - baseline["by_modality"][task]["accuracy"] for task in tasks},
                         "mean_real_memory_attention_mass": {task: baseline["mean_memory_attention_mass"][task][idx][head] for task in tasks}})
    report = {"protocol": "Inference-time zeroing of one RETRO cross-attention output head before its output projection; prefix and self-attention stay on. Negative delta means ablation improved NLL. Attention mass is descriptive, not causal.",
              "examples": args.examples, "tasks": tasks, "baseline": baseline, "heads": rows}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2) + "\n")
    draw_heatmap(rows, tasks, args.plot)
    print(f"Measured {len(rows)} heads across {len(tasks)} modalities; wrote {args.output} and {args.plot}.")


if __name__ == "__main__":
    main()
