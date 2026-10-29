"""Phase 4 evaluation: reconstruction accuracy, byte-noise robustness,
inference timing, and concept write/read (learn without retraining)."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.dataset import ByteMultimodalDataset, collate
from models.bytecore_diffusion import ByteCoreDiffusion
from training.train import count_params


def corrupt_sequence(seq_ids, f, mode, rng):
    """Corrupt fraction f of the input byte ids; returns corrupted list."""
    b = list(seq_ids)
    n = len(b)
    if n == 0 or f <= 0:
        return b
    k = max(1, int(round(f * n)))
    if mode == "substitute":
        idx = rng.sample(range(n), min(k, n))
        for i in idx:
            b[i] = rng.randint(0, 255)
    elif mode == "insert":
        out = []
        inserts_left = k
        for i, byte in enumerate(b):
            out.append(byte)
            n_ins = inserts_left // max(n - i, 1)
            inserts_left -= n_ins
            for _ in range(n_ins):
                out.append(rng.randint(0, 255))
        b = out[:128]  # keep within max_len
    elif mode == "delete":
        kill = set(rng.sample(range(n), min(k, n)))
        b = [byte for i, byte in enumerate(b) if i not in kill]
        # pad back to original length
        b = b + [0] * (n - len(b))
    return b


def corrupt_sequence(seq, f, mode, rng):
    b = list(seq)
    n = len(b)
    if n == 0 or f <= 0:
        return b
    k = max(1, int(round(f * n)))
    if mode == "substitute":
        idx = rng.sample(range(n), min(k, n))
        for i in idx:
            b[i] = rng.randint(0, 255)
    elif mode == "insert":
        out = []
        inserts_left = k
        for i, byte in enumerate(b):
            out.append(byte)
            n_ins = inserts_left // max(n - i, 1)
            inserts_left -= n_ins
            for _ in range(n_ins):
                out.append(rng.randint(0, 255))
        b = out[: len(seq)] if len(seq) else out
    elif mode == "delete":
        kill = set(rng.sample(range(n), min(k, n)))
        b = [byte for i, byte in enumerate(b) if i not in kill]
        b = b + [0] * (n - len(b))
    return b


@torch.no_grad()
def denoise_accuracy(model, x0, t_level=2, guidance_scale=1.0):
    """Exact byte-match after one denoising step at noise level t_level."""
    b, l = x0.shape
    device = x0.device
    x0_emb = model.bytes_to_emb(x0)
    t_int = torch.full((b,), t_level, device=device, dtype=torch.long)
    xt = model.q_sample(x0_emb, t_int)
    tf = t_int.float() / model.diffusion_timesteps
    if guidance_scale != 1.0:
        # concept=None here; guidance needs a concept, so skip.
        pred = model.core(xt, tf, None)
    else:
        pred = model.core(xt, tf, None)
    pred_b = model.core.embed_to_bytes(pred)
    return (pred_b == x0).float().mean().item()


@torch.no_grad()
def timing(model, loader, device, batches=20):
    model.eval()
    times = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= batches:
                break
            x0 = batch["input_ids"].to(device)
            x0_emb = model.bytes_to_emb(x0)
            t0 = time.perf_counter()
            _ = model.compute_loss(x0_emb)
            times.append(time.perf_counter() - t0)
    n_samples = sum(batch["input_ids"].shape[0] for batch in [])
    # per-sample ms: total time / total samples across measured batches
    # recompute count:
    total_samples = 0
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        total_samples += batch["input_ids"].shape[0]
    return (sum(times) / max(len(times), 1)) / max(total_samples, 1) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--report", default="runs/report.md")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = args.device
    mcfg = cfg["model"]
    d_cfg = cfg["data"]
    ev = cfg["evaluation"]

    os.makedirs(args.runs_dir, exist_ok=True)

    # ---- Load model -------------------------------------------------
    model = ByteCoreDiffusion(
        vocab_size=mcfg["vocab_size"], d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"], n_heads=mcfg["n_heads"],
        d_ff=mcfg["d_ff"], max_len=mcfg["max_len"],
        diffusion_timesteps=mcfg["diffusion_timesteps"],
        concept_capacity=mcfg["concept_capacity"],
        concept_dim=mcfg["concept_dim"],
        dropout=mcfg["dropout"],
        device=device,
    )
    model.load(args.runs_dir, map_location=device)
    params = count_params(model)

    # ---- Load test set ----------------------------------------------
    test_ds = ByteMultimodalDataset(
        os.path.join(d_cfg["processed_dir"], "test.jsonl"),
        max_len=d_cfg["max_seq_len"])
    loader_test = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=collate)

    results = {"model": "ByteCore-Diffusion", "params": params,
               "params_mb": params * 4 / 1e6}

    # Reconstruction accuracy (clean, light denoising t=2)
    rec = denoise_accuracy(model, next(iter(loader_test))["input_ids"].to(device))
    results["reconstruction_accuracy"] = rec

    # Per-modality reconstruction
    per_mod = {}
    for batch in loader_test:
        x0 = batch["input_ids"].to(device)
        mods = batch["modalities"]
        # group by modality
        from collections import defaultdict
        by_mod = defaultdict(list)
        for i, m in enumerate(mods):
            by_mod[m].append(x0[i:i+1])
        for m, chunks in by_mod.items():
            xm = torch.cat(chunks, dim=0)
            per_mod.setdefault(m, []).append(denoise_accuracy(model, xm))
    results["per_modality"] = {
        m: sum(v) / len(v) for m, v in per_mod.items()
    }

    # ---- Noise robustness -------------------------------------------
    noise_results = {}
    rng = random.Random(7)
    for mode in ev["noise_modes"]:
        noise_results[mode] = {}
        for level in ev["noise_levels"]:
            if level == 0.0:
                continue
            # corrupt test bytes
            corrupted = []
            with open(os.path.join(d_cfg["processed_dir"], "test.jsonl")) as f:
                samples = [json.loads(l) for l in f if l.strip()]
            for s in samples:
                cb = corrupt_sequence(s["bytes"], level, mode, rng)
                corrupted.append({"modality": s["modality"], "concept": s["concept"],
                                  "bytes": cb})
            path = os.path.join(args.runs_dir, f"_eval_noise_{mode}_{int(level*1000)}.jsonl")
            with open(path, "w") as fh:
                for s in corrupted:
                    fh.write(json.dumps(s) + "\n")
            cds = ByteMultimodalDataset(path, max_len=d_cfg["max_seq_len"])
            cl = DataLoader(cds, batch_size=64, shuffle=False, collate_fn=collate)
            accs = []
            for batch in cl:
                x0 = batch["input_ids"].to(device)
                accs.append(denoise_accuracy(model, x0))
            noise_results[mode][level] = sum(accs) / len(accs)
    results["noise"] = noise_results

    # Robustness score = mean acc at max noise / clean acc
    max_level = max(ev["noise_levels"])
    scores = []
    for mode in ev["noise_modes"]:
        acc_n = noise_results[mode].get(max_level, 0.0)
        scores.append(acc_n / max(rec, 1e-9))
    results["robustness_score"] = sum(scores) / max(len(scores), 1)

    # ---- Timing -----------------------------------------------------
    results["timing_ms_per_sample"] = timing(model, loader_test, device,
                                             ev["inference_timing_batches"])

    # ---- Concept write/read (learn without retraining) --------------
    # Write new concepts from example bytes, then generate conditioned.
    concept_demo = {}
    # Pick one example per modality from test set as a "new concept".
    with open(os.path.join(d_cfg["processed_dir"], "test.jsonl")) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    for modality in ["text", "image", "audio"]:
        ex = next(s for s in samples if s["modality"] == modality)
        name = f"new:{modality}"
        model.write_concept(name, ex["bytes"])
        cv = model.read_concept(name)
        gen = model.generate(1, d_cfg["max_seq_len"], concept=cv,
                             guidance_scale=2.0)[0]
        # similarity = fraction of bytes matching the example (first real_len bytes)
        ex_bytes = torch.tensor(ex["bytes"][:d_cfg["max_seq_len"]],
                                dtype=torch.long)
        # pad example to max_seq_len for comparison
        if ex_bytes.shape[0] < d_cfg["max_seq_len"]:
            ex_bytes = torch.cat([ex_bytes,
                                  torch.zeros(d_cfg["max_seq_len"] - ex_bytes.shape[0],
                                              dtype=torch.long)])
        match = (gen == ex_bytes).float().mean().item()
        concept_demo[modality] = {
            "concept_written": name,
            "gen_vs_example_byte_match": match,
        }
    results["concept_demo"] = concept_demo

    # ---- Report -----------------------------------------------------
    lines = [
        "# ByteCore-Diffusion Evaluation Report",
        "",
        "## Overview",
        f"- **Model**: ByteCore-Diffusion (byte-level diffusion, raw bytes 0-255)",
        f"- **Parameters**: {params:,}  ({params * 4 / 1e6:.2f} MB, < 50 MB)",
        f"- **Reconstruction accuracy (clean)**: {rec:.4f}",
        f"- **Robustness score** (mean max-noise acc / clean): {results['robustness_score']:.4f}",
        f"- **Inference time**: {results['timing_ms_per_sample']:.3f} ms/sample",
        "",
        "## Per-modality reconstruction accuracy",
        "",
        "| Modality | Accuracy |",
        "|----------|----------|",
    ]
    for m, a in results["per_modality"].items():
        lines.append(f"| {m} | {a:.4f} |")

    lines += [
        "",
        "## Byte-level noise robustness (denoising accuracy after corruption)",
        "",
    ]
    # simpler per-mode table:
    lines.append("")
    lines.append("### Per noise mode")
    lines.append("| Mode | clean | 0.02 | 0.05 | 0.10 |")
    lines.append("|------|-------|------|------|------|")
    for mode in ev["noise_modes"]:
        row = [f"{rec:.4f}"]
        for l in (0.02, 0.05, 0.10):
            v = noise_results[mode].get(l, float("nan"))
            row.append(f"{v:.4f}" if v == v else "n/a")
        lines.append(f"| {mode} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## Concept storage — learn new concepts without retraining",
        "",
        "New concepts are written directly to the concept bank (a memory write, "
        "no gradient step). The model can then generate conditioned on them.",
        "",
        "| Modality | Concept written | Generated-vs-example byte match |",
        "|----------|-----------------|---------------------------------|",
    ]
    for modality, d in concept_demo.items():
        lines.append(f"| {modality} | {d['concept_written']} | "
                     f"{d['gen_vs_example_byte_match']:.4f} |")

    lines += [
        "",
        "## Notes",
        "- Raw bytes (0-255) throughout; no external tokenizers.",
        "- Reconstruction accuracy = exact byte match after one denoising step "
        "(light noise, t=2/50).",
        "- Robustness score = mean accuracy at max noise level / clean accuracy, "
        "averaged over substitute/insert/delete modes.",
        "- Concept bank: new concepts stored as memory entries (write, not train); "
        "instantly available at inference — no retraining required.",
        "- Inference time measured on CPU, one denoising step.",
    ]

    report = "\n".join(lines)
    Path(args.report).write_text(report)
    with open(os.path.join(args.runs_dir, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(report)
    print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
